# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Walk-forward backtest Sniper:
- Carrega todos os period_* de um wf_XXX
- Para cada timestamp, escolhe o modelo cujo train_end_utc está antes do timestamp
  (preferindo o menor tail válido, i.e., o mais "recente" que não vazou)
- Roda o ciclo usando p_entry/p_danger pré-computados em batch
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

from .sniper_simulator import (
    SniperBacktestResult,
    SniperTrade,
    _finalize_monthly_returns,
)
from modules.config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


try:
    # quando importado como pacote (ex.: backtest.*)
    from modules.trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:
    try:
        # fallback legado (quando o repo_root está no sys.path com nome de pacote)
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # type: ignore[import]
    except Exception:
        # modo comum: rodar a partir do repo_root (sys.path inclui modules/)
        from modules.trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # type: ignore[import]


@dataclass
class PeriodModel:
    period_days: int
    train_end_utc: pd.Timestamp
    entry_model_long: Any
    entry_model_short: Any
    entry_models_long: dict[str, Any]
    entry_models_short: dict[str, Any]
    # Campos abaixo têm defaults para manter compatibilidade com loaders legados
    # e evitar erro de dataclass por ordem de argumentos default/non-default.
    entry_model: Any | None = None
    entry_models: dict[str, Any] | None = None
    danger_model: Any | None = None
    exit_model: Any | None = None
    entry_cols: List[str] = field(default_factory=list)
    entry_cols_map_long: dict[str, List[str]] = field(default_factory=dict)
    entry_cols_map_short: dict[str, List[str]] = field(default_factory=dict)
    entry_cols_map: dict[str, List[str]] | None = None
    entry_pred_bias_map_long: dict[str, float] = field(default_factory=dict)
    entry_pred_bias_map_short: dict[str, float] = field(default_factory=dict)
    entry_pred_bias_map: dict[str, float] | None = None
    entry_calibration_map_long: dict[str, dict[str, float]] = field(default_factory=dict)
    entry_calibration_map_short: dict[str, dict[str, float]] = field(default_factory=dict)
    entry_calibration_map: dict[str, dict[str, float]] | None = None
    # marca quando o "entry_model" veio de classificador (fallback cls-only)
    entry_from_cls_map_long: dict[str, bool] = field(default_factory=dict)
    entry_from_cls_map_short: dict[str, bool] = field(default_factory=dict)
    entry_cls_model_long: Any | None = None
    entry_cls_model_short: Any | None = None
    entry_cls_models_long: dict[str, Any] = field(default_factory=dict)
    entry_cls_models_short: dict[str, Any] = field(default_factory=dict)
    entry_cls_cols_map_long: dict[str, List[str]] = field(default_factory=dict)
    entry_cls_cols_map_short: dict[str, List[str]] = field(default_factory=dict)
    entry_cls_calibration_map_long: dict[str, dict[str, float]] = field(default_factory=dict)
    entry_cls_calibration_map_short: dict[str, dict[str, float]] = field(default_factory=dict)
    danger_cols: List[str] = field(default_factory=list)
    exit_cols: List[str] = field(default_factory=list)
    tau_entry_long: float = 0.0
    tau_entry_short: float = 0.0
    tau_entry_long_map: dict[str, float] = field(default_factory=dict)
    tau_entry_short_map: dict[str, float] = field(default_factory=dict)
    tau_entry: float | None = None
    tau_entry_map: dict[str, float] | None = None
    tau_danger: float = 1.0
    tau_exit: float = 1.0
    tau_add: float = 0.0
    tau_danger_add: float = 1.0
    entry_label_scale: float = 1.0


def _is_catboost_model_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(4)
        return head.startswith(b"CBM")
    except Exception:
        return False


def _load_booster(path_json: Path) -> Any:
    ubj = path_json.with_suffix(".ubj")
    use_path = ubj if ubj.exists() and ubj.stat().st_size > 0 else path_json
    if _is_catboost_model_file(use_path):
        if CatBoostRegressor is None:
            raise RuntimeError("catboost nao instalado para carregar modelo")
        model = CatBoostRegressor()
        model.load_model(str(use_path))
        return model
    if xgb is None:
        raise RuntimeError("xgboost nao instalado para carregar modelo")
    bst = xgb.Booster()
    bst.load_model(str(use_path))
    return bst


def _load_classifier_model(path_json: Path) -> Any:
    ubj = path_json.with_suffix(".ubj")
    use_path = ubj if ubj.exists() and ubj.stat().st_size > 0 else path_json
    if _is_catboost_model_file(use_path):
        if CatBoostClassifier is None:
            raise RuntimeError("catboost nao instalado para carregar classificador")
        model = CatBoostClassifier()
        model.load_model(str(use_path))
        return model
    if xgb is None:
        raise RuntimeError("xgboost nao instalado para carregar classificador")
    bst = xgb.Booster()
    bst.load_model(str(use_path))
    return bst


def _entry_specs() -> list[tuple[str, int]]:
    windows = list(getattr(DEFAULT_TRADE_CONTRACT, "entry_label_windows_minutes", []) or [])
    if len(windows) < 1:
        raise ValueError("entry_label_windows_minutes deve ter ao menos 1 valor")
    return [(f"ret_exp_{int(w)}m", int(w)) for w in windows]


def select_entry_mid(p_entry_map: dict[str, np.ndarray]) -> np.ndarray:
    # fallback: primeiro disponível
    for v in p_entry_map.values():
        return v
    return np.array([], dtype=np.float32)


def load_period_models(
    run_dir: Path,
    *,
    tau_add_multiplier: float = 1.10,
    tau_danger_add_multiplier: float = 0.90,
) -> List[PeriodModel]:
    periods: List[PeriodModel] = []
    for pd_dir in sorted([p for p in Path(run_dir).iterdir() if p.is_dir() and p.name.startswith("period_") and p.name.endswith("d")]):
        period_days = int(pd_dir.name.replace("period_", "").replace("d", ""))
        meta_path = pd_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        train_end = meta.get("train_end_utc")
        if not train_end:
            continue
        train_end_ts = pd.to_datetime(train_end)

        entry_label_scale = float((meta.get("entry") or {}).get("label_scale") or 1.0)
        entry_models_long: dict[str, Any] = {}
        entry_models_short: dict[str, Any] = {}
        entry_cols_map_long: dict[str, list[str]] = {}
        entry_cols_map_short: dict[str, list[str]] = {}
        entry_pred_bias_map_long: dict[str, float] = {}
        entry_pred_bias_map_short: dict[str, float] = {}
        entry_calibration_map_long: dict[str, dict[str, float]] = {}
        entry_calibration_map_short: dict[str, dict[str, float]] = {}
        entry_from_cls_map_long: dict[str, bool] = {}
        entry_from_cls_map_short: dict[str, bool] = {}
        entry_cls_models_long: dict[str, Any] = {}
        entry_cls_models_short: dict[str, Any] = {}
        entry_cls_cols_map_long: dict[str, list[str]] = {}
        entry_cls_cols_map_short: dict[str, list[str]] = {}
        entry_cls_calibration_map_long: dict[str, dict[str, float]] = {}
        entry_cls_calibration_map_short: dict[str, dict[str, float]] = {}

        # tenta carregar modelos multi-janela a partir das pastas (long/short)
        for d in pd_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.startswith("entry_model_long_") and d.name.endswith("m"):
                w_str = d.name.replace("entry_model_long_", "").replace("m", "")
                if not w_str.isdigit():
                    continue
                w = int(w_str)
                mdir = d / "model_entry.json"
                if not mdir.exists():
                    continue
                name = f"ret_exp_{w}m"
                entry_models_long[name] = _load_booster(mdir)
                entry_from_cls_map_long[name] = False
                meta_key = f"entry_long_{int(w)}m"
                mk = meta.get(meta_key) or {}
                entry_cols_map_long[name] = list(mk.get("feature_cols") or meta["entry"]["feature_cols"])
                entry_pred_bias_map_long[name] = float(mk.get("pred_bias", 0.0) or 0.0)
                cal = mk.get("calibration") if isinstance(mk, dict) else None
                if isinstance(cal, dict):
                    try:
                        entry_calibration_map_long[name] = {
                            "a": float(cal.get("a", 1.0)),
                            "b": float(cal.get("b", 0.0)),
                            "s_pos": float(cal.get("s_pos", 1.0)),
                            "s_neg": float(cal.get("s_neg", 1.0)),
                        }
                    except Exception:
                        pass
            elif d.name.startswith("entry_model_short_") and d.name.endswith("m"):
                w_str = d.name.replace("entry_model_short_", "").replace("m", "")
                if not w_str.isdigit():
                    continue
                w = int(w_str)
                mdir = d / "model_entry.json"
                if not mdir.exists():
                    continue
                name = f"ret_exp_{w}m"
                entry_models_short[name] = _load_booster(mdir)
                entry_from_cls_map_short[name] = False
                meta_key = f"entry_short_{int(w)}m"
                mk = meta.get(meta_key) or {}
                entry_cols_map_short[name] = list(mk.get("feature_cols") or meta["entry"]["feature_cols"])
                entry_pred_bias_map_short[name] = float(mk.get("pred_bias", 0.0) or 0.0)
                cal = mk.get("calibration") if isinstance(mk, dict) else None
                if isinstance(cal, dict):
                    try:
                        entry_calibration_map_short[name] = {
                            "a": float(cal.get("a", 1.0)),
                            "b": float(cal.get("b", 0.0)),
                            "s_pos": float(cal.get("s_pos", 1.0)),
                            "s_neg": float(cal.get("s_neg", 1.0)),
                        }
                    except Exception:
                        pass
            elif d.name.startswith("entry_cls_model_long_") and d.name.endswith("m"):
                w_str = d.name.replace("entry_cls_model_long_", "").replace("m", "")
                if not w_str.isdigit():
                    continue
                w = int(w_str)
                mdir = d / "model_entry_gate.json"
                if not mdir.exists():
                    continue
                name = f"ret_exp_{w}m"
                entry_cls_models_long[name] = _load_classifier_model(mdir)
                mk = meta.get(f"entry_cls_long_{int(w)}m") or {}
                entry_cls_cols_map_long[name] = list(mk.get("feature_cols") or (meta.get("entry_cls") or {}).get("feature_cols") or meta["entry"]["feature_cols"])
                cal = mk.get("calibration") if isinstance(mk, dict) else None
                if isinstance(cal, dict):
                    try:
                        entry_cls_calibration_map_long[name] = {
                            "kind": str(cal.get("kind", "platt")),
                            "a": float(cal.get("a", 1.0)),
                            "b": float(cal.get("b", 0.0)),
                        }
                    except Exception:
                        pass
            elif d.name.startswith("entry_cls_model_short_") and d.name.endswith("m"):
                w_str = d.name.replace("entry_cls_model_short_", "").replace("m", "")
                if not w_str.isdigit():
                    continue
                w = int(w_str)
                mdir = d / "model_entry_gate.json"
                if not mdir.exists():
                    continue
                name = f"ret_exp_{w}m"
                entry_cls_models_short[name] = _load_classifier_model(mdir)
                mk = meta.get(f"entry_cls_short_{int(w)}m") or {}
                entry_cls_cols_map_short[name] = list(mk.get("feature_cols") or (meta.get("entry_cls") or {}).get("feature_cols") or meta["entry"]["feature_cols"])
                cal = mk.get("calibration") if isinstance(mk, dict) else None
                if isinstance(cal, dict):
                    try:
                        entry_cls_calibration_map_short[name] = {
                            "kind": str(cal.get("kind", "platt")),
                            "a": float(cal.get("a", 1.0)),
                            "b": float(cal.get("b", 0.0)),
                        }
                    except Exception:
                        pass

        # fallback cls-only: quando nao houver entry_model_*, usa entry_cls_* como score principal
        if (not entry_models_long) and entry_cls_models_long:
            for name, model in entry_cls_models_long.items():
                entry_models_long[name] = model
                entry_cols_map_long[name] = list(entry_cls_cols_map_long.get(name) or meta["entry"]["feature_cols"])
                entry_pred_bias_map_long[name] = 0.0
                entry_from_cls_map_long[name] = True
        if (not entry_models_short) and entry_cls_models_short:
            for name, model in entry_cls_models_short.items():
                entry_models_short[name] = model
                entry_cols_map_short[name] = list(entry_cls_cols_map_short.get(name) or meta["entry"]["feature_cols"])
                entry_pred_bias_map_short[name] = 0.0
                entry_from_cls_map_short[name] = True

        # fallback legado (single)
        if not entry_models_long and not entry_models_short:
            specs = _entry_specs()
            fallback_w = int(specs[0][1]) if specs else 60
            name = f"ret_exp_{fallback_w}m"
            legacy_model = _load_booster(pd_dir / "entry_model" / "model_entry.json")
            entry_models_long[name] = legacy_model
            entry_models_short[name] = legacy_model
            entry_cols_map_long[name] = list(meta["entry"]["feature_cols"])
            entry_cols_map_short[name] = list(meta["entry"]["feature_cols"])
            entry_pred_bias_map_long[name] = 0.0
            entry_pred_bias_map_short[name] = 0.0
            entry_from_cls_map_long[name] = False
            entry_from_cls_map_short[name] = False

        if not entry_models_long and entry_models_short:
            first_name = next(iter(entry_models_short.keys()))
            entry_models_long[first_name] = entry_models_short[first_name]
            entry_cols_map_long[first_name] = list(entry_cols_map_short.get(first_name) or meta["entry"]["feature_cols"])
            entry_pred_bias_map_long[first_name] = float(entry_pred_bias_map_short.get(first_name, 0.0) or 0.0)
            entry_from_cls_map_long[first_name] = bool(entry_from_cls_map_short.get(first_name, False))
        if not entry_models_short and entry_models_long:
            first_name = next(iter(entry_models_long.keys()))
            entry_models_short[first_name] = entry_models_long[first_name]
            entry_cols_map_short[first_name] = list(entry_cols_map_long.get(first_name) or meta["entry"]["feature_cols"])
            entry_pred_bias_map_short[first_name] = float(entry_pred_bias_map_long.get(first_name, 0.0) or 0.0)
            entry_from_cls_map_short[first_name] = bool(entry_from_cls_map_long.get(first_name, False))

        entry_model_long = list(entry_models_long.values())[0]
        entry_model_short = list(entry_models_short.values())[0]
        entry_cls_model_long = list(entry_cls_models_long.values())[0] if entry_cls_models_long else None
        entry_cls_model_short = list(entry_cls_models_short.values())[0] if entry_cls_models_short else None
        # modelos de danger/exit removidos (pipeline entry-only)
        danger_model = None
        exit_model = None
        entry_cols = list(meta["entry"]["feature_cols"])
        danger_cols: list[str] = []
        exit_cols: list[str] = []
        tau_entry = DEFAULT_THRESHOLD_OVERRIDES.tau_entry
        tau_danger = 1.0
        tau_exit = 1.0
        cls_as_entry = bool(
            any(entry_from_cls_map_long.values()) or any(entry_from_cls_map_short.values())
        )
        cls_default_thr = float((meta.get("entry_cls") or {}).get("positive_threshold", 50.0) or 50.0) / 100.0
        if cls_as_entry and tau_entry is not None and float(tau_entry) < 0.05:
            tau_entry = float(cls_default_thr)
        if tau_entry is None:
            tau_entry = float(meta["entry"].get("threshold", 0.5))
        scale_mult = float(entry_label_scale) if float(entry_label_scale) > 1.0 else 1.0
        tau_entry_scaled = float(tau_entry) * scale_mult
        tau_add = float(min(0.99 * scale_mult, max(0.01 * scale_mult, tau_entry_scaled * float(tau_add_multiplier))))
        tau_danger_add = 1.0
        tau_entry_long_map = {name: float(tau_entry_scaled) for name in entry_models_long.keys()}
        tau_entry_short_map = {name: float(tau_entry_scaled) for name in entry_models_short.keys()}

        periods.append(
            PeriodModel(
                period_days=period_days,
                train_end_utc=train_end_ts,
                entry_model_long=entry_model_long,
                entry_model_short=entry_model_short,
                entry_models_long=entry_models_long,
                entry_models_short=entry_models_short,
                entry_model=entry_model_long,
                entry_models=entry_models_long,
                danger_model=danger_model,
                exit_model=exit_model,
                entry_cols=entry_cols,
                entry_cols_map_long=entry_cols_map_long,
                entry_cols_map_short=entry_cols_map_short,
                entry_cols_map=entry_cols_map_long,
                entry_pred_bias_map_long=entry_pred_bias_map_long,
                entry_pred_bias_map_short=entry_pred_bias_map_short,
                entry_pred_bias_map=entry_pred_bias_map_long,
                entry_calibration_map_long=entry_calibration_map_long,
                entry_calibration_map_short=entry_calibration_map_short,
                entry_calibration_map=entry_calibration_map_long,
                entry_from_cls_map_long=entry_from_cls_map_long,
                entry_from_cls_map_short=entry_from_cls_map_short,
                entry_cls_model_long=entry_cls_model_long,
                entry_cls_model_short=entry_cls_model_short,
                entry_cls_models_long=entry_cls_models_long,
                entry_cls_models_short=entry_cls_models_short,
                entry_cls_cols_map_long=entry_cls_cols_map_long,
                entry_cls_cols_map_short=entry_cls_cols_map_short,
                entry_cls_calibration_map_long=entry_cls_calibration_map_long,
                entry_cls_calibration_map_short=entry_cls_calibration_map_short,
                danger_cols=danger_cols,
                exit_cols=exit_cols,
                tau_entry_long=float(tau_entry_scaled),
                tau_entry_short=float(tau_entry_scaled),
                tau_entry_long_map=tau_entry_long_map,
                tau_entry_short_map=tau_entry_short_map,
                tau_entry=float(tau_entry_scaled),
                tau_entry_map=tau_entry_long_map,
                tau_danger=tau_danger,
                tau_exit=tau_exit,
                tau_add=tau_add,
                tau_danger_add=tau_danger_add,
                entry_label_scale=float(entry_label_scale),
            )
        )
    # mais recente primeiro (menor tail) => train_end maior
    periods.sort(key=lambda p: p.train_end_utc, reverse=True)
    return periods


def apply_threshold_overrides(
    periods: List[PeriodModel],
    *,
    tau_entry: float | None = None,
    tau_add_multiplier: float = 1.10,
) -> List[PeriodModel]:
    """
    Aplica overrides (simulação) em todos os períodos, mantendo consistência de thresholds derivados.
    """
    out: List[PeriodModel] = []
    for pm in periods:
        scale_mult = float(pm.entry_label_scale) if float(pm.entry_label_scale) > 1.0 else 1.0
        te_raw = pm.tau_entry_long if tau_entry is None else float(tau_entry) * scale_mult
        ta = float(min(0.99 * scale_mult, max(0.01 * scale_mult, te_raw * float(tau_add_multiplier))))
        tau_entry_long_map = dict(pm.tau_entry_long_map or {})
        tau_entry_short_map = dict(pm.tau_entry_short_map or {})
        if tau_entry is not None:
            keys = list(tau_entry_long_map.keys()) or [n for n, _w in _entry_specs()]
            tau_entry_long_map = {k: float(te_raw) for k in keys}
            tau_entry_short_map = {k: float(te_raw) for k in keys}
        out.append(
            replace(
                pm,
                tau_entry_long=float(te_raw),
                tau_entry_short=float(te_raw),
                tau_entry_long_map=tau_entry_long_map,
                tau_entry_short_map=tau_entry_short_map,
                tau_add=float(ta),
            )
        )
    return out


def _build_matrix_rows(df: pd.DataFrame, cols: List[str], rows: np.ndarray) -> np.ndarray:
    """
    Monta matriz float32 apenas para as linhas `rows`, evitando `df.loc[mask]` (muito caro).
    """
    rows = np.asarray(rows, dtype=np.int64)
    mat = np.zeros((int(rows.size), len(cols)), dtype=np.float32)
    if rows.size == 0 or (len(cols) == 0):
        return mat
    df_cols = set(df.columns)
    for j, c in enumerate(cols):
        if c not in df_cols:
            continue
        v = df[c].to_numpy()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        mat[:, j] = v[rows]
    return mat


def _predict_entry_model(model: Any, X: np.ndarray) -> np.ndarray:
    if xgb is not None and isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X), validate_features=False)
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(np.float32, copy=False)
            if proba.ndim == 2 and proba.shape[1] == 1:
                return proba[:, 0].astype(np.float32, copy=False)
        except Exception:
            pass
    return model.predict(X)


def _predict_classifier_model(model: Any, X: np.ndarray) -> np.ndarray:
    if xgb is not None and isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(X), validate_features=False)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32, copy=False)
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0].astype(np.float32, copy=False)
    pred = np.asarray(model.predict(X), dtype=np.float32)
    if pred.ndim > 1:
        pred = pred.reshape(-1)
    return np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)


def _apply_cls_calibration(prob: np.ndarray, cal: dict[str, float] | None) -> np.ndarray:
    if not isinstance(cal, dict):
        return prob.astype(np.float32, copy=False)
    if str(cal.get("kind", "platt")).lower() != "platt":
        return prob.astype(np.float32, copy=False)
    a = float(cal.get("a", 1.0))
    b = float(cal.get("b", 0.0))
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    z = np.log(p / (1.0 - p))
    t = np.clip((a * z) + b, -40.0, 40.0)
    out = 1.0 / (1.0 + np.exp(-t))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _clip_entry_pred(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Mantem previsoes na mesma faixa do label quando configurado.
    Por padrao, clipa em [0, scale] para modo split_0_100.
    """
    try:
        use_clip = str(os.getenv("SNIPER_CLIP_ENTRY_PRED", "1")).strip().lower() not in {"0", "false", "no", "off"}
    except Exception:
        use_clip = True
    if not use_clip:
        return arr
    try:
        lo_env = os.getenv("SNIPER_ENTRY_PRED_CLIP_MIN", "").strip()
        hi_env = os.getenv("SNIPER_ENTRY_PRED_CLIP_MAX", "").strip()
        lo = float(lo_env) if lo_env else 0.0
        hi = float(hi_env) if hi_env else float(scale)
    except Exception:
        lo, hi = 0.0, float(scale)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = float(scale)
    if hi < lo:
        lo, hi = hi, lo
    return np.clip(arr, lo, hi).astype(np.float32, copy=False)


def predict_scores_walkforward(
    df: pd.DataFrame,
    *,
    periods: List[PeriodModel],
    return_period_id: bool = False,
    return_cls_maps: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel] | tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel, dict[str, np.ndarray], dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, PeriodModel, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Prediz p_entry/p_danger para cada timestamp, escolhendo o modelo mais recente com train_end_utc < t.
    Retorna p_entry por janela, p_danger e o primeiro período efetivamente usado.
    """
    idx = pd.to_datetime(df.index)
    n = len(idx)
    entry_names = sorted({name for pm in periods for name in (pm.entry_models_long or {}).keys()})
    if not entry_names:
        entry_names = [name for name, _w in _entry_specs()]
    p_entry_long_map = {name: np.full(n, np.nan, dtype=np.float32) for name in entry_names}
    p_entry_short_map = {name: np.full(n, np.nan, dtype=np.float32) for name in entry_names}
    p_entry_cls_long_map = {name: np.full(n, np.nan, dtype=np.float32) for name in entry_names}
    p_entry_cls_short_map = {name: np.full(n, np.nan, dtype=np.float32) for name in entry_names}
    p_danger = np.zeros(n, dtype=np.float32)  # danger removido
    # p_exit removido (mantido nulo para compatibilidade)
    p_exit = np.zeros(n, dtype=np.float32)
    period_id = np.full(n, -1, dtype=np.int16)

    used_any: PeriodModel | None = None

    for pid, pm in enumerate(periods):
        mask = idx > pm.train_end_utc
        # só preenche onde ainda não foi preenchido (usa base_name como proxy)
        base_name = entry_names[0] if entry_names else "mid"
        mask &= ~np.isfinite(p_entry_long_map.get(base_name, np.full(n, np.nan, dtype=np.float32)))
        if not mask.any():
            continue
        mask_np = np.asarray(mask, dtype=bool)
        rows = np.flatnonzero(mask_np)
        pdg = np.zeros(rows.size, dtype=np.float32)
        apply_bias = str(os.getenv("SNIPER_APPLY_PRED_BIAS", "0")).strip().lower() not in {"0", "false", "no", "off"}
        # entry por janela (long/short)
        for name in entry_names:
            model_long = pm.entry_models_long.get(name, pm.entry_model_long)
            model_short = pm.entry_models_short.get(name, pm.entry_model_short)
            cols_long = pm.entry_cols_map_long.get(name, pm.entry_cols)
            cols_short = pm.entry_cols_map_short.get(name, pm.entry_cols)
            src_cls_long = bool((pm.entry_from_cls_map_long or {}).get(name, False))
            src_cls_short = bool((pm.entry_from_cls_map_short or {}).get(name, False))
            X_e = _build_matrix_rows(df, cols_long, rows)
            pe_long = _predict_entry_model(model_long, X_e).astype(np.float32, copy=False)
            if src_cls_long:
                cls_scale = float(pm.entry_label_scale) if float(pm.entry_label_scale) > 1.0 else 1.0
                pe_long = np.clip(pe_long, 0.0, 1.0).astype(np.float32, copy=False) * np.float32(cls_scale)
            cal_l = pm.entry_calibration_map_long.get(name)
            if cal_l is not None:
                a = np.float32(float(cal_l.get("a", 1.0)))
                b = np.float32(float(cal_l.get("b", 0.0)))
                pe_long = (pe_long * a) + b
                try:
                    s_pos = float(cal_l.get("s_pos", 1.0))
                    s_neg = float(cal_l.get("s_neg", 1.0))
                    if s_pos != 1.0 or s_neg != 1.0:
                        mpos = pe_long >= 0
                        if np.any(mpos):
                            pe_long[mpos] = pe_long[mpos] * np.float32(s_pos)
                        if np.any(~mpos):
                            pe_long[~mpos] = pe_long[~mpos] * np.float32(s_neg)
                except Exception:
                    pass
            if apply_bias:
                pe_long = pe_long - np.float32(pm.entry_pred_bias_map_long.get(name, 0.0))
            pe_long = _clip_entry_pred(pe_long, pm.entry_label_scale)

            X_s = _build_matrix_rows(df, cols_short, rows)
            pe_short = _predict_entry_model(model_short, X_s).astype(np.float32, copy=False)
            if src_cls_short:
                cls_scale = float(pm.entry_label_scale) if float(pm.entry_label_scale) > 1.0 else 1.0
                pe_short = np.clip(pe_short, 0.0, 1.0).astype(np.float32, copy=False) * np.float32(cls_scale)
            cal_s = pm.entry_calibration_map_short.get(name)
            if cal_s is not None:
                a = np.float32(float(cal_s.get("a", 1.0)))
                b = np.float32(float(cal_s.get("b", 0.0)))
                pe_short = (pe_short * a) + b
                try:
                    s_pos = float(cal_s.get("s_pos", 1.0))
                    s_neg = float(cal_s.get("s_neg", 1.0))
                    if s_pos != 1.0 or s_neg != 1.0:
                        mpos = pe_short >= 0
                        if np.any(mpos):
                            pe_short[mpos] = pe_short[mpos] * np.float32(s_pos)
                        if np.any(~mpos):
                            pe_short[~mpos] = pe_short[~mpos] * np.float32(s_neg)
                except Exception:
                    pass
            if apply_bias:
                pe_short = pe_short - np.float32(pm.entry_pred_bias_map_short.get(name, 0.0))
            pe_short = _clip_entry_pred(pe_short, pm.entry_label_scale)
            p_entry_long_map[name][rows] = pe_long
            p_entry_short_map[name][rows] = pe_short
            if return_cls_maps:
                cls_long = pm.entry_cls_models_long.get(name)
                cls_short = pm.entry_cls_models_short.get(name)
                if cls_long is not None:
                    cols_cls_long = pm.entry_cls_cols_map_long.get(name, cols_long)
                    X_cl = _build_matrix_rows(df, cols_cls_long, rows)
                    p_cl = _predict_classifier_model(cls_long, X_cl).astype(np.float32, copy=False)
                    p_cl = _apply_cls_calibration(p_cl, pm.entry_cls_calibration_map_long.get(name))
                    p_entry_cls_long_map[name][rows] = np.clip(p_cl, 0.0, 1.0)
                if cls_short is not None:
                    cols_cls_short = pm.entry_cls_cols_map_short.get(name, cols_short)
                    X_cs = _build_matrix_rows(df, cols_cls_short, rows)
                    p_cs = _predict_classifier_model(cls_short, X_cs).astype(np.float32, copy=False)
                    p_cs = _apply_cls_calibration(p_cs, pm.entry_cls_calibration_map_short.get(name))
                    p_entry_cls_short_map[name][rows] = np.clip(p_cs, 0.0, 1.0)
        p_danger[rows] = pdg
        period_id[rows] = np.int16(pid)
        if used_any is None:
            used_any = pm

    if used_any is None:
        if len(periods) == 0:
            raise RuntimeError("Nenhum período encontrado (periods vazio).")
        t0 = pd.to_datetime(idx.min())
        t1 = pd.to_datetime(idx.max())
        te_max = max(p.train_end_utc for p in periods)
        te_min = min(p.train_end_utc for p in periods)
        raise RuntimeError(
            "Nenhum período do run_dir é válido para o range de timestamps fornecido. "
            f"df=[{t0}..{t1}] train_end_range=[{te_min}..{te_max}]"
        )
    if return_cls_maps and return_period_id:
        return p_entry_long_map, p_entry_short_map, p_danger, p_exit, used_any, period_id, p_entry_cls_long_map, p_entry_cls_short_map
    if return_cls_maps:
        return p_entry_long_map, p_entry_short_map, p_danger, p_exit, used_any, p_entry_cls_long_map, p_entry_cls_short_map
    if return_period_id:
        return p_entry_long_map, p_entry_short_map, p_danger, p_exit, used_any, period_id
    return p_entry_long_map, p_entry_short_map, p_danger, p_exit, used_any





def simulate_sniper_from_scores(
    df: pd.DataFrame,
    *,
    p_entry: np.ndarray,
    p_entry_short: np.ndarray | None = None,
    entry_best_win_mins: np.ndarray | None = None,
    entry_best_win_mins_short: np.ndarray | None = None,
    p_danger: np.ndarray,
    p_exit: np.ndarray | None = None,
    thresholds: PeriodModel,
    periods: List[PeriodModel] | None = None,
    period_id: np.ndarray | None = None,
    contract: TradeContract | None = None,
    candle_sec: int = 60,
    # Config explícita (sem depender de env vars)
    exit_min_hold_bars: int = 0,
    exit_confirm_bars: int = 1,
    tau_entry_long: float | None = None,
    tau_entry_short: float | None = None,
) -> SniperBacktestResult:
    """
    Simula ciclo Sniper usando scores pré-computados (mais rápido) e thresholds do período selecionado.
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    close = df["close"].to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    idx = df.index
    n = len(df)


    eq = 1.0
    eq_curve = np.ones(n, dtype=np.float64)
    trades: List[SniperTrade] = []

    in_pos = False
    pos_side: str | None = None
    entry_i = 0
    entry_price = 0.0
    avg_price = 0.0
    last_fill = 0.0
    total_size = 0.0
    num_adds = 0
    # confirmação simples do exit
    exit_min_hold = int(max(0, exit_min_hold_bars))
    exit_confirm = int(exit_confirm_bars)
    if exit_confirm <= 0:
        exit_confirm = 1
    exit_streak = 0
    ema_span = exit_ema_span_from_window(contract, int(candle_sec))
    use_ema_exit = ema_span > 0
    ema_alpha = 2.0 / float(ema_span + 1) if use_ema_exit else 0.0
    ema_offset = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)
    ema = 0.0
    ema_alpha_use = ema_alpha

    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))


    for i in range(n):
        pm = thresholds
        pid_i = 0
        if periods is not None and period_id is not None:
            try:
                pid_i = int(period_id[i])
                if 0 <= pid_i < len(periods):
                    pm = periods[pid_i]
            except Exception:
                pm = thresholds
                pid_i = 0

        px = close[i]
        if not np.isfinite(px) or px <= 0.0:
            eq_curve[i] = eq
            continue
        pe_long = float(p_entry[i]) if np.isfinite(p_entry[i]) else 0.0
        pe_short = 0.0
        if p_entry_short is not None and i < len(p_entry_short) and np.isfinite(p_entry_short[i]):
            pe_short = float(p_entry_short[i])

        if not in_pos:
            exit_streak = 0
            if i + int(exit_min_hold) >= (n - 1):
                eq_curve[i] = eq
                continue
            tau_long = float(tau_entry_long) if tau_entry_long is not None else float(pm.tau_entry_long)
            tau_short = float(tau_entry_short) if tau_entry_short is not None else float(pm.tau_entry_short)
            long_ok = pe_long >= tau_long
            short_ok = pe_short >= tau_short
            pick_side = None
            if long_ok and short_ok:
                pick_side = "long" if (pe_long - tau_long) >= (pe_short - tau_short) else "short"
            elif long_ok:
                pick_side = "long"
            elif short_ok:
                pick_side = "short"
            if pick_side is not None:
                in_pos = True
                pos_side = pick_side
                entry_i = i
                entry_price = px
                avg_price = px
                last_fill = px
                total_size = size_sched[0]
                num_adds = 0
                if use_ema_exit:
                    if pick_side == "short":
                        if entry_best_win_mins_short is not None:
                            try:
                                win = float(entry_best_win_mins_short[i])
                            except Exception:
                                win = 0.0
                        else:
                            win = 0.0
                    else:
                        if entry_best_win_mins is not None:
                            try:
                                win = float(entry_best_win_mins[i])
                            except Exception:
                                win = 0.0
                        else:
                            win = 0.0
                    if win > 0.0:
                        span_use = int(max(1, round((win * 60.0) / float(candle_sec))))
                        ema_alpha_use = 2.0 / float(span_use + 1)
                    ema = float(entry_price) * (1.0 - ema_offset)
            eq_curve[i] = eq
            continue

        time_in_trade = i - entry_i
        lo = low[i] if np.isfinite(low[i]) else px
        reason = None
        exit_px = None
        if use_ema_exit:
            ema = ema + (ema_alpha_use * (px - ema))
            if pos_side == "short":
                exit_streak = exit_streak + 1 if (px > ema) else 0
            else:
                exit_streak = exit_streak + 1 if (px < ema) else 0

        if use_ema_exit and (exit_streak >= exit_confirm):
            reason = "EMA"
            exit_px = px

        if reason is not None and exit_px is not None:
            entries = 1 + num_adds
            sides = entries + 1
            costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
            r_gross = (avg_price / exit_px) - 1.0 if pos_side == "short" else (exit_px / avg_price) - 1.0
            r_net = float(r_gross) - float(costs)
            eq = eq * (1.0 + r_net)
            trades.append(
                SniperTrade(
                    entry_ts=pd.to_datetime(idx[entry_i]),
                    exit_ts=pd.to_datetime(idx[i]),
                    entry_price=float(entry_price),
                    exit_price=float(exit_px),
                    num_adds=int(num_adds),
                    reason=str(reason),
                    r_net=float(r_net),
                    avg_entry_price=float(avg_price),
                    entries=int(entries),
                    sides=int(sides),
                    costs=float(costs),
                    r_gross=float(r_gross),
                    side=str(pos_side or "long"),
                )
            )
            in_pos = False
            pos_side = None
            exit_streak = 0
            entry_i = 0
            entry_price = 0.0
            avg_price = 0.0
            last_fill = 0.0
            total_size = 0.0
            num_adds = 0
            eq_curve[i] = eq
            continue

        # add logic com scores précomputados
        if pos_side != "short" and num_adds < int(contract.max_adds):
            trigger = last_fill * (1.0 - float(contract.add_spacing_pct))
            if trigger > 0 and lo <= trigger:
                next_size = size_sched[num_adds + 1]
                risk_after = 0.0
                if (pe_long >= pm.tau_add) and (risk_after <= contract.risk_max_cycle_pct + 1e-9):
                    new_total = total_size + next_size
                    avg_price = (avg_price * total_size + trigger * next_size) / new_total
                    total_size = new_total
                    last_fill = trigger
                    num_adds += 1

        eq_curve[i] = eq

    # métricas
    if in_pos and n > 0:
        exit_px = float(close[-1])
        entries = 1 + num_adds
        sides = entries + 1
        costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
        r_gross = (avg_price / exit_px) - 1.0 if (pos_side == "short" and avg_price > 0) else ((exit_px / avg_price) - 1.0 if avg_price > 0 else 0.0)
        r_net = float(r_gross) - float(costs)
        eq = eq * (1.0 + r_net)
        trades.append(
            SniperTrade(
                entry_ts=pd.to_datetime(idx[entry_i]),
                exit_ts=pd.to_datetime(idx[-1]),
                entry_price=float(entry_price),
                exit_price=float(exit_px),
                num_adds=int(num_adds),
                reason="EOD",
                r_net=float(r_net),
                avg_entry_price=float(avg_price),
                entries=int(entries),
                sides=int(sides),
                costs=float(costs),
                r_gross=float(r_gross),
                side=str(pos_side or "long"),
            )
        )
        eq_curve[-1] = eq

    eq_max = np.maximum.accumulate(eq_curve) if len(eq_curve) else eq_curve
    dd = (eq_max - eq_curve) / np.where(eq_max > 0, eq_max, 1.0) if len(eq_curve) else np.array([0.0])
    max_dd = float(np.nanmax(dd)) if len(dd) else 0.0
    dd2 = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    ulcer = float(np.sqrt(float(np.mean(dd2 * dd2)))) if len(dd2) else 0.0
    # maior sequência de drawdown (dd > 0)
    if len(dd2):
        x = (dd2 > 1e-12).astype(np.int8, copy=False)
        dx = np.diff(np.concatenate(([0], x, [0])))
        starts = np.flatnonzero(dx == 1)
        ends = np.flatnonzero(dx == -1)
        dd_dur = float((ends - starts).max()) / float(len(dd2)) if starts.size and ends.size else 0.0
    else:
        dd_dur = 0.0
    if trades:
        wins = [t.r_net for t in trades if t.r_net > 0]
        losses = [-t.r_net for t in trades if t.r_net <= 0]
        win_rate = float(len(wins) / len(trades))
        pf = float(sum(wins) / max(1e-12, sum(losses))) if losses else float("inf")
    else:
        win_rate = 0.0
        pf = 0.0
    monthly = _finalize_monthly_returns(df.index, eq_curve)
    return SniperBacktestResult(
        trades=trades,
        equity_curve=eq_curve,
        timestamps=df.index,
        monthly_returns=monthly,
        max_dd=max_dd,
        ulcer_index=float(ulcer),
        dd_duration_ratio=float(dd_dur),
        win_rate=win_rate,
        profit_factor=pf,
    )


__all__ = [
    "PeriodModel",
    "load_period_models",
    "predict_scores_walkforward",
    "select_entry_mid",
    "simulate_sniper_from_scores",
]
