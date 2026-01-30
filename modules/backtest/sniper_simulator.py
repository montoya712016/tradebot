# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Motor de backtest Sniper (ciclo stateful):
- Entrada apenas se EntryScore >= tau_entry
- Sem adds/danger/exit model (entry + EMA exit)
"""

from dataclasses import dataclass
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
from modules.config.thresholds import DEFAULT_THRESHOLD_OVERRIDES


@dataclass
class SniperModels:
    entry_model: "xgb.Booster"
    entry_models: Dict[str, "xgb.Booster"]
    danger_model: "xgb.Booster | None"  # legado (não usado)
    exit_model: "xgb.Booster | None"  # legado (não usado)
    entry_feature_cols: List[str]
    entry_feature_cols_map: Dict[str, List[str]]
    danger_feature_cols: List[str]
    exit_feature_cols: List[str]
    entry_calib: dict
    entry_calib_map: Dict[str, dict]
    danger_calib: dict
    exit_calib: dict
    tau_entry: float
    tau_entry_map: Dict[str, float]
    tau_danger: float
    tau_exit: float
    tau_add: float
    tau_danger_add: float
    entry_windows_minutes: Tuple[int, ...] | None = None


def _entry_specs() -> list[tuple[str, int]]:
    windows = list(getattr(DEFAULT_TRADE_CONTRACT, "entry_label_windows_minutes", []) or [])
    if len(windows) < 1:
        raise ValueError("entry_label_windows_minutes deve ter ao menos 1 valor")
    return [("mid", int(windows[0]))]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_calibration(p: np.ndarray, calib: dict) -> np.ndarray:
    if not isinstance(calib, dict):
        return p
    if calib.get("type") != "platt":
        return p
    a = float(calib.get("coef", 1.0))
    b = float(calib.get("intercept", 0.0))
    z = a * p + b
    return _sigmoid(z)


def _load_booster(path_json: Path) -> "xgb.Booster":
    bst = xgb.Booster()
    p_ubj = path_json.with_suffix(".ubj")
    if p_ubj.exists() and p_ubj.stat().st_size > 0:
        bst.load_model(str(p_ubj))
        return bst
    bst.load_model(str(path_json))
    try:
        bst.save_model(str(p_ubj))
    except Exception:
        pass
    return bst


def load_sniper_models(
    run_dir: Path,
    period_days: int,
    *,
    tau_add_multiplier: float = 1.10,
) -> SniperModels:
    """
    Carrega EntryScore e DangerScore + meta (features, calibração).

    tau_add e tau_danger_add são derivados do threshold principal por multiplicadores
    (thresholds definidos manualmente em config/thresholds.py).
    """
    pd_dir = Path(run_dir) / f"period_{int(period_days)}d"
    meta_path = pd_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    entry_path = pd_dir / "entry_model" / "model_entry.json"
    danger_path = pd_dir / "danger_model" / "model_danger.json"
    exit_path = pd_dir / "exit_model" / "model_exit.json"

    entry_models: Dict[str, xgb.Booster] = {}
    entry_feature_cols_map: Dict[str, List[str]] = {}
    entry_calib_map: Dict[str, dict] = {}
    for name, w in _entry_specs():
        p = pd_dir / f"entry_model_{int(w)}m" / "model_entry.json"
        if p.exists():
            entry_models[name] = _load_booster(p)
            meta_key = f"entry_{int(w)}m"
            entry_feature_cols_map[name] = list((meta.get(meta_key) or {}).get("feature_cols") or meta["entry"]["feature_cols"])
            entry_calib_map[name] = dict((meta.get(meta_key) or {}).get("calibration") or meta["entry"].get("calibration") or {"type": "identity"})

    if not entry_models:
        entry_models["mid"] = _load_booster(entry_path)
        entry_feature_cols_map["mid"] = list(meta["entry"]["feature_cols"])
        entry_calib_map["mid"] = dict(meta["entry"].get("calibration") or {"type": "identity"})

    entry_model = entry_models.get("mid") or list(entry_models.values())[0]
    danger_model = None
    exit_model = None

    entry_cols = list(meta["entry"]["feature_cols"])
    danger_cols: list[str] = []
    exit_cols: list[str] = []
    entry_calib = dict(meta["entry"].get("calibration") or {"type": "identity"})
    danger_calib: dict = {}
    exit_calib: dict = {}
    tau_entry = DEFAULT_THRESHOLD_OVERRIDES.tau_entry
    tau_danger = 1.0
    tau_exit = 1.0
    if tau_entry is None:
        tau_entry = float(meta["entry"].get("threshold", 0.5))

    tau_add = float(min(0.99, max(0.01, tau_entry * float(tau_add_multiplier))))
    tau_danger_add = 1.0
    tau_entry_map = {"mid": float(tau_entry)}
    entry_windows = tuple(int(w) for _n, w in _entry_specs())

    return SniperModels(
        entry_model=entry_model,
        entry_models=entry_models,
        danger_model=danger_model,
        exit_model=exit_model,
        entry_feature_cols=entry_cols,
        entry_feature_cols_map=entry_feature_cols_map,
        danger_feature_cols=danger_cols,
        exit_feature_cols=exit_cols,
        entry_calib=entry_calib,
        entry_calib_map=entry_calib_map,
        danger_calib=danger_calib,
        exit_calib=exit_calib,
        tau_entry=tau_entry,
        tau_entry_map=tau_entry_map,
        tau_danger=tau_danger,
        tau_exit=tau_exit,
        tau_add=tau_add,
        tau_danger_add=tau_danger_add,
        entry_windows_minutes=entry_windows,
    )


def _predict_1row(bst: "xgb.Booster", row: np.ndarray) -> float:
    # IMPORTANT (XGBoost CUDA):
    # `inplace_predict` com booster em CUDA e input numpy (CPU) gera warning
    # "mismatched devices" e cai em DMatrix internamente. Evitamos isso aqui.
    dm = xgb.DMatrix(row.reshape(1, -1))
    p = bst.predict(dm, validate_features=False)
    return float(p[0])


def _build_row_vector(
    df: pd.DataFrame,
    i: int,
    feat_cols: List[str],
    *,
    cycle_state: dict,
) -> np.ndarray:
    out = np.zeros(len(feat_cols), dtype=np.float32)
    for k, col in enumerate(feat_cols):
        if col.startswith("cycle_"):
            out[k] = float(cycle_state.get(col, 0.0))
            continue
        try:
            v = df[col].iat[i]
            out[k] = float(v) if np.isfinite(v) else 0.0
        except Exception:
            out[k] = 0.0
    return out


@dataclass
class SniperTrade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    num_adds: int
    reason: str
    r_net: float
    # Extras para diagnóstico/plots (opcionais para compat)
    avg_entry_price: float | None = None
    entries: int | None = None
    sides: int | None = None
    costs: float | None = None
    r_gross: float | None = None


@dataclass
class SniperBacktestResult:
    trades: List[SniperTrade]
    equity_curve: np.ndarray
    timestamps: pd.DatetimeIndex
    monthly_returns: List[float]
    max_dd: float
    win_rate: float
    profit_factor: float
    # métricas adicionais (baratas; calculadas junto com dd)
    ulcer_index: float = 0.0
    dd_duration_ratio: float = 0.0


def _max_true_run_np(flags: np.ndarray) -> int:
    """
    Maior sequência consecutiva de True em um array booleano.
    """
    if flags.size == 0:
        return 0
    x = flags.astype(np.int8, copy=False)
    # detecta inícios/fins de runs com diff
    dx = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(dx == 1)
    ends = np.flatnonzero(dx == -1)
    if starts.size == 0 or ends.size == 0:
        return 0
    lens = ends - starts
    return int(lens.max()) if lens.size else 0


def _finalize_monthly_returns(idx: pd.DatetimeIndex, eq: np.ndarray) -> List[float]:
    if len(idx) == 0:
        return []
    month_first: Dict[Tuple[int, int], float] = {}
    month_last: Dict[Tuple[int, int], float] = {}
    for t, e in zip(idx, eq):
        key = (int(t.year), int(t.month))
        month_first.setdefault(key, float(e))
        month_last[key] = float(e)
    out: List[float] = []
    for k in sorted(month_first.keys()):
        s = month_first[k]
        e = month_last.get(k, s)
        if s > 0:
            out.append((e / s) - 1.0)
    return out


def simulate_sniper_cycle(
    df: pd.DataFrame,
    *,
    models: SniperModels,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
    capital_per_cycle: float = 1.0,
    # Config explícita (sem depender de env vars)
    exit_min_hold_bars: int = 0,
    exit_confirm_bars: int = 1,
) -> SniperBacktestResult:
    """
    Simula um único símbolo usando ciclo stateful e os modelos treinados.
    Espera que `df` já contenha as features usadas no treino (mesmos nomes).
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    candle_sec = int(candle_sec or 60)

    close = df["close"].to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    idx = df.index
    n = len(df)

    eq = 1.0
    eq_curve = np.ones(n, dtype=np.float64)
    trades: List[SniperTrade] = []

    in_pos = False
    entry_i = 0
    entry_price = 0.0
    avg_price = 0.0
    last_fill = 0.0
    total_size = 0.0
    num_adds = 0
    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))

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
    ema_span_use = ema_span

    for i in range(n):
        px = close[i]
        if not np.isfinite(px) or px <= 0.0:
            eq_curve[i] = eq
            continue

        # estado do ciclo observável (somente entry; sem adds/danger)
        time_in_trade = (i - entry_i) if in_pos else 0
        dd_pct = (px / avg_price - 1.0) if (in_pos and avg_price > 0) else 0.0
        cycle_state = {
            "cycle_is_add": 0.0,
            "cycle_num_adds": 0.0,
            "cycle_time_in_trade": float(time_in_trade),
            "cycle_dd_pct": float(dd_pct),
            "cycle_avg_entry_price": float(avg_price if in_pos else px),
            "cycle_last_fill_price": float(last_fill if in_pos else px),
        }
        p_d = 0.0

        if not in_pos:
            exit_streak = 0
            if i + int(exit_min_hold) >= (n - 1):
                eq_curve[i] = eq
                continue
            best_name = "mid"
            best_pe = 0.0
            best_tau = float(models.tau_entry)
            best_win = None
            for name, w in _entry_specs():
                feat_cols = models.entry_feature_cols_map.get(name, models.entry_feature_cols)
                model = models.entry_models.get(name, models.entry_model)
                calib = models.entry_calib_map.get(name, models.entry_calib)
                x_e = _build_row_vector(df, i, feat_cols, cycle_state=cycle_state)
                p_e = _predict_1row(model, x_e)
                p_e = float(_apply_calibration(np.array([p_e], dtype=np.float64), calib)[0])
                if p_e >= best_pe:
                    best_pe = p_e
                    best_name = name
                    best_tau = float(models.tau_entry_map.get(name, models.tau_entry))
                    best_win = w
            if best_pe >= best_tau:
                in_pos = True
                entry_i = i
                entry_price = px
                avg_price = px
                last_fill = px
                total_size = size_sched[0]
                num_adds = 0
                if use_ema_exit:
                    ema_span_use = int(max(1, round((float(best_win or 0.0) * 60.0) / float(candle_sec)))) if best_win else ema_span
                    ema_alpha = 2.0 / float(ema_span_use + 1) if ema_span_use > 0 else 0.0
                    ema = float(entry_price) * (1.0 - ema_offset)
            eq_curve[i] = eq
            continue

        # in position: checa exits (ordem conservadora SL -> EMA -> timeout)
        lo = low[i] if np.isfinite(low[i]) else px
        reason = None
        exit_px = None
        if use_ema_exit:
            ema = ema + (ema_alpha * (px - ema))
            if px < ema:
                exit_streak += 1
            else:
                exit_streak = 0

        if use_ema_exit and (px < ema) and (exit_streak >= exit_confirm):
            reason = "EMA"
            exit_px = px

        if reason is not None and exit_px is not None:
            entries = 1 + num_adds
            sides = entries + 1
            costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
            r = (exit_px / avg_price) - 1.0
            r_net = r - costs
            eq = eq * (1.0 + capital_per_cycle * r_net)
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
                    r_gross=float(r),
                )
            )
            in_pos = False
            exit_streak = 0
            entry_i = 0
            entry_price = 0.0
            avg_price = 0.0
            last_fill = 0.0
            total_size = 0.0
            num_adds = 0
            eq_curve[i] = eq
            continue

        eq_curve[i] = eq

    # métricas
    if in_pos and n > 0:
        exit_px = float(close[-1])
        entries = 1 + num_adds
        sides = entries + 1
        costs = sides * (contract.fee_pct_per_side + contract.slippage_pct)
        r = (exit_px / avg_price) - 1.0 if avg_price > 0 else 0.0
        r_net = r - costs
        eq = eq * (1.0 + capital_per_cycle * r_net)
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
                r_gross=float(r),
            )
        )
        eq_curve[-1] = eq

    if len(eq_curve):
        eq_max = np.maximum.accumulate(eq_curve)
        dd = (eq_max - eq_curve) / np.where(eq_max > 0, eq_max, 1.0)
        max_dd = float(np.nanmax(dd))
        dd2 = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
        ulcer = float(np.sqrt(float(np.mean(dd2 * dd2))))
        dd_dur = float(_max_true_run_np(dd2 > 1e-12)) / float(len(dd2))
    else:
        max_dd = 0.0
        ulcer = 0.0
        dd_dur = 0.0
    if trades:
        wins = [t.r_net for t in trades if t.r_net > 0]
        losses = [-t.r_net for t in trades if t.r_net <= 0]
        win_rate = float(len(wins) / len(trades))
        pf = float(sum(wins) / max(1e-12, sum(losses))) if losses else float("inf")
    else:
        win_rate = 0.0
        pf = 0.0

    monthly = _finalize_monthly_returns(idx, eq_curve)
    return SniperBacktestResult(
        trades=trades,
        equity_curve=eq_curve.astype(np.float64, copy=False),
        timestamps=idx,
        monthly_returns=monthly,
        max_dd=max_dd,
        ulcer_index=float(ulcer),
        dd_duration_ratio=float(dd_dur),
        win_rate=win_rate,
        profit_factor=pf,
    )


__all__ = [
    "SniperModels",
    "SniperTrade",
    "SniperBacktestResult",
    "load_sniper_models",
    "simulate_sniper_cycle",
]


