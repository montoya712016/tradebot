# -*- coding: utf-8 -*-
"""
Treinamento dos modelos Sniper (EntryScore + ExitSpan) usando walk-forward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import hashlib
import json
import os
import pickle
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:
    raise

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except Exception:
    go = None  # type: ignore
    make_subplots = None  # type: ignore

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None  # type: ignore

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None  # type: ignore

try:
    from .sniper_dataflow import (
        prepare_sniper_dataset,
        prepare_sniper_dataset_from_cache,
        ensure_feature_cache,
        GLOBAL_FLAGS_FULL,
        SniperBatch,
        SniperDataPack,
    )
except Exception:
    import sys
    from pathlib import Path

    _HERE = Path(__file__).resolve()
    for _p in _HERE.parents:
        if _p.name.lower() == "modules":
            _sp = str(_p)
            if _sp not in sys.path:
                sys.path.insert(0, _sp)
            break
    from train.sniper_dataflow import (
        prepare_sniper_dataset,
        prepare_sniper_dataset_from_cache,
        ensure_feature_cache,
        GLOBAL_FLAGS_FULL,
        SniperBatch,
        SniperDataPack,
    )
try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]


try:
    from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

    SAVE_ROOT = _models_root_for_asset
except Exception:
    # fallback antigo
    SAVE_ROOT = None
DEFAULT_SYMBOLS_FILE = Path(__file__).resolve().parents[1] / "top_market_cap.txt"
DEFAULT_STOCKS_SYMBOLS_FILE = Path(__file__).resolve().parents[2] / "data" / "generated" / "tiingo_universe_seed.txt"

try:
    from config.symbols import load_market_caps
except Exception:
    # fallback para execução fora de pacote
    from config.symbols import load_market_caps  # type: ignore[import]


@dataclass
class TrainConfig:
    total_days: int = 365 * 5
    offsets_days: Sequence[int] = (180, 360, 540, 720, 900, 1_080, 1_260, 1_440, 1_620, 1_800)
    mcap_min_usd: float = 100_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    symbols_file: Path | None = None
    # 0 = sem limite (usa todas elegíveis por market cap)
    max_symbols: int = 0
    # Se o período ficar com poucos símbolos com dados válidos, pula o treino (evita modelo escasso).
    min_symbols_used_per_period: int = 30
    entry_params: dict = None
    exit_params: dict = None
    contract: TradeContract = DEFAULT_TRADE_CONTRACT
    # dataset sizing (VRAM/RAM)
    max_rows_entry: int = 2_000_000
    max_rows_exit: int = 2_000_000
    entry_ratio_neg_per_pos: float = 6.0
    use_feature_cache: bool = True
    # modo/asset
    asset_class: str = "crypto"
    # lista explícita de símbolos (prioritária a symbols_file/top_market_cap)
    symbols: Sequence[str] | None = None
    # flags de features custom (senão escolhe default por asset)
    feature_flags: dict | None = None
    # onde salvar/ler cache de features (senão usa padrão por asset)
    feature_cache_dir: Path | None = None
    use_full_entry_pool: bool = True
    full_pool_max_rows_entry: int = 4_000_000
    # Thresholds são definidos manualmente em config/thresholds.py (sem calibrar no treino).


DEFAULT_ENTRY_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 9,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    # Observação: QuantileDMatrix costuma usar 256 bins por padrão; manter consistente evita
    # erro "Inconsistent max_bin (512 vs 256)" e permite treinar na GPU.
    "max_bin": 256,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "device": "cuda:0",
}

DEFAULT_EXIT_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "max_bin": 256,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "device": "cuda:0",
}


def _default_symbols_file(asset_class: str) -> Path:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        return DEFAULT_STOCKS_SYMBOLS_FILE
    return DEFAULT_SYMBOLS_FILE


def _select_symbols(cfg: TrainConfig) -> List[str]:
    # prioridade: lista explícita
    if getattr(cfg, "symbols", None):
        raw = [str(s).strip().upper() for s in cfg.symbols if str(s).strip()]
        uniq: List[str] = []
        for s in raw:
            if s and s not in uniq:
                uniq.append(s)
        max_symbols = int(getattr(cfg, "max_symbols", 0) or 0)
        if max_symbols > 0:
            uniq = uniq[:max_symbols]
        return uniq

    asset = str(getattr(cfg, "asset_class", "crypto") or "crypto").lower()
    symbols_file = cfg.symbols_file or _default_symbols_file(asset)
    max_symbols = int(getattr(cfg, "max_symbols", 0) or 0)

    if asset == "stocks":
        if not symbols_file.exists():
            raise RuntimeError(f"Lista de símbolos para stocks não encontrada: {symbols_file}")
        lines = [line.strip().upper() for line in symbols_file.read_text(encoding="utf-8").splitlines()]
        syms: List[str] = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            tok = line.split(",", 1)[0].split(":", 1)[0].strip()
            if tok and tok not in syms:
                syms.append(tok)
            if max_symbols > 0 and len(syms) >= max_symbols:
                break
        if not syms:
            raise RuntimeError("Nenhum símbolo encontrado no arquivo de stocks")
        return syms

    cap_map = load_market_caps(symbols_file)
    lo = min(cfg.mcap_min_usd, cfg.mcap_max_usd)
    hi = max(cfg.mcap_min_usd, cfg.mcap_max_usd)
    # ordena por market cap desc
    ranked = sorted(cap_map.items(), key=lambda kv: kv[1], reverse=True)
    symbols: List[str] = []
    for sym, cap in ranked:
        if cap < lo or cap > hi:
            continue
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        symbols.append(sym)
        if max_symbols > 0 and len(symbols) >= max_symbols:
            break
    if not symbols:
        raise RuntimeError("Nenhum símbolo elegível para o intervalo de market cap")
    return symbols


def _split_train_val(batch: SniperBatch, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n = batch.X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(batch.ts)
    cut = max(1, int(round(n * (1 - val_frac))))
    tr_idx = order[:cut]
    va_idx = order[cut:]
    if va_idx.size == 0:
        va_idx = tr_idx[-max(1, min(1000, tr_idx.size // 5)) :]
        tr_idx = tr_idx[: tr_idx.size - va_idx.size]
    return tr_idx, va_idx


def _entry_objective_mode() -> str:
    v = str(os.getenv("SNIPER_ENTRY_OBJECTIVE_MODE", "binary") or "binary").strip().lower()
    if v in {"rank", "rank_pairwise", "pairwise"}:
        return "rank_pairwise"
    if v in {"rank_ndcg", "ndcg", "lambdarank"}:
        return "rank_ndcg"
    return "binary"


def _entry_rank_group_minutes() -> int:
    try:
        v = int(float(os.getenv("SNIPER_ENTRY_RANK_GROUP_MINUTES", "4320") or "4320"))
    except Exception:
        v = 4320
    return int(max(60, v))


def _build_entry_rank_qid(ts: np.ndarray, sym_id: np.ndarray, *, group_minutes: int) -> np.ndarray:
    if ts.size == 0:
        return np.empty((0,), dtype=np.uint32)
    gmin = int(max(1, group_minutes))
    ts_min = np.asarray(ts, dtype="datetime64[m]").astype(np.int64, copy=False)
    bucket = np.floor_divide(ts_min, np.int64(gmin)).astype(np.int64, copy=False)
    bucket0 = bucket - int(np.min(bucket))
    span = int(np.max(bucket0)) + 1 if bucket0.size else 1
    sid = np.asarray(sym_id, dtype=np.int64)
    qid = sid * np.int64(max(1, span)) + bucket0
    return np.asarray(qid, dtype=np.uint32)


def _sort_rank_arrays(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    qid: np.ndarray,
    w: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    order = np.lexsort((np.asarray(ts, dtype="datetime64[ns]").astype(np.int64, copy=False), np.asarray(qid, dtype=np.uint64)))
    Xo = X[order]
    yo = y[order]
    tso = ts[order]
    qido = qid[order]
    wo = (w[order] if w is not None else None)
    return Xo, yo, tso, qido, wo


def _entry_top_metric_specs() -> list[float]:
    raw = str(os.getenv("SNIPER_ENTRY_TOP_METRIC_QS", "0.001,0.0025,0.005") or "").strip()
    out: list[float] = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            q = float(tok)
        except Exception:
            continue
        if 0.0 < q < 1.0:
            out.append(float(q))
    if not out:
        out = [0.001, 0.0025, 0.005]
    uniq = sorted({float(q) for q in out})
    return uniq


def _precision_at_top_fraction(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    top_frac: float,
    min_count: int,
) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) <= 0:
        return float("nan")
    yt = yt[m]
    yp = yp[m]
    n = int(yt.size)
    if n <= 0:
        return float("nan")
    k = int(round(float(top_frac) * float(n)))
    k = max(int(min_count), k)
    if k > n:
        k = n
    if k <= 0:
        return float("nan")
    idx = np.argpartition(-yp, k - 1)[:k]
    return float(np.mean(yt[idx] >= 0.5))


def _entry_top_combo_metric(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> tuple[float, dict[str, float]]:
    qs = _entry_top_metric_specs()
    try:
        min_count = int(os.getenv("SNIPER_ENTRY_TOP_METRIC_MIN_COUNT", "64") or "64")
    except Exception:
        min_count = 64
    min_count = max(1, min_count)
    vals: list[float] = []
    parts: dict[str, float] = {}
    for q in qs:
        v = _precision_at_top_fraction(y_true, pred, top_frac=float(q), min_count=min_count)
        if np.isfinite(v):
            vals.append(float(v))
            parts[f"p_at_{q:.4f}"] = float(v)
    if not vals:
        return float("nan"), parts
    return float(np.mean(vals)), parts


def _slice_booster_to_best(booster: xgb.Booster, best_iteration: int | None) -> xgb.Booster:
    try:
        bi = int(best_iteration if best_iteration is not None else -1)
    except Exception:
        bi = -1
    if bi < 0:
        return booster
    try:
        return booster[: bi + 1]
    except Exception:
        return booster


def _entry_weight_bin_max() -> float:
    try:
        v = float(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_MAX", "") or "nan")
    except Exception:
        v = float("nan")
    if np.isfinite(v) and v > 0.0:
        return float(v)
    try:
        v2 = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX", "7.0") or "7.0")
    except Exception:
        v2 = 7.0
    if (not np.isfinite(v2)) or v2 <= 0.0:
        v2 = 7.0
    return float(v2)


def _build_entry_classifier_weights(
    ytr: np.ndarray,
    yva: np.ndarray,
    wtr: np.ndarray | None,
    wva: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ytr = np.asarray(ytr, dtype=np.float32)
    yva = np.asarray(yva, dtype=np.float32)
    wtr0 = np.asarray(wtr, dtype=np.float32) if wtr is not None else np.ones(ytr.size, dtype=np.float32)
    wva0 = np.asarray(wva, dtype=np.float32) if wva is not None else np.ones(yva.size, dtype=np.float32)
    wbin_max = _entry_weight_bin_max()
    wtr0 = np.clip(np.nan_to_num(wtr0, nan=0.0, posinf=wbin_max, neginf=0.0), 0.0, None).astype(np.float32, copy=False)
    wva0 = np.clip(np.nan_to_num(wva0, nan=0.0, posinf=wbin_max, neginf=0.0), 0.0, None).astype(np.float32, copy=False)

    enabled = str(os.getenv("SNIPER_ENTRY_CLASS_BALANCE_WEIGHTS", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    strength = float(os.getenv("SNIPER_ENTRY_CLASS_BALANCE_STRENGTH", "1.0") or "1.0")
    clip_min = float(os.getenv("SNIPER_ENTRY_CLASS_BALANCE_CLIP_MIN", "0.35") or "0.35")
    clip_max = float(os.getenv("SNIPER_ENTRY_CLASS_BALANCE_CLIP_MAX", "3.00") or "3.00")

    if not enabled:
        return wtr0, wva0, {"enabled": False}

    pos_tr = ytr >= 0.5
    neg_tr = ~pos_tr
    if (not bool(np.any(pos_tr))) or (not bool(np.any(neg_tr))):
        return wtr0, wva0, {"enabled": False, "reason": "single_class_train"}

    mu_pos = float(np.mean(wtr0[pos_tr]))
    mu_neg = float(np.mean(wtr0[neg_tr]))
    if (not np.isfinite(mu_pos)) or (not np.isfinite(mu_neg)) or mu_pos <= 0.0 or mu_neg <= 0.0:
        return wtr0, wva0, {"enabled": False, "reason": "invalid_class_weight_mean"}

    p = float(max(0.0, strength))
    tgt = float(np.sqrt(mu_pos * mu_neg))
    s_pos = float((tgt / mu_pos) ** p) if mu_pos > 0.0 else 1.0
    s_neg = float((tgt / mu_neg) ** p) if mu_neg > 0.0 else 1.0
    lo = float(max(0.05, min(clip_min, clip_max)))
    hi = float(max(lo, clip_max))
    s_pos = float(np.clip(s_pos, lo, hi))
    s_neg = float(np.clip(s_neg, lo, hi))

    wtr1 = np.asarray(wtr0, dtype=np.float64)
    wtr1[pos_tr] *= s_pos
    wtr1[neg_tr] *= s_neg
    wtr1 = wtr1.astype(np.float32, copy=False)

    wva1 = np.asarray(wva0, dtype=np.float64)
    pos_va = yva >= 0.5
    neg_va = ~pos_va
    if bool(np.any(pos_va)):
        wva1[pos_va] *= s_pos
    if bool(np.any(neg_va)):
        wva1[neg_va] *= s_neg
    wva1 = wva1.astype(np.float32, copy=False)

    mu_tr0 = float(np.mean(wtr0)) if wtr0.size else 1.0
    mu_va0 = float(np.mean(wva0)) if wva0.size else 1.0
    mu_tr1 = float(np.mean(wtr1)) if wtr1.size else 1.0
    mu_va1 = float(np.mean(wva1)) if wva1.size else 1.0
    if np.isfinite(mu_tr0) and np.isfinite(mu_tr1) and mu_tr1 > 0.0:
        wtr1 = (wtr1.astype(np.float64) * (mu_tr0 / mu_tr1)).astype(np.float32, copy=False)
    if np.isfinite(mu_va0) and np.isfinite(mu_va1) and mu_va1 > 0.0:
        wva1 = (wva1.astype(np.float64) * (mu_va0 / mu_va1)).astype(np.float32, copy=False)

    meta = {
        "enabled": True,
        "strength": float(p),
        "clip": [float(lo), float(hi)],
        "scale_pos": float(s_pos),
        "scale_neg": float(s_neg),
        "train_w_pos_mean_before": float(mu_pos),
        "train_w_neg_mean_before": float(mu_neg),
    }
    return wtr1, wva1, meta


def _weighted_positive_rate(y: np.ndarray, w: np.ndarray | None = None) -> float:
    yy = np.asarray(y, dtype=np.float64)
    if yy.size == 0:
        return float("nan")
    m = np.isfinite(yy)
    if int(m.sum()) <= 0:
        return float("nan")
    yb = (yy[m] >= 0.5).astype(np.float64, copy=False)
    if w is None:
        return float(np.mean(yb))
    ww0 = np.asarray(w, dtype=np.float64)
    if ww0.shape[0] != yy.shape[0]:
        return float(np.mean(yb))
    wbin_max = _entry_weight_bin_max()
    ww = np.clip(np.nan_to_num(ww0[m], nan=1.0, posinf=wbin_max, neginf=1.0), 1e-9, None)
    sw = float(np.sum(ww))
    if (not np.isfinite(sw)) or sw <= 0.0:
        return float(np.mean(yb))
    return float(np.sum(ww * yb) / sw)


def _normalize_entry_loss_weights(y: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float32)
    if w is None:
        return np.ones(yy.size, dtype=np.float32)
    ww = np.asarray(w, dtype=np.float64)
    if ww.shape[0] != yy.shape[0]:
        return np.ones(yy.size, dtype=np.float32)
    wbin_max = _entry_weight_bin_max()
    ww = np.clip(np.nan_to_num(ww, nan=0.0, posinf=wbin_max, neginf=0.0), 0.0, wbin_max)
    wpow = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER", "1.0") or "1.0")
    wpow_pos = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER_POS", str(wpow)) or str(wpow))
    wpow_neg = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER_NEG", str(wpow)) or str(wpow))
    pos = yy >= 0.5
    neg = ~pos
    if np.isfinite(wpow_pos) and wpow_pos > 0.0 and abs(wpow_pos - 1.0) > 1e-6 and bool(np.any(pos)):
        ww[pos] = np.power(ww[pos], wpow_pos)
    if np.isfinite(wpow_neg) and wpow_neg > 0.0 and abs(wpow_neg - 1.0) > 1e-6 and bool(np.any(neg)):
        ww[neg] = np.power(ww[neg], wpow_neg)
    class_norm = str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLASS_NORM", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    if class_norm:
        if bool(np.any(pos)) and bool(np.any(neg)):
            mu_pos = float(np.mean(ww[pos]))
            mu_neg = float(np.mean(ww[neg]))
            if np.isfinite(mu_pos) and np.isfinite(mu_neg) and mu_pos > 1e-12 and mu_neg > 1e-12:
                tgt = 0.5 * (mu_pos + mu_neg)
                s_pos = tgt / mu_pos
                s_neg = tgt / mu_neg
                ww[pos] *= s_pos
                ww[neg] *= s_neg
    mean_norm = str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_MEAN_NORM", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    target_mean = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_MEAN_TARGET", "1.0") or "1.0")
    if (not np.isfinite(target_mean)) or target_mean <= 0.0:
        target_mean = 1.0
    if mean_norm:
        # Opcional: mantem escala media controlada sem destruir a relacao entre bins.
        mu = float(np.mean(ww)) if ww.size else 1.0
        if np.isfinite(mu) and mu > 1e-12:
            ww = ww * (target_mean / mu)
    wmin = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MIN", "0.0") or "0.0")
    wmax = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX", str(wbin_max)) or str(wbin_max))
    wmax_pos = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_POS", str(wmax)) or str(wmax))
    wmax_neg = float(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_NEG", str(wmax)) or str(wmax))
    if wmax < wmin:
        wmax = wmin
    if wmax_pos < wmin:
        wmax_pos = wmin
    if wmax_neg < wmin:
        wmax_neg = wmin
    if bool(np.any(pos)):
        ww[pos] = np.clip(ww[pos], wmin, wmax_pos)
    if bool(np.any(neg)):
        ww[neg] = np.clip(ww[neg], wmin, wmax_neg)
    if (not np.isfinite(np.sum(ww))) or float(np.sum(ww)) <= 0.0:
        ww = np.ones(yy.size, dtype=np.float64)
    return ww.astype(np.float32, copy=False)


def _entry_weight_bin_table(y: np.ndarray, w: np.ndarray, *, step: float = 0.5) -> pd.DataFrame:
    yy = np.asarray(y, dtype=np.float32)
    ww = np.asarray(w, dtype=np.float64)
    if yy.size == 0 or ww.size != yy.size:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "count_pos", "count_neg", "mass_pos", "mass_neg"])
    st = float(step)
    if (not np.isfinite(st)) or st <= 0.0:
        st = 0.5
    wbin_max = _entry_weight_bin_max()
    n_bins = int(max(4, round(wbin_max / st)))
    wb = np.clip(np.nan_to_num(ww, nan=0.0, posinf=wbin_max, neginf=0.0), 0.0, wbin_max)
    b = np.floor(wb / st).astype(np.int32, copy=False)
    b = np.clip(b, 0, n_bins - 1)
    pos = yy >= 0.5
    neg = ~pos
    rows: list[dict] = []
    for i in range(n_bins):
        lo = float(i * st)
        hi = float(min(wbin_max, (i + 1) * st))
        m = b == i
        mp = m & pos
        mn = m & neg
        cp = int(np.sum(mp))
        cn = int(np.sum(mn))
        sp = float(np.sum(wb[mp])) if cp > 0 else 0.0
        sn = float(np.sum(wb[mn])) if cn > 0 else 0.0
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count_pos": cp,
                "count_neg": cn,
                "mass_pos": sp,
                "mass_neg": sn,
            }
        )
    return pd.DataFrame(rows)


def _entry_weight_bin_mass_str(y: np.ndarray, w: np.ndarray, *, step: float = 0.5) -> str:
    tb = _entry_weight_bin_table(y, w, step=step)
    if tb.empty:
        return "bins=na"
    nz = tb[(tb["count_pos"] + tb["count_neg"]) > 0].copy()
    if nz.empty:
        return "bins=empty"
    parts: list[str] = []
    for _, r in nz.iterrows():
        lo = float(r["bin_lo"])
        hi = float(r["bin_hi"])
        cp = int(r["count_pos"])
        cn = int(r["count_neg"])
        sp = float(r["mass_pos"])
        sn = float(r["mass_neg"])
        parts.append(f"[{lo:.1f},{hi:.1f}) p={cp}/{sp:.0f} n={cn}/{sn:.0f}")
    max_bins = int(os.getenv("SNIPER_ENTRY_WEIGHT_LOG_MAX_BINS", "16") or "16")
    if max_bins < 4:
        max_bins = 4
    if len(parts) <= max_bins:
        return " | ".join(parts)
    h = max_bins // 2
    t = max_bins - h
    shown = parts[:h] + ["..."] + parts[-t:]
    return " | ".join(shown)


def _save_entry_weight_bins_plot(
    tables: dict[str, pd.DataFrame],
    *,
    out_html: Path,
    title: str,
) -> None:
    if not tables:
        return
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Sempre salva CSV por cenario para auditoria/reprodutibilidade.
        for name, tb in tables.items():
            if tb is None or tb.empty:
                continue
            csv_path = out_html.with_name(f"{out_html.stem}_{name}.csv")
            tb.to_csv(csv_path, index=False)
    except Exception:
        pass
    if go is None or make_subplots is None:
        return
    names = [k for k, v in tables.items() if v is not None and not v.empty]
    if not names:
        return
    cols = len(names)
    fig = make_subplots(
        rows=2,
        cols=cols,
        subplot_titles=tuple(names + names),
        vertical_spacing=0.14,
        horizontal_spacing=0.05,
    )
    for ci, name in enumerate(names, start=1):
        tb = tables[name].copy()
        x = [f"{float(lo):.1f}-{float(hi):.1f}" for lo, hi in zip(tb["bin_lo"].to_numpy(), tb["bin_hi"].to_numpy())]
        cpos = tb["count_pos"].to_numpy(dtype=np.float64, copy=False)
        cneg = tb["count_neg"].to_numpy(dtype=np.float64, copy=False)
        mpos = tb["mass_pos"].to_numpy(dtype=np.float64, copy=False)
        mneg = tb["mass_neg"].to_numpy(dtype=np.float64, copy=False)
        fig.add_trace(
            go.Bar(x=x, y=cpos, name=f"{name}: count_pos", marker_color="#2ecc71", showlegend=(ci == 1)),
            row=1,
            col=ci,
        )
        fig.add_trace(
            go.Bar(x=x, y=cneg, name=f"{name}: count_neg", marker_color="#e74c3c", showlegend=(ci == 1)),
            row=1,
            col=ci,
        )
        fig.add_trace(
            go.Bar(x=x, y=mpos, name=f"{name}: mass_pos", marker_color="#27ae60", showlegend=(ci == 1)),
            row=2,
            col=ci,
        )
        fig.add_trace(
            go.Bar(x=x, y=mneg, name=f"{name}: mass_neg", marker_color="#c0392b", showlegend=(ci == 1)),
            row=2,
            col=ci,
        )
        fig.update_xaxes(tickangle=-65, row=1, col=ci)
        fig.update_xaxes(tickangle=-65, row=2, col=ci)
    fig.update_layout(
        title=str(title),
        barmode="group",
        template="plotly_dark",
        height=800,
        margin=dict(l=40, r=20, t=70, b=70),
    )
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(title_text="weight mass", row=2, col=1)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _train_xgb_classifier(batch: SniperBatch, params: dict) -> tuple[xgb.Booster, dict]:
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    device = str(params.get("device", "cpu")).lower()
    objective_mode = _entry_objective_mode()
    is_ranking = objective_mode in {"rank_pairwise", "rank_ndcg"}
    use_quantile = device.startswith("cuda") and (not is_ranking)
    Xtr = batch.X[tr_idx]
    ytr = batch.y[tr_idx]
    Xva = batch.X[va_idx]
    yva = batch.y[va_idx]
    use_label_weights = str(os.getenv("SNIPER_ENTRY_USE_LABEL_WEIGHTS", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    wtr = (batch.w[tr_idx] if (use_label_weights and getattr(batch, "w", None) is not None) else None)
    wva = (batch.w[va_idx] if (use_label_weights and getattr(batch, "w", None) is not None) else None)
    wtr_eff, wva_eff, wmeta = _build_entry_classifier_weights(ytr, yva, wtr, wva)
    loss_bin_stats: dict[str, object] = {}
    if use_label_weights:
        wtr_eff = _normalize_entry_loss_weights(ytr, wtr_eff)
        wva_eff = _normalize_entry_loss_weights(yva, wva_eff)
        try:
            bin_step = float(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_STEP_X10", "1") or "1") / 10.0
            tb_tr = _entry_weight_bin_table(ytr, wtr_eff, step=bin_step)
            tb_va = _entry_weight_bin_table(yva, wva_eff, step=bin_step)
            loss_bin_stats = {
                "step": float(bin_step),
                "train": tb_tr.to_dict(orient="records"),
                "val": tb_va.to_dict(orient="records"),
            }
            msg_tr = _entry_weight_bin_mass_str(ytr, wtr_eff, step=bin_step)
            msg_va = _entry_weight_bin_mass_str(yva, wva_eff, step=bin_step)
            print(f"[xgb] loss_w train bins: {msg_tr}", flush=True)
            print(f"[xgb] loss_w val   bins: {msg_va}", flush=True)
        except Exception:
            pass
    try:
        base_unw = _weighted_positive_rate(yva, None)
        base_w = _weighted_positive_rate(yva, wva_eff)
        if np.isfinite(base_unw) and np.isfinite(base_w):
            print(
                f"[xgb] val baseline: pos_rate={base_unw:.4f} weighted_pos_rate={base_w:.4f} class_w_balance={bool((wmeta or {}).get('enabled', False))}",
                flush=True,
            )
    except Exception:
        pass

    params = dict(params)
    if is_ranking:
        params["objective"] = ("rank:ndcg" if objective_mode == "rank_ndcg" else "rank:pairwise")
        params["eval_metric"] = str(os.getenv("SNIPER_ENTRY_RANK_EVAL_METRIC", "ndcg@32") or "ndcg@32").strip()
    else:
        # Blindagem: evita erro de base_score quando o dataset ficar com 1-classe por sampling.
        y_mean = float(np.mean(ytr)) if ytr.size else 0.5
        if not (0.0 < y_mean < 1.0):
            # Se acontecer, o treino não é informativo (sem negativos/positivos).
            # Não quebra: define base_score neutro e segue (ou você pode optar por "pular período").
            print(f"[xgb] AVISO: y_mean={y_mean:.6f} (1-classe). Ajustando base_score=0.5", flush=True)
            params["base_score"] = 0.5
        elif "base_score" not in params:
            # ajuda convergência / estabilidade
            eps = 1e-6
            params["base_score"] = float(min(1.0 - eps, max(eps, y_mean)))

    # Para GPU + hist, QuantileDMatrix reduz VRAM.
    # IMPORTANTE: o dvalid precisa referenciar o dtrain via `ref=...`.
    if is_ranking:
        group_minutes = _entry_rank_group_minutes()
        qid_tr = _build_entry_rank_qid(batch.ts[tr_idx], batch.sym_id[tr_idx], group_minutes=group_minutes)
        qid_va = _build_entry_rank_qid(batch.ts[va_idx], batch.sym_id[va_idx], group_minutes=group_minutes)
        Xtr, ytr, ts_tr, qid_tr, _ = _sort_rank_arrays(Xtr, ytr, batch.ts[tr_idx], qid_tr, None)
        Xva, yva, ts_va, qid_va, _ = _sort_rank_arrays(Xva, yva, batch.ts[va_idx], qid_va, None)
        dtrain = xgb.DMatrix(Xtr, label=ytr, qid=qid_tr)
        dvalid = xgb.DMatrix(Xva, label=yva, qid=qid_va)
        try:
            ngrp_tr = int(np.unique(qid_tr).size)
            ngrp_va = int(np.unique(qid_va).size)
            print(
                f"[xgb] ranking groups: train={ngrp_tr:,} val={ngrp_va:,} group_min={group_minutes}".replace(",", "."),
                flush=True,
            )
        except Exception:
            pass
        wtr_eff = None
        wva_eff = None
    elif use_quantile:
        mb = int(params.get("max_bin", 256))
        # garante consistência entre Booster params e QuantileDMatrix bins
        params["max_bin"] = mb
        try:
            # Nem todas as versões expõem max_bin no construtor; por isso try/except.
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr_eff, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva_eff, ref=dtrain, max_bin=mb)
        except TypeError:
            # fallback: usa default do QuantileDMatrix (geralmente 256) e ajusta params
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr_eff)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva_eff, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr_eff)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva_eff)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr_eff)
        dvalid = xgb.DMatrix(Xva, label=yva, weight=wva_eff)
    watch = [(dtrain, "train"), (dvalid, "val")]
    print(
        f"[xgb] train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,} objective={params.get('objective')}".replace(",", "."),
        flush=True,
    )
    metric_mode = str(os.getenv("SNIPER_ENTRY_MODEL_SELECTION", "aucpr") or "aucpr").strip().lower()
    custom_metric = None
    callbacks = []
    if metric_mode in {"top_precision", "top_precision_combo", "topq", "precision_top"}:
        def _xgb_top_precision_metric(predt: np.ndarray, dmat: xgb.DMatrix) -> tuple[str, float]:
            y_eval = dmat.get_label()
            score, _parts = _entry_top_combo_metric(y_eval, predt)
            if not np.isfinite(score):
                score = 0.0
            return "top_precision_combo", float(score)

        custom_metric = _xgb_top_precision_metric
        try:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=400,
                    metric_name="top_precision_combo",
                    data_name="val",
                    maximize=True,
                    save_best=True,
                )
            )
        except Exception:
            callbacks = []
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=watch,
            custom_metric=custom_metric,
            early_stopping_rounds=(None if callbacks else 200),
            callbacks=(callbacks or None),
            verbose_eval=50,
        )
    except Exception as e:
        # fallback adicional: se GPU falhar em runtime (driver/VRAM), tenta CPU
        if device.startswith("cuda"):
            print(f"[xgb] treino em GPU falhou ({type(e).__name__}: {e}) -> tentando CPU", flush=True)
            params_cpu = dict(params)
            params_cpu["device"] = "cpu"
            params_cpu["tree_method"] = "hist"
            if is_ranking:
                dtrain = xgb.DMatrix(Xtr, label=ytr, qid=qid_tr)
                dvalid = xgb.DMatrix(Xva, label=yva, qid=qid_va)
            else:
                dtrain = xgb.DMatrix(Xtr, label=ytr)
                dvalid = xgb.DMatrix(Xva, label=yva)
            watch = [(dtrain, "train"), (dvalid, "val")]
            booster = xgb.train(
                params=params_cpu,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=watch,
                custom_metric=custom_metric,
                early_stopping_rounds=(None if callbacks else 200),
                callbacks=(callbacks or None),
                verbose_eval=50,
            )
        else:
            raise
    best_iteration = int(getattr(booster, "best_iteration", -1))
    if metric_mode in {"top_precision", "top_precision_combo", "topq", "precision_top"} and best_iteration >= 0:
        booster = _slice_booster_to_best(booster, best_iteration)
    val_pred = booster.predict(dvalid)
    val_true = batch.y[va_idx]
    top_metric_value, top_metric_parts = _entry_top_combo_metric(val_true, val_pred)
    target_pos_rate = getattr(batch, "natural_pos_rate", None)
    calib = _fit_platt(val_pred, val_true, sample_weight=wva_eff, target_pos_rate=target_pos_rate)
    try:
        p_raw = np.asarray(val_pred, dtype=np.float64)
        p_cal = np.asarray(calib["transform"](p_raw), dtype=np.float64)
        q_raw = [float(np.nanquantile(p_raw, q)) for q in (0.50, 0.90, 0.99)]
        q_cal = [float(np.nanquantile(p_cal, q)) for q in (0.50, 0.90, 0.99)]
        cp = dict(calib.get("params") or {})
        print(
            (
                "[xgb] calib entry: "
                f"type={str(cp.get('type', 'identity'))} "
                f"space={str(cp.get('space', 'prob'))} "
                f"prior={float(cp.get('sample_pos_rate', np.nan)):.4f}->{float(cp.get('target_pos_rate', np.nan)):.4f} "
                f"q50/q90/q99 raw={q_raw[0]:.4f}/{q_raw[1]:.4f}/{q_raw[2]:.4f} "
                f"cal={q_cal[0]:.4f}/{q_cal[1]:.4f}/{q_cal[2]:.4f}"
            ),
            flush=True,
        )
    except Exception:
        pass
    try:
        if np.isfinite(top_metric_value):
            top_msg = " ".join([f"{k}={v:.4f}" for k, v in sorted(top_metric_parts.items())])
            print(f"[xgb] top_metric val={top_metric_value:.4f} {top_msg}".strip(), flush=True)
    except Exception:
        pass
    meta = {
        "calibrator": calib["params"],
        "best_iteration": best_iteration,
        "selection_metric": metric_mode,
        "objective_mode": objective_mode,
        "top_metric_value": float(top_metric_value) if np.isfinite(top_metric_value) else None,
        "top_metric_parts": top_metric_parts,
        "loss_weight_bins": loss_bin_stats,
    }
    return booster, meta


def _fit_reg_balance_map(
    y: np.ndarray,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(yv)
    if int(m.sum()) < 64:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    yy = yv[m]
    nb = int(max(3, bins))
    q = np.linspace(0.0, 1.0, nb + 1, dtype=np.float64)
    try:
        edges = np.quantile(yy, q)
    except Exception:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    edges = np.unique(np.asarray(edges, dtype=np.float64))
    if edges.size < 4:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    n_bins = int(edges.size - 1)
    bidx = np.digitize(yy, edges[1:-1], right=False).astype(np.int32, copy=False)
    counts = np.bincount(bidx, minlength=n_bins).astype(np.float64, copy=False)
    pos = counts > 0.0
    if int(np.sum(pos)) < 2:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    target = float(np.mean(counts[pos]))
    scales = np.ones(n_bins, dtype=np.float64)
    scales[pos] = np.sqrt(target / counts[pos])
    return edges, scales


def _apply_reg_balance_map(
    y: np.ndarray,
    *,
    edges: np.ndarray,
    scales: np.ndarray,
    strength: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    out = np.ones(np.asarray(y).shape[0], dtype=np.float32)
    if edges.size < 3 or scales.size < 2:
        return out
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(yv)
    if not bool(np.any(m)):
        return out
    idx = np.digitize(yv[m], edges[1:-1], right=False).astype(np.int32, copy=False)
    idx = np.clip(idx, 0, int(scales.size) - 1)
    sc = scales[idx]
    p = float(max(0.0, strength))
    if p != 1.0:
        sc = np.power(np.maximum(sc, 1e-6), p)
    lo = float(max(0.01, clip_min))
    hi = float(max(lo, clip_max))
    sc = np.clip(sc, lo, hi)
    out[m] = sc.astype(np.float32, copy=False)
    return out


def _build_exit_regression_weights(
    ytr: np.ndarray,
    yva: np.ndarray,
    wtr: np.ndarray | None,
    wva: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    ytr = np.asarray(ytr, dtype=np.float32)
    yva = np.asarray(yva, dtype=np.float32)
    wtr0 = np.asarray(wtr, dtype=np.float32) if wtr is not None else np.ones(ytr.size, dtype=np.float32)
    wva0 = np.asarray(wva, dtype=np.float32) if wva is not None else np.ones(yva.size, dtype=np.float32)
    wtr0 = np.clip(np.nan_to_num(wtr0, nan=1.0, posinf=7.0, neginf=1.0), 1e-3, None).astype(np.float32, copy=False)
    wva0 = np.clip(np.nan_to_num(wva0, nan=1.0, posinf=7.0, neginf=1.0), 1e-3, None).astype(np.float32, copy=False)

    # Default em 0: labels de exit já carregam peso de confiança; balance extra tende a puxar
    # o regressor para spans raros/extremos.
    enabled = str(os.getenv("SNIPER_EXIT_BALANCE_ENABLE", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    bins = int(os.getenv("SNIPER_EXIT_BALANCE_BINS", "24") or "24")
    strength = float(os.getenv("SNIPER_EXIT_BALANCE_STRENGTH", "0.35") or "0.35")
    clip_min = float(os.getenv("SNIPER_EXIT_BALANCE_CLIP_MIN", "0.50") or "0.50")
    clip_max = float(os.getenv("SNIPER_EXIT_BALANCE_CLIP_MAX", "2.50") or "2.50")

    if not enabled:
        return wtr0, wva0, {
            "enabled": False,
            "bins_req": int(bins),
        }

    edges, scales = _fit_reg_balance_map(ytr, bins=int(bins))
    if edges.size < 3 or scales.size < 2:
        return wtr0, wva0, {
            "enabled": False,
            "reason": "insufficient_span_diversity",
            "bins_req": int(bins),
        }

    s_tr = _apply_reg_balance_map(
        ytr,
        edges=edges,
        scales=scales,
        strength=float(strength),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )
    s_va = _apply_reg_balance_map(
        yva,
        edges=edges,
        scales=scales,
        strength=float(strength),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )

    wtr1 = (wtr0.astype(np.float64) * s_tr.astype(np.float64)).astype(np.float32, copy=False)
    wva1 = (wva0.astype(np.float64) * s_va.astype(np.float64)).astype(np.float32, copy=False)

    mu_tr0 = float(np.mean(wtr0)) if wtr0.size else 1.0
    mu_va0 = float(np.mean(wva0)) if wva0.size else 1.0
    mu_tr1 = float(np.mean(wtr1)) if wtr1.size else 1.0
    mu_va1 = float(np.mean(wva1)) if wva1.size else 1.0
    if np.isfinite(mu_tr0) and np.isfinite(mu_tr1) and mu_tr1 > 0.0:
        wtr1 = (wtr1.astype(np.float64) * (mu_tr0 / mu_tr1)).astype(np.float32, copy=False)
    if np.isfinite(mu_va0) and np.isfinite(mu_va1) and mu_va1 > 0.0:
        wva1 = (wva1.astype(np.float64) * (mu_va0 / mu_va1)).astype(np.float32, copy=False)

    meta = {
        "enabled": True,
        "bins_req": int(bins),
        "bins_used": int(scales.size),
        "strength": float(strength),
        "clip": [float(clip_min), float(clip_max)],
        "train_scale_p10": float(np.nanpercentile(s_tr, 10)) if s_tr.size else 1.0,
        "train_scale_p50": float(np.nanpercentile(s_tr, 50)) if s_tr.size else 1.0,
        "train_scale_p90": float(np.nanpercentile(s_tr, 90)) if s_tr.size else 1.0,
    }
    return wtr1, wva1, meta


def _apply_exit_isotonic(pred: np.ndarray, calib: dict | None) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float64)
    if not isinstance(calib, dict):
        return out.astype(np.float32, copy=False)
    if str(calib.get("type", "identity")).strip().lower() != "isotonic":
        return out.astype(np.float32, copy=False)
    x = np.asarray(calib.get("x", []), dtype=np.float64)
    y = np.asarray(calib.get("y", []), dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return out.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        out2 = np.interp(out, x, y, left=float(y[0]), right=float(y[-1]))
    return out2.astype(np.float32, copy=False)


def _fit_exit_isotonic(pred: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None) -> dict:
    if IsotonicRegression is None:
        return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}
    px = np.asarray(pred, dtype=np.float64)
    ty = np.asarray(target, dtype=np.float64)
    m = np.isfinite(px) & np.isfinite(ty)
    if int(m.sum()) < 64:
        return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}
    px = px[m]
    ty = ty[m]
    if np.unique(px).size < 8:
        return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}

    sw = None
    if sample_weight is not None:
        sw0 = np.asarray(sample_weight, dtype=np.float64)
        if sw0.shape[0] == m.shape[0]:
            sw = np.clip(np.nan_to_num(sw0[m], nan=1.0, posinf=1.0, neginf=1.0), 1e-6, None)
    try:
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(px, ty, sample_weight=sw)
        xk = np.asarray(getattr(ir, "X_thresholds_", []), dtype=np.float64)
        yk = np.asarray(getattr(ir, "y_thresholds_", []), dtype=np.float64)
        if xk.size < 2 or yk.size != xk.size:
            return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}
        max_knots = int(os.getenv("SNIPER_EXIT_CALIB_MAX_KNOTS", "128") or "128")
        max_knots = max(16, min(512, max_knots))
        if xk.size > max_knots:
            kidx = np.linspace(0, xk.size - 1, num=max_knots, dtype=np.int32)
            xk = xk[kidx]
            yk = yk[kidx]
        params = {
            "type": "isotonic",
            "x": [float(v) for v in xk.tolist()],
            "y": [float(v) for v in yk.tolist()],
        }
        return {
            "params": params,
            "transform": lambda p: _apply_exit_isotonic(np.asarray(p, dtype=np.float64), params),
        }
    except Exception:
        return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}


def _train_xgb_regressor(batch: SniperBatch, params: dict, *, y_transform: str = "log1p") -> tuple[xgb.Booster, dict]:
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    device = str(params.get("device", "cpu")).lower()
    use_quantile = device.startswith("cuda")
    Xtr = batch.X[tr_idx]
    ytr0 = batch.y[tr_idx].astype(np.float32, copy=False)
    Xva = batch.X[va_idx]
    yva0 = batch.y[va_idx].astype(np.float32, copy=False)
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else None
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else None
    wtr_eff, wva_eff, balance_meta = _build_exit_regression_weights(ytr0, yva0, wtr, wva)

    def _apply_transform(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float32)
        if y_transform == "none":
            return yy
        if y_transform == "log1p":
            yy = np.maximum(0.0, yy)
            return np.log1p(yy)
        raise ValueError(f"y_transform inválido: {y_transform}")

    def _inverse_transform(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float64)
        if y_transform == "none":
            return yy.astype(np.float32, copy=False)
        if y_transform == "log1p":
            with np.errstate(over="ignore", invalid="ignore"):
                return np.expm1(yy).astype(np.float32, copy=False)
        raise ValueError(f"y_transform invalido: {y_transform}")

    ytr = _apply_transform(ytr0)
    yva = _apply_transform(yva0)

    if use_quantile:
        mb = int(params.get("max_bin", 256))
        params = dict(params)
        params["max_bin"] = mb
        try:
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr_eff, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva_eff, ref=dtrain, max_bin=mb)
        except TypeError:
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr_eff)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva_eff, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr_eff)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva_eff)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr_eff)
        dvalid = xgb.DMatrix(Xva, label=yva, weight=wva_eff)

    watch = [(dtrain, "train"), (dvalid, "val")]
    print(f"[xgb] (reg) train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,}".replace(",", "."), flush=True)
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2500,
            evals=watch,
            early_stopping_rounds=250,
            verbose_eval=50,
        )
    except Exception as e:
        if device.startswith("cuda"):
            print(f"[xgb] treino reg em GPU falhou ({type(e).__name__}: {e}) -> tentando CPU", flush=True)
            params_cpu = dict(params)
            params_cpu["device"] = "cpu"
            params_cpu["tree_method"] = "hist"
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr_eff)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva_eff)
            watch = [(dtrain, "train"), (dvalid, "val")]
            booster = xgb.train(
                params=params_cpu,
                dtrain=dtrain,
                num_boost_round=2500,
                evals=watch,
                early_stopping_rounds=250,
                verbose_eval=50,
            )
        else:
            raise

    pred_t = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1)).astype(np.float32, copy=False)
    pred_raw = np.maximum(0.0, _inverse_transform(pred_t))
    calib = _fit_exit_isotonic(pred_raw, yva0, sample_weight=wva_eff)
    pred_cal = np.maximum(0.0, _apply_exit_isotonic(pred_raw, calib["params"]))

    err = pred_raw.astype(np.float64) - yva0.astype(np.float64)
    err_cal = pred_cal.astype(np.float64) - yva0.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    rmse_cal = float(np.sqrt(np.mean(err_cal * err_cal))) if err_cal.size else float("nan")
    mae_cal = float(np.mean(np.abs(err_cal))) if err_cal.size else float("nan")
    meta = {
        "transform": y_transform,
        "best_iteration": booster.best_iteration,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "rmse_cal": rmse_cal,
            "mae_cal": mae_cal,
        },
        "calibration": calib["params"],
        "balance": balance_meta,
    }
    return booster, meta


def _fit_platt(
    preds: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    target_pos_rate: float | None = None,
) -> dict:
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = labels.astype(np.int32)
    method = str(os.getenv("SNIPER_ENTRY_CALIB_METHOD", "platt") or "platt").strip().lower()
    if method == "identity":
        return {"params": {"type": "identity"}, "transform": lambda p: np.asarray(p, dtype=np.float32)}
    use_weighted_cal = str(os.getenv("SNIPER_ENTRY_CALIB_USE_WEIGHTS", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    sw = None
    if use_weighted_cal and (sample_weight is not None):
        sw0 = np.asarray(sample_weight, dtype=np.float64)
        if sw0.shape[0] == labels.shape[0]:
            sw = np.clip(np.nan_to_num(sw0, nan=1.0, posinf=1.0, neginf=1.0), 1e-6, None)

    if method == "isotonic" and IsotonicRegression is not None:
        m = np.isfinite(p) & np.isfinite(labels.astype(np.float64))
        if int(m.sum()) >= 64:
            px = p[m]
            yx = labels[m].astype(np.float64, copy=False)
            if np.unique(px).size >= 8:
                try:
                    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
                    ir.fit(px, yx, sample_weight=(sw[m] if sw is not None else None))
                    xk = np.asarray(getattr(ir, "X_thresholds_", []), dtype=np.float64)
                    yk = np.asarray(getattr(ir, "y_thresholds_", []), dtype=np.float64)
                    if xk.size >= 2 and yk.size == xk.size:
                        max_knots = int(os.getenv("SNIPER_ENTRY_CALIB_MAX_KNOTS", "128") or "128")
                        max_knots = max(16, min(512, max_knots))
                        if xk.size > max_knots:
                            kidx = np.linspace(0, xk.size - 1, num=max_knots, dtype=np.int32)
                            xk = xk[kidx]
                            yk = yk[kidx]
                        params = {
                            "type": "isotonic",
                            "x": [float(v) for v in xk.tolist()],
                            "y": [float(v) for v in yk.tolist()],
                            "weighted": bool(use_weighted_cal),
                        }

                        def _iso_transform(x: np.ndarray) -> np.ndarray:
                            xx = np.asarray(x, dtype=np.float64)
                            out = np.interp(xx, xk, yk, left=float(yk[0]), right=float(yk[-1]))
                            return out.astype(np.float32, copy=False)

                        return {"params": params, "transform": _iso_transform}
                except Exception:
                    pass

    if LogisticRegression is None:
        return {"params": {"type": "identity"}, "transform": lambda p: p}
    try:
        # Platt em espaco logit tende a preservar melhor dinamica de cauda
        # do que regressao linear diretamente em probabilidade.
        eps = 1e-6
        p_clip = np.clip(p, eps, 1.0 - eps)
        x = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        # Calibracao robusta: regularizacao forte evita coeficientes extremos
        # que saturam probabilidades em alguns periodos WF.
        c_reg = float(os.getenv("SNIPER_ENTRY_CALIB_PLATT_C", "0.1") or "0.1")
        if (not np.isfinite(c_reg)) or c_reg <= 0.0:
            c_reg = 0.1
        clf = LogisticRegression(max_iter=200, C=float(c_reg), solver="lbfgs")
        clf.fit(x, labels, sample_weight=sw)
        a = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
        a_lo = float(os.getenv("SNIPER_ENTRY_CALIB_COEF_MIN", "0.25") or "0.25")
        a_hi = float(os.getenv("SNIPER_ENTRY_CALIB_COEF_MAX", "2.50") or "2.50")
        b_lo = float(os.getenv("SNIPER_ENTRY_CALIB_INTERCEPT_MIN", "-4.00") or "-4.00")
        b_hi = float(os.getenv("SNIPER_ENTRY_CALIB_INTERCEPT_MAX", "0.50") or "0.50")
        if a_hi < a_lo:
            a_hi = a_lo
        if b_hi < b_lo:
            b_hi = b_lo
        a = float(np.clip(a, a_lo, a_hi))
        b = float(np.clip(b, b_lo, b_hi))
        tail_blend = float(os.getenv("SNIPER_ENTRY_CALIB_TAIL_BLEND", "0.0") or "0.0")
        tail_start = float(os.getenv("SNIPER_ENTRY_CALIB_TAIL_START", "0.70") or "0.70")
        tail_power = float(os.getenv("SNIPER_ENTRY_CALIB_TAIL_POWER", "1.0") or "1.0")
        tail_boost = float(os.getenv("SNIPER_ENTRY_CALIB_TAIL_BOOST", "0.0") or "0.0")
        if (not np.isfinite(tail_blend)) or tail_blend < 0.0:
            tail_blend = 0.0
        if tail_blend > 1.0:
            tail_blend = 1.0
        if (not np.isfinite(tail_start)) or tail_start < 0.0:
            tail_start = 0.70
        if tail_start >= 1.0:
            tail_start = 0.999
        if (not np.isfinite(tail_power)) or tail_power <= 0.0:
            tail_power = 1.0
        if (not np.isfinite(tail_boost)) or tail_boost < 0.0:
            tail_boost = 0.0
        if sw is not None:
            sw_sum = float(np.sum(sw))
            sample_prior = float(np.sum(sw * labels) / sw_sum) if sw_sum > 0.0 else float(np.mean(labels))
        else:
            sample_prior = float(np.mean(labels)) if labels.size else 0.5
        prior_target = target_pos_rate
        if prior_target is None or (not np.isfinite(float(prior_target))):
            prior_target = sample_prior
        prior_target = float(np.clip(float(prior_target), eps, 1.0 - eps))
        sample_prior = float(np.clip(float(sample_prior), eps, 1.0 - eps))
        prior_strength = float(os.getenv("SNIPER_ENTRY_CALIB_PRIOR_STRENGTH", "1.0") or "1.0")
        prior_shift_clip = float(os.getenv("SNIPER_ENTRY_CALIB_PRIOR_SHIFT_CLIP", "6.0") or "6.0")
        if not np.isfinite(prior_strength):
            prior_strength = 1.0
        if (not np.isfinite(prior_shift_clip)) or prior_shift_clip <= 0.0:
            prior_shift_clip = 6.0
        try:
            prior_delta = float(
                np.clip(
                    float(prior_strength)
                    * (
                        np.log(prior_target / (1.0 - prior_target))
                        - np.log(sample_prior / (1.0 - sample_prior))
                    ),
                    -float(prior_shift_clip),
                    float(prior_shift_clip),
                )
            )
        except Exception:
            prior_delta = 0.0

        def _transform(x: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=np.float64).reshape(-1)
            xx = np.clip(xx, eps, 1.0 - eps)
            z = np.log(xx / (1.0 - xx)).reshape(-1, 1)
            zz = a * z.reshape(-1) + b + float(prior_delta)
            out = (1.0 / (1.0 + np.exp(-zz))).astype(np.float64, copy=False)
            if tail_blend > 0.0:
                denom = max(1e-9, 1.0 - float(tail_start))
                rel = np.clip((xx - float(tail_start)) / denom, 0.0, 1.0)
                if tail_power != 1.0:
                    rel = np.power(rel, float(tail_power))
                out = out + float(tail_blend) * rel * (xx - out)
                if tail_boost > 0.0:
                    out = out + float(tail_boost) * rel * (1.0 - out)
            return np.clip(out, 0.0, 1.0).astype(np.float64, copy=False)

        params = {
            "type": "platt",
            "coef": float(a),
            "intercept": float(b),
            "space": "logit",
            "weighted": bool(use_weighted_cal),
            "c_reg": float(c_reg),
            "coef_clip": [float(a_lo), float(a_hi)],
            "intercept_clip": [float(b_lo), float(b_hi)],
            "tail_blend": float(tail_blend),
            "tail_start": float(tail_start),
            "tail_power": float(tail_power),
            "tail_boost": float(tail_boost),
            "sample_pos_rate": float(sample_prior),
            "target_pos_rate": float(prior_target),
            "prior_shift": float(prior_delta),
        }
        return {
            "params": params,
            "transform": _transform,
        }
    except Exception:
        return {"params": {"type": "identity"}, "transform": lambda p: p}


def _calib_stability_enabled() -> bool:
    return str(os.getenv("SNIPER_ENTRY_CALIB_STABILITY_ENABLE", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    vv = np.asarray(values, dtype=np.float64)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return 0.0, 1.0
    center = float(np.median(vv))
    mad = float(np.median(np.abs(vv - center)))
    sigma = max(1e-6, 1.4826 * mad)
    return center, sigma


def _local_weighted_mean(values: np.ndarray, idx: int) -> float:
    n = int(values.size)
    idxs = [j for j in (idx - 1, idx, idx + 1) if 0 <= j < n]
    if not idxs:
        return float(values[idx])
    ww = np.asarray([2.0 if j == idx else 1.0 for j in idxs], dtype=np.float64)
    return float(np.average(values[np.asarray(idxs, dtype=np.int32)], weights=ww))


def _stabilize_saved_entry_calibrations(period_records: list[tuple[int, Path]]) -> None:
    if len(period_records) < 3 or (not _calib_stability_enabled()):
        return
    try:
        history_blend = float(os.getenv("SNIPER_ENTRY_CALIB_STABILITY_GLOBAL_BLEND", "0.20") or "0.20")
    except Exception:
        history_blend = 0.20
    try:
        neighbor_blend = float(os.getenv("SNIPER_ENTRY_CALIB_STABILITY_NEIGHBOR_BLEND", "0.35") or "0.35")
    except Exception:
        neighbor_blend = 0.35
    try:
        clip_sigma = float(os.getenv("SNIPER_ENTRY_CALIB_STABILITY_ROBUST_CLIP_SIGMA", "2.0") or "2.0")
    except Exception:
        clip_sigma = 2.0
    history_blend = float(np.clip(history_blend, 0.0, 1.0))
    neighbor_blend = float(np.clip(neighbor_blend, 0.0, 1.0))
    if history_blend + neighbor_blend > 1.0:
        total = history_blend + neighbor_blend
        history_blend = float(history_blend / total)
        neighbor_blend = float(neighbor_blend / total)
    clip_sigma = max(0.0, float(clip_sigma))

    metas: list[dict] = []
    for tail, period_dir in sorted(period_records, key=lambda x: int(x[0])):
        meta_path = period_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metas.append(
            {
                "tail": int(tail),
                "period_dir": period_dir,
                "meta_path": meta_path,
                "meta": meta,
                "dirty": False,
            }
        )
    if len(metas) < 3:
        return

    for side in ("long", "short"):
        rows: list[dict] = []
        for item in metas:
            side_meta = item["meta"].get(f"entry_{side}")
            if not isinstance(side_meta, dict):
                continue
            calib = dict(side_meta.get("calibration") or {})
            if str(calib.get("type", "")).strip().lower() != "platt":
                continue
            try:
                coef = float(calib.get("coef", 1.0))
                intercept = float(calib.get("intercept", 0.0))
            except Exception:
                continue
            if (not np.isfinite(coef)) or (not np.isfinite(intercept)):
                continue
            rows.append(
                {
                    "item": item,
                    "tail": int(item["tail"]),
                    "calib": calib,
                    "coef_raw": float(coef),
                    "intercept_raw": float(intercept),
                }
            )
        if len(rows) < 3:
            continue

        rows = sorted(rows, key=lambda r: int(r["tail"]), reverse=True)
        coef_raw = np.asarray([r["coef_raw"] for r in rows], dtype=np.float64)
        intercept_raw = np.asarray([r["intercept_raw"] for r in rows], dtype=np.float64)
        coef_final = coef_raw.copy()
        intercept_final = intercept_raw.copy()
        for idx in range(len(rows)):
            if idx == 0:
                continue
            ref_coef = coef_final[:idx]
            ref_intercept = intercept_final[:idx]
            coef_center, coef_sigma = _robust_center_scale(ref_coef)
            intercept_center, intercept_sigma = _robust_center_scale(ref_intercept)
            if clip_sigma > 0.0 and idx >= 3:
                coef_lo = coef_center - clip_sigma * coef_sigma
                coef_hi = coef_center + clip_sigma * coef_sigma
                int_lo = intercept_center - clip_sigma * intercept_sigma
                int_hi = intercept_center + clip_sigma * intercept_sigma
            else:
                coef_lo, coef_hi = -np.inf, np.inf
                int_lo, int_hi = -np.inf, np.inf
            coef_clip = float(np.clip(coef_raw[idx], coef_lo, coef_hi))
            intercept_clip = float(np.clip(intercept_raw[idx], int_lo, int_hi))
            coef_neighbor = float(coef_final[idx - 1])
            intercept_neighbor = float(intercept_final[idx - 1])
            if idx == 1:
                local_history_blend = 0.0
                local_neighbor_blend = min(0.12, neighbor_blend)
            elif idx == 2:
                local_history_blend = min(0.10, history_blend)
                local_neighbor_blend = min(0.18, neighbor_blend)
            else:
                local_history_blend = history_blend
                local_neighbor_blend = neighbor_blend
            keep_w = max(0.0, 1.0 - local_history_blend - local_neighbor_blend)
            coef_v = keep_w * coef_clip + local_history_blend * coef_center + local_neighbor_blend * coef_neighbor
            int_v = keep_w * intercept_clip + local_history_blend * intercept_center + local_neighbor_blend * intercept_neighbor
            coef_final[idx] = float(np.clip(coef_v, coef_lo, coef_hi))
            intercept_final[idx] = float(np.clip(int_v, int_lo, int_hi))

        side_changes = 0
        for idx, row in enumerate(rows):
            new_coef = float(coef_final[idx])
            new_intercept = float(intercept_final[idx])
            old_coef = float(row["coef_raw"])
            old_intercept = float(row["intercept_raw"])
            if abs(new_coef - old_coef) < 1e-9 and abs(new_intercept - old_intercept) < 1e-9:
                continue
            calib = dict(row["calib"])
            calib["coef_raw"] = float(old_coef)
            calib["intercept_raw"] = float(old_intercept)
            calib["coef"] = float(new_coef)
            calib["intercept"] = float(new_intercept)
            calib["stability_adjusted"] = True
            calib["stability_causal"] = True
            calib["stability_blend"] = {
                "history": float(history_blend),
                "neighbor": float(neighbor_blend),
                "clip_sigma": float(clip_sigma),
            }
            item = row["item"]
            item["meta"][f"entry_{side}"]["calibration"] = calib
            base_side = "long" if "entry_long" in item["meta"] else ("short" if "entry_short" in item["meta"] else "")
            if base_side == side and isinstance(item["meta"].get("entry"), dict):
                item["meta"]["entry"]["calibration"] = dict(calib)
            item["dirty"] = True
            side_changes += 1
            print(
                f"[sniper-train] calib-causal {side} T-{int(row['tail'])}d: "
                f"coef {old_coef:.3f}->{new_coef:.3f} "
                f"intercept {old_intercept:.3f}->{new_intercept:.3f}",
                flush=True,
            )
        if side_changes > 0:
            print(
                f"[sniper-train] calib-causal {side}: adjusted={side_changes} "
                f"history_blend={history_blend:.2f} neighbor_blend={neighbor_blend:.2f}",
                flush=True,
            )

    for item in metas:
        if not bool(item["dirty"]):
            continue
        try:
            item["meta_path"].write_text(
                json.dumps(item["meta"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(
                f"[sniper-train] calib-stability save failed {item['meta_path']}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )


def _next_run_dir(base: Path, prefix: str = "wf_") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.glob(f"{prefix}*") if p.is_dir()]
    max_idx = 0
    for p in existing:
        try:
            idx = int(p.name.replace(prefix, ""))
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    run_dir = base / f"{prefix}{max_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _full_pool_cache_root() -> Path:
    v = os.getenv("SNIPER_FULL_POOL_CACHE_DIR", "").strip()
    if v:
        return Path(v).expanduser().resolve()
    try:
        from utils.paths import workspace_root as _workspace_root  # type: ignore

        return _workspace_root() / "cache_sniper" / "full_pool"
    except Exception:
        return Path(__file__).resolve().parents[2].parent / "cache_sniper" / "full_pool"


def _cache_map_fingerprint(cache_map: dict | None) -> str:
    if not cache_map:
        return "nocache"
    mode = str(os.getenv("SNIPER_FULL_POOL_FINGERPRINT_MODE", "size") or "size").strip().lower()
    use_mtime = mode in {"mtime", "size_mtime", "full"}
    h = hashlib.sha1()
    for sym in sorted(cache_map.keys()):
        p = Path(str(cache_map[sym]))
        try:
            st = p.stat()
            if use_mtime:
                token = f"{sym}|{p.name}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
            else:
                # default "size": permite reaproveitar pool mesmo apos refresh que so regrava
                # os arquivos sem alterar conteudo efetivo.
                token = f"{sym}|{p.name}|{int(st.st_size)}"
        except Exception:
            token = f"{sym}|{p.name}|missing"
        h.update(token.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _full_pool_cache_key(
    *,
    asset_class: str,
    symbols: Sequence[str],
    cfg: TrainConfig,
    contract: TradeContract,
    feature_flags: dict,
    cache_fp: str,
    pool_rows: int,
) -> str:
    payload = {
        "asset_class": str(asset_class),
        "symbols": [str(s) for s in symbols],
        "total_days": int(getattr(cfg, "total_days", 0) or 0),
        "entry_ratio_neg_per_pos": float(getattr(cfg, "entry_ratio_neg_per_pos", 0.0) or 0.0),
        "max_rows_exit": int(getattr(cfg, "max_rows_exit", 0) or 0),
        "pool_rows_entry": int(pool_rows),
        "contract": asdict(contract),
        "feature_flags": dict(feature_flags or {}),
        "cache_fp": str(cache_fp),
        "train_exit_model": str(os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1") or "1"),
        "bin_step_x10": str(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_STEP_X10", "1") or "1"),
        "bin_max": str(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_MAX", "7.0") or "7.0"),
        "pos_keep_fraction": str(os.getenv("SNIPER_ENTRY_POS_KEEP_FRACTION", "1.0") or "1.0"),
        "pos_min_weight": str(os.getenv("SNIPER_ENTRY_POS_MIN_WEIGHT", "0.0") or "0.0"),
        "neg_min_weight": str(os.getenv("SNIPER_ENTRY_NEG_MIN_WEIGHT", "0.0") or "0.0"),
        "pos_favor_high": str(os.getenv("SNIPER_ENTRY_POS_FAVOR_HIGH", "1") or "1"),
        "neg_favor_high": str(os.getenv("SNIPER_ENTRY_NEG_FAVOR_HIGH", "0") or "0"),
        "use_weight_bins": str(os.getenv("SNIPER_ENTRY_USE_WEIGHT_BINS", "1") or "1"),
        "pf_entry_label_net_profit_thr": str(os.getenv("PF_ENTRY_LABEL_NET_PROFIT_THR", "0.03") or "0.03"),
        "pf_entry_label_min_future_avg_net": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_AVG_NET", "0.0") or "0.0"),
        "pf_entry_label_early_window_min": str(os.getenv("PF_ENTRY_LABEL_EARLY_WINDOW_MIN", "15") or "15"),
        "pf_entry_label_min_future_early_avg_net": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_EARLY_AVG_NET", "0.0") or "0.0"),
        "pf_entry_label_min_future_early_worst_net": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_EARLY_WORST_NET", "-1.0") or "-1.0"),
        "pf_entry_label_min_risk_score": str(os.getenv("PF_ENTRY_LABEL_MIN_RISK_SCORE", "0.0") or "0.0"),
        "pf_entry_label_min_future_end_net": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_END_NET", "0.0") or "0.0"),
        "pf_entry_label_min_future_eff": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_EFF", "0.0") or "0.0"),
        "pf_entry_label_min_future_pos_frac": str(os.getenv("PF_ENTRY_LABEL_MIN_FUTURE_POS_FRAC", "0.0") or "0.0"),
        "pf_entry_label_min_consistency_score": str(os.getenv("PF_ENTRY_LABEL_MIN_CONSISTENCY_SCORE", "0.0") or "0.0"),
        "pf_entry_label_min_contract_mae": str(os.getenv("PF_ENTRY_LABEL_MIN_CONTRACT_MAE", "-1.0") or "-1.0"),
        "pf_entry_label_enable_neutral": str(os.getenv("PF_ENTRY_LABEL_ENABLE_NEUTRAL", "1") or "1"),
        "pf_entry_label_neg_net_profit_thr": str(os.getenv("PF_ENTRY_LABEL_NEG_NET_PROFIT_THR", "0.0") or "0.0"),
        "pf_entry_label_neg_max_future_avg_net": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_AVG_NET", "0.0") or "0.0"),
        "pf_entry_label_neg_max_future_early_avg_net": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_EARLY_AVG_NET", "0.0") or "0.0"),
        "pf_entry_label_neg_max_future_early_worst_net": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_EARLY_WORST_NET", "-1.0") or "-1.0"),
        "pf_entry_label_neg_max_risk_score": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_RISK_SCORE", "0.0") or "0.0"),
        "pf_entry_label_neg_max_future_end_net": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_END_NET", "0.0") or "0.0"),
        "pf_entry_label_neg_max_future_eff": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_EFF", "1.0") or "1.0"),
        "pf_entry_label_neg_max_future_pos_frac": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_FUTURE_POS_FRAC", "1.0") or "1.0"),
        "pf_entry_label_neg_max_consistency_score": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_CONSISTENCY_SCORE", "0.0") or "0.0"),
        "pf_entry_label_neg_max_contract_mae": str(os.getenv("PF_ENTRY_LABEL_NEG_MAX_CONTRACT_MAE", "-1.0") or "-1.0"),
        "pf_entry_label_neg_min_failures": str(os.getenv("PF_ENTRY_LABEL_NEG_MIN_FAILURES", "1") or "1"),
        "pf_entry_label_any_canonical": str(os.getenv("PF_ENTRY_LABEL_ANY_CANONICAL", "1") or "1"),
        "pf_entry_weight_mode": str(os.getenv("PF_ENTRY_WEIGHT_MODE", "ret_curve") or "ret_curve"),
        "pf_entry_weight_future_window_min": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_WINDOW_MIN", "60") or "60"),
        "pf_entry_weight_future_dd_penalty": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_DD_PENALTY", "1.5") or "1.5"),
        "pf_entry_weight_future_avg_softcap": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_AVG_SOFTCAP", "0.03") or "0.03"),
        "pf_entry_weight_future_end_softcap": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_END_SOFTCAP", "0.04") or "0.04"),
        "pf_entry_weight_future_early_avg_softcap": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_EARLY_AVG_SOFTCAP", "0.01") or "0.01"),
        "pf_entry_weight_future_end_gain": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_END_GAIN", "0.0") or "0.0"),
        "pf_entry_weight_future_early_gain": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_EARLY_GAIN", "0.0") or "0.0"),
        "pf_entry_weight_future_early_dd_penalty": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_EARLY_DD_PENALTY", "0.0") or "0.0"),
        "pf_entry_weight_future_eff_gain": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_EFF_GAIN", "0.0") or "0.0"),
        "pf_entry_weight_future_pos_frac_gain": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_POS_FRAC_GAIN", "0.0") or "0.0"),
        "pf_entry_weight_future_pos_scale": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_POS_SCALE", "1.0") or "1.0"),
        "pf_entry_weight_future_neg_scale": str(os.getenv("PF_ENTRY_WEIGHT_FUTURE_NEG_SCALE", "1.0") or "1.0"),
        "pf_entry_weight_pct_power": str(os.getenv("PF_ENTRY_WEIGHT_PCT_POWER", "1.0") or "1.0"),
        "entry_loss_use_label_weights": str(os.getenv("SNIPER_ENTRY_USE_LABEL_WEIGHTS", "0") or "0"),
        "entry_loss_weight_power": str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER", "1.0") or "1.0"),
        "entry_loss_weight_power_pos": str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER_POS", "") or ""),
        "entry_loss_weight_power_neg": str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_POWER_NEG", "") or ""),
        "entry_loss_weight_clip_max_pos": str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_POS", "") or ""),
        "entry_loss_weight_clip_max_neg": str(os.getenv("SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_NEG", "") or ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _load_full_pool_cache(path: Path) -> SniperDataPack | None:
    def _coerce_batch(obj, *, allow_none: bool = False) -> SniperBatch | None:
        try:
            if obj is None:
                return None if allow_none else None
            if isinstance(obj, SniperBatch):
                return obj
            d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return None
            X = np.asarray(d.get("X", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
            y = np.asarray(d.get("y", np.empty((0,), dtype=np.float32)), dtype=np.float32)
            w = np.asarray(d.get("w", np.empty((0,), dtype=np.float32)), dtype=np.float32)
            ts = np.asarray(d.get("ts", np.empty((0,), dtype="datetime64[ns]")), dtype="datetime64[ns]")
            sym_id = np.asarray(d.get("sym_id", np.empty((0,), dtype=np.int32)), dtype=np.int32)
            feat_cols = list(d.get("feature_cols", []) or [])
            return SniperBatch(
                X=X,
                y=y,
                w=w,
                ts=ts,
                sym_id=sym_id,
                feature_cols=feat_cols,
            )
        except Exception:
            return None

    def _coerce_contract(obj) -> TradeContract:
        if isinstance(obj, TradeContract):
            return obj
        try:
            d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                return TradeContract(**d)
        except Exception:
            pass
        return DEFAULT_TRADE_CONTRACT

    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, SniperDataPack):
            return obj
        d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", None)
        if not isinstance(d, dict):
            return None
        req = ("entry", "entry_short", "entry_mid", "entry_long", "danger", "exit", "contract", "symbols")
        if not all(k in d for k in req):
            return None
        b_entry = _coerce_batch(d.get("entry"))
        b_short = _coerce_batch(d.get("entry_short"), allow_none=True)
        b_mid = _coerce_batch(d.get("entry_mid"))
        b_long = _coerce_batch(d.get("entry_long"))
        b_danger = _coerce_batch(d.get("danger"))
        b_exit = _coerce_batch(d.get("exit"))
        if any(v is None for v in (b_entry, b_mid, b_long, b_danger, b_exit)):
            return None
        t_end = d.get("train_end_utc", None)
        try:
            t_end = pd.to_datetime(t_end) if t_end is not None else None
        except Exception:
            t_end = None
        return SniperDataPack(
            entry=b_entry,  # type: ignore[arg-type]
            entry_short=b_short,  # type: ignore[arg-type]
            entry_mid=b_mid,  # type: ignore[arg-type]
            entry_long=b_long,  # type: ignore[arg-type]
            danger=b_danger,  # type: ignore[arg-type]
            exit=b_exit,  # type: ignore[arg-type]
            contract=_coerce_contract(d.get("contract")),
            symbols=list(d.get("symbols", []) or []),
            symbols_used=(list(d.get("symbols_used", []) or []) or None),
            symbols_skipped=(list(d.get("symbols_skipped", []) or []) or None),
            train_end_utc=t_end,
        )
    except Exception:
        return None
    return None


def _save_full_pool_cache(path: Path, pack: SniperDataPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _prune_full_pool_cache(cache_root: Path, asset_class: str) -> None:
    keep = str(os.getenv("SNIPER_FULL_POOL_KEEP_LAST", "3") or "3").strip()
    try:
        keep_n = max(0, int(keep))
    except Exception:
        keep_n = 3
    if keep_n <= 0:
        return
    d = cache_root / str(asset_class or "crypto").lower()
    if not d.exists():
        return
    files = sorted(d.glob("pool_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep_n:]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def _load_latest_full_pool_cache(cache_root: Path, asset_class: str) -> tuple[SniperDataPack | None, Path | None]:
    d = cache_root / str(asset_class or "crypto").lower()
    if not d.exists():
        return None, None
    files = sorted(d.glob("pool_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        obj = _load_full_pool_cache(p)
        if obj is not None:
            return obj, p
    return None, None


def _save_model(booster: xgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    try:
        ubj = path.with_suffix(".ubj")
        booster.save_model(str(ubj))
    except Exception:
        pass


def _empty_batch_like(b: SniperBatch | None = None) -> SniperBatch:
    n_feats = int(b.X.shape[1]) if (b is not None and getattr(b, "X", None) is not None and b.X.ndim == 2) else 0
    out = SniperBatch(
        X=np.empty((0, n_feats), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=list(getattr(b, "feature_cols", []) or []),
    )
    nat = getattr(b, "natural_pos_rate", None)
    if nat is not None:
        try:
            setattr(out, "natural_pos_rate", float(nat))
        except Exception:
            pass
    return out


def _sample_indices_binary(
    y: np.ndarray,
    *,
    ratio_neg_per_pos: float,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if y.size == 0:
        return np.empty((0,), dtype=np.int64)
    pos_idx = np.flatnonzero(y >= 0.5)
    neg_idx = np.flatnonzero(y < 0.5)
    if pos_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    ratio = float(max(0.0, ratio_neg_per_pos))
    max_rows = int(max_rows)
    pos_keep_n = int(pos_idx.size)
    if max_rows > 0:
        max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
        pos_keep_n = min(pos_keep_n, max_pos)
    pos_keep = rng.choice(pos_idx, size=pos_keep_n, replace=False) if pos_keep_n < pos_idx.size else pos_idx
    neg_target = int(round(pos_keep_n * ratio))
    if max_rows > 0:
        neg_target = min(neg_target, max(0, max_rows - pos_keep_n))
    if neg_target <= 0 or neg_idx.size == 0:
        keep = pos_keep
    else:
        neg_keep = rng.choice(neg_idx, size=neg_target, replace=False) if neg_target < neg_idx.size else neg_idx
        keep = np.concatenate([pos_keep, neg_keep])
    keep.sort()
    return keep.astype(np.int64, copy=False)


def _slice_and_rebalance_batch(
    b: SniperBatch | None,
    *,
    cutoff_ts: np.datetime64 | None,
    ratio_neg_per_pos: float,
    max_rows: int,
    seed: int,
) -> SniperBatch:
    if b is None or b.X.size == 0 or b.ts.size == 0:
        return _empty_batch_like(b)
    mask = np.ones(int(b.ts.shape[0]), dtype=bool)
    if cutoff_ts is not None:
        mask &= (b.ts <= cutoff_ts)
    if not bool(np.any(mask)):
        return _empty_batch_like(b)
    idx0 = np.flatnonzero(mask)
    y0 = b.y[idx0]
    rng = np.random.default_rng(int(seed))
    keep_rel = _sample_indices_binary(
        y0,
        ratio_neg_per_pos=float(ratio_neg_per_pos),
        max_rows=int(max_rows),
        rng=rng,
    )
    if keep_rel.size == 0:
        return _empty_batch_like(b)
    idx = idx0[keep_rel]
    out = SniperBatch(
        X=b.X[idx],
        y=b.y[idx],
        w=b.w[idx],
        ts=b.ts[idx],
        sym_id=b.sym_id[idx],
        feature_cols=list(b.feature_cols),
    )
    nat = getattr(b, "natural_pos_rate", None)
    if nat is None:
        try:
            nat = float(np.mean(y0 >= 0.5)) if y0.size else None
        except Exception:
            nat = None
    if nat is not None:
        try:
            setattr(out, "natural_pos_rate", float(nat))
        except Exception:
            pass
    return out


def _slice_regression_batch(
    b: SniperBatch | None,
    *,
    cutoff_ts: np.datetime64 | None,
    max_rows: int,
    seed: int,
) -> SniperBatch:
    if b is None or b.X.size == 0 or b.ts.size == 0:
        return _empty_batch_like(b)
    mask = np.ones(int(b.ts.shape[0]), dtype=bool)
    if cutoff_ts is not None:
        mask &= (b.ts <= cutoff_ts)
    if not bool(np.any(mask)):
        return _empty_batch_like(b)
    idx = np.flatnonzero(mask)
    if int(max_rows) > 0 and idx.size > int(max_rows):
        rng = np.random.default_rng(int(seed))
        yv = np.asarray(b.y[idx], dtype=np.float32)
        if getattr(b, "w", None) is not None and b.w.size == b.ts.size:
            wv = np.asarray(b.w[idx], dtype=np.float64)
        else:
            wv = np.ones(idx.size, dtype=np.float64)
        wv = np.clip(np.nan_to_num(wv, nan=1.0, posinf=1.0, neginf=1.0), 1e-6, None)
        use_bal = str(os.getenv("SNIPER_EXIT_BALANCE_SAMPLE", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        if use_bal and yv.size > 0:
            bins = int(os.getenv("SNIPER_EXIT_BALANCE_BINS", "24") or "24")
            strength = float(os.getenv("SNIPER_EXIT_BALANCE_STRENGTH", "0.70") or "0.70")
            clip_min = float(os.getenv("SNIPER_EXIT_BALANCE_CLIP_MIN", "0.50") or "0.50")
            clip_max = float(os.getenv("SNIPER_EXIT_BALANCE_CLIP_MAX", "2.50") or "2.50")
            edges, scales = _fit_reg_balance_map(yv, bins=int(bins))
            if edges.size >= 3 and scales.size >= 2:
                bal = _apply_reg_balance_map(
                    yv,
                    edges=edges,
                    scales=scales,
                    strength=float(strength),
                    clip_min=float(clip_min),
                    clip_max=float(clip_max),
                )
                wv = wv * np.asarray(bal, dtype=np.float64)
        sw = float(np.sum(wv))
        if np.isfinite(sw) and sw > 0.0:
            pick = rng.choice(idx, size=int(max_rows), replace=False, p=(wv / sw))
        else:
            pick = rng.choice(idx, size=int(max_rows), replace=False)
        pick.sort()
        idx = pick.astype(np.int64, copy=False)
    return SniperBatch(
        X=b.X[idx],
        y=b.y[idx],
        w=b.w[idx],
        ts=b.ts[idx],
        sym_id=b.sym_id[idx],
        feature_cols=list(b.feature_cols),
    )


def _slice_pack_from_full_pool(
    pool: SniperDataPack,
    *,
    remove_tail_days: int,
    ratio_neg_per_pos: float,
    max_rows_entry: int,
    max_rows_exit: int,
    seed: int,
) -> SniperDataPack:
    ts_candidates: list[np.ndarray] = []
    for _b in (pool.entry_long, pool.entry_short):
        if _b is not None and _b.ts.size:
            ts_candidates.append(_b.ts)
    max_ts = max((np.max(x) for x in ts_candidates), default=None)
    cutoff_ts = None
    if max_ts is not None:
        cutoff_ts = np.datetime64(pd.to_datetime(max_ts) - pd.Timedelta(days=int(remove_tail_days)))

    side_cap = int(max_rows_entry)
    b_long = _slice_and_rebalance_batch(
        pool.entry_long,
        cutoff_ts=cutoff_ts,
        ratio_neg_per_pos=float(ratio_neg_per_pos),
        max_rows=side_cap,
        seed=int(seed) + 11,
    )
    b_short = _slice_and_rebalance_batch(
        pool.entry_short,
        cutoff_ts=cutoff_ts,
        ratio_neg_per_pos=float(ratio_neg_per_pos),
        max_rows=side_cap,
        seed=int(seed) + 23,
    )
    b_exit = _slice_regression_batch(
        pool.exit,
        cutoff_ts=cutoff_ts,
        max_rows=int(max_rows_exit),
        seed=int(seed) + 37,
    )
    entry_mid = b_long if b_long.X.size > 0 else b_short
    entry = entry_mid if entry_mid is not None else _empty_batch_like(pool.entry_long or pool.entry_short)
    empty = _empty_batch_like(pool.entry_long or pool.entry_short)

    ids_parts = [np.unique(b.sym_id) for b in (b_long, b_short) if b is not None and b.sym_id.size]
    syms_all = list(getattr(pool, "symbols", []) or [])
    if ids_parts:
        ids = np.unique(np.concatenate(ids_parts))
        symbols_used = [syms_all[int(i)] for i in ids if 0 <= int(i) < len(syms_all)]
    else:
        symbols_used = []
    used_set = set(symbols_used)
    symbols_skipped = [s for s in syms_all if s not in used_set]

    train_end = None
    for _b in (b_long, b_short, b_exit):
        if _b is not None and _b.ts.size:
            t = pd.to_datetime(_b.ts.max())
            train_end = t if train_end is None else max(train_end, t)

    return SniperDataPack(
        entry=entry,
        entry_short=b_short,
        entry_mid=entry_mid,
        entry_long=b_long,
        danger=empty,
        exit=b_exit,
        contract=pool.contract,
        symbols=syms_all,
        symbols_used=(symbols_used or None),
        symbols_skipped=(symbols_skipped or None),
        train_end_utc=train_end,
    )


def train_sniper_models(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    asset_class = str(getattr(cfg, "asset_class", "crypto") or "crypto").lower()

    symbols = _select_symbols(cfg)
    print(
        f"[sniper-train] símbolos={len(symbols)} asset={asset_class} (max_symbols={cfg.max_symbols})",
        flush=True,
    )
    if SAVE_ROOT is None:
        base_root = Path(__file__).resolve().parents[2].parent / "models_sniper"
        save_root = base_root / asset_class
    else:
        save_root = SAVE_ROOT(asset_class)
    run_dir = _next_run_dir(save_root)
    entry_params = dict(DEFAULT_ENTRY_PARAMS if cfg.entry_params is None else cfg.entry_params)
    exit_params = dict(DEFAULT_EXIT_PARAMS if cfg.exit_params is None else cfg.exit_params)
    if "device" in entry_params:
        exit_params["device"] = entry_params["device"]
    train_exit_model = str(os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not train_exit_model:
        print("[sniper-train] mode=entry_only (exit desabilitado)", flush=True)

    def _default_feature_flags() -> dict:
        if asset_class == "stocks":
            try:
                from stocks.prepare_features_stocks import FLAGS_STOCKS  # type: ignore

                return dict(FLAGS_STOCKS)
            except Exception:
                pass
        return dict(GLOBAL_FLAGS_FULL)

    feature_flags = dict(cfg.feature_flags or _default_feature_flags())
    contract = cfg.contract

    cache_map = None
    if bool(getattr(cfg, "use_feature_cache", True)):
        print(
            f"[sniper-train] cache: garantindo features+labels 1x (total_days={int(cfg.total_days)})",
            flush=True,
        )
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(cfg.total_days),
            contract=contract,
            flags=feature_flags,
            cache_dir=getattr(cfg, "feature_cache_dir", None),
            asset_class=asset_class,
            # Se você pediu histórico completo (total_days<=0), garanta que o cache também foi feito assim.
            strict_total_days=(int(cfg.total_days) <= 0),
        )
        # se algum símbolo falhar no cache, removemos aqui para evitar erro nos períodos
        symbols = [s for s in symbols if s in cache_map]
        print(f"[sniper-train] cache pronto: símbolos_ok={len(symbols)}", flush=True)
        if not symbols:
            raise RuntimeError("Nenhum símbolo restou após gerar cache (ver logs [cache])")

    full_pool_pack = None
    if bool(getattr(cfg, "use_full_entry_pool", True)):
        pool_rows = int(getattr(cfg, "full_pool_max_rows_entry", 0) or 0)
        base_rows = int(getattr(cfg, "max_rows_entry", 2_000_000) or 2_000_000)
        if pool_rows <= base_rows:
            pool_rows = max(base_rows * 2, base_rows)
        reuse_pool = str(os.getenv("SNIPER_FULL_POOL_REUSE", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        force_rebuild = str(os.getenv("SNIPER_FULL_POOL_REBUILD", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        cache_fp = _cache_map_fingerprint(cache_map)
        cache_key = _full_pool_cache_key(
            asset_class=asset_class,
            symbols=symbols,
            cfg=cfg,
            contract=contract,
            feature_flags=feature_flags,
            cache_fp=cache_fp,
            pool_rows=int(pool_rows),
        )
        pool_cache_file = _full_pool_cache_root() / asset_class / f"pool_{cache_key}.pkl"
        if reuse_pool and (not force_rebuild) and pool_cache_file.exists():
            full_pool_pack = _load_full_pool_cache(pool_cache_file)
            if full_pool_pack is not None:
                print(f"[sniper-train] pool-full: reutilizando cache ({pool_cache_file})", flush=True)
            else:
                print(f"[sniper-train] pool-full: cache invalido, reconstruindo ({pool_cache_file})", flush=True)
        use_last = str(os.getenv("SNIPER_FULL_POOL_USE_LAST", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        if full_pool_pack is None and reuse_pool and (not force_rebuild) and use_last:
            last_pack, last_path = _load_latest_full_pool_cache(_full_pool_cache_root(), asset_class)
            if last_pack is not None and last_path is not None:
                full_pool_pack = last_pack
                print(f"[sniper-train] pool-full: usando ultimo pool salvo ({last_path})", flush=True)
                # Materializa tambem no hash atual para nao depender sempre de fallback.
                try:
                    _save_full_pool_cache(pool_cache_file, full_pool_pack)
                    _prune_full_pool_cache(_full_pool_cache_root(), asset_class)
                    print(f"[sniper-train] pool-full: vinculado ao hash atual ({pool_cache_file})", flush=True)
                except Exception:
                    pass
        if full_pool_pack is None:
            print(f"[sniper-train] pool-full: construindo 1x (rows_target={pool_rows})", flush=True)
            if bool(getattr(cfg, "use_feature_cache", True)):
                full_pool_pack = prepare_sniper_dataset_from_cache(
                    symbols,
                    total_days=int(cfg.total_days),
                    remove_tail_days=0,
                    contract=contract,
                    cache_map=cache_map,
                    entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                    max_rows_entry=int(pool_rows),
                    max_rows_exit=int(getattr(cfg, "max_rows_exit", 2_000_000)),
                    seed=7331,
                    asset_class=asset_class,
                    feature_flags=feature_flags,
                )
            else:
                full_pool_pack = prepare_sniper_dataset(
                    symbols,
                    total_days=int(cfg.total_days),
                    remove_tail_days=0,
                    contract=contract,
                    entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                    max_rows_entry=int(pool_rows),
                    max_rows_exit=int(getattr(cfg, "max_rows_exit", 2_000_000)),
                    seed=7331,
                    asset_class=asset_class,
                    feature_flags=feature_flags,
                )
            if reuse_pool and full_pool_pack is not None:
                try:
                    _save_full_pool_cache(pool_cache_file, full_pool_pack)
                    _prune_full_pool_cache(_full_pool_cache_root(), asset_class)
                    print(f"[sniper-train] pool-full: cache salvo ({pool_cache_file})", flush=True)
                except Exception as e:
                    print(f"[sniper-train] pool-full: falha ao salvar cache ({type(e).__name__}: {e})", flush=True)
        try:
            for _side_name, _b in (("long", full_pool_pack.entry_long), ("short", full_pool_pack.entry_short)):
                if _b is not None and _b.X.size > 0:
                    _pos = int((_b.y >= 0.5).sum())
                    _neg = int((_b.y < 0.5).sum())
                    print((f"[sniper-train] pool-full entry_{_side_name}: pos={_pos:,} neg={_neg:,}").replace(",", "."), flush=True)
        except Exception:
            pass

    saved_periods: list[tuple[int, Path]] = []

    for tail in cfg.offsets_days:
        print(f"[sniper-train] periodo T-{tail}d", flush=True)
        if full_pool_pack is not None:
            pack = _slice_pack_from_full_pool(
                full_pool_pack,
                remove_tail_days=int(tail),
                ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                max_rows_exit=int(getattr(cfg, "max_rows_exit", 2_000_000)),
                seed=1337 + int(tail),
            )
        elif bool(getattr(cfg, "use_feature_cache", True)):
            pack = prepare_sniper_dataset_from_cache(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                cache_map=cache_map,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                max_rows_exit=int(getattr(cfg, "max_rows_exit", 2_000_000)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
            )
        else:
            pack = prepare_sniper_dataset(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                max_rows_exit=int(getattr(cfg, "max_rows_exit", 2_000_000)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
            )
        try:
            n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            n_skip = len(pack.symbols_skipped) if getattr(pack, "symbols_skipped", None) else 0
            te = str(pd.to_datetime(pack.train_end_utc)) if getattr(pack, "train_end_utc", None) is not None else "None"
            print(f"[sniper-train] {tail}d: train_end_utc={te} | symbols_used={n_used} skipped={n_skip}", flush=True)
        except Exception:
            pass
        # protecao anti-escassez: se poucos simbolos sobreviveram nesse tail, nao vale treinar.
        try:
            min_used = int(getattr(cfg, "min_symbols_used_per_period", 0) or 0)
        except Exception:
            min_used = 0
        if min_used > 0:
            try:
                n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            except Exception:
                n_used = 0
            if int(n_used) < int(min_used):
                print(f"[sniper-train] {tail}d: symbols_used={n_used} < min_symbols_used_per_period={min_used} -> pulando", flush=True)
                continue
        if (pack.entry_long is None or pack.entry_long.X.size == 0) and (pack.entry_short is None or pack.entry_short.X.size == 0):
            print(f"[sniper-train] {tail}d: dataset vazio, pulando")
            continue
        dataset_bin_tables: dict[str, pd.DataFrame] = {}
        for side_name, side_batch in (("long", pack.entry_long), ("short", pack.entry_short)):
            try:
                if side_batch is not None and side_batch.X.size > 0:
                    pos_e = int((side_batch.y >= 0.5).sum())
                    neg_e = int((side_batch.y < 0.5).sum())
                    print((f"[sniper-train] dataset entry_{side_name}: pos={pos_e:,} neg={neg_e:,}").replace(",", "."), flush=True)
                    if getattr(side_batch, "w", None) is not None and side_batch.w.size == side_batch.y.size:
                        bin_step = float(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_STEP_X10", "1") or "1") / 10.0
                        dataset_bin_tables[side_name] = _entry_weight_bin_table(side_batch.y, side_batch.w, step=bin_step)
                        msg = _entry_weight_bin_mass_str(side_batch.y, side_batch.w, step=bin_step)
                        print(f"[sniper-train] dataset entry_{side_name} w-bins: {msg}", flush=True)
            except Exception:
                pass
        if train_exit_model:
            try:
                if pack.exit is not None and pack.exit.X.size > 0:
                    yx = np.asarray(pack.exit.y, dtype=np.float32)
                    print(
                        (
                            f"[sniper-train] dataset exit: rows={yx.size:,} "
                            f"span_p50={float(np.nanpercentile(yx, 50)):.1f} "
                            f"span_p90={float(np.nanpercentile(yx, 90)):.1f}"
                        ).replace(",", "."),
                        flush=True,
                    )
            except Exception:
                pass

        entry_specs = [("long", "long"), ("short", "short")]
        entry_batches = {"long": pack.entry_long, "short": pack.entry_short}
        entry_models: dict[str, tuple[xgb.Booster, dict]] = {}
        for name, tag in entry_specs:
            b = entry_batches.get(name)
            if b is None or b.X.size == 0:
                print(f"[sniper-train] EntryScore {name}: dataset vazio, pulando", flush=True)
                continue
            print(f"[sniper-train] treinando EntryScore {name}...", flush=True)
            m, m_meta = _train_xgb_classifier(b, entry_params)
            entry_models[name] = (m, m_meta)
            print(f"[sniper-train] EntryScore {name}: best_iter={m_meta['best_iteration']}", flush=True)

        exit_model_pack: tuple[xgb.Booster, dict] | None = None
        if (not train_exit_model):
            print("[sniper-train] ExitSpan desabilitado (SNIPER_TRAIN_EXIT_MODEL=0)", flush=True)
        elif pack.exit is not None and pack.exit.X.size > 0:
            exit_y_transform = str(os.getenv("SNIPER_EXIT_Y_TRANSFORM", "none") or "none").strip().lower()
            if exit_y_transform not in {"none", "log1p"}:
                exit_y_transform = "none"
            print("[sniper-train] treinando ExitSpan (reg)...", flush=True)
            m_exit, m_exit_meta = _train_xgb_regressor(pack.exit, exit_params, y_transform=exit_y_transform)
            exit_model_pack = (m_exit, m_exit_meta)
            print(
                f"[sniper-train] ExitSpan: best_iter={m_exit_meta.get('best_iteration')} "
                f"rmse={float((m_exit_meta.get('metrics') or {}).get('rmse', float('nan'))):.4f} "
                f"rmse_cal={float((m_exit_meta.get('metrics') or {}).get('rmse_cal', float('nan'))):.4f}",
                flush=True,
            )

        if not entry_models:
            print(f"[sniper-train] {tail}d: nenhum modelo treinado, pulando", flush=True)
            continue

        period_train_end = None
        try:
            if getattr(pack, "train_end_utc", None) is not None:
                period_train_end = pd.to_datetime(pack.train_end_utc)
            elif pack.entry_long is not None and pack.entry_long.ts.size:
                period_train_end = pd.to_datetime(pack.entry_long.ts.max())
            elif pack.entry_short is not None and pack.entry_short.ts.size:
                period_train_end = pd.to_datetime(pack.entry_short.ts.max())
        except Exception:
            period_train_end = None

        period_dir = run_dir / f"period_{int(tail)}d"
        # salva modelos entry por lado (sem alias legado "mid")
        for name, _tag in entry_specs:
            if name not in entry_models:
                continue
            model, _m_meta = entry_models[name]
            d = period_dir / f"entry_model_{name}"
            d.mkdir(parents=True, exist_ok=True)
            _save_model(model, d / "model_entry.json")
        if exit_model_pack is not None:
            d_exit = period_dir / "exit_model"
            d_exit.mkdir(parents=True, exist_ok=True)
            _save_model(exit_model_pack[0], d_exit / "model_exit.json")
        save_bin_plot = str(os.getenv("SNIPER_ENTRY_BIN_PLOT", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        bin_plot_paths: dict[str, str] = {}
        if save_bin_plot:
            for name, _tag in entry_specs:
                if name not in entry_models:
                    continue
                _m, _m_meta = entry_models[name]
                tables: dict[str, pd.DataFrame] = {}
                tb_ds = dataset_bin_tables.get(name)
                if tb_ds is not None and (not tb_ds.empty):
                    tables["dataset"] = tb_ds
                lb = dict((_m_meta or {}).get("loss_weight_bins") or {})
                rec_tr = list(lb.get("train") or [])
                rec_va = list(lb.get("val") or [])
                if rec_tr:
                    tables["loss_train"] = pd.DataFrame(rec_tr)
                if rec_va:
                    tables["loss_val"] = pd.DataFrame(rec_va)
                if not tables:
                    continue
                out_html = period_dir / f"entry_bins_{name}.html"
                try:
                    _save_entry_weight_bins_plot(
                        tables,
                        out_html=out_html,
                        title=f"entry {name} | T-{int(tail)}d",
                    )
                    if out_html.exists():
                        bin_plot_paths[name] = str(out_html)
                        print(f"[sniper-train] entry_{name} bins plot: {out_html}", flush=True)
                except Exception as e:
                    print(f"[sniper-train] entry_{name} bins plot falhou: {type(e).__name__}: {e}", flush=True)
        base_name = "long" if "long" in entry_models else "short"
        base_batch = entry_batches[base_name]
        base_meta = entry_models[base_name][1]
        meta = {
            "entry": {
                "feature_cols": base_batch.feature_cols,
                "calibration": base_meta.get("calibrator", {"type": "identity"}),
            },
            # Ponto final do treino (auditavel): preferimos o valor deterministico vindo do dataflow
            # (cutoff - lookahead). Se nao existir, cai no max(ts) do dataset amostrado.
            "train_end_utc": (
                str(pd.to_datetime(period_train_end))
                if period_train_end is not None
                else (np.datetime_as_string(np.max(base_batch.ts), unit="s") if base_batch.ts.size else None)
            ),
            "symbols": (pack.symbols_used if getattr(pack, "symbols_used", None) else pack.symbols),
            "symbols_total": int(len(pack.symbols) if getattr(pack, "symbols", None) else 0),
            "symbols_used": (pack.symbols_used or None),
            "symbols_skipped": (pack.symbols_skipped or None),
        }
        if exit_model_pack is not None:
            try:
                y_exit = np.asarray(pack.exit.y, dtype=np.float32)
                clip_min = float(np.nanpercentile(y_exit, 2)) if y_exit.size else 3.0
                clip_max = float(np.nanpercentile(y_exit, 98)) if y_exit.size else 600.0
                if not np.isfinite(clip_min):
                    clip_min = 3.0
                if not np.isfinite(clip_max):
                    clip_max = max(clip_min + 1.0, 600.0)
                if clip_max <= clip_min:
                    clip_max = clip_min + 1.0
            except Exception:
                clip_min, clip_max = 3.0, 600.0
            meta["exit"] = {
                "feature_cols": list(pack.exit.feature_cols),
                "transform": str((exit_model_pack[1] or {}).get("transform", "none")),
                "metrics": dict((exit_model_pack[1] or {}).get("metrics", {}) or {}),
                "calibration": dict((exit_model_pack[1] or {}).get("calibration", {}) or {"type": "identity"}),
                "balance": dict((exit_model_pack[1] or {}).get("balance", {}) or {}),
                "target": "ema_span_bars",
                "target_clip_bars": [float(clip_min), float(clip_max)],
            }
        # meta extra por lado
        for name, _tag in entry_specs:
            if name not in entry_models:
                continue
            _m, _m_meta = entry_models[name]
            meta[f"entry_{name}"] = {
                "feature_cols": entry_batches[name].feature_cols,
                "calibration": _m_meta["calibrator"],
                "loss_weight_bins": dict((_m_meta or {}).get("loss_weight_bins") or {}),
                "dataset_weight_bins": (
                    dataset_bin_tables[name].to_dict(orient="records")
                    if name in dataset_bin_tables and (not dataset_bin_tables[name].empty)
                    else []
                ),
                "bins_plot_file": bin_plot_paths.get(name, ""),
            }
        (period_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sniper-train] {tail}d salvo em {period_dir}")
        saved_periods.append((int(tail), period_dir))

    _stabilize_saved_entry_calibrations(saved_periods)

    return run_dir


if __name__ == "__main__":
    run = train_sniper_models()
    print(f"[sniper-train] modelos salvos em: {run}")

