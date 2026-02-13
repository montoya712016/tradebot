# -*- coding: utf-8 -*-
"""
Treinamento dos modelos Sniper (EntryScore) usando walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Any

import os
import sys
import time
import json


import numpy as np
import pandas as pd
try:
    from utils.adaptive_parallel import AdaptiveParallelPolicy, run_adaptive_thread_map
except Exception:
    from utils.adaptive_parallel import AdaptiveParallelPolicy, run_adaptive_thread_map  # type: ignore[import]
try:
    from utils.thermal_guard import ThermalGuard
except Exception:
    from utils.thermal_guard import ThermalGuard  # type: ignore[import]

def _import_xgb():
    import xgboost as xgb  # type: ignore

    return xgb


def _import_catboost_regressor():
    from catboost import CatBoostRegressor  # type: ignore

    return CatBoostRegressor


def _import_catboost_classifier():
    from catboost import CatBoostClassifier  # type: ignore

    return CatBoostClassifier


def _import_catboost_pool():
    from catboost import Pool  # type: ignore

    return Pool

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return bool(default)
    return v.strip().lower() not in {"0", "false", "no", "off"}

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _print_dist_stats(tag: str, arr: np.ndarray, *, weights: np.ndarray | None = None) -> None:
    try:
        x = np.asarray(arr, dtype=np.float64)
        m = np.isfinite(x)
        if not np.any(m):
            return
        x = x[m]
        q = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
        msg = (
            f"[sniper-train] {tag}: n={x.size} mean={np.mean(x):+.5f} std={np.std(x):.5f} "
            f"q01={q[0]:+.5f} q05={q[1]:+.5f} q50={q[2]:+.5f} q95={q[3]:+.5f} q99={q[4]:+.5f} "
            f"min={np.min(x):+.5f} max={np.max(x):+.5f}"
        )
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape == np.asarray(arr).shape:
                w = w[m]
                wm = np.isfinite(w) & (w > 0)
                if np.any(wm):
                    msg += f" wmean={np.average(x[wm], weights=w[wm]):+.5f}"
        print(msg, flush=True)
    except Exception:
        return


def _print_regression_balance_stats(tag: str, arr: np.ndarray, *, weights: np.ndarray | None = None) -> None:
    try:
        x = np.asarray(arr, dtype=np.float64)
        m = np.isfinite(x)
        if not np.any(m):
            return
        x = x[m]
        nz = float(np.mean(np.abs(x) > 1e-12))
        p10 = float(np.mean(x >= 10.0))
        p20 = float(np.mean(x >= 20.0))
        p40 = float(np.mean(x >= 40.0))
        msg = f"[sniper-train] {tag}: nz={nz:.3f} ge10={p10:.3f} ge20={p20:.3f} ge40={p40:.3f}"
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape == np.asarray(arr).shape:
                w = w[m]
                wm = np.isfinite(w) & (w > 0)
                if np.any(wm):
                    ww = w[wm]
                    xx = x[wm]
                    sw = float(np.sum(ww))
                    if sw > 0:
                        wz = float(np.sum(ww[np.abs(xx) > 1e-12]) / sw)
                        wp20 = float(np.sum(ww[xx >= 20.0]) / sw)
                        msg += f" w_nz={wz:.3f} w_ge20={wp20:.3f}"
        print(msg, flush=True)
    except Exception:
        return


def _print_binary_balance_stats(tag: str, arr: np.ndarray, *, weights: np.ndarray | None = None) -> None:
    try:
        y = np.asarray(arr, dtype=np.float64)
        m = np.isfinite(y)
        if not np.any(m):
            return
        y = y[m]
        yb = y >= 0.5
        n = int(yb.size)
        pos = int(np.sum(yb))
        neg = int(n - pos)
        pos_ratio = (float(pos) / float(n)) if n > 0 else 0.0
        msg = f"[sniper-train] {tag}: n={n} pos={pos} neg={neg} pos_ratio={pos_ratio:.3f}"
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape == np.asarray(arr).shape:
                w = w[m]
                wm = np.isfinite(w) & (w > 0)
                if np.any(wm):
                    ww = w[wm]
                    yy = yb[wm]
                    sw = float(np.sum(ww))
                    if sw > 0:
                        w_pos_ratio = float(np.sum(ww[yy]) / sw)
                        msg += f" w_pos_ratio={w_pos_ratio:.3f}"
        print(msg, flush=True)
    except Exception:
        return

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return float(np.nanmedian(x)) if x.size else 0.0
    x = x[m]
    w = w[m]
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    csum = np.cumsum(ws)
    cutoff = 0.5 * float(csum[-1])
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = max(0, min(idx, xs.size - 1))
    return float(xs[idx])

def _compute_pred_bias(pred: np.ndarray, weights: np.ndarray | None) -> float:
    method = str(os.getenv("SNIPER_PRED_BIAS_METHOD", "median")).strip().lower()
    p = np.asarray(pred, dtype=np.float64)
    if p.size == 0:
        return 0.0
    if method == "mean":
        if weights is not None and np.size(weights) == np.size(p):
            w = np.asarray(weights, dtype=np.float64)
            m = np.isfinite(p) & np.isfinite(w) & (w > 0)
            if np.any(m):
                return float(np.average(p[m], weights=w[m]))
        m = np.isfinite(p)
        return float(np.mean(p[m])) if np.any(m) else 0.0
    if weights is not None and np.size(weights) == np.size(p):
        return _weighted_median(p, np.asarray(weights, dtype=np.float64))
    m = np.isfinite(p)
    if not np.any(m):
        return 0.0
    return float(np.median(p[m]))

def _fit_affine_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None,
) -> dict:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        m &= np.isfinite(w) & (w > 0.0)
        ww = w[m]
    else:
        ww = np.ones(int(np.sum(m)), dtype=np.float64)
    xt = yp[m]
    yy = yt[m]
    if xt.size < 8:
        return {"kind": "affine", "a": 1.0, "b": 0.0}
    sw = float(np.sum(ww))
    if sw <= 0.0:
        return {"kind": "affine", "a": 1.0, "b": 0.0}
    mx = float(np.sum(ww * xt) / sw)
    my = float(np.sum(ww * yy) / sw)
    dx = xt - mx
    dy = yy - my
    var_x = float(np.sum(ww * dx * dx) / sw)
    cov_xy = float(np.sum(ww * dx * dy) / sw)
    if not np.isfinite(var_x) or var_x <= 1e-12:
        a = 1.0
    else:
        a = float(cov_xy / var_x)
    b = float(my - a * mx)
    if not np.isfinite(a):
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return {"kind": "affine", "a": float(a), "b": float(b)}


def _fit_piecewise_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Ajuste simples por lados:
    - escala positiva para alinhar p90
    - escala negativa para alinhar p10
    Retorna s_pos, s_neg e rmse apos escala.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size < 8:
        err = yp - yt
        rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
        return {"s_pos": 1.0, "s_neg": 1.0, "rmse": rmse}
    pos = yp > 0
    neg = yp < 0
    s_pos = 1.0
    s_neg = 1.0
    try:
        if np.any(pos):
            q_pred = float(np.quantile(yp[pos], 0.90))
            q_true = float(np.quantile(yt[pos], 0.90))
            if np.isfinite(q_pred) and abs(q_pred) > 1e-12:
                s_pos = float(q_true / q_pred)
        if np.any(neg):
            q_pred = float(np.quantile(yp[neg], 0.10))
            q_true = float(np.quantile(yt[neg], 0.10))
            if np.isfinite(q_pred) and abs(q_pred) > 1e-12:
                s_neg = float(q_true / q_pred)
    except Exception:
        s_pos = 1.0
        s_neg = 1.0
    yp_pw = yp.copy()
    if np.any(pos):
        yp_pw[pos] *= s_pos
    if np.any(neg):
        yp_pw[neg] *= s_neg
    err = yp_pw - yt
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    return {"s_pos": float(s_pos), "s_neg": float(s_neg), "rmse": float(rmse)}


def _clamp_scale(v: float, lo: float = 0.5, hi: float = 3.0) -> float:
    try:
        if not np.isfinite(v):
            return 1.0
    except Exception:
        return 1.0
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _enable_vt_mode(stream) -> bool:
    """
    Try to enable ANSI/VT processing on Windows so in-place progress works.
    Returns True if VT is enabled (or not needed), False otherwise.
    """
    if os.name != "nt":
        return True
    try:
        import msvcrt
        import ctypes

        handle = msvcrt.get_osfhandle(stream.fileno())
        kernel32 = ctypes.windll.kernel32
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VT = 0x0004
        new_mode = mode.value | ENABLE_VT
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False
        return True
    except Exception:
        return False


_RAM_GUARD_WARNED = False
_THERMAL_GUARD = ThermalGuard.from_env()


def _check_ram_or_abort(threshold_pct: float, *, where: str) -> None:
    if threshold_pct <= 0:
        return
    if psutil is None:
        global _RAM_GUARD_WARNED
        if not _RAM_GUARD_WARNED:
            print("[sniper-train] aviso: psutil ausente, ram_guard desativado", flush=True)
            _RAM_GUARD_WARNED = True
        return
    try:
        used = float(psutil.virtual_memory().percent)
    except Exception:
        return
    if used >= float(threshold_pct):
        raise RuntimeError(f"RAM guard: uso {used:.1f}% >= {float(threshold_pct):.1f}% em {where}")


def _wait_ram_below_threshold(threshold_pct: float, *, where: str) -> None:
    """
    Soft RAM guard: aguarda RAM baixar em vez de abortar.
    Usado em trechos paralelos de build do pool para evitar falha do processo inteiro.
    """
    if threshold_pct <= 0 or psutil is None:
        return
    sleep_s = float(os.getenv("SNIPER_POOL_BUILD_RAM_WAIT_S", "2.0") or "2.0")
    log_every = int(os.getenv("SNIPER_POOL_BUILD_RAM_LOG_EVERY", "5") or "5")
    it = 0
    while True:
        try:
            used = float(psutil.virtual_memory().percent)
        except Exception:
            return
        if used < float(threshold_pct):
            return
        it += 1
        if (it % max(1, log_every)) == 0:
            print(
                f"[sniper-train] RAM throttle: uso {used:.1f}% >= {float(threshold_pct):.1f}% em {where}; aguardando...",
                flush=True,
            )
        time.sleep(max(0.2, sleep_s))


def _sample_temperatures(*, force: bool = False) -> dict[str, float | None]:
    s = _THERMAL_GUARD.sample(force=force)
    return {"cpu_c": s.cpu_c, "gpu_c": s.gpu_c}


def _thermal_guard_wait_if_needed(*, where: str, force_sample: bool = False) -> None:
    _THERMAL_GUARD.wait_until_safe(
        where=where,
        force_sample=force_sample,
        logger=lambda msg: print(f"[sniper-train] {msg}", flush=True),
    )


def _xgb_thermal_callbacks(xgb_module: Any, *, where: str) -> list[Any]:
    if not _env_bool("SNIPER_THERMAL_GUARD", default=False):
        return []
    callback_base = getattr(getattr(xgb_module, "callback", None), "TrainingCallback", None)
    if callback_base is None:
        return []

    class _ThermalCallback(callback_base):  # type: ignore[misc, valid-type]
        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            _thermal_guard_wait_if_needed(where=f"{where}:iter={int(epoch)}")
            return False

    return [_ThermalCallback()]


class _CatBoostThermalCallback:
    def __init__(self, where: str):
        self.where = str(where)

    def after_iteration(self, info) -> bool:
        it = getattr(info, "iteration", None)
        if it is None:
            _thermal_guard_wait_if_needed(where=self.where)
        else:
            _thermal_guard_wait_if_needed(where=f"{self.where}:iter={int(it)}")
        return True


def _fit_catboost_with_optional_thermal(model: Any, *, where: str, **fit_kwargs) -> None:
    if not _env_bool("SNIPER_THERMAL_GUARD", default=False):
        model.fit(**fit_kwargs)
        return

    # CatBoost em GPU não suporta callbacks de usuário.
    # Mantemos thermal guard no pré-fit e tentamos callback apenas em CPU.
    task_type = ""
    try:
        params = model.get_params() if hasattr(model, "get_params") else {}
        task_type = str((params or {}).get("task_type", "")).strip().upper()
    except Exception:
        task_type = ""

    if task_type == "GPU":
        model.fit(**fit_kwargs)
        return

    fit_kwargs["callbacks"] = [_CatBoostThermalCallback(where)]
    try:
        model.fit(**fit_kwargs)
        return
    except Exception as e:
        msg = str(e).lower()
        # versões/paths diferentes podem lançar TypeError ou CatBoostError
        # para callback não suportado.
        if ("callback" not in msg) and ("user defined callbacks are not supported for gpu" not in msg):
            raise
        print("[sniper-train] aviso: CatBoost sem suporte a callbacks; thermal_guard só no pré-fit", flush=True)
        fit_kwargs.pop("callbacks", None)
        model.fit(**fit_kwargs)

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
    # overrides opcionais por lado (ex.: {"long": {...}, "short": {...}})
    entry_params_by_side: dict | None = None
    contract: TradeContract = DEFAULT_TRADE_CONTRACT
    # dataset sizing (VRAM/RAM)
    max_rows_entry: int = 2_000_000
    entry_pool_full: bool = False
    entry_pool_dir: Path | None = None
    entry_pool_prefiltered: bool = True
    # se False, roda apenas o classificador (entry_cls_*), sem regressor entry_model_*
    entry_reg_enabled: bool = True
    # modelo para regressao (ex.: "catboost" ou "xgb")
    entry_model_type: str = "xgb"
    # regressao: janela usada para target (minutos). Se 0, usa contrato.
    entry_reg_window_min: int = 0
    # regressao: pesos por amostra
    entry_reg_weight_alpha: float = 4.0
    entry_reg_weight_power: float = 1.2
    # regressao: bins para balanceamento do target
    entry_reg_balance_bins: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 80.0)
    # modo do label: "signed" (antigo) ou "split_0_100" (long/short)
    entry_label_mode: str = "split_0_100"
    entry_label_scale: float = 100.0
    entry_sides: Sequence[str] = ("long", "short")
    # labels por lado para regressao/classificacao
    entry_reg_label_col_template: str = "edge_label_{side}"
    entry_reg_weight_col_template: str = ""
    entry_cls_enabled: bool = True
    entry_cls_model_type: str = "catboost"
    entry_cls_params: dict | None = None
    entry_cls_params_by_side: dict | None = None
    entry_cls_label_col_template: str = "entry_gate_{side}"
    entry_cls_positive_threshold: float = 50.0
    entry_cls_balance_bins: Sequence[float] = (0.5,)
    entry_cls_weight_col_template: str = ""
    entry_cls_target_pos_ratio: float = 0.5
    entry_cls_pos_weight: float = 1.0
    entry_cls_neg_weight: float = 1.0
    # regressao: balanceamento por distancia do zero (0 = uniforme)
    entry_reg_balance_distance_power: float = 1.0
    # regressao: fração mínima de amostras por bin (relativo ao bin mais pesado)
    entry_reg_balance_min_frac: float = 0.1
    # aborta se uso de RAM ultrapassar esse percentual (<=0 desativa)
    abort_ram_pct: float = 99.0
    use_feature_cache: bool = True
    # modo/asset
    asset_class: str = "crypto"
    # lista explícita de símbolos (prioritária a symbols_file/top_market_cap)
    symbols: Sequence[str] | None = None
    # flags de features custom (senão escolhe default por asset)
    feature_flags: dict | None = None
    # onde salvar/ler cache de features (senão usa padrão por asset)
    feature_cache_dir: Path | None = None
    # força reutilizar diretório de run (para retomar wf_XXX)
    run_dir: Path | None = None
    # Thresholds são definidos manualmente em config/thresholds.py (sem calibrar no treino).


DEFAULT_ENTRY_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 7,
    "min_child_weight": 5.0,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Observação: QuantileDMatrix costuma usar 256 bins por padrão; manter consistente evita
    # erro "Inconsistent max_bin (512 vs 256)" e permite treinar na GPU.
    "max_bin": 256,
    "lambda": 2.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "device": "cuda:0",
}

DEFAULT_ENTRY_CLS_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 3.0,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 256,
    "lambda": 2.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "device": "cuda:0",
}

def _sample_indices_regression(
    y: np.ndarray,
    bins: Sequence[float] | None,
    max_rows: int,
    rng: np.random.Generator,
    *,
    distance_power: float = 0.0,
    min_frac: float = 1.0,
) -> np.ndarray:
    if y.size == 0:
        return np.empty((0,), dtype=np.int64)
    if max_rows <= 0 or y.size <= max_rows:
        return np.arange(y.size, dtype=np.int64)
    edges = list(bins or [])
    edges = sorted(float(x) for x in edges)
    edges = [-np.inf] + edges + [np.inf]
    bins_n = max(1, len(edges) - 1)
    use_weighted = float(distance_power) > 0.0
    weights: list[float] | None = None
    if use_weighted:
        weights = []
        for i in range(bins_n):
            lo = edges[i]
            hi = edges[i + 1]
            if np.isfinite(lo) and np.isfinite(hi):
                center = 0.5 * (lo + hi)
            elif np.isfinite(hi):
                center = float(hi)
            elif np.isfinite(lo):
                center = float(lo)
            else:
                center = 0.0
            w = float(max(abs(center), 1e-6) ** float(distance_power))
            weights.append(w)
        max_w = max(weights) if weights else 1.0
        min_w = max_w * float(max(0.0, min_frac))
        weights = [max(w, min_w) for w in weights]
        total_w = float(sum(weights)) if weights else 1.0
    else:
        target_per_bin = max(1, int(round(max_rows / bins_n)))
    keep_idx: list[int] = []
    for i in range(bins_n):
        lo = edges[i]
        hi = edges[i + 1]
        sel = (y >= lo) & (y < hi)
        idx_bin = np.flatnonzero(sel)
        if use_weighted:
            target_per_bin = max(1, int(round(max_rows * (weights[i] / total_w))))
        if idx_bin.size <= target_per_bin:
            keep_idx.extend(idx_bin.tolist())
        else:
            pick = rng.choice(idx_bin, size=target_per_bin, replace=False)
            keep_idx.extend(pick.tolist())
    if len(keep_idx) > max_rows:
        keep_idx = rng.choice(np.array(keep_idx), size=max_rows, replace=False).tolist()
    keep = np.array(sorted(set(keep_idx)), dtype=np.int64)
    return keep


def _sample_indices_binary(
    y: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if y.size == 0:
        return np.empty((0,), dtype=np.int64)
    if max_rows <= 0 or y.size <= max_rows:
        return np.arange(y.size, dtype=np.int64)
    yb = y >= 0.5
    pos_idx = np.flatnonzero(yb)
    neg_idx = np.flatnonzero(~yb)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return rng.choice(np.arange(y.size, dtype=np.int64), size=max_rows, replace=False).astype(np.int64, copy=False)
    tgt_pos = int(max_rows // 2)
    tgt_neg = int(max_rows - tgt_pos)
    if pos_idx.size < tgt_pos:
        tgt_pos = int(pos_idx.size)
        tgt_neg = int(min(neg_idx.size, max_rows - tgt_pos))
    if neg_idx.size < tgt_neg:
        tgt_neg = int(neg_idx.size)
        tgt_pos = int(min(pos_idx.size, max_rows - tgt_neg))
    keep_parts: list[np.ndarray] = []
    if tgt_pos > 0:
        keep_parts.append(rng.choice(pos_idx, size=tgt_pos, replace=False))
    if tgt_neg > 0:
        keep_parts.append(rng.choice(neg_idx, size=tgt_neg, replace=False))
    if not keep_parts:
        return np.empty((0,), dtype=np.int64)
    keep = np.concatenate(keep_parts, axis=0)
    if keep.size > max_rows:
        keep = rng.choice(keep, size=max_rows, replace=False)
    return np.array(sorted(set(keep.tolist())), dtype=np.int64)


def _rebalance_binary_to_target_ratio(
    y: np.ndarray,
    target_pos_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Downsample da classe majoritária para aproximar y~Bernoulli(target_pos_ratio).
    Nunca faz upsample.
    """
    yb = np.asarray(y, dtype=np.float32) >= 0.5
    n = int(yb.size)
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    r = float(target_pos_ratio)
    if not np.isfinite(r) or r <= 0.0 or r >= 1.0:
        return np.arange(n, dtype=np.int64)
    pos_idx = np.flatnonzero(yb)
    neg_idx = np.flatnonzero(~yb)
    pos = int(pos_idx.size)
    neg = int(neg_idx.size)
    if pos == 0 or neg == 0:
        return np.arange(n, dtype=np.int64)
    cur_r = float(pos) / float(pos + neg)
    if abs(cur_r - r) < 1e-6:
        return np.arange(n, dtype=np.int64)
    if cur_r > r:
        # positivos demais -> reduz positivos
        keep_neg = neg_idx
        tgt_pos = int(round((r / (1.0 - r)) * float(neg)))
        tgt_pos = max(1, min(pos, tgt_pos))
        keep_pos = rng.choice(pos_idx, size=tgt_pos, replace=False)
    else:
        # negativos demais -> reduz negativos
        keep_pos = pos_idx
        tgt_neg = int(round(((1.0 - r) / r) * float(pos)))
        tgt_neg = max(1, min(neg, tgt_neg))
        keep_neg = rng.choice(neg_idx, size=tgt_neg, replace=False)
    keep = np.concatenate([keep_pos, keep_neg], axis=0)
    return np.array(sorted(set(keep.tolist())), dtype=np.int64)


def _save_entry_pool_symbol(pool_dir: Path, symbol: str, entry_batches: dict[str, SniperBatch]) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    base_batch = None
    for b in entry_batches.values():
        if b is not None and b.X.size > 0:
            base_batch = b
            break
    if base_batch is None:
        return
    df = pd.DataFrame(base_batch.X, columns=list(base_batch.feature_cols))
    df.index = pd.to_datetime(base_batch.ts)
    # salva label_entry/weight (regressão)
    b = base_batch
    df["label_entry"] = b.y.astype(np.float32, copy=False)
    df["weight"] = b.w.astype(np.float32, copy=False)
    out_dir = pool_dir / "entry_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{str(symbol).upper()}.parquet"
    df.to_parquet(out_path, index=True)



def _write_pool_meta(pool_dir: Path, windows: Sequence[int], max_ts: pd.Timestamp | None) -> None:
    meta = {
        "windows": {f"w{int(w)}": int(w) for w in windows},
        "layout": "per_symbol_combined",
        "max_ts": None if max_ts is None else str(pd.to_datetime(max_ts)),
    }
    (pool_dir / "pool_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")



def _load_entry_pool(pool_dir: Path) -> tuple[dict[str, list[Path]], pd.Timestamp | None, str]:
    meta_path = pool_dir / "pool_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"pool_meta.json n??o encontrado em {pool_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layout = str(meta.get("layout") or "per_symbol_combined")
    out: dict[str, list[Path]] = {}
    if layout == "per_symbol_combined":
        d = pool_dir / "entry_pool"
        out["_combined"] = sorted(d.glob("*.parquet")) if d.exists() else []
    else:
        for name in (meta.get("windows") or {}).keys():
            d = pool_dir / f"entry_pool_{name}"
            if not d.exists():
                continue
            out[name] = sorted(d.glob("*.parquet"))
    max_ts = meta.get("max_ts")
    return out, (pd.to_datetime(max_ts) if max_ts else None), layout


def _default_symbols_file(asset_class: str) -> Path:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        return DEFAULT_STOCKS_SYMBOLS_FILE
    return DEFAULT_SYMBOLS_FILE


def _reservoir_merge(
    keys_old: np.ndarray,
    idx_old: np.ndarray,
    keys_new: np.ndarray,
    idx_new: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if keys_old.size == 0:
        if keys_new.size <= k:
            return keys_new, idx_new
        sel = np.argpartition(keys_new, k - 1)[:k]
        return keys_new[sel], idx_new[sel]
    # concat and keep best k (smallest keys)
    keys_all = np.concatenate([keys_old, keys_new])
    idx_all = np.concatenate([idx_old, idx_new])
    if keys_all.size <= k:
        return keys_all, idx_all
    sel = np.argpartition(keys_all, k - 1)[:k]
    return keys_all[sel], idx_all[sel]


def _reservoir_update(
    keys_res: np.ndarray,
    X_res: np.ndarray,
    y_res: np.ndarray,
    w_res: np.ndarray,
    ts_res: np.ndarray,
    keys_new: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    w_new: np.ndarray,
    ts_new: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0 or keys_new.size == 0:
        return keys_res, X_res, y_res, w_res, ts_res
    # if reservoir not full, just take up to k
    if keys_res.size == 0:
        if keys_new.size <= k:
            return keys_new, X_new, y_new, w_new, ts_new
        sel = np.argpartition(keys_new, k - 1)[:k]
        return keys_new[sel], X_new[sel], y_new[sel], w_new[sel], ts_new[sel]
    if keys_res.size >= k:
        # keep only candidates better than current threshold
        thr = np.partition(keys_res, k - 1)[k - 1]
        better = keys_new < thr
        if not np.any(better):
            return keys_res, X_res, y_res, w_res, ts_res
        keys_new = keys_new[better]
        X_new = X_new[better]
        y_new = y_new[better]
        w_new = w_new[better]
        ts_new = ts_new[better]
    if keys_new.size == 0:
        return keys_res, X_res, y_res, w_res, ts_res
    keys_all = np.concatenate([keys_res, keys_new])
    X_all = np.concatenate([X_res, X_new])
    y_all = np.concatenate([y_res, y_new])
    w_all = np.concatenate([w_res, w_new])
    ts_all = np.concatenate([ts_res, ts_new])
    if keys_all.size <= k:
        return keys_all, X_all, y_all, w_all, ts_all
    sel = np.argpartition(keys_all, k - 1)[:k]
    return keys_all[sel], X_all[sel], y_all[sel], w_all[sel], ts_all[sel]


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


def _train_xgb_regressor(batch: SniperBatch, params: dict, *, y_transform: str = "log1p") -> tuple[Any, dict]:
    xgb = _import_xgb()
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

    def _apply_transform(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float32)
        if y_transform == "none":
            return yy
        if y_transform == "log1p":
            yy = np.maximum(0.0, yy)
            return np.log1p(yy)
        raise ValueError(f"y_transform inválido: {y_transform}")

    ytr = _apply_transform(ytr0)
    yva = _apply_transform(yva0)

    if use_quantile:
        mb = int(params.get("max_bin", 256))
        params = dict(params)
        params["max_bin"] = mb
        try:
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=mb)
        except TypeError:
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)

    watch = [(dtrain, "train"), (dvalid, "val")]
    print(f"[xgb] (reg) train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,}".replace(",", "."), flush=True)
    _thermal_guard_wait_if_needed(where="xgb_reg:pre_fit", force_sample=True)
    xgb_callbacks = _xgb_thermal_callbacks(xgb, where="xgb_reg")
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2500,
            evals=watch,
            early_stopping_rounds=250,
            verbose_eval=50,
            callbacks=xgb_callbacks,
        )
    except Exception as e:
        if device.startswith("cuda"):
            print(f"[xgb] treino reg em GPU falhou ({type(e).__name__}: {e}) -> tentando CPU", flush=True)
            params_cpu = dict(params)
            params_cpu["device"] = "cpu"
            params_cpu["tree_method"] = "hist"
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
            watch = [(dtrain, "train"), (dvalid, "val")]
            booster = xgb.train(
                params=params_cpu,
                dtrain=dtrain,
                num_boost_round=2500,
                evals=watch,
                early_stopping_rounds=250,
                verbose_eval=50,
                callbacks=xgb_callbacks,
            )
        else:
            raise

    pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    err = pred.astype(np.float64) - yva.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    try:
        pred_bias = _compute_pred_bias(pred.astype(np.float64), wva.astype(np.float64) if wva is not None else None)
    except Exception:
        pred_bias = 0.0
    calibration = None
    if _env_bool("SNIPER_ENABLE_CALIBRATION", default=False):
        try:
            cal_try = _fit_affine_calibration(yva.astype(np.float64), pred.astype(np.float64), wva)
            a = float(cal_try.get("a", 1.0))
            b = float(cal_try.get("b", 0.0))
            pred_cal = (a * pred.astype(np.float64)) + b
            rmse_cal = float(np.sqrt(np.mean((pred_cal - yva.astype(np.float64)) ** 2)))
            try:
                err_raw = pred.astype(np.float64) - yva.astype(np.float64)
                err_cal = pred_cal - yva.astype(np.float64)
                if wva is not None:
                    ww = np.asarray(wva, dtype=np.float64)
                    m = np.isfinite(ww) & (ww > 0)
                    bias_raw = float(np.average(err_raw[m], weights=ww[m])) if np.any(m) else float(np.mean(err_raw))
                    bias_cal = float(np.average(err_cal[m], weights=ww[m])) if np.any(m) else float(np.mean(err_cal))
                else:
                    bias_raw = float(np.mean(err_raw))
                    bias_cal = float(np.mean(err_cal))
            except Exception:
                bias_raw = 0.0
                bias_cal = 0.0
            keep = (rmse_cal <= (rmse * 0.999)) or (rmse_cal <= (rmse * 1.01) and abs(bias_cal) <= abs(bias_raw) * 0.9)
            calib_msg = (
                f"[sniper-train] calib affine: a={a:+.5f} b={b:+.5f} rmse_raw={rmse:.5f} rmse_cal={rmse_cal:.5f} "
                f"bias_raw={bias_raw:+.5f} bias_cal={bias_cal:+.5f} keep={keep}"
            )
            if keep:
                pw = _fit_piecewise_scale(yva.astype(np.float64), pred_cal.astype(np.float64))
                rmse_pw = float(pw.get("rmse", rmse_cal))
                s_pos = _clamp_scale(float(pw.get("s_pos", 1.0)))
                s_neg = _clamp_scale(float(pw.get("s_neg", 1.0)))
                force_pw = _env_bool("SNIPER_FORCE_PIECEWISE_CALIB", default=False)
                use_pw = force_pw or (rmse_pw <= (rmse_cal * 1.02) and (abs(s_pos - 1.0) > 0.02 or abs(s_neg - 1.0) > 0.02))
                calib_msg += f" rmse_pw={rmse_pw:.5f} s_pos={s_pos:+.3f} s_neg={s_neg:+.3f} use_pw={use_pw}"
                if use_pw:
                    calibration = {"kind": "affine_piecewise", "a": a, "b": b, "s_pos": s_pos, "s_neg": s_neg}
                else:
                    calibration = cal_try
            print(calib_msg, flush=True)
        except Exception:
            calibration = None
    meta = {
        "transform": y_transform,
        "best_iteration": booster.best_iteration,
        "metrics": {"rmse": rmse, "mae": mae},
        "pred_bias": pred_bias,
        "calibration": calibration,
    }
    return booster, meta


def _train_catboost_regressor(batch: SniperBatch, params: dict | None = None) -> tuple[object, dict]:
    CatBoostRegressor = _import_catboost_regressor()
    if CatBoostRegressor is None:
        raise RuntimeError("catboost nao instalado")
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    Xtr = batch.X[tr_idx]
    ytr = batch.y[tr_idx]
    Xva = batch.X[va_idx]
    yva = batch.y[va_idx]
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else None
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else None

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1200,
        "learning_rate": 0.05,
        "depth": 8,
        "subsample": 0.8,
        "random_seed": 1337,
        "verbose": 50,
        "allow_writing_files": False,
    }
    if params:
        allow = {
            "loss_function",
            "eval_metric",
            "iterations",
            "learning_rate",
            "depth",
            "subsample",
            "bootstrap_type",
            "random_seed",
            "l2_leaf_reg",
            "bagging_temperature",
            "border_count",
            "min_data_in_leaf",
            "verbose",
            "od_type",
            "od_wait",
            "allow_writing_files",
            "task_type",
            "devices",
        }
        clean = {k: v for k, v in dict(params).items() if k in allow}
        cb_params.update(clean)
    # subsample exige bootstrap_type compatÃ­vel (Bayesian nÃ£o suporta subsample)
    if cb_params.get("subsample") is not None:
        bt = str(cb_params.get("bootstrap_type", "")).strip().lower()
        if bt in ("", "bayesian"):
            cb_params["bootstrap_type"] = "Bernoulli"
        if str(cb_params.get("bootstrap_type", "")).strip().lower() != "bayesian":
            cb_params.pop("bagging_temperature", None)
    # força GPU quando o device do entry_params estiver em CUDA
    try:
        dev = str((params or {}).get("device", "")).lower()
    except Exception:
        dev = ""
    if dev.startswith("cuda"):
        cb_params.setdefault("task_type", "GPU")
        # catboost usa string tipo "0" ou "0:1"
        cb_params.setdefault("devices", dev.replace("cuda:", "") or "0")
    # CatBoost exige métricas com case correto (ex.: "RMSE")
    try:
        if str(cb_params.get("eval_metric", "")).strip().lower() == "rmse":
            cb_params["eval_metric"] = "RMSE"
    except Exception:
        pass
    model = CatBoostRegressor(**cb_params)
    if wva is not None:
        Pool = _import_catboost_pool()
        eval_pool = Pool(Xva, yva, weight=wva)
        _thermal_guard_wait_if_needed(where="catboost_reg:pre_fit", force_sample=True)
        _fit_catboost_with_optional_thermal(
            model,
            where="catboost_reg",
            X=Xtr,
            y=ytr,
            sample_weight=wtr,
            eval_set=eval_pool,
            use_best_model=True,
        )
    else:
        _thermal_guard_wait_if_needed(where="catboost_reg:pre_fit", force_sample=True)
        _fit_catboost_with_optional_thermal(
            model,
            where="catboost_reg",
            X=Xtr,
            y=ytr,
            sample_weight=wtr,
            eval_set=(Xva, yva),
            use_best_model=True,
        )
    pred = model.predict(Xva)
    err = pred.astype(np.float64) - yva.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    try:
        pred_bias = _compute_pred_bias(pred.astype(np.float64), wva.astype(np.float64) if wva is not None else None)
    except Exception:
        pred_bias = 0.0
    calibration = None
    if _env_bool("SNIPER_ENABLE_CALIBRATION", default=False):
        try:
            cal_try = _fit_affine_calibration(yva.astype(np.float64), pred.astype(np.float64), wva)
            a = float(cal_try.get("a", 1.0))
            b = float(cal_try.get("b", 0.0))
            pred_cal = (a * pred.astype(np.float64)) + b
            rmse_cal = float(np.sqrt(np.mean((pred_cal - yva.astype(np.float64)) ** 2)))
            try:
                err_raw = pred.astype(np.float64) - yva.astype(np.float64)
                err_cal = pred_cal - yva.astype(np.float64)
                if wva is not None:
                    ww = np.asarray(wva, dtype=np.float64)
                    m = np.isfinite(ww) & (ww > 0)
                    bias_raw = float(np.average(err_raw[m], weights=ww[m])) if np.any(m) else float(np.mean(err_raw))
                    bias_cal = float(np.average(err_cal[m], weights=ww[m])) if np.any(m) else float(np.mean(err_cal))
                else:
                    bias_raw = float(np.mean(err_raw))
                    bias_cal = float(np.mean(err_cal))
            except Exception:
                bias_raw = 0.0
                bias_cal = 0.0
            keep = (rmse_cal <= (rmse * 0.999)) or (rmse_cal <= (rmse * 1.01) and abs(bias_cal) <= abs(bias_raw) * 0.9)
            calib_msg = (
                f"[sniper-train] calib affine: a={a:+.5f} b={b:+.5f} rmse_raw={rmse:.5f} rmse_cal={rmse_cal:.5f} "
                f"bias_raw={bias_raw:+.5f} bias_cal={bias_cal:+.5f} keep={keep}"
            )
            if keep:
                pw = _fit_piecewise_scale(yva.astype(np.float64), pred_cal.astype(np.float64))
                rmse_pw = float(pw.get("rmse", rmse_cal))
                s_pos = _clamp_scale(float(pw.get("s_pos", 1.0)))
                s_neg = _clamp_scale(float(pw.get("s_neg", 1.0)))
                force_pw = _env_bool("SNIPER_FORCE_PIECEWISE_CALIB", default=False)
                use_pw = force_pw or (rmse_pw <= (rmse_cal * 1.02) and (abs(s_pos - 1.0) > 0.02 or abs(s_neg - 1.0) > 0.02))
                calib_msg += f" rmse_pw={rmse_pw:.5f} s_pos={s_pos:+.3f} s_neg={s_neg:+.3f} use_pw={use_pw}"
                if use_pw:
                    calibration = {"kind": "affine_piecewise", "a": a, "b": b, "s_pos": s_pos, "s_neg": s_neg}
                else:
                    calibration = cal_try
            print(calib_msg, flush=True)
        except Exception:
            calibration = None
    meta = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "metrics": {"rmse": rmse, "mae": mae},
        "pred_bias": pred_bias,
        "calibration": calibration,
    }
    return model, meta


def _binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float("nan")
    yt = yt[m]
    yp = np.clip(yp[m], 1e-7, 1.0 - 1e-7)
    return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))

def _fit_platt_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict | None:
    """
    Platt scaling sobre logit(prob): p_cal = sigmoid(a*logit(p)+b)
    Ajuste via Newton-Raphson em regressao logistica com 1 feature.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        m &= np.isfinite(w) & (w > 0.0)
        ww = w[m]
    else:
        ww = None
    y = y[m]
    p = np.clip(p[m], 1e-6, 1.0 - 1e-6)
    if y.size < 100:
        return None
    # precisa de ambas classes
    pos = float(np.sum(y >= 0.5))
    neg = float(y.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        return None
    z = np.log(p / (1.0 - p))
    a = 1.0
    b = 0.0
    for _ in range(60):
        t = a * z + b
        t = np.clip(t, -40.0, 40.0)
        s = 1.0 / (1.0 + np.exp(-t))
        if ww is None:
            e = s - y
            r = s * (1.0 - s)
            g0 = float(np.sum(e * z))
            g1 = float(np.sum(e))
            h00 = float(np.sum(r * z * z)) + 1e-9
            h01 = float(np.sum(r * z))
            h11 = float(np.sum(r)) + 1e-9
        else:
            e = (s - y) * ww
            r = s * (1.0 - s) * ww
            g0 = float(np.sum(e * z))
            g1 = float(np.sum(e))
            h00 = float(np.sum(r * z * z)) + 1e-9
            h01 = float(np.sum(r * z))
            h11 = float(np.sum(r)) + 1e-9
        det = (h00 * h11) - (h01 * h01)
        if not np.isfinite(det) or abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a_new = a - da
        b_new = b - db
        if (not np.isfinite(a_new)) or (not np.isfinite(b_new)):
            break
        if abs(da) < 1e-7 and abs(db) < 1e-7:
            a, b = float(a_new), float(b_new)
            break
        a, b = float(a_new), float(b_new)
    return {"kind": "platt", "a": float(a), "b": float(b)}


def _apply_platt_calibration(y_prob: np.ndarray, calibration: dict | None) -> np.ndarray:
    if not isinstance(calibration, dict):
        return np.asarray(y_prob, dtype=np.float64)
    if str(calibration.get("kind", "")).lower() != "platt":
        return np.asarray(y_prob, dtype=np.float64)
    a = float(calibration.get("a", 1.0))
    b = float(calibration.get("b", 0.0))
    p = np.asarray(y_prob, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = np.log(p / (1.0 - p))
    t = np.clip((a * z) + b, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-t))


def _train_xgb_classifier(batch: SniperBatch, params: dict | None = None) -> tuple[Any, dict]:
    xgb = _import_xgb()
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    Xtr = batch.X[tr_idx]
    Xva = batch.X[va_idx]
    ytr = (batch.y[tr_idx] >= 0.5).astype(np.float32, copy=False)
    yva = (batch.y[va_idx] >= 0.5).astype(np.float32, copy=False)
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else np.ones_like(ytr, dtype=np.float32)
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else np.ones_like(yva, dtype=np.float32)

    cls_params = dict(DEFAULT_ENTRY_CLS_PARAMS if params is None else params)
    cls_params.setdefault("objective", "binary:logistic")
    cls_params.setdefault("eval_metric", "logloss")
    try:
        auto_spw = str(os.getenv("SNIPER_ENTRY_CLS_AUTO_POS_WEIGHT", "0")).strip().lower() not in {"0", "false", "no", "off"}
        pos = float(np.sum(ytr > 0.5))
        neg = float(np.sum(ytr <= 0.5))
        if auto_spw and pos > 0.0 and neg > 0.0 and "scale_pos_weight" not in cls_params:
            cls_params["scale_pos_weight"] = float(neg / pos)
    except Exception:
        pass

    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
    watch = [(dtrain, "train"), (dvalid, "val")]
    print(f"[xgb] (cls) train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,}".replace(",", "."), flush=True)
    _thermal_guard_wait_if_needed(where="xgb_cls:pre_fit", force_sample=True)
    xgb_callbacks = _xgb_thermal_callbacks(xgb, where="xgb_cls")
    booster = xgb.train(
        params=cls_params,
        dtrain=dtrain,
        num_boost_round=2500,
        evals=watch,
        early_stopping_rounds=250,
        verbose_eval=50,
        callbacks=xgb_callbacks,
    )
    proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    calibration = None
    if str(os.getenv("SNIPER_ENABLE_CLS_CALIB", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        calibration = _fit_platt_calibration(yva.astype(np.float32), proba.astype(np.float32), wva.astype(np.float32))
    if calibration is not None:
        proba = _apply_platt_calibration(proba, calibration)
    pred_bin = (proba >= 0.5).astype(np.float32)
    acc = float(np.mean(pred_bin == yva)) if yva.size else float("nan")
    logloss = _binary_logloss(yva, proba)
    brier = float(np.mean((proba.astype(np.float64) - yva.astype(np.float64)) ** 2)) if yva.size else float("nan")
    meta = {
        "best_iteration": int(booster.best_iteration),
        "metrics": {"logloss": logloss, "brier": brier, "acc@0.5": acc},
        "problem_type": "binary_classification",
        "calibration": calibration,
    }
    return booster, meta


def _train_catboost_classifier(batch: SniperBatch, params: dict | None = None) -> tuple[object, dict]:
    CatBoostClassifier = _import_catboost_classifier()
    if CatBoostClassifier is None:
        raise RuntimeError("catboost nao instalado")
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    Xtr = batch.X[tr_idx]
    Xva = batch.X[va_idx]
    ytr = (batch.y[tr_idx] >= 0.5).astype(np.int32, copy=False)
    yva = (batch.y[va_idx] >= 0.5).astype(np.int32, copy=False)
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else np.ones((len(tr_idx),), dtype=np.float32)
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else np.ones((len(va_idx),), dtype=np.float32)

    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 6,
        "subsample": 0.8,
        "random_seed": 1337,
        "verbose": 50,
        "allow_writing_files": False,
    }
    if params:
        allow = {
            "loss_function",
            "eval_metric",
            "iterations",
            "learning_rate",
            "depth",
            "subsample",
            "bootstrap_type",
            "random_seed",
            "l2_leaf_reg",
            "bagging_temperature",
            "border_count",
            "min_data_in_leaf",
            "verbose",
            "od_type",
            "od_wait",
            "allow_writing_files",
            "task_type",
            "devices",
            "auto_class_weights",
        }
        clean = {k: v for k, v in dict(params).items() if k in allow}
        cb_params.update(clean)
    # compatibilidade CatBoost:
    # - bootstrap_type=Bayesian não aceita subsample
    # - se subsample estiver presente sem bootstrap_type explícito, use Bernoulli
    try:
        btype = str(cb_params.get("bootstrap_type", "") or "").strip().lower()
        has_subsample = ("subsample" in cb_params) and (cb_params.get("subsample", None) is not None)
        if has_subsample and btype == "bayesian":
            cb_params.pop("subsample", None)
        elif has_subsample and not btype:
            cb_params["bootstrap_type"] = "Bernoulli"
    except Exception:
        pass
    try:
        dev = str((params or {}).get("device", "")).lower()
    except Exception:
        dev = ""
    if dev.startswith("cuda"):
        cb_params.setdefault("task_type", "GPU")
        cb_params.setdefault("devices", dev.replace("cuda:", "") or "0")
    # Nao forca balanceamento por classe por padrao (evita probs achatadas em ~0.5).
    try:
        auto_cw = str(os.getenv("SNIPER_ENTRY_CLS_AUTO_CLASS_WEIGHTS", "")).strip()
        if auto_cw and "auto_class_weights" not in cb_params:
            cb_params["auto_class_weights"] = auto_cw
    except Exception:
        pass

    model = CatBoostClassifier(**cb_params)
    Pool = _import_catboost_pool()
    eval_pool = Pool(Xva, yva, weight=wva)
    _thermal_guard_wait_if_needed(where="catboost_cls:pre_fit", force_sample=True)
    _fit_catboost_with_optional_thermal(
        model,
        where="catboost_cls",
        X=Xtr,
        y=ytr,
        sample_weight=wtr,
        eval_set=eval_pool,
        use_best_model=True,
    )
    proba = model.predict_proba(Xva)[:, 1]
    calibration = None
    if str(os.getenv("SNIPER_ENABLE_CLS_CALIB", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        calibration = _fit_platt_calibration(yva.astype(np.float32), proba.astype(np.float32), wva.astype(np.float32))
    if calibration is not None:
        proba = _apply_platt_calibration(proba, calibration)
    pred_bin = (proba >= 0.5).astype(np.int32)
    acc = float(np.mean(pred_bin == yva)) if yva.size else float("nan")
    logloss = _binary_logloss(yva.astype(np.float32), proba.astype(np.float32))
    brier = float(np.mean((proba.astype(np.float64) - yva.astype(np.float64)) ** 2)) if yva.size else float("nan")
    meta = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "metrics": {"logloss": logloss, "brier": brier, "acc@0.5": acc},
        "problem_type": "binary_classification",
        "calibration": calibration,
    }
    return model, meta


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


def _save_model(booster: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    try:
        ubj = path.with_suffix(".ubj")
        booster.save_model(str(ubj))
    except Exception:
        pass


def train_sniper_models(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    asset_class = str(getattr(cfg, "asset_class", "crypto") or "crypto").lower()
    timing_on = _env_bool("SNIPER_TIMINGS", default=False)

    t0 = time.perf_counter()
    symbols = _select_symbols(cfg)
    if timing_on:
        print(f"[sniper-train][timing] select_symbols_s={time.perf_counter() - t0:.2f}", flush=True)
    print(
        f"[sniper-train] símbolos={len(symbols)} asset={asset_class} (max_symbols={cfg.max_symbols})",
        flush=True,
    )
    if SAVE_ROOT is None:
        base_root = Path(__file__).resolve().parents[2].parent / "models_sniper"
        save_root = base_root / asset_class
    else:
        save_root = SAVE_ROOT(asset_class)
    if getattr(cfg, "run_dir", None):
        run_dir = Path(cfg.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sniper-train] usando run_dir existente: {run_dir}", flush=True)
    else:
        run_dir = _next_run_dir(save_root)
    entry_params = dict(DEFAULT_ENTRY_PARAMS if cfg.entry_params is None else cfg.entry_params)
    entry_params_by_side = dict(getattr(cfg, "entry_params_by_side", None) or {})
    reg_enabled = bool(getattr(cfg, "entry_reg_enabled", True))
    cls_enabled = bool(getattr(cfg, "entry_cls_enabled", True))
    cls_model_type = str(getattr(cfg, "entry_cls_model_type", "catboost") or "catboost").strip().lower()
    cls_params = dict(DEFAULT_ENTRY_CLS_PARAMS if getattr(cfg, "entry_cls_params", None) is None else getattr(cfg, "entry_cls_params"))
    cls_params_by_side = dict(getattr(cfg, "entry_cls_params_by_side", None) or {})
    cls_params.setdefault("device", str(entry_params.get("device", "cpu")))
    reg_label_tpl = str(getattr(cfg, "entry_reg_label_col_template", "edge_label_{side}") or "edge_label_{side}")
    reg_weight_tpl = str(getattr(cfg, "entry_reg_weight_col_template", "") or "")
    cls_label_tpl = str(getattr(cfg, "entry_cls_label_col_template", "entry_gate_{side}") or "entry_gate_{side}")
    cls_weight_tpl = str(getattr(cfg, "entry_cls_weight_col_template", "") or "")
    cls_thr = float(getattr(cfg, "entry_cls_positive_threshold", 50.0) or 50.0)
    cls_target_pos_ratio = float(getattr(cfg, "entry_cls_target_pos_ratio", 0.5) or 0.5)
    cls_pos_w = float(getattr(cfg, "entry_cls_pos_weight", 1.0) or 1.0)
    cls_neg_w = float(getattr(cfg, "entry_cls_neg_weight", 1.0) or 1.0)
    _raw_cls_bins = getattr(cfg, "entry_cls_balance_bins", None)
    if _raw_cls_bins is None:
        cls_bins = (0.5,)
    else:
        cls_bins = tuple(float(x) for x in _raw_cls_bins)
    if cls_enabled:
        print(
            f"[sniper-train] entry_cls config: thr={cls_thr:.2f} target_pos_ratio={cls_target_pos_ratio:.3f} "
            f"pos_w={cls_pos_w:.3f} neg_w={cls_neg_w:.3f} bins={cls_bins if len(cls_bins) else '()'}",
            flush=True,
        )
    print(
        f"[sniper-train] entry_reg config: enabled={int(reg_enabled)} "
        f"label_tpl={reg_label_tpl} weight_tpl={(reg_weight_tpl if reg_weight_tpl else '<auto>')}",
        flush=True,
    )
    label_mode = str(getattr(cfg, "entry_label_mode", "signed") or "signed").strip().lower()
    label_scale = float(getattr(cfg, "entry_label_scale", 1.0) or 1.0)

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
        t_cache = time.perf_counter()
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(cfg.total_days),
            contract=contract,
            flags=feature_flags,
            cache_dir=getattr(cfg, "feature_cache_dir", None),
            asset_class=asset_class,
            # Se você pediu histórico completo (total_days<=0), garanta que o cache também foi feito assim.
            strict_total_days=(int(cfg.total_days) <= 0),
            abort_ram_pct=float(getattr(cfg, "abort_ram_pct", 85.0)),
        )
        if timing_on:
            print(f"[sniper-train][timing] cache_build_s={time.perf_counter() - t_cache:.2f}", flush=True)
        # se algum símbolo falhar no cache, removemos aqui para evitar erro nos períodos
        symbols = [s for s in symbols if s in cache_map]
        print(f"[sniper-train] cache pronto: símbolos_ok={len(symbols)}", flush=True)
        if not symbols:
            raise RuntimeError("Nenhum símbolo restou após gerar cache (ver logs [cache])")

    window_min = int(getattr(cfg, "entry_reg_window_min", 0) or 0)
    if window_min <= 0:
        window_min = 60
    windows = [int(window_min)]
    entry_specs_full = [
        {
            "name": f"ret_exp_{int(window_min)}m",
            "window": int(window_min),
        }
    ]
    base_spec = entry_specs_full[0]
    base_name = str(base_spec.get("name") or "")
    entry_sides = list(getattr(cfg, "entry_sides", None) or ("long", "short"))
    entry_sides = [str(s).strip().lower() for s in entry_sides if str(s).strip()]
    if not entry_sides:
        entry_sides = ["long", "short"]

    pool_dir = None
    pool_df_map: dict[str, dict[str, list[Path]]] | None = None
    pool_max_ts: dict[str, pd.Timestamp | None] | None = None
    pool_layout: dict[str, str] | None = None
    if bool(getattr(cfg, "entry_pool_full", False)) and bool(reg_enabled):
        pool_dir = Path(cfg.entry_pool_dir) if getattr(cfg, "entry_pool_dir", None) else (run_dir / "entry_pool_full")
        sides = list(entry_sides)
        pool_df_map = {}
        pool_max_ts = {}
        pool_layout = {}
        from utils.progress import ProgressPrinter
        # Tenta habilitar VT no Windows para permitir atualizaÃ§Ã£o in-place.
        vt_ok = _enable_vt_mode(sys.stderr)
        for side in sides:
            side_key = str(side or "").strip().lower() or "long"
            reg_weight_col = ""
            if reg_weight_tpl:
                try:
                    reg_weight_col = str(reg_weight_tpl).format(side=side_key)
                except Exception:
                    reg_weight_col = str(reg_weight_tpl)
            side_dir = pool_dir / side_key
            if (side_dir / "pool_meta.json").exists():
                print(f"[sniper-train] pool_full: usando cache {side_key} em {side_dir}", flush=True)
                df_map, max_ts, layout = _load_entry_pool(side_dir)
                pool_df_map[side_key] = df_map
                pool_max_ts[side_key] = max_ts
                pool_layout[side_key] = layout
                continue
            print(f"[sniper-train] pool_full: gerando dataset {side_key} (sem limites)...", flush=True)
            max_ts = None
            progress = ProgressPrinter(
                prefix=f"[sniper-train] pool_full {side_key}:",
                total=len(symbols),
                stream=sys.stdout,
                print_every_s=5.0,
                force_inplace=vt_ok,
            )
            def _build_and_save_symbol(sym: str) -> tuple[str, pd.Timestamp | None]:
                _wait_ram_below_threshold(
                    float(getattr(cfg, "abort_ram_pct", 0.0) or 0.0),
                    where=f"pool_full:{side_key}:{sym}",
                )
                _thermal_guard_wait_if_needed(
                    where=f"pool_full:{side_key}:{sym}",
                    force_sample=False,
                )
                if bool(getattr(cfg, "use_feature_cache", True)):
                    pack_sym = prepare_sniper_dataset_from_cache(
                        [sym],
                        total_days=int(cfg.total_days),
                        remove_tail_days=0,
                        contract=contract,
                        cache_map=cache_map,
                        max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                        full_entry_pool=True,
                        seed=1337,
                        asset_class=asset_class,
                        feature_flags=feature_flags,
                        entry_side=side_key,
                        entry_label_col=str(reg_label_tpl).format(side=side_key),
                        entry_weight_col=(reg_weight_col or None),
                        entry_label_binary=False,
                        entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                        entry_reg_weight_alpha=float(getattr(cfg, "entry_reg_weight_alpha", 4.0)),
                        entry_reg_weight_power=float(getattr(cfg, "entry_reg_weight_power", 1.2)),
                        entry_reg_balance_bins=getattr(cfg, "entry_reg_balance_bins", None),
                    )
                else:
                    pack_sym = prepare_sniper_dataset(
                        [sym],
                        total_days=int(cfg.total_days),
                        remove_tail_days=0,
                        contract=contract,
                        max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                        full_entry_pool=True,
                        seed=1337,
                        asset_class=asset_class,
                        feature_flags=feature_flags,
                        entry_side=side_key,
                        entry_label_col=str(reg_label_tpl).format(side=side_key),
                        entry_weight_col=(reg_weight_col or None),
                        entry_label_binary=False,
                        entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                        entry_reg_weight_alpha=float(getattr(cfg, "entry_reg_weight_alpha", 4.0)),
                        entry_reg_weight_power=float(getattr(cfg, "entry_reg_weight_power", 1.2)),
                        entry_reg_balance_bins=getattr(cfg, "entry_reg_balance_bins", None),
                    )
                entry_batches_sym = pack_sym.entry_map or {}
                if not entry_batches_sym:
                    entry_batches_sym = {base_name: pack_sym.entry}
                _save_entry_pool_symbol(side_dir, sym, entry_batches_sym)
                ts_max_local = None
                for b in entry_batches_sym.values():
                    if b is None or b.ts.size == 0:
                        continue
                    cur = pd.to_datetime(b.ts).max()
                    if ts_max_local is None or cur > ts_max_local:
                        ts_max_local = cur
                return sym, ts_max_local

            workers = max(1, int(os.getenv("SNIPER_POOL_BUILD_WORKERS", "0") or "0"))
            if workers <= 0:
                workers = min(8, max(1, (os.cpu_count() or 4) // 2))
            if workers == 1:
                for i, symbol in enumerate(symbols):
                    sym_done, ts_max_local = _build_and_save_symbol(symbol)
                    if ts_max_local is not None and (max_ts is None or ts_max_local > max_ts):
                        max_ts = ts_max_local
                    progress.update(i + 1, suffix=sym_done)
            else:
                build_ram_cap = float(
                    os.getenv(
                        "SNIPER_POOL_BUILD_RAM_PCT",
                        str(float(getattr(cfg, "abort_ram_pct", 85.0) or 85.0)),
                    )
                    or "85"
                )
                build_min_free_mb = float(os.getenv("SNIPER_POOL_BUILD_MIN_FREE_MB", "2048") or "2048")
                build_per_worker_mb = float(os.getenv("SNIPER_POOL_BUILD_PER_WORKER_MB", "768") or "768")
                build_policy = AdaptiveParallelPolicy(
                    max_ram_pct=build_ram_cap,
                    min_free_mb=build_min_free_mb,
                    per_worker_mem_mb=build_per_worker_mb,
                    min_workers=1,
                    poll_interval_s=0.5,
                    log_every_s=15.0,
                )
                print(
                    f"[sniper-train] pool_full {side_key}: workers={workers} (SNIPER_POOL_BUILD_WORKERS) "
                    f"ram_cap={build_policy.max_ram_pct:.1f}% min_free_mb={build_policy.min_free_mb:.0f} per_worker_mb={build_policy.per_worker_mem_mb:.0f}",
                    flush=True,
                )
                done = 0
                for sym_submitted, fut in run_adaptive_thread_map(
                    symbols,
                    _build_and_save_symbol,
                    max_workers=workers,
                    policy=build_policy,
                    task_name=f"pool-build-{side_key}",
                ):
                    sym_done, ts_max_local = fut.result()
                    if ts_max_local is not None and (max_ts is None or ts_max_local > max_ts):
                        max_ts = ts_max_local
                    done += 1
                    progress.update(done, suffix=sym_done or sym_submitted)
            progress.close()
            _write_pool_meta(side_dir, windows, max_ts)
            df_map, max_ts_loaded, layout = _load_entry_pool(side_dir)
            pool_df_map[side_key] = df_map
            pool_max_ts[side_key] = max_ts_loaded
            pool_layout[side_key] = layout
    for tail in cfg.offsets_days:
        period_dir = run_dir / f"period_{int(tail)}d"
        reg_done = (not reg_enabled) or all((period_dir / f"entry_model_{side}" / "model_entry.json").exists() for side in entry_sides)
        cls_done = (not cls_enabled) or all((period_dir / f"entry_cls_model_{side}" / "model_entry_gate.json").exists() for side in entry_sides)
        if reg_done and cls_done:
            print(f"[sniper-train] {tail}d: ja existe ({period_dir}), pulando", flush=True)
            continue
        print(f"[sniper-train] periodo T-{tail}d", flush=True)
        _check_ram_or_abort(
            float(getattr(cfg, "abort_ram_pct", 0.0) or 0.0),
            where=f"period_start:{int(tail)}d",
        )
        _thermal_guard_wait_if_needed(where=f"period_start:{int(tail)}d", force_sample=True)
        pack_by_side: dict[str, SniperDataPack | None] = {}
        if reg_enabled and pool_df_map is None and bool(getattr(cfg, "use_feature_cache", True)):
            t_pack = time.perf_counter()
            for side in entry_sides:
                side_key = str(side).strip().lower()
                reg_weight_col = ""
                if reg_weight_tpl:
                    try:
                        reg_weight_col = str(reg_weight_tpl).format(side=side_key)
                    except Exception:
                        reg_weight_col = str(reg_weight_tpl)
                pack_by_side[side] = prepare_sniper_dataset_from_cache(
                    symbols,
                    total_days=int(cfg.total_days),
                    remove_tail_days=int(tail),
                    contract=contract,
                    cache_map=cache_map,
                    max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                    full_entry_pool=bool(getattr(cfg, "entry_pool_full", False)),
                    seed=1337,
                    asset_class=asset_class,
                    feature_flags=feature_flags,
                    entry_side=side_key,
                    entry_label_col=str(reg_label_tpl).format(side=side_key),
                    entry_weight_col=(reg_weight_col or None),
                    entry_label_binary=False,
                    entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                    entry_reg_weight_alpha=float(getattr(cfg, "entry_reg_weight_alpha", 4.0)),
                    entry_reg_weight_power=float(getattr(cfg, "entry_reg_weight_power", 1.2)),
                    entry_reg_balance_bins=getattr(cfg, "entry_reg_balance_bins", None),
                )
            if timing_on:
                print(f"[sniper-train][timing] pack_from_cache_s={time.perf_counter() - t_pack:.2f}", flush=True)
        elif reg_enabled and pool_df_map is None:
            t_pack = time.perf_counter()
            for side in entry_sides:
                side_key = str(side).strip().lower()
                reg_weight_col = ""
                if reg_weight_tpl:
                    try:
                        reg_weight_col = str(reg_weight_tpl).format(side=side_key)
                    except Exception:
                        reg_weight_col = str(reg_weight_tpl)
                pack_by_side[side] = prepare_sniper_dataset(
                    symbols,
                    total_days=int(cfg.total_days),
                    remove_tail_days=int(tail),
                    contract=contract,
                    max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                    full_entry_pool=bool(getattr(cfg, "entry_pool_full", False)),
                    seed=1337,
                    asset_class=asset_class,
                    feature_flags=feature_flags,
                    entry_side=side_key,
                    entry_label_col=str(reg_label_tpl).format(side=side_key),
                    entry_weight_col=(reg_weight_col or None),
                    entry_label_binary=False,
                    entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                    entry_reg_weight_alpha=float(getattr(cfg, "entry_reg_weight_alpha", 4.0)),
                    entry_reg_weight_power=float(getattr(cfg, "entry_reg_weight_power", 1.2)),
                    entry_reg_balance_bins=getattr(cfg, "entry_reg_balance_bins", None),
                )
            if timing_on:
                print(f"[sniper-train][timing] pack_build_s={time.perf_counter() - t_pack:.2f}", flush=True)
        else:
            # usa pool completo salvo em disco e filtra por data
            pack_by_side = {}
        if pack_by_side:
            try:
                sample_pack = next(iter(pack_by_side.values()))
                if sample_pack is not None:
                    n_used = len(sample_pack.symbols_used) if getattr(sample_pack, "symbols_used", None) else len(sample_pack.symbols)
                    n_skip = len(sample_pack.symbols_skipped) if getattr(sample_pack, "symbols_skipped", None) else 0
                    te = str(pd.to_datetime(sample_pack.train_end_utc)) if getattr(sample_pack, "train_end_utc", None) is not None else "None"
                    print(f"[sniper-train] {tail}d: train_end_utc={te} | symbols_used={n_used} skipped={n_skip}", flush=True)
            except Exception:
                pass
        # protecao anti-escassez: se poucos simbolos sobreviveram nesse tail, nao vale treinar.
        try:
            min_used = int(getattr(cfg, "min_symbols_used_per_period", 0) or 0)
        except Exception:
            min_used = 0
        if pack_by_side and min_used > 0:
            try:
                sample_pack = next(iter(pack_by_side.values()))
                n_used = len(sample_pack.symbols_used) if getattr(sample_pack, "symbols_used", None) else len(sample_pack.symbols)
            except Exception:
                n_used = 0
            if int(n_used) < int(min_used):
                print(f"[sniper-train] {tail}d: symbols_used={n_used} < min_symbols_used_per_period={min_used} -> pulando", flush=True)
                continue
        entry_specs = [(str(s.get("name")), int(s.get("window") or 0)) for s in entry_specs_full]
        spec_by_name = {str(s.get("name")): s for s in entry_specs_full}
        entry_batches_by_side: dict[str, dict[str, SniperBatch]] = {}
        base_batch_by_side: dict[str, SniperBatch] = {}
        if reg_enabled and pool_df_map is None:
            for side in entry_sides:
                pack = pack_by_side.get(side)
                if pack is None:
                    continue
                entry_batches = pack.entry_map or {base_name: pack.entry}
                base_batch = entry_batches.get(base_name) or pack.entry
                entry_batches_by_side[side] = entry_batches
                base_batch_by_side[side] = base_batch
        elif reg_enabled:
            def _pool_for_side_name(side: str, name: str) -> tuple[dict[str, list[Path]] | None, pd.Timestamp | None, str | None]:
                df_map = pool_df_map.get(side) if pool_df_map else None
                max_ts = pool_max_ts.get(side) if pool_max_ts else None
                layout = pool_layout.get(side) if pool_layout else None
                return df_map, max_ts, layout

            def _load_pool_batch_combined(side: str, name: str, max_rows: int, rng: np.random.Generator) -> SniperBatch | None:
                df_map, max_ts, layout = _pool_for_side_name(side, name)
                files = df_map.get("_combined", []) if df_map else []
                if not files:
                    return None
                print(
                    f"[sniper-train] pool_load {side} {name}: start files={len(files)} tail={int(tail)}d",
                    flush=True,
                )
                cutoff = None
                if max_ts is not None:
                    cutoff = max_ts - pd.Timedelta(days=int(tail))
                # cache por período + janela (evita reler todos os parquets)
                cache_dir = period_dir / "entry_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_key = f"{int(tail)}d_{side}_{name}_{int(max_rows)}.npz"
                cache_path = cache_dir / cache_key
                if cache_path.exists():
                    try:
                        loaded = np.load(cache_path, allow_pickle=False)
                        X = loaded["X"]
                        y = loaded["y"]
                        w = loaded["w"]
                        ts = loaded["ts"]
                        feat_cols = loaded["feat_cols"].tolist()
                        sym_id = np.zeros((X.shape[0],), dtype=np.int32)
                        return SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))
                    except Exception:
                        pass
                if True:
                    k = int(max_rows)
                    keys_res = np.empty((0,), dtype=np.float64)
                    X_res = np.empty((0, 0), dtype=np.float32)
                    y_res = np.empty((0,), dtype=np.float32)
                    w_res = np.empty((0,), dtype=np.float32)
                    ts_res = np.empty((0,), dtype="datetime64[ns]")
                    base_feat_cols: list[str] | None = None
                    cols_needed: list[str] | None = None
                    try:
                        df0 = pd.read_parquet(files[0])
                        base_feat_cols = [c for c in df0.columns if c not in {"label_entry", "weight", "sym_id"}]
                        cols_needed = list(base_feat_cols) + ["label_entry"]
                        if "weight" in df0.columns:
                            cols_needed.append("weight")
                        del df0
                    except Exception:
                        cols_needed = None

                    def _load_one_reg(fp: Path) -> pd.DataFrame:
                        df = pd.read_parquet(fp, columns=cols_needed) if cols_needed else pd.read_parquet(fp)
                        if cutoff is not None:
                            df = df.loc[pd.to_datetime(df.index) < cutoff]
                        return df

                    max_workers = int(os.getenv("SNIPER_POOL_READERS", "8") or "8")
                    max_workers = max(1, min(32, max_workers))
                    total_files = len(files)
                    last_report = -1
                    try:
                        progress_every_pct = int(os.getenv("SNIPER_POOL_LOAD_PROGRESS_PCT", "5") or "5")
                    except Exception:
                        progress_every_pct = 5
                    progress_every_pct = max(1, min(50, progress_every_pct))
                    load_ram_cap = float(
                        os.getenv(
                            "SNIPER_POOL_LOAD_RAM_PCT",
                            str(float(getattr(cfg, "abort_ram_pct", 85.0) or 85.0)),
                        )
                        or "85"
                    )
                    load_min_free_mb = float(os.getenv("SNIPER_POOL_LOAD_MIN_FREE_MB", "1536") or "1536")
                    load_per_worker_mb = float(os.getenv("SNIPER_POOL_LOAD_PER_WORKER_MB", "384") or "384")
                    load_policy = AdaptiveParallelPolicy(
                        max_ram_pct=load_ram_cap,
                        min_free_mb=load_min_free_mb,
                        per_worker_mem_mb=load_per_worker_mb,
                        min_workers=1,
                        poll_interval_s=0.3,
                        log_every_s=20.0,
                    )
                    idx = -1
                    for _fp_submitted, fut in run_adaptive_thread_map(
                        files,
                        _load_one_reg,
                        max_workers=max_workers,
                        policy=load_policy,
                        task_name=f"pool-load-{side}-{name}",
                    ):
                        idx += 1
                        df = fut.result()
                        if df.empty or "label_entry" not in df.columns:
                            continue
                        if base_feat_cols is None:
                            base_feat_cols = [c for c in df.columns if c not in {"label_entry", "weight", "sym_id"}]
                        chunk_rows = int(os.getenv("SNIPER_POOL_CHUNK_ROWS", "250000") or "250000")
                        chunk_rows = max(10_000, chunk_rows)
                        if k > 0 and df.shape[0] > chunk_rows:
                            for start in range(0, df.shape[0], chunk_rows):
                                end = min(df.shape[0], start + chunk_rows)
                                df_chunk = df.iloc[start:end]
                                if df_chunk.empty:
                                    continue
                                X = df_chunk[base_feat_cols].to_numpy(np.float32, copy=True)
                                y = df_chunk["label_entry"].to_numpy(dtype=np.float32, copy=False)
                                if "weight" in df_chunk.columns:
                                    w = df_chunk["weight"].to_numpy(dtype=np.float32, copy=False)
                                else:
                                    w = np.ones((df_chunk.shape[0],), dtype=np.float32)
                                ts = pd.to_datetime(df_chunk.index).to_numpy(dtype="datetime64[ns]")
                                u = rng.random(size=y.shape[0], dtype=np.float64)
                                w_safe = np.maximum(w.astype(np.float64, copy=False), 1e-12)
                                keys = -np.log(u) / w_safe
                                keys_res, X_res, y_res, w_res, ts_res = _reservoir_update(
                                    keys_res, X_res, y_res, w_res, ts_res, keys, X, y, w, ts, k
                                )
                        else:
                            X = df[base_feat_cols].to_numpy(np.float32, copy=True)
                            y = df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                            if "weight" in df.columns:
                                w = df["weight"].to_numpy(dtype=np.float32, copy=False)
                            else:
                                w = np.ones((df.shape[0],), dtype=np.float32)
                            ts = pd.to_datetime(df.index).to_numpy(dtype="datetime64[ns]")
                            if k <= 0:
                                keys_res = np.concatenate([keys_res, np.zeros((y.shape[0],), dtype=np.float64)])
                                X_res = X if X_res.size == 0 else np.concatenate([X_res, X])
                                y_res = np.concatenate([y_res, y])
                                w_res = np.concatenate([w_res, w])
                                ts_res = np.concatenate([ts_res, ts])
                            else:
                                u = rng.random(size=y.shape[0], dtype=np.float64)
                                w_safe = np.maximum(w.astype(np.float64, copy=False), 1e-12)
                                keys = -np.log(u) / w_safe
                                keys_res, X_res, y_res, w_res, ts_res = _reservoir_update(
                                    keys_res, X_res, y_res, w_res, ts_res, keys, X, y, w, ts, k
                                )
                        del df
                        if total_files:
                            pct = int((idx + 1) * 100 / total_files)
                            if pct != last_report and (pct % progress_every_pct == 0 or pct == 100):
                                print(
                                    f"[sniper-train] pool_load {side} {name}: {pct}% ({idx + 1}/{total_files})",
                                    end="\r",
                                    flush=True,
                                )
                                last_report = pct
                    if y_res.size == 0 or base_feat_cols is None:
                        print(f"\n[sniper-train] pool_load {side} {name}: vazio", flush=True)
                        return None
                    sym_id = np.zeros((y_res.shape[0],), dtype=np.int32)
                    try:
                        t_save = time.perf_counter()
                        use_compress = _env_bool("SNIPER_POOL_CACHE_COMPRESS", default=False)
                        print(
                            f"\n[sniper-train] pool_load {side} {name}: cache_save start compress={int(use_compress)} path={cache_path.name}",
                            flush=True,
                        )
                        if use_compress:
                            np.savez_compressed(
                                cache_path,
                                X=X_res,
                                y=y_res,
                                w=w_res,
                                ts=ts_res,
                                feat_cols=np.array(base_feat_cols, dtype=object),
                            )
                        else:
                            np.savez(
                                cache_path,
                                X=X_res,
                                y=y_res,
                                w=w_res,
                                ts=ts_res,
                                feat_cols=np.array(base_feat_cols, dtype=object),
                            )
                        print(
                            f"[sniper-train] pool_load {side} {name}: cache_save done sec={time.perf_counter() - t_save:.2f}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    print(
                        f"\n[sniper-train] pool_load {side} {name}: done rows={int(y_res.size)} feats={int(X_res.shape[1] if X_res.ndim == 2 else 0)}",
                        flush=True,
                    )
                    return SniperBatch(X=X_res, y=y_res, w=w_res, ts=ts_res, sym_id=sym_id, feature_cols=list(base_feat_cols))
            def _load_pool_batch_legacy(side: str, name: str) -> SniperBatch | None:
                df_map, max_ts, _layout = _pool_for_side_name(side, name)
                files = df_map.get(name, []) if df_map else []
                if not files:
                    return None
                cutoff = None
                if max_ts is not None:
                    cutoff = max_ts - pd.Timedelta(days=int(tail))
                parts = []
                for fp in files:
                    df = pd.read_parquet(fp)
                    if cutoff is not None:
                        df = df.loc[pd.to_datetime(df.index) < cutoff]
                    if df.empty:
                        continue
                    parts.append(df)
                if not parts:
                    return None
                df_f = pd.concat(parts, axis=0, ignore_index=False)
                feat_cols = [c for c in df_f.columns if c not in {"label_entry", "weight", "sym_id"}]
                X = df_f[feat_cols].to_numpy(np.float32, copy=True)
                y = df_f["label_entry"].to_numpy(dtype=np.float32, copy=False)
                if "weight" in df_f.columns:
                    w = df_f["weight"].to_numpy(dtype=np.float32, copy=False)
                else:
                    w = np.ones((df_f.shape[0],), dtype=np.float32)
                ts = pd.to_datetime(df_f.index).to_numpy(dtype="datetime64[ns]")
                if "sym_id" in df_f.columns:
                    sym_id = df_f["sym_id"].to_numpy(dtype=np.int32, copy=False)
                else:
                    sym_id = np.zeros((df_f.shape[0],), dtype=np.int32)
                return SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))

            for side in entry_sides:
                _df_map_base, _max_ts_base, base_layout = _pool_for_side_name(side, base_name)
                if base_layout == "per_symbol_combined":
                    base_batch = _load_pool_batch_combined(
                        side,
                        base_name,
                        int(getattr(cfg, "max_rows_entry", 2_000_000)),
                        np.random.default_rng(1337),
                    )
                else:
                    base_batch = _load_pool_batch_legacy(side, base_name)
                if base_batch is not None and base_batch.X.size > 0:
                    base_batch_by_side[side] = base_batch
                    entry_batches_by_side[side] = {}
        if reg_enabled and (not base_batch_by_side):
            print(f"[sniper-train] {tail}d: dataset vazio, pulando", flush=True)
            continue
        entry_cls_batches_by_side: dict[str, SniperBatch] = {}
        if cls_enabled:
            cls_max_rows = int(
                os.getenv(
                    "SNIPER_MAX_ROWS_ENTRY_CLS",
                    str(int(getattr(cfg, "max_rows_entry", 2_000_000))),
                )
                or int(getattr(cfg, "max_rows_entry", 2_000_000))
            )
            cls_max_rows = max(50_000, cls_max_rows)
            t_cls_pack = time.perf_counter()
            for side in entry_sides:
                side_key = str(side).strip().lower()
                cls_label_col = str(cls_label_tpl).format(side=side_key)
                cls_weight_col = ""
                if cls_weight_tpl:
                    try:
                        cls_weight_col = str(cls_weight_tpl).format(side=side_key)
                    except Exception:
                        cls_weight_col = str(cls_weight_tpl)
                try:
                    if bool(getattr(cfg, "use_feature_cache", True)):
                        cls_pack = prepare_sniper_dataset_from_cache(
                            symbols,
                            total_days=int(cfg.total_days),
                            remove_tail_days=int(tail),
                            contract=contract,
                            cache_map=cache_map,
                            max_rows_entry=int(cls_max_rows),
                            full_entry_pool=False,
                            seed=7331,
                            asset_class=asset_class,
                            feature_flags=feature_flags,
                            entry_side=side_key,
                            entry_label_col=cls_label_col,
                            entry_label_binary=True,
                            entry_label_positive_threshold=float(cls_thr),
                            entry_weight_col=(cls_weight_col or None),
                            entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                            entry_reg_weight_alpha=0.0,
                            entry_reg_weight_power=1.0,
                            entry_reg_balance_bins=(cls_bins if len(cls_bins) > 0 else None),
                        )
                    else:
                        cls_pack = prepare_sniper_dataset(
                            symbols,
                            total_days=int(cfg.total_days),
                            remove_tail_days=int(tail),
                            contract=contract,
                            max_rows_entry=int(cls_max_rows),
                            full_entry_pool=False,
                            seed=7331,
                            asset_class=asset_class,
                            feature_flags=feature_flags,
                            entry_side=side_key,
                            entry_label_col=cls_label_col,
                            entry_label_binary=True,
                            entry_label_positive_threshold=float(cls_thr),
                            entry_weight_col=(cls_weight_col or None),
                            entry_reg_window_min=int(getattr(cfg, "entry_reg_window_min", 0)),
                            entry_reg_weight_alpha=0.0,
                            entry_reg_weight_power=1.0,
                            entry_reg_balance_bins=(cls_bins if len(cls_bins) > 0 else None),
                        )
                except Exception as e:
                    print(
                        f"[sniper-train] cls dataset {side_key}: falhou ({type(e).__name__}: {e})",
                        flush=True,
                    )
                    continue
                cls_entry_map = cls_pack.entry_map or {base_name: cls_pack.entry}
                cls_batch = cls_entry_map.get(base_name) or cls_pack.entry
                if cls_batch is None or cls_batch.X.size == 0:
                    print(f"[sniper-train] cls dataset {side_key}: vazio", flush=True)
                    continue
                y_cls = (cls_batch.y >= 0.5)
                pos = int(np.sum(y_cls))
                neg = int(y_cls.size - pos)
                if pos == 0 or neg == 0:
                    print(
                        f"[sniper-train] cls dataset {side_key}: classe unica (pos={pos} neg={neg}), pulando",
                        flush=True,
                    )
                    continue
                # pesos por classe para penalizar mais falso-positivo:
                # aumentar peso da classe negativa tende a empurrar prob para baixo fora dos bons eventos.
                if cls_pos_w != 1.0 or cls_neg_w != 1.0:
                    w_base = cls_batch.w if getattr(cls_batch, "w", None) is not None else np.ones_like(cls_batch.y, dtype=np.float32)
                    w_base = np.asarray(w_base, dtype=np.float32)
                    w_cls = np.where(y_cls, np.float32(cls_pos_w), np.float32(cls_neg_w)).astype(np.float32, copy=False)
                    w_new = (w_base * w_cls).astype(np.float32, copy=False)
                    cls_batch = SniperBatch(
                        X=cls_batch.X,
                        y=cls_batch.y,
                        w=w_new,
                        ts=cls_batch.ts,
                        sym_id=cls_batch.sym_id,
                        feature_cols=cls_batch.feature_cols,
                    )
                entry_cls_batches_by_side[side] = cls_batch
                _print_binary_balance_stats(
                    f"{side_key} cls dataset_raw",
                    cls_batch.y,
                    weights=(cls_batch.w if getattr(cls_batch, "w", None) is not None else None),
                )
            if timing_on:
                print(f"[sniper-train][timing] pack_cls_s={time.perf_counter() - t_cls_pack:.2f}", flush=True)
        entry_models_by_side: dict[str, dict[str, tuple[Any, dict]]] = {s: {} for s in entry_sides}
        entry_cls_models_by_side: dict[str, dict[str, tuple[Any, dict]]] = {s: {} for s in entry_sides}
        for side in (entry_sides if reg_enabled else []):
            base_batch = base_batch_by_side.get(side)
            entry_batches = entry_batches_by_side.get(side, {})
            if base_batch is None or base_batch.X.size == 0:
                continue
            try:
                yb = base_batch.y.astype(np.float64, copy=False)
                print(
                    f"[sniper-train] dataset entry {side}: n={int(yb.size)} mean={yb.mean():.5f} std={yb.std():.5f} "
                    f"min={yb.min():.5f} max={yb.max():.5f}",
                    flush=True,
                )
                _print_regression_balance_stats(
                    f"{side} dataset_entry_raw",
                    base_batch.y,
                    weights=(base_batch.w if getattr(base_batch, "w", None) is not None else None),
                )
            except Exception:
                pass
            for name, w in entry_specs:
                _check_ram_or_abort(
                    float(getattr(cfg, "abort_ram_pct", 0.0) or 0.0),
                    where=f"train_entry:{side}:{name}",
                )
                if pool_df_map is not None:
                    _df_map_cur, _max_ts_cur, cur_layout = _pool_for_side_name(side, name)
                    if cur_layout == "per_symbol_combined":
                        if name == base_name and base_batch is not None:
                            b = base_batch
                        else:
                            b = _load_pool_batch_combined(
                                side,
                                name,
                                int(getattr(cfg, "max_rows_entry", 2_000_000)),
                                np.random.default_rng(1337 + int(w)),
                            )
                    else:
                        b = _load_pool_batch_legacy(side, name)
                    if b is None:
                        b = base_batch
                else:
                    b = entry_batches.get(name, base_batch)
                if b is None or b.X.size == 0:
                    print(f"[sniper-train] EntryScore {side} {name} ({w}m): dataset vazio, pulando", flush=True)
                    continue
                try:
                    yb = b.y.astype(np.float64, copy=False)
                    print(
                        f"[sniper-train] {side} {name}: n={int(yb.size)} mean={yb.mean():.5f} std={yb.std():.5f} "
                        f"min={yb.min():.5f} max={yb.max():.5f}",
                        flush=True,
                    )
                    _print_regression_balance_stats(
                        f"{side} {name} raw_balance",
                        b.y,
                        weights=(b.w if getattr(b, "w", None) is not None else None),
                    )
                except Exception:
                    pass
                # se pool full, amostra por modelo aqui
                if bool(getattr(cfg, "entry_pool_full", False)):
                    max_rows = int(getattr(cfg, "max_rows_entry", 2_000_000))
                    if max_rows > 0 and b.y.size > max_rows:
                        rng = np.random.default_rng(1337 + int(w))
                        keep = _sample_indices_regression(
                            b.y.astype(np.float32, copy=False),
                            getattr(cfg, "entry_reg_balance_bins", None),
                            max_rows,
                            rng=rng,
                            distance_power=float(getattr(cfg, "entry_reg_balance_distance_power", 0.0) or 0.0),
                            min_frac=float(getattr(cfg, "entry_reg_balance_min_frac", 0.0) or 0.0),
                        )
                        if keep.size > 0:
                            b = SniperBatch(
                                X=b.X[keep],
                                y=b.y[keep],
                                w=b.w[keep],
                                ts=b.ts[keep],
                                sym_id=b.sym_id[keep],
                                feature_cols=b.feature_cols,
                            )
                alpha = float(getattr(cfg, "entry_reg_weight_alpha", 4.0))
                power = float(getattr(cfg, "entry_reg_weight_power", 1.2))
                base_w = b.w.astype(np.float32, copy=False) if getattr(b, "w", None) is not None else np.ones_like(b.y, dtype=np.float32)
                if _env_bool("SNIPER_ENABLE_REG_SHAPE_WEIGHT", default=True):
                    reg_w = (1.0 + alpha * np.power(np.abs(b.y.astype(np.float32, copy=False)), power)).astype(np.float32)
                else:
                    reg_w = np.ones_like(base_w, dtype=np.float32)
                tail_abs = float(_env_float("SNIPER_TAIL_ABS", 0.04))
                tail_mult = float(_env_float("SNIPER_TAIL_WEIGHT_MULT", 1.0))
                tail_abs_use = float(tail_abs) * (float(label_scale) if label_mode == "split_0_100" else 1.0)
                if tail_mult > 1.0:
                    tail_mask = np.abs(b.y.astype(np.float32, copy=False)) >= float(tail_abs_use)
                    if np.any(tail_mask):
                        reg_w[tail_mask] = (reg_w[tail_mask] * float(tail_mult)).astype(np.float32, copy=False)
                w_final = (base_w * reg_w).astype(np.float32, copy=False)
                if label_mode == "signed" and _env_bool("SNIPER_BALANCE_SIGN", default=True):
                    try:
                        yv = b.y.astype(np.float32, copy=False)
                        pos = yv > 0
                        neg = yv < 0
                        sum_pos = float(np.sum(w_final[pos])) if np.any(pos) else 0.0
                        sum_neg = float(np.sum(w_final[neg])) if np.any(neg) else 0.0
                        if sum_pos > 0.0 and sum_neg > 0.0:
                            tgt = 0.5 * (sum_pos + sum_neg)
                            w_final = w_final.copy()
                            w_final[pos] *= float(tgt / sum_pos)
                            w_final[neg] *= float(tgt / sum_neg)
                    except Exception:
                        pass
                if label_mode == "signed" and _env_bool("SNIPER_BALANCE_TAIL_SIGN", default=True):
                    try:
                        yv = b.y.astype(np.float32, copy=False)
                        tail_sign_mask = np.abs(yv) >= float(tail_abs_use)
                        pos = tail_sign_mask & (yv > 0)
                        neg = tail_sign_mask & (yv < 0)
                        sum_pos = float(np.sum(w_final[pos])) if np.any(pos) else 0.0
                        sum_neg = float(np.sum(w_final[neg])) if np.any(neg) else 0.0
                        if sum_pos > 0.0 and sum_neg > 0.0:
                            tgt = 0.5 * (sum_pos + sum_neg)
                            w_final = w_final.copy()
                            w_final[pos] *= float(tgt / sum_pos)
                            w_final[neg] *= float(tgt / sum_neg)
                    except Exception:
                        pass
                if label_mode == "signed" and _env_bool("SNIPER_BALANCE_BINS_SIGN", default=False):
                    try:
                        edges = list(getattr(cfg, "entry_reg_balance_bins", None) or [])
                        if edges:
                            edges = [-np.inf] + sorted(float(x) for x in edges) + [np.inf]
                            yv = b.y.astype(np.float32, copy=False)
                            bin_idx = np.digitize(yv, edges) - 1
                            bins_n = max(1, len(edges) - 1)
                            for i in range(int(bins_n // 2)):
                                j = (bins_n - 1) - i
                                if i == j:
                                    continue
                                m_i = bin_idx == i
                                m_j = bin_idx == j
                                if not (np.any(m_i) and np.any(m_j)):
                                    continue
                                sum_i = float(np.sum(w_final[m_i]))
                                sum_j = float(np.sum(w_final[m_j]))
                                if sum_i > 0.0 and sum_j > 0.0:
                                    tgt = 0.5 * (sum_i + sum_j)
                                    w_final = w_final.copy()
                                    w_final[m_i] *= float(tgt / sum_i)
                                    w_final[m_j] *= float(tgt / sum_j)
                    except Exception:
                        pass
                max_mult = float(_env_float("SNIPER_WEIGHT_MAX_MULT", 0.0))
                if max_mult > 1.0:
                    try:
                        med = float(np.median(w_final[np.isfinite(w_final)]))
                        cap = med * max_mult if med > 0 else 0.0
                        if cap > 0.0:
                            w_final = np.clip(w_final, 0.0, cap).astype(np.float32, copy=False)
                    except Exception:
                        pass
                y_final = b.y.astype(np.float32, copy=False)
                if label_mode == "signed" and _env_bool("SNIPER_TARGET_CENTER_WEIGHTED", default=True):
                    try:
                        ww = w_final.astype(np.float64, copy=False)
                        yy = y_final.astype(np.float64, copy=False)
                        center_method = str(os.getenv("SNIPER_TARGET_CENTER_METHOD", "median")).strip().lower()
                        if center_method == "mean":
                            den = float(np.sum(ww))
                            if den > 0.0:
                                y_bias = float(np.sum(yy * ww) / den)
                            else:
                                y_bias = float(np.mean(yy))
                        else:
                            y_bias = _weighted_median(yy, ww)
                        if np.isfinite(y_bias):
                            y_final = (y_final - np.float32(y_bias)).astype(np.float32, copy=False)
                            y_final = np.clip(y_final, -0.2, 0.2).astype(np.float32, copy=False)
                            print(
                                f"[sniper-train] target_center {name}: method={center_method} bias={y_bias:+.5f}",
                                flush=True,
                            )
                    except Exception:
                        pass
                b = SniperBatch(
                    X=b.X,
                    y=y_final,
                    w=w_final.astype(np.float32, copy=False),
                    ts=b.ts,
                    sym_id=b.sym_id,
                    feature_cols=b.feature_cols,
                )
                _print_dist_stats(f"{side} {name} y_final", b.y, weights=b.w)
                _print_dist_stats(f"{side} {name} w_final", b.w)
                _print_regression_balance_stats(f"{side} {name} final_balance", b.y, weights=b.w)
                print(f"[sniper-train] treinando EntryScore {side} {name} ({w}m)...", flush=True)
                t_train = time.perf_counter()
                params_side = dict(entry_params)
                try:
                    side_override = entry_params_by_side.get(str(side).strip().lower(), None)
                    if isinstance(side_override, dict):
                        params_side.update(side_override)
                except Exception:
                    pass
                if str(getattr(cfg, "entry_model_type", "xgb")).strip().lower() == "catboost":
                    m, m_meta = _train_catboost_regressor(b, params_side)
                else:
                    reg_params = dict(params_side)
                    reg_params.setdefault("objective", "reg:squarederror")
                    m, m_meta = _train_xgb_regressor(b, reg_params, y_transform="none")
                if timing_on:
                    print(
                        f"[sniper-train][timing] train_entry_{side}_{name}_{int(w)}m_s={time.perf_counter() - t_train:.2f}",
                        flush=True,
                    )
                entry_models_by_side[side][name] = (m, m_meta)
                print(f"[sniper-train] EntryScore {side} {name}: best_iter={m_meta['best_iteration']}", flush=True)

        if cls_enabled:
            for side in entry_sides:
                cls_base_batch = entry_cls_batches_by_side.get(side)
                if cls_base_batch is None or cls_base_batch.X.size == 0:
                    continue
                for name, w in entry_specs:
                    b_cls = cls_base_batch
                    max_rows_cls = int(
                        os.getenv(
                            "SNIPER_MAX_ROWS_ENTRY_CLS",
                            str(int(getattr(cfg, "max_rows_entry", 2_000_000))),
                        )
                        or int(getattr(cfg, "max_rows_entry", 2_000_000))
                    )
                    if max_rows_cls > 0 and b_cls.y.size > max_rows_cls:
                        rng_cls = np.random.default_rng(7331 + int(w))
                        keep_cls = _sample_indices_binary(
                            b_cls.y.astype(np.float32, copy=False),
                            max_rows_cls,
                            rng_cls,
                        )
                        if keep_cls.size > 0:
                            b_cls = SniperBatch(
                                X=b_cls.X[keep_cls],
                                y=b_cls.y[keep_cls],
                                w=b_cls.w[keep_cls],
                                ts=b_cls.ts[keep_cls],
                                sym_id=b_cls.sym_id[keep_cls],
                                feature_cols=b_cls.feature_cols,
                            )
                y_cls = (b_cls.y >= 0.5)
                pos = int(np.sum(y_cls))
                neg = int(y_cls.size - pos)
                if pos == 0 or neg == 0:
                    print(
                        f"[sniper-train] EntryGate {side} {name}: classe unica (pos={pos} neg={neg}), pulando",
                        flush=True,
                    )
                    continue
                # Ajusta proporcao pos/neg para o alvo (ex.: 0.20 => 1:4), sem mexer no label.
                keep_ratio = _rebalance_binary_to_target_ratio(
                    b_cls.y.astype(np.float32, copy=False),
                    float(cls_target_pos_ratio),
                    np.random.default_rng(8441 + int(w)),
                )
                if keep_ratio.size > 0 and keep_ratio.size < b_cls.y.size:
                    b_cls = SniperBatch(
                        X=b_cls.X[keep_ratio],
                        y=b_cls.y[keep_ratio],
                        w=b_cls.w[keep_ratio],
                        ts=b_cls.ts[keep_ratio],
                        sym_id=b_cls.sym_id[keep_ratio],
                        feature_cols=b_cls.feature_cols,
                    )
                    y_cls = (b_cls.y >= 0.5)
                    pos = int(np.sum(y_cls))
                    neg = int(y_cls.size - pos)
                    print(
                        f"[sniper-train] EntryGate {side} {name}: rebalanced pos={pos} neg={neg} pos_ratio={(float(pos)/float(max(1,pos+neg))):.3f}",
                        flush=True,
                    )
                w_cls = b_cls.w.astype(np.float32, copy=False) if getattr(b_cls, "w", None) is not None else np.ones_like(b_cls.y, dtype=np.float32)
                if _env_bool("SNIPER_CLS_BALANCE_CLASS_WEIGHT", default=True):
                    try:
                        sum_pos = float(np.sum(w_cls[y_cls]))
                        sum_neg = float(np.sum(w_cls[~y_cls]))
                        if sum_pos > 0.0 and sum_neg > 0.0:
                            tgt = 0.5 * (sum_pos + sum_neg)
                            w_cls = w_cls.copy()
                            w_cls[y_cls] *= float(tgt / sum_pos)
                            w_cls[~y_cls] *= float(tgt / sum_neg)
                    except Exception:
                        pass
                b_cls = SniperBatch(
                    X=b_cls.X,
                    y=(y_cls.astype(np.float32, copy=False)),
                    w=w_cls.astype(np.float32, copy=False),
                    ts=b_cls.ts,
                    sym_id=b_cls.sym_id,
                    feature_cols=b_cls.feature_cols,
                )
                _print_binary_balance_stats(f"{side} {name} cls_final_balance", b_cls.y, weights=b_cls.w)
                print(f"[sniper-train] treinando EntryGate {side} {name} ({w}m)...", flush=True)
                t_train_cls = time.perf_counter()
                params_cls_side = dict(cls_params)
                try:
                    side_override = cls_params_by_side.get(str(side).strip().lower(), None)
                    if isinstance(side_override, dict):
                        params_cls_side.update(side_override)
                except Exception:
                    pass
                if cls_model_type == "catboost":
                    m_cls, m_cls_meta = _train_catboost_classifier(b_cls, params_cls_side)
                else:
                    m_cls, m_cls_meta = _train_xgb_classifier(b_cls, params_cls_side)
                if timing_on:
                    print(
                        f"[sniper-train][timing] train_entry_cls_{side}_{name}_{int(w)}m_s={time.perf_counter() - t_train_cls:.2f}",
                        flush=True,
                    )
                entry_cls_models_by_side[side][name] = (m_cls, m_cls_meta)
                print(f"[sniper-train] EntryGate {side} {name}: best_iter={m_cls_meta['best_iteration']}", flush=True)

        has_reg_models = any(entry_models_by_side[s] for s in entry_sides)
        has_cls_models = (bool(cls_enabled) and any(entry_cls_models_by_side[s] for s in entry_sides))
        if (not has_reg_models) and (not has_cls_models):
            print(f"[sniper-train] {tail}d: nenhum modelo treinado, pulando", flush=True)
            continue

        period_train_end = None
        try:
            sample_pack = None
            if pack_by_side:
                sample_pack = pack_by_side.get(entry_sides[0]) or next(iter(pack_by_side.values()), None)
            sample_batch = base_batch_by_side.get(entry_sides[0]) if base_batch_by_side else None
            if sample_pack is not None and getattr(sample_pack, "train_end_utc", None) is not None:
                period_train_end = pd.to_datetime(sample_pack.train_end_utc)
            elif sample_batch is not None and sample_batch.ts.size:
                period_train_end = pd.to_datetime(sample_batch.ts.max())
        except Exception:
            period_train_end = None

        # salva modelos de regressao (opcional)
        if reg_enabled:
            for side in entry_sides:
                entry_models = entry_models_by_side.get(side, {})
                for name, w in entry_specs:
                    if name not in entry_models:
                        continue
                    model, _m_meta = entry_models[name]
                    spec = spec_by_name.get(name, {})
                    w_int = int(spec.get("window") or w)
                    d = period_dir / f"entry_model_{side}_{w_int}m"
                    d.mkdir(parents=True, exist_ok=True)
                    _save_model(model, d / "model_entry.json")
                if base_name in entry_models:
                    (period_dir / f"entry_model_{side}").mkdir(parents=True, exist_ok=True)
                    _save_model(entry_models[base_name][0], period_dir / f"entry_model_{side}" / "model_entry.json")

        if cls_enabled:
            for side in entry_sides:
                cls_models = entry_cls_models_by_side.get(side, {})
                for name, w in entry_specs:
                    if name not in cls_models:
                        continue
                    model_cls, _m_meta_cls = cls_models[name]
                    spec = spec_by_name.get(name, {})
                    w_int = int(spec.get("window") or w)
                    d = period_dir / f"entry_cls_model_{side}_{w_int}m"
                    d.mkdir(parents=True, exist_ok=True)
                    _save_model(model_cls, d / "model_entry_gate.json")
                if base_name in cls_models:
                    (period_dir / f"entry_cls_model_{side}").mkdir(parents=True, exist_ok=True)
                    _save_model(cls_models[base_name][0], period_dir / f"entry_cls_model_{side}" / "model_entry_gate.json")

        sample_batch = base_batch_by_side.get(entry_sides[0]) if base_batch_by_side else None
        sample_cls_batch = entry_cls_batches_by_side.get(entry_sides[0]) if entry_cls_batches_by_side else None
        entry_feat_cols = (
            sample_batch.feature_cols
            if sample_batch is not None
            else (sample_cls_batch.feature_cols if sample_cls_batch is not None else [])
        )
        meta = {
            "entry": {
                "feature_cols": entry_feat_cols,
                "label_mode": str(label_mode),
                "label_scale": float(label_scale),
                "reg_enabled": bool(reg_enabled),
                "sides": list(entry_sides),
            },
            "entry_cls": {
                "enabled": bool(cls_enabled),
                "feature_cols": (sample_cls_batch.feature_cols if sample_cls_batch is not None else []),
                "label_template": str(cls_label_tpl),
                "positive_threshold": float(cls_thr),
                "model_type": str(cls_model_type),
                "sides": list(entry_sides),
            },
            # Ponto final do treino (auditavel): preferimos o valor deterministico vindo do dataflow
            # (cutoff - lookahead). Se nao existir, cai no max(ts) do dataset amostrado.
            "train_end_utc": (
                str(pd.to_datetime(period_train_end))
                if period_train_end is not None
                else (
                    np.datetime_as_string(np.max(sample_batch.ts), unit="s")
                    if sample_batch is not None and sample_batch.ts.size
                    else (
                        np.datetime_as_string(np.max(sample_cls_batch.ts), unit="s")
                        if sample_cls_batch is not None and sample_cls_batch.ts.size
                        else None
                    )
                )
            ),
            "symbols": (
                sample_pack.symbols_used if (sample_pack is not None and getattr(sample_pack, "symbols_used", None)) else (sample_pack.symbols if sample_pack is not None else list(symbols))
            ),
            "symbols_total": int(len(sample_pack.symbols) if sample_pack is not None and getattr(sample_pack, "symbols", None) else len(symbols)),
            "symbols_used": (sample_pack.symbols_used or None) if sample_pack is not None else None,
            "symbols_skipped": (sample_pack.symbols_skipped or None) if sample_pack is not None else None,
        }
        # meta extra por janela e lado (regressao)
        if reg_enabled:
            for side in entry_sides:
                entry_models = entry_models_by_side.get(side, {})
                entry_batches = entry_batches_by_side.get(side, {})
                base_batch = base_batch_by_side.get(side)
                for name, w in entry_specs:
                    if name not in entry_models:
                        continue
                    if name in entry_batches:
                        feat_cols = entry_batches[name].feature_cols
                    else:
                        # fallback para o batch base carregado on-demand
                        feat_cols = base_batch.feature_cols if base_batch is not None else []
                    _model, _m_meta = entry_models[name]
                    spec = spec_by_name.get(name, {})
                    w_int = int(spec.get("window") or w)
                    meta_key = f"entry_{side}_{w_int}m"
                    meta[meta_key] = {
                        "feature_cols": feat_cols,
                        "best_iteration": int((_m_meta or {}).get("best_iteration", 0) or 0),
                        "metrics": ((_m_meta or {}).get("metrics", None) or None),
                        "pred_bias": float((_m_meta or {}).get("pred_bias", 0.0) or 0.0),
                        "calibration": ((_m_meta or {}).get("calibration", None) or None),
                    }
        if cls_enabled:
            for side in entry_sides:
                cls_models = entry_cls_models_by_side.get(side, {})
                cls_base_batch = entry_cls_batches_by_side.get(side)
                for name, w in entry_specs:
                    if name not in cls_models:
                        continue
                    _cls_model, _cls_meta = cls_models[name]
                    spec = spec_by_name.get(name, {})
                    w_int = int(spec.get("window") or w)
                    cls_meta_key = f"entry_cls_{side}_{w_int}m"
                    meta[cls_meta_key] = {
                        "feature_cols": (cls_base_batch.feature_cols if cls_base_batch is not None else []),
                        "best_iteration": int((_cls_meta or {}).get("best_iteration", 0) or 0),
                        "metrics": ((_cls_meta or {}).get("metrics", None) or None),
                        "problem_type": ((_cls_meta or {}).get("problem_type", None) or None),
                        "calibration": ((_cls_meta or {}).get("calibration", None) or None),
                    }
        (period_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sniper-train] {tail}d salvo em {period_dir}")

    return run_dir


if __name__ == "__main__":
    run = train_sniper_models()
    print(f"[sniper-train] modelos salvos em: {run}")

