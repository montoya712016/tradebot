# -*- coding: utf-8 -*-
"""
Timing-adjusted regression labels for Sniper (regression-only).
"""
from __future__ import annotations

from typing import Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd

from numba import njit


if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = Path(__file__).resolve().parents[2].parent / "cache_sniper" / "numba"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache_dir)


# ===== TIMING-ADJUSTED REGRESSION LABEL =====
# Label de regressão que penaliza entradas antecipadas
# Janela maior para capturar swings mais amplos (ex.: 360m = 6h)
TIMING_HORIZON_PROFIT = 360   # minutos à frente para lucro médio realizável
TIMING_K_LOOKAHEAD = 10       # minutos à frente para procurar pontos melhores
TIMING_TOP_N = 3              # quantidade de melhores pontos futuros
TIMING_ALPHA = 0.0            # peso da penalização por ponto melhor à frente (0 = sem penalização)
TIMING_VOL_WINDOW = 14        # janela para volatilidade (usado em weights)
TIMING_LABEL_CLIP = 0.20      # clip labels a ±20% para evitar outliers
TIMING_SOFTMAX_TEMP = 0.01    # temperatura do softmax (menor = mais "máximo")
TIMING_USE_SOFTMAX = True     # usa softmax-weighted future return em vez de top-n
TIMING_USE_DOMINANT = True    # usa movimento dominante (pos/neg) quando mais forte
TIMING_DOMINANT_MIX = 0.50    # mistura entre profit_now e dominante (0..1)
TIMING_WEIGHT_LABEL_MULT = 20.0  # multiplicador de peso por |label|
TIMING_LABEL_SCALE = 100.0    # escala dos labels long/short (0..100)
TIMING_WEIGHT_VOL_MULT = 1.0  # multiplicador de peso por volatilidade
TIMING_WEIGHT_MIN = 0.1       # peso mínimo para evitar zeros
TIMING_WEIGHT_MAX = 10.0      # peso máximo para evitar dominância
TIMING_SIDE_MAE_PENALTY = 1.25  # penaliza excursão adversa antes do melhor ponto
TIMING_SIDE_TIME_PENALTY = 0.35  # penaliza demora para atingir o melhor ponto
TIMING_SIDE_CROSS_PENALTY = 0.85  # penaliza um lado quando o retorno medio do lado oposto domina


@njit(cache=True)
def _compute_profit_now(close: np.ndarray, idx: int, horizon: int) -> float:
    """Calcula retorno médio futuro a partir do índice idx."""
    n = close.size
    if idx >= n - 1:
        return 0.0
    px0 = close[idx]
    if not np.isfinite(px0) or px0 <= 0.0:
        return 0.0
    end_idx = min(idx + horizon, n - 1)
    if end_idx <= idx:
        return 0.0
    total_ret = 0.0
    count = 0
    for j in range(idx + 1, end_idx + 1):
        pxj = close[j]
        if np.isfinite(pxj) and pxj > 0.0:
            ret = (pxj / px0) - 1.0
            total_ret += ret
            count += 1
    if count == 0:
        return 0.0
    return total_ret / float(count)


@njit(cache=True)
def _compute_timing_adjusted_label_numba(
    close: np.ndarray,
    horizon_profit: int,
    k_lookahead: int,
    top_n: int,
    alpha: float,
    label_clip: float,
    use_softmax: bool,
    softmax_temp: float,
    use_dominant: bool,
    dominant_mix: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula label de regressão ajustado a timing.

    Retorna:
        labels: array float32 com label contínuo
        profit_now_arr: array float32 com retorno médio futuro bruto
    """
    n = close.size
    labels = np.zeros(n, np.float32)
    profit_now_arr = np.zeros(n, np.float32)

    for i in range(n):
        profit_now = _compute_profit_now(close, i, horizon_profit)
        profit_now_arr[i] = np.float32(profit_now)

        future_profits = np.zeros(k_lookahead, np.float64)
        valid_count = 0
        for k in range(k_lookahead):
            future_idx = i + 1 + k
            if future_idx < n:
                pf = _compute_profit_now(close, future_idx, horizon_profit)
                future_profits[valid_count] = pf
                valid_count += 1

        best_future = 0.0
        best_future_pos = 0.0
        best_future_neg = 0.0
        if valid_count > 0:
            if use_softmax and softmax_temp > 0.0:
                # softmax-weighted future return, separado por sinal (evita viés de sinal)
                inv_t = 1.0 / softmax_temp

                # positivos
                max_pf_pos = -1e9
                for j in range(valid_count):
                    pf = future_profits[j]
                    if pf >= 0.0 and pf > max_pf_pos:
                        max_pf_pos = pf
                if max_pf_pos > -1e8:
                    denom = 0.0
                    num = 0.0
                    for j in range(valid_count):
                        pf = future_profits[j]
                        if pf >= 0.0:
                            w = np.exp((pf - max_pf_pos) * inv_t)
                            denom += w
                            num += w * pf
                    if denom > 0.0:
                        best_future_pos = num / denom

                # negativos (softmax em -pf para enfatizar perdas maiores)
                max_pf_neg = -1e9
                for j in range(valid_count):
                    pf = future_profits[j]
                    if pf <= 0.0:
                        npf = -pf
                        if npf > max_pf_neg:
                            max_pf_neg = npf
                if max_pf_neg > 0.0:
                    denom = 0.0
                    num = 0.0
                    for j in range(valid_count):
                        pf = future_profits[j]
                        if pf <= 0.0:
                            npf = -pf
                            w = np.exp((npf - max_pf_neg) * inv_t)
                            denom += w
                            num += w * pf
                    if denom > 0.0:
                        best_future_neg = num / denom
            else:
                # top-N separado por sinal
                actual_top_n = min(top_n, valid_count)
                # positivos
                pos_vals = np.zeros(valid_count, np.float64)
                pos_count = 0
                neg_vals = np.zeros(valid_count, np.float64)
                neg_count = 0
                for j in range(valid_count):
                    pf = future_profits[j]
                    if pf >= 0.0:
                        pos_vals[pos_count] = pf
                        pos_count += 1
                    else:
                        neg_vals[neg_count] = pf
                        neg_count += 1
                if pos_count > 0:
                    top_sum = 0.0
                    used = 0
                    for _ in range(min(actual_top_n, pos_count)):
                        max_val = -1e9
                        max_idx = -1
                        for j in range(pos_count):
                            if pos_vals[j] > max_val:
                                max_val = pos_vals[j]
                                max_idx = j
                        if max_idx >= 0:
                            top_sum += pos_vals[max_idx]
                            pos_vals[max_idx] = -1e9
                            used += 1
                    if used > 0:
                        best_future_pos = top_sum / float(used)
                if neg_count > 0:
                    top_sum = 0.0
                    used = 0
                    for _ in range(min(actual_top_n, neg_count)):
                        min_val = 1e9
                        min_idx = -1
                        for j in range(neg_count):
                            if neg_vals[j] < min_val:
                                min_val = neg_vals[j]
                                min_idx = j
                        if min_idx >= 0:
                            top_sum += neg_vals[min_idx]
                            neg_vals[min_idx] = 1e9
                            used += 1
                    if used > 0:
                        best_future_neg = top_sum / float(used)

        best_future = best_future_pos if profit_now >= 0.0 else best_future_neg

        profit_use = profit_now
        if use_dominant:
            # escolhe direcao dominante pelo maior modulo (pos vs neg)
            if abs(best_future_pos) >= abs(best_future_neg):
                dom = best_future_pos
            else:
                dom = best_future_neg
            if dominant_mix > 0.0:
                # se sinais alinham, mistura; se divergem, usa dominante se for mais forte
                if (profit_now >= 0.0 and dom >= 0.0) or (profit_now <= 0.0 and dom <= 0.0):
                    mix = dominant_mix
                    if mix > 1.0:
                        mix = 1.0
                    profit_use = (1.0 - mix) * profit_now + mix * dom
                else:
                    if abs(dom) > abs(profit_now):
                        profit_use = dom

        if profit_use >= 0:
            penalty = alpha * max(0.0, best_future - profit_use)
            raw_label = profit_use - penalty
        else:
            penalty = alpha * max(0.0, profit_use - best_future)
            raw_label = profit_use + penalty

        if raw_label > label_clip:
            raw_label = label_clip
        elif raw_label < -label_clip:
            raw_label = -label_clip

        labels[i] = np.float32(raw_label)

    return labels, profit_now_arr


@njit(cache=True)
def _compute_timing_weights_numba(
    labels: np.ndarray,
    vol: np.ndarray,
    label_mult: float,
    vol_mult: float,
    weight_min: float,
    weight_max: float,
) -> np.ndarray:
    """
    Calcula pesos para cada amostra baseado em |label| e volatilidade.
    """
    n = labels.size
    weights = np.zeros(n, np.float32)

    vol_mean = 0.0
    vol_count = 0
    for i in range(n):
        if np.isfinite(vol[i]) and vol[i] > 0:
            vol_mean += vol[i]
            vol_count += 1
    if vol_count > 0:
        vol_mean /= float(vol_count)
    else:
        vol_mean = 1.0

    for i in range(n):
        abs_label = abs(labels[i])
        w = weight_min + label_mult * abs_label
        if vol_mult > 0 and np.isfinite(vol[i]) and vol[i] > 0 and vol_mean > 0:
            vol_norm = vol[i] / vol_mean
            w += vol_mult * vol_norm
        if w < weight_min:
            w = weight_min
        if w > weight_max:
            w = weight_max
        weights[i] = np.float32(w)

    return weights


@njit(cache=True)
def _compute_side_labels_from_future_excursion_numba(
    close: np.ndarray,
    horizon_bars: int,
    label_clip: float,
    side_mae_penalty: float,
    side_time_penalty: float,
    side_cross_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula labels por lado com foco em retorno medio esperado (nao melhor caso).
    - long_base: media de PnL long futuro no horizonte, truncada em >= 0
    - short_base: media de PnL short futuro no horizonte, truncada em >= 0
    Depois aplica penalidade de arrependimento por excursao adversa antes do melhor ponto.
    """
    n = close.size
    long_raw = np.zeros(n, np.float32)
    short_raw = np.zeros(n, np.float32)

    for i in range(n):
        if i >= n - 1:
            continue
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue

        end_idx = min(i + horizon_bars, n - 1)
        if end_idx <= i:
            continue

        hlen = end_idx - i
        long_pnls = np.zeros(hlen, np.float64)
        short_pnls = np.zeros(hlen, np.float64)
        best_long = -1e18
        best_long_idx = -1
        best_short = -1e18
        best_short_idx = -1
        sum_long = 0.0
        sum_short = 0.0
        valid = 0

        k = 0
        for j in range(i + 1, end_idx + 1):
            pxj = close[j]
            if not np.isfinite(pxj) or pxj <= 0.0:
                long_pnls[k] = 0.0
                short_pnls[k] = 0.0
                k += 1
                continue
            long_pnl = (pxj / px0) - 1.0
            short_pnl = (px0 / pxj) - 1.0
            long_pnls[k] = long_pnl
            short_pnls[k] = short_pnl
            sum_long += long_pnl
            sum_short += short_pnl
            valid += 1
            if long_pnl > best_long:
                best_long = long_pnl
                best_long_idx = k + 1
            if short_pnl > best_short:
                best_short = short_pnl
                best_short_idx = k + 1
            k += 1

        if valid <= 0:
            continue

        long_base = sum_long / float(valid)
        if long_base < 0.0:
            long_base = 0.0
        short_base = sum_short / float(valid)
        if short_base < 0.0:
            short_base = 0.0

        long_val = 0.0
        if long_base > 0.0 and best_long_idx > 0:
            mae_before = 0.0
            for t in range(best_long_idx):
                pnl_t = long_pnls[t]
                if pnl_t < 0.0:
                    dd = -pnl_t
                    if dd > mae_before:
                        mae_before = dd
            time_frac = float(best_long_idx) / float(hlen if hlen > 0 else 1)
            penalty = side_mae_penalty * mae_before + side_time_penalty * (long_base * time_frac)
            long_val = long_base - penalty

        short_val = 0.0
        if short_base > 0.0 and best_short_idx > 0:
            mae_before_short = 0.0
            for t in range(best_short_idx):
                pnl_t = short_pnls[t]
                if pnl_t < 0.0:
                    dd = -pnl_t
                    if dd > mae_before_short:
                        mae_before_short = dd
            time_frac = float(best_short_idx) / float(hlen if hlen > 0 else 1)
            penalty = side_mae_penalty * mae_before_short + side_time_penalty * (short_base * time_frac)
            short_val = short_base - penalty

        if long_val < 0.0:
            long_val = 0.0
        if short_val < 0.0:
            short_val = 0.0

        # Penalizacao cruzada: evita manter label alto no lado "perdedor"
        # quando o retorno medio do lado oposto ja domina (reversao).
        if side_cross_penalty > 0.0:
            long_val = long_val - side_cross_penalty * short_base
            short_val = short_val - side_cross_penalty * long_base
            if long_val < 0.0:
                long_val = 0.0
            if short_val < 0.0:
                short_val = 0.0

        if long_val > label_clip:
            long_val = label_clip
        if short_val > label_clip:
            short_val = label_clip

        long_raw[i] = np.float32(long_val)
        short_raw[i] = np.float32(short_val)

    return long_raw, short_raw


def apply_timing_regression_labels(
    df: pd.DataFrame,
    *,
    horizon_profit: int | None = None,
    k_lookahead: int | None = None,
    top_n: int | None = None,
    alpha: float | None = None,
    label_clip: float | None = None,
    weight_label_mult: float | None = None,
    weight_vol_mult: float | None = None,
    weight_min: float | None = None,
    weight_max: float | None = None,
    vol_window: int | None = None,
    candle_sec: int = 60,
    label_center: bool | None = None,
    use_softmax: bool | None = None,
    softmax_temp: float | None = None,
    use_dominant: bool | None = None,
    dominant_mix: float | None = None,
    side_mae_penalty: float | None = None,
    side_time_penalty: float | None = None,
    side_cross_penalty: float | None = None,
) -> pd.DataFrame:
    """
    Aplica labels de regressão ajustados a timing.
    """
    horizon = horizon_profit if horizon_profit is not None else TIMING_HORIZON_PROFIT
    k_look = k_lookahead if k_lookahead is not None else TIMING_K_LOOKAHEAD
    topn = top_n if top_n is not None else TIMING_TOP_N
    alph = alpha if alpha is not None else TIMING_ALPHA
    clip = label_clip if label_clip is not None else TIMING_LABEL_CLIP
    w_label = weight_label_mult if weight_label_mult is not None else TIMING_WEIGHT_LABEL_MULT
    w_vol = weight_vol_mult if weight_vol_mult is not None else TIMING_WEIGHT_VOL_MULT
    w_min = weight_min if weight_min is not None else TIMING_WEIGHT_MIN
    w_max = weight_max if weight_max is not None else TIMING_WEIGHT_MAX
    vol_win = vol_window if vol_window is not None else TIMING_VOL_WINDOW
    use_sm = use_softmax if use_softmax is not None else TIMING_USE_SOFTMAX
    sm_temp = softmax_temp if softmax_temp is not None else TIMING_SOFTMAX_TEMP
    side_mae = float(side_mae_penalty) if side_mae_penalty is not None else float(TIMING_SIDE_MAE_PENALTY)
    side_time = float(side_time_penalty) if side_time_penalty is not None else float(TIMING_SIDE_TIME_PENALTY)
    side_cross = float(side_cross_penalty) if side_cross_penalty is not None else float(TIMING_SIDE_CROSS_PENALTY)
    if use_dominant is None:
        env_dom = os.getenv("SNIPER_TIMING_USE_DOMINANT", "").strip().lower()
        if env_dom:
            use_dom = env_dom not in {"0", "false", "no", "off"}
        else:
            use_dom = bool(TIMING_USE_DOMINANT)
    else:
        use_dom = bool(use_dominant)
    if dominant_mix is None:
        env_mix = os.getenv("SNIPER_TIMING_DOMINANT_MIX", "").strip()
        if env_mix:
            try:
                dom_mix = float(env_mix)
            except Exception:
                dom_mix = float(TIMING_DOMINANT_MIX)
        else:
            dom_mix = float(TIMING_DOMINANT_MIX)
    else:
        dom_mix = float(dominant_mix)

    bars_per_min = 60.0 / float(candle_sec)
    horizon_bars = max(1, int(round(float(horizon) * bars_per_min)))
    k_bars = max(1, int(round(float(k_look) * bars_per_min)))
    vol_bars = max(1, int(round(float(vol_win) * bars_per_min)))

    close = df["close"].to_numpy(np.float64, copy=False)
    labels, profit_now = _compute_timing_adjusted_label_numba(
        close,
        horizon_bars,
        k_bars,
        topn,
        alph,
        clip,
        bool(use_sm),
        float(sm_temp),
        bool(use_dom),
        float(dom_mix),
    )
    # centraliza labels (opcional) para evitar bias de mercado
    if label_center is None:
        env_center = os.getenv("SNIPER_LABEL_CENTER", "").strip().lower()
        label_center = bool(env_center) and env_center not in {"0", "false", "no", "off"}
    if label_center:
        try:
            mean_val = float(np.mean(labels.astype(np.float64)))
            if np.isfinite(mean_val) and abs(mean_val) > 0.0:
                labels = labels.astype(np.float32, copy=True)
                labels -= np.float32(mean_val)
                # reclip
                labels = np.clip(labels, -float(clip), float(clip)).astype(np.float32, copy=False)
        except Exception:
            pass

    log_ret = np.zeros(len(close), dtype=np.float64)
    log_ret[1:] = np.log(close[1:] / np.maximum(1e-12, close[:-1]))
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    vol = pd.Series(log_ret).rolling(window=vol_bars, min_periods=1).std().fillna(0.0).to_numpy(np.float64)
    weights = _compute_timing_weights_numba(
        labels,
        vol,
        w_label,
        w_vol,
        w_min,
        w_max,
    )

    # labels separados (long/short) em escala 0..100 (independentes por lado)
    clip_f = float(clip) if float(clip) > 0 else 1.0
    scale = float(TIMING_LABEL_SCALE)
    label_long_raw, label_short_raw = _compute_side_labels_from_future_excursion_numba(
        close,
        horizon_bars,
        clip_f,
        float(side_mae),
        float(side_time),
        float(side_cross),
    )
    label_long = (label_long_raw / clip_f * scale).astype(np.float32, copy=False)
    label_short = (label_short_raw / clip_f * scale).astype(np.float32, copy=False)
    # pesos por lado (mantém mesma lógica de peso por |label| + vol)
    weights_long = _compute_timing_weights_numba(
        label_long_raw,
        vol,
        w_label,
        w_vol,
        w_min,
        w_max,
    )
    weights_short = _compute_timing_weights_numba(
        label_short_raw,
        vol,
        w_label,
        w_vol,
        w_min,
        w_max,
    )

    df["timing_label"] = pd.Series(labels.astype(np.float32), index=df.index)
    df["timing_profit_now"] = pd.Series(profit_now.astype(np.float32), index=df.index)
    df["timing_weight"] = pd.Series(weights.astype(np.float32), index=df.index)
    df["timing_label_pct"] = pd.Series((labels * 100.0).astype(np.float32), index=df.index)
    df["timing_profit_now_pct"] = pd.Series((profit_now * 100.0).astype(np.float32), index=df.index)
    df["timing_label_long"] = pd.Series(label_long, index=df.index)
    df["timing_label_short"] = pd.Series(label_short, index=df.index)
    df["timing_weight_long"] = pd.Series(weights_long.astype(np.float32), index=df.index)
    df["timing_weight_short"] = pd.Series(weights_short.astype(np.float32), index=df.index)
    return df


__all__ = [
    "apply_timing_regression_labels",
    "TIMING_HORIZON_PROFIT",
    "TIMING_K_LOOKAHEAD",
    "TIMING_TOP_N",
    "TIMING_ALPHA",
    "TIMING_VOL_WINDOW",
    "TIMING_LABEL_CLIP",
    "TIMING_LABEL_SCALE",
    "TIMING_USE_DOMINANT",
    "TIMING_DOMINANT_MIX",
    "TIMING_WEIGHT_LABEL_MULT",
    "TIMING_WEIGHT_VOL_MULT",
    "TIMING_WEIGHT_MIN",
    "TIMING_WEIGHT_MAX",
]
