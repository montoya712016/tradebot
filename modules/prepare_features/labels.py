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
TIMING_HORIZON_PROFIT = 360   # minutos à frente para qualidade da entrada
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
TIMING_SIDE_MAE_PENALTY = 1.10
TIMING_SIDE_TIME_PENALTY = 0.55
TIMING_SIDE_GIVEBACK_PENALTY = 0.65
TIMING_SIDE_CROSS_PENALTY = 0.35
TIMING_SIDE_REV_LOOKBACK_MIN = 90
TIMING_SIDE_CHASE_PENALTY = 0.90
TIMING_SIDE_REVERSAL_BONUS = 0.55
TIMING_SIDE_CONFIRM_MIN = 20
TIMING_SIDE_CONFIRM_MOVE = 0.0025
TIMING_SIDE_PRECONF_SUPPRESS = 0.15

# ===== EDGE + GATES (SUPERVISED PIPELINE) =====
# Edge labels: potencial de lucro (MFE) menos risco (MAE) dentro de um horizonte.
EDGE_HORIZON_MIN = 720
EDGE_DD_LAMBDA = 1.0
EDGE_LABEL_CLIP = 0.20
EDGE_LABEL_SCALE = 100.0

# Entry gates: 1 quando existe TP antes de "estourar" SL (aprox. por MFE/MAE dentro do horizonte).
ENTRY_GATE_TMAX_MIN = 720
ENTRY_GATE_TP_PCT = 0.02
ENTRY_GATE_SL_PCT = 0.01
ENTRY_GATE_SCALE = 100.0
ENTRY_GATE_REVERSAL_ONLY = True
ENTRY_GATE_PRELOOKBACK_MIN = 120
ENTRY_GATE_PREMOVE_ATR_SOFT = 0.80
ENTRY_GATE_PREMOVE_ATR_HARD = 2.00
ENTRY_GATE_TP_ATR = 1.80
ENTRY_GATE_SL_ATR = 1.20
ENTRY_GATE_NEAR_EXTREMA_ATR = 0.35
ENTRY_GATE_TIMEOUT_RET_ATR_MIN = 0.25
ENTRY_GATE_ATR_SPAN = 48
ENTRY_GATE_WEIGHT_MIN = 0.20
ENTRY_GATE_WEIGHT_MAX = 4.00


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
def _future_window_max_min_numba(close: np.ndarray, horizon_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Para cada i, retorna:
      - fmax[i] = max(close[i+1 : i+1+horizon])
      - fmin[i] = min(close[i+1 : i+1+horizon])

    Implementação O(n) com filas monotônicas no domínio reverso (evita cópia do array).
    """
    n = close.size
    fmax_rev = np.empty(n, np.float32)
    fmin_rev = np.empty(n, np.float32)
    for j in range(n):
        fmax_rev[j] = np.nan
        fmin_rev[j] = np.nan
    if n == 0 or horizon_bars <= 0:
        out_max = np.empty(n, np.float32)
        out_min = np.empty(n, np.float32)
        for i in range(n):
            out_max[i] = np.nan
            out_min[i] = np.nan
        return out_max, out_min

    dq_max = np.empty(n, np.int64)
    dq_min = np.empty(n, np.int64)
    hmax = 0
    tmax = 0
    hmin = 0
    tmin = 0

    for j in range(n):
        lim = j - horizon_bars
        while hmax < tmax and dq_max[hmax] < lim:
            hmax += 1
        while hmin < tmin and dq_min[hmin] < lim:
            hmin += 1

        if hmax < tmax:
            fmax_rev[j] = np.float32(close[n - 1 - dq_max[hmax]])
        else:
            fmax_rev[j] = np.nan
        if hmin < tmin:
            fmin_rev[j] = np.float32(close[n - 1 - dq_min[hmin]])
        else:
            fmin_rev[j] = np.nan

        vj = close[n - 1 - j]
        while hmax < tmax:
            idx = dq_max[tmax - 1]
            if close[n - 1 - idx] <= vj:
                tmax -= 1
            else:
                break
        dq_max[tmax] = j
        tmax += 1

        while hmin < tmin:
            idx = dq_min[tmin - 1]
            if close[n - 1 - idx] >= vj:
                tmin -= 1
            else:
                break
        dq_min[tmin] = j
        tmin += 1

    out_max = np.empty(n, np.float32)
    out_min = np.empty(n, np.float32)
    for i in range(n):
        out_max[i] = fmax_rev[n - 1 - i]
        out_min[i] = fmin_rev[n - 1 - i]
    return out_max, out_min


@njit(cache=True)
def _compute_edge_labels_numba(
    close: np.ndarray,
    horizon_bars: int,
    dd_lambda: float,
    clip_pct: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    fmax, fmin = _future_window_max_min_numba(close, horizon_bars)
    n = close.size
    out_long = np.zeros(n, np.float32)
    out_short = np.zeros(n, np.float32)
    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue
        mx = float(fmax[i])
        mn = float(fmin[i])
        if (not np.isfinite(mx)) or mx <= 0.0:
            mx = px0
        if (not np.isfinite(mn)) or mn <= 0.0:
            mn = px0

        # long: MFE - lambda*MAE
        mfe_l = (mx / px0) - 1.0
        mae_l = 1.0 - (mn / px0)
        if mae_l < 0.0:
            mae_l = 0.0
        score_l = mfe_l - dd_lambda * mae_l
        if score_l < 0.0:
            score_l = 0.0

        # short: MFE(short) - lambda*MAE(short)
        mfe_s = (px0 / mn) - 1.0
        mae_s = (mx / px0) - 1.0
        if mae_s < 0.0:
            mae_s = 0.0
        score_s = mfe_s - dd_lambda * mae_s
        if score_s < 0.0:
            score_s = 0.0

        if clip_pct > 0.0:
            if score_l > clip_pct:
                score_l = clip_pct
            if score_s > clip_pct:
                score_s = clip_pct
            out_long[i] = np.float32((score_l / clip_pct) * scale)
            out_short[i] = np.float32((score_s / clip_pct) * scale)
        else:
            out_long[i] = np.float32(score_l * scale)
            out_short[i] = np.float32(score_s * scale)
    return out_long, out_short


@njit(cache=True)
def _future_window_mean_numba(close: np.ndarray, horizon_bars: int) -> np.ndarray:
    """
    Media de close na janela futura [i+1 .. i+horizon_bars], ignorando nao-finitos.
    """
    n = close.size
    out = np.zeros(n, np.float32)
    csum = np.zeros(n + 1, np.float64)
    ccnt = np.zeros(n + 1, np.int64)
    for i in range(n):
        v = close[i]
        csum[i + 1] = csum[i]
        ccnt[i + 1] = ccnt[i]
        if np.isfinite(v):
            csum[i + 1] += float(v)
            ccnt[i + 1] += 1
    for i in range(n):
        s = i + 1
        e = i + 1 + horizon_bars
        if e > n:
            e = n
        if s >= e:
            out[i] = np.float32(close[i]) if np.isfinite(close[i]) else np.float32(0.0)
            continue
        tot = csum[e] - csum[s]
        cnt = ccnt[e] - ccnt[s]
        if cnt <= 0:
            out[i] = np.float32(close[i]) if np.isfinite(close[i]) else np.float32(0.0)
        else:
            out[i] = np.float32(tot / float(cnt))
    return out


@njit(cache=True)
def _compute_entry_gate_labels_numba(
    close: np.ndarray,
    horizon_bars: int,
    tp_pct: float,
    sl_pct: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    fmax, fmin = _future_window_max_min_numba(close, horizon_bars)
    fmean = _future_window_mean_numba(close, horizon_bars)
    n = close.size
    out_long = np.zeros(n, np.float32)
    out_short = np.zeros(n, np.float32)
    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue
        mx = float(fmax[i])
        mn = float(fmin[i])
        if (not np.isfinite(mx)) or mx <= 0.0:
            mx = px0
        if (not np.isfinite(mn)) or mn <= 0.0:
            mn = px0

        # long gate:
        # - TP via media futura (evita positivo por pico isolado)
        # - SL via minimo futuro (adversidade real)
        mean_px = float(fmean[i])
        if (not np.isfinite(mean_px)) or mean_px <= 0.0:
            mean_px = px0
        tp_l = (mean_px / px0) - 1.0
        mae_l = 1.0 - (mn / px0)
        if mae_l < 0.0:
            mae_l = 0.0
        if tp_l >= tp_pct and mae_l <= sl_pct:
            out_long[i] = np.float32(scale)

        # short gate:
        # - TP via media futura (evita positivo por fundo isolado)
        # - SL via maximo futuro (adversidade real)
        tp_s = (px0 / mean_px) - 1.0 if mean_px > 0.0 else 0.0
        mae_s = (mx / px0) - 1.0
        if mae_s < 0.0:
            mae_s = 0.0
        if tp_s >= tp_pct and mae_s <= sl_pct:
            out_short[i] = np.float32(scale)

    return out_long, out_short


def _compute_atr_abs_from_ohlc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    span_bars: int,
) -> np.ndarray:
    h = pd.Series(np.asarray(high, dtype=np.float64))
    l = pd.Series(np.asarray(low, dtype=np.float64))
    c = pd.Series(np.asarray(close, dtype=np.float64))
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    span = max(2, int(span_bars))
    atr = tr.ewm(span=span, adjust=False, min_periods=span).mean().bfill().ffill()
    atr_np = atr.to_numpy(dtype=np.float64, copy=False)
    floor = np.maximum(1e-8, np.abs(np.asarray(close, dtype=np.float64)) * 1e-6)
    return np.maximum(atr_np, floor).astype(np.float64, copy=False)


@njit(cache=True)
def _compute_entry_gate_reversal_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_abs: np.ndarray,
    past_high: np.ndarray,
    past_low: np.ndarray,
    fut_high: np.ndarray,
    fut_low: np.ndarray,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    pre_soft_atr: float,
    pre_hard_atr: float,
    near_ext_atr: float,
    timeout_ret_atr: float,
    scale: float,
    w_min: float,
    w_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    out_long = np.zeros(n, np.float32)
    out_short = np.zeros(n, np.float32)
    w_long = np.full(n, np.float32(max(0.0, w_min)), np.float32)
    w_short = np.full(n, np.float32(max(0.0, w_min)), np.float32)
    eps = 1e-9
    pre_h = pre_hard_atr if pre_hard_atr > pre_soft_atr else pre_soft_atr
    if pre_h <= 0.0:
        pre_h = 1.0
    if near_ext_atr <= 0.0:
        near_ext_atr = 0.25
    for i in range(n):
        px0 = close[i]
        atr = atr_abs[i]
        if (not np.isfinite(px0)) or (not np.isfinite(atr)) or px0 <= 0.0 or atr <= 0.0:
            continue
        end = i + horizon_bars
        if end >= n:
            end = n - 1
        if end <= i:
            continue

        ph = past_high[i]
        pl = past_low[i]
        fh = fut_high[i]
        fl = fut_low[i]

        drop = 0.0
        rise = 0.0
        dist_bottom = 1e9
        dist_top = 1e9
        if np.isfinite(ph):
            drop = (ph - px0) / atr
        if np.isfinite(pl):
            rise = (px0 - pl) / atr
        if np.isfinite(fl):
            dist_bottom = (px0 - fl) / atr
        if np.isfinite(fh):
            dist_top = (fh - px0) / atr

        cand_long = (drop >= pre_soft_atr) and (dist_bottom <= near_ext_atr)
        cand_short = (rise >= pre_soft_atr) and (dist_top <= near_ext_atr)

        if cand_long:
            tp_px = px0 + tp_atr * atr
            sl_px = px0 - sl_atr * atr
            hit = 0
            for j in range(i + 1, end + 1):
                hj = high[j]
                lj = low[j]
                cj = close[j]
                hit_tp = np.isfinite(hj) and (hj >= tp_px)
                hit_sl = np.isfinite(lj) and (lj <= sl_px)
                if hit_tp and hit_sl:
                    if np.isfinite(cj) and (cj >= px0):
                        hit = 1
                    else:
                        hit = -1
                    break
                if hit_tp:
                    hit = 1
                    break
                if hit_sl:
                    hit = -1
                    break
            if hit == 0:
                cend = close[end]
                if np.isfinite(cend):
                    r_to = (cend - px0) / atr
                    if r_to >= timeout_ret_atr:
                        hit = 1
            if hit > 0:
                out_long[i] = np.float32(scale)
            strength = drop / pre_h
            if strength < 0.0:
                strength = 0.0
            if strength > 1.5:
                strength = 1.5
            near = 1.0 - (dist_bottom / (near_ext_atr + eps))
            if near < 0.0:
                near = 0.0
            if near > 1.0:
                near = 1.0
            bonus = 0.5 if hit > 0 else 0.0
            ww = 0.6 + strength + near + bonus
            if ww < w_min:
                ww = w_min
            if ww > w_max:
                ww = w_max
            w_long[i] = np.float32(ww)

        if cand_short:
            tp_px = px0 - tp_atr * atr
            sl_px = px0 + sl_atr * atr
            hit = 0
            for j in range(i + 1, end + 1):
                hj = high[j]
                lj = low[j]
                cj = close[j]
                hit_tp = np.isfinite(lj) and (lj <= tp_px)
                hit_sl = np.isfinite(hj) and (hj >= sl_px)
                if hit_tp and hit_sl:
                    if np.isfinite(cj) and (cj <= px0):
                        hit = 1
                    else:
                        hit = -1
                    break
                if hit_tp:
                    hit = 1
                    break
                if hit_sl:
                    hit = -1
                    break
            if hit == 0:
                cend = close[end]
                if np.isfinite(cend):
                    r_to = (px0 - cend) / atr
                    if r_to >= timeout_ret_atr:
                        hit = 1
            if hit > 0:
                out_short[i] = np.float32(scale)
            strength = rise / pre_h
            if strength < 0.0:
                strength = 0.0
            if strength > 1.5:
                strength = 1.5
            near = 1.0 - (dist_top / (near_ext_atr + eps))
            if near < 0.0:
                near = 0.0
            if near > 1.0:
                near = 1.0
            bonus = 0.5 if hit > 0 else 0.0
            ww = 0.6 + strength + near + bonus
            if ww < w_min:
                ww = w_min
            if ww > w_max:
                ww = w_max
            w_short[i] = np.float32(ww)
    return out_long, out_short, w_long, w_short


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
    rev_lookback_bars: int,
    label_clip: float,
    side_mae_penalty: float,
    side_time_penalty: float,
    side_giveback_penalty: float,
    side_cross_penalty: float,
    side_chase_penalty: float,
    side_reversal_bonus: float,
    confirm_bars: int,
    confirm_move: float,
    preconfirm_suppress: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula labels por lado com foco em qualidade da entrada no horizonte.
    Score por lado:
      quality = MFE
                - mae_penalty * MAE_ate_pico
                - time_penalty * (MFE * frac_tempo_ate_pico)
                - giveback_penalty * giveback_ate_fim
                - cross_penalty * MFE_lado_oposto
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
        valid = 0
        for j in range(i + 1, end_idx + 1):
            pxj = close[j]
            if not np.isfinite(pxj) or pxj <= 0.0:
                long_pnls[valid] = 0.0
                short_pnls[valid] = 0.0
                valid += 1
                continue
            long_pnl = (pxj / px0) - 1.0
            short_pnl = (px0 / pxj) - 1.0
            long_pnls[valid] = long_pnl
            short_pnls[valid] = short_pnl
            valid += 1

        if valid <= 0:
            continue

        # contexto recente para favorecer reversão e punir "chasing"
        prev_min = 1e18
        prev_max = -1e18
        lb = max(0, i - max(1, int(rev_lookback_bars)))
        has_prev = False
        for j in range(lb, i):
            pxp = close[j]
            if not np.isfinite(pxp) or pxp <= 0.0:
                continue
            has_prev = True
            if pxp < prev_min:
                prev_min = pxp
            if pxp > prev_max:
                prev_max = pxp
        # move de alta já ocorrido até o candle de entrada (chasing long)
        prev_rise_into_entry = 0.0
        # move de baixa já ocorrido até o candle de entrada (chasing short)
        prev_drop_into_entry = 0.0
        if has_prev:
            if prev_min > 0.0:
                prev_rise_into_entry = (px0 / prev_min) - 1.0
                if prev_rise_into_entry < 0.0:
                    prev_rise_into_entry = 0.0
            if px0 > 0.0 and prev_max > 0.0:
                prev_drop_into_entry = (prev_max / px0) - 1.0
                if prev_drop_into_entry < 0.0:
                    prev_drop_into_entry = 0.0

        # confirmação curta de reversão logo após a entrada.
        # evita label alto "cedo demais" durante movimento contrário ainda vivo.
        conf_n = min(valid, max(1, int(confirm_bars)))
        max_up_early = 0.0
        max_dn_early = 0.0
        for t in range(conf_n):
            lp = long_pnls[t]
            sp = short_pnls[t]
            if lp > max_up_early:
                max_up_early = lp
            if sp > max_dn_early:
                max_dn_early = sp
        long_confirm = 0.0
        short_confirm = 0.0
        if max_up_early >= confirm_move and max_up_early >= (0.6 * max_dn_early):
            long_confirm = 1.0
        if max_dn_early >= confirm_move and max_dn_early >= (0.6 * max_up_early):
            short_confirm = 1.0

        # --- long quality ---
        best_long = -1e18
        best_long_idx = -1
        for t in range(valid):
            pnl_t = long_pnls[t]
            if pnl_t > best_long:
                best_long = pnl_t
                best_long_idx = t
        long_mfe = best_long if best_long > 0.0 else 0.0
        long_mae = 0.0
        long_giveback = 0.0
        if long_mfe > 0.0 and best_long_idx >= 0:
            for t in range(best_long_idx + 1):
                pnl_t = long_pnls[t]
                if pnl_t < 0.0:
                    dd = -pnl_t
                    if dd > long_mae:
                        long_mae = dd
            pnl_end = long_pnls[valid - 1]
            gb = long_mfe - pnl_end
            if gb > 0.0:
                long_giveback = gb
        long_time_frac = float(best_long_idx + 1) / float(valid if valid > 0 else 1)
        long_val = (
            long_mfe
            - side_mae_penalty * long_mae
            - side_time_penalty * (long_mfe * long_time_frac)
            - side_giveback_penalty * long_giveback
        )
        # reversão long: bônus após queda recente; penalidade por entrar atrasado em alta esticada
        long_val = long_val + side_reversal_bonus * prev_drop_into_entry - side_chase_penalty * prev_rise_into_entry

        # --- short quality ---
        best_short = -1e18
        best_short_idx = -1
        for t in range(valid):
            pnl_t = short_pnls[t]
            if pnl_t > best_short:
                best_short = pnl_t
                best_short_idx = t
        short_mfe = best_short if best_short > 0.0 else 0.0
        short_mae = 0.0
        short_giveback = 0.0
        if short_mfe > 0.0 and best_short_idx >= 0:
            for t in range(best_short_idx + 1):
                pnl_t = short_pnls[t]
                if pnl_t < 0.0:
                    dd = -pnl_t
                    if dd > short_mae:
                        short_mae = dd
            pnl_end = short_pnls[valid - 1]
            gb = short_mfe - pnl_end
            if gb > 0.0:
                short_giveback = gb
        short_time_frac = float(best_short_idx + 1) / float(valid if valid > 0 else 1)
        short_val = (
            short_mfe
            - side_mae_penalty * short_mae
            - side_time_penalty * (short_mfe * short_time_frac)
            - side_giveback_penalty * short_giveback
        )
        # reversão short: bônus após alta recente; penalidade por entrar atrasado em queda esticada
        short_val = short_val + side_reversal_bonus * prev_rise_into_entry - side_chase_penalty * prev_drop_into_entry

        if long_val < 0.0:
            long_val = 0.0
        if short_val < 0.0:
            short_val = 0.0

        # sem confirmação precoce, comprime fortemente o label (não zera total para manter sinal fraco)
        if long_confirm < 0.5:
            long_val = long_val * preconfirm_suppress
        if short_confirm < 0.5:
            short_val = short_val * preconfirm_suppress

        # Penalizacao cruzada: reduz score do lado quando o lado oposto
        # tambem teve oportunidade forte no mesmo horizonte.
        if side_cross_penalty > 0.0:
            long_val = long_val - side_cross_penalty * short_mfe
            short_val = short_val - side_cross_penalty * long_mfe
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
    side_giveback_penalty: float | None = None,
    side_cross_penalty: float | None = None,
    side_rev_lookback_min: int | None = None,
    side_chase_penalty: float | None = None,
    side_reversal_bonus: float | None = None,
    side_confirm_min: int | None = None,
    side_confirm_move: float | None = None,
    side_preconfirm_suppress: float | None = None,
    # edge + gates (pipeline supervisionado)
    compute_edge_labels: bool | None = None,
    edge_horizon_min: int | None = None,
    edge_dd_lambda: float | None = None,
    edge_clip: float | None = None,
    edge_scale: float | None = None,
    compute_entry_gates: bool | None = None,
    entry_gate_tmax_min: int | None = None,
    entry_gate_tp_pct: float | None = None,
    entry_gate_sl_pct: float | None = None,
    entry_gate_scale: float | None = None,
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
    side_giveback = float(side_giveback_penalty) if side_giveback_penalty is not None else float(TIMING_SIDE_GIVEBACK_PENALTY)
    side_cross = float(side_cross_penalty) if side_cross_penalty is not None else float(TIMING_SIDE_CROSS_PENALTY)
    side_rev_lb_min = int(side_rev_lookback_min) if side_rev_lookback_min is not None else int(TIMING_SIDE_REV_LOOKBACK_MIN)
    side_chase = float(side_chase_penalty) if side_chase_penalty is not None else float(TIMING_SIDE_CHASE_PENALTY)
    side_rev_bonus = float(side_reversal_bonus) if side_reversal_bonus is not None else float(TIMING_SIDE_REVERSAL_BONUS)
    side_confirm_min_v = int(side_confirm_min) if side_confirm_min is not None else int(TIMING_SIDE_CONFIRM_MIN)
    side_confirm_move_v = float(side_confirm_move) if side_confirm_move is not None else float(TIMING_SIDE_CONFIRM_MOVE)
    side_preconfirm_sup = float(side_preconfirm_suppress) if side_preconfirm_suppress is not None else float(TIMING_SIDE_PRECONF_SUPPRESS)
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
    side_rev_lb_bars = max(1, int(round(float(side_rev_lb_min) * bars_per_min)))
    side_confirm_bars = max(1, int(round(float(side_confirm_min_v) * bars_per_min)))

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
        side_rev_lb_bars,
        clip_f,
        float(side_mae),
        float(side_time),
        float(side_giveback),
        float(side_cross),
        float(side_chase),
        float(side_rev_bonus),
        int(side_confirm_bars),
        float(side_confirm_move_v),
        float(side_preconfirm_sup),
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

    # ===== Edge labels (regressor) + Entry gates (classificador) =====
    if compute_edge_labels is None:
        v = os.getenv("SNIPER_COMPUTE_EDGE_LABELS", "").strip().lower()
        compute_edge_labels = bool(v) and v not in {"0", "false", "no", "off"}
    if compute_entry_gates is None:
        v = os.getenv("SNIPER_COMPUTE_ENTRY_GATES", "").strip().lower()
        compute_entry_gates = not bool(v) or v not in {"0", "false", "no", "off"}

    # Pesos base por tarefa.
    if bool(compute_edge_labels):
        df["edge_weight_long"] = pd.Series(weights_long.astype(np.float32), index=df.index)
        df["edge_weight_short"] = pd.Series(weights_short.astype(np.float32), index=df.index)
    else:
        df.drop(columns=["edge_weight_long", "edge_weight_short"], inplace=True, errors="ignore")
    df["entry_gate_weight_long"] = pd.Series(weights_long.astype(np.float32), index=df.index)
    df["entry_gate_weight_short"] = pd.Series(weights_short.astype(np.float32), index=df.index)
    df["entry_gate_weight"] = pd.Series(weights.astype(np.float32), index=df.index)

    if bool(compute_edge_labels) or bool(compute_entry_gates):
        edge_h_min = int(edge_horizon_min) if edge_horizon_min is not None else int(os.getenv("SNIPER_EDGE_HORIZON_MIN", str(int(EDGE_HORIZON_MIN))) or EDGE_HORIZON_MIN)
        edge_clip_use = float(edge_clip) if edge_clip is not None else float(os.getenv("SNIPER_EDGE_CLIP", str(float(EDGE_LABEL_CLIP))) or EDGE_LABEL_CLIP)
        edge_scale_use = float(edge_scale) if edge_scale is not None else float(os.getenv("SNIPER_EDGE_SCALE", str(float(EDGE_LABEL_SCALE))) or EDGE_LABEL_SCALE)
        dd_lam = float(edge_dd_lambda) if edge_dd_lambda is not None else float(os.getenv("SNIPER_EDGE_LAMBDA_DD", str(float(EDGE_DD_LAMBDA))) or EDGE_DD_LAMBDA)
        gate_tmax = int(entry_gate_tmax_min) if entry_gate_tmax_min is not None else int(os.getenv("SNIPER_ENTRY_GATE_TMAX_MIN", str(int(ENTRY_GATE_TMAX_MIN))) or ENTRY_GATE_TMAX_MIN)
        gate_tp = float(entry_gate_tp_pct) if entry_gate_tp_pct is not None else float(os.getenv("SNIPER_ENTRY_GATE_TP_PCT", str(float(ENTRY_GATE_TP_PCT))) or ENTRY_GATE_TP_PCT)
        gate_sl = float(entry_gate_sl_pct) if entry_gate_sl_pct is not None else float(os.getenv("SNIPER_ENTRY_GATE_SL_PCT", str(float(ENTRY_GATE_SL_PCT))) or ENTRY_GATE_SL_PCT)
        gate_scale_use = float(entry_gate_scale) if entry_gate_scale is not None else float(os.getenv("SNIPER_ENTRY_GATE_SCALE", str(float(ENTRY_GATE_SCALE))) or ENTRY_GATE_SCALE)

        edge_h_bars = int(max(1, round(float(edge_h_min) * 60.0 / float(candle_sec))))
        gate_h_bars = int(max(1, round(float(gate_tmax) * 60.0 / float(candle_sec))))

        if bool(compute_edge_labels):
            edge_long, edge_short = _compute_edge_labels_numba(
                close,
                int(edge_h_bars),
                float(dd_lam),
                float(edge_clip_use),
                float(edge_scale_use),
            )
            df["edge_label_long"] = pd.Series(edge_long, index=df.index)
            df["edge_label_short"] = pd.Series(edge_short, index=df.index)
        else:
            df.drop(columns=["edge_label_long", "edge_label_short"], inplace=True, errors="ignore")

        if bool(compute_entry_gates):
            rev_only = bool(ENTRY_GATE_REVERSAL_ONLY)
            v_rev = os.getenv("SNIPER_ENTRY_GATE_REVERSAL_ONLY", "").strip().lower()
            if v_rev:
                rev_only = v_rev not in {"0", "false", "no", "off"}

            if rev_only and ("high" in df.columns) and ("low" in df.columns):
                pre_lb_min = int(os.getenv("SNIPER_ENTRY_GATE_PRELOOKBACK_MIN", str(int(ENTRY_GATE_PRELOOKBACK_MIN))) or ENTRY_GATE_PRELOOKBACK_MIN)
                pre_soft_atr = float(os.getenv("SNIPER_ENTRY_GATE_PREMOVE_ATR_SOFT", str(float(ENTRY_GATE_PREMOVE_ATR_SOFT))) or ENTRY_GATE_PREMOVE_ATR_SOFT)
                pre_hard_atr = float(os.getenv("SNIPER_ENTRY_GATE_PREMOVE_ATR_HARD", str(float(ENTRY_GATE_PREMOVE_ATR_HARD))) or ENTRY_GATE_PREMOVE_ATR_HARD)
                tp_atr = float(os.getenv("SNIPER_ENTRY_GATE_TP_ATR", str(float(ENTRY_GATE_TP_ATR))) or ENTRY_GATE_TP_ATR)
                sl_atr = float(os.getenv("SNIPER_ENTRY_GATE_SL_ATR", str(float(ENTRY_GATE_SL_ATR))) or ENTRY_GATE_SL_ATR)
                near_ext_atr = float(os.getenv("SNIPER_ENTRY_GATE_NEAR_EXTREMA_ATR", str(float(ENTRY_GATE_NEAR_EXTREMA_ATR))) or ENTRY_GATE_NEAR_EXTREMA_ATR)
                timeout_ret_atr = float(os.getenv("SNIPER_ENTRY_GATE_TIMEOUT_RET_ATR_MIN", str(float(ENTRY_GATE_TIMEOUT_RET_ATR_MIN))) or ENTRY_GATE_TIMEOUT_RET_ATR_MIN)
                atr_span = int(os.getenv("SNIPER_ENTRY_GATE_ATR_SPAN", str(int(ENTRY_GATE_ATR_SPAN))) or ENTRY_GATE_ATR_SPAN)
                gate_w_min = float(os.getenv("SNIPER_ENTRY_GATE_WEIGHT_MIN", str(float(ENTRY_GATE_WEIGHT_MIN))) or ENTRY_GATE_WEIGHT_MIN)
                gate_w_max = float(os.getenv("SNIPER_ENTRY_GATE_WEIGHT_MAX", str(float(ENTRY_GATE_WEIGHT_MAX))) or ENTRY_GATE_WEIGHT_MAX)

                high_arr = df["high"].to_numpy(dtype=np.float64, copy=False)
                low_arr = df["low"].to_numpy(dtype=np.float64, copy=False)
                atr_abs = _compute_atr_abs_from_ohlc(high_arr, low_arr, close, max(2, int(atr_span)))
                fut_high, _ = _future_window_max_min_numba(high_arr, int(gate_h_bars))
                _, fut_low = _future_window_max_min_numba(low_arr, int(gate_h_bars))
                pre_lb_bars = int(max(2, round(float(pre_lb_min) * 60.0 / float(candle_sec))))
                past_high = pd.Series(high_arr).rolling(pre_lb_bars, min_periods=pre_lb_bars).max().shift(1).to_numpy(dtype=np.float64, copy=False)
                past_low = pd.Series(low_arr).rolling(pre_lb_bars, min_periods=pre_lb_bars).min().shift(1).to_numpy(dtype=np.float64, copy=False)

                gate_long, gate_short, gate_w_long, gate_w_short = _compute_entry_gate_reversal_numba(
                    close.astype(np.float64, copy=False),
                    high_arr.astype(np.float64, copy=False),
                    low_arr.astype(np.float64, copy=False),
                    atr_abs.astype(np.float64, copy=False),
                    past_high.astype(np.float64, copy=False),
                    past_low.astype(np.float64, copy=False),
                    fut_high.astype(np.float64, copy=False),
                    fut_low.astype(np.float64, copy=False),
                    int(gate_h_bars),
                    float(tp_atr),
                    float(sl_atr),
                    float(pre_soft_atr),
                    float(pre_hard_atr),
                    float(near_ext_atr),
                    float(timeout_ret_atr),
                    float(gate_scale_use),
                    float(gate_w_min),
                    float(gate_w_max),
                )
                df["entry_gate_long"] = pd.Series(gate_long, index=df.index)
                df["entry_gate_short"] = pd.Series(gate_short, index=df.index)
                df["entry_gate_weight_long"] = pd.Series(gate_w_long.astype(np.float32), index=df.index)
                df["entry_gate_weight_short"] = pd.Series(gate_w_short.astype(np.float32), index=df.index)
                df["entry_gate_weight"] = pd.Series(
                    np.maximum(gate_w_long.astype(np.float32), gate_w_short.astype(np.float32)),
                    index=df.index,
                )
            else:
                gate_long, gate_short = _compute_entry_gate_labels_numba(
                    close,
                    int(gate_h_bars),
                    float(gate_tp),
                    float(gate_sl),
                    float(gate_scale_use),
                )
                df["entry_gate_long"] = pd.Series(gate_long, index=df.index)
                df["entry_gate_short"] = pd.Series(gate_short, index=df.index)
        else:
            df.drop(
                columns=[
                    "entry_gate_long",
                    "entry_gate_short",
                    "entry_gate_weight",
                    "entry_gate_weight_long",
                    "entry_gate_weight_short",
                ],
                inplace=True,
                errors="ignore",
            )
    else:
        df.drop(columns=["edge_label_long", "edge_label_short"], inplace=True, errors="ignore")
        df.drop(
            columns=[
                "entry_gate_long",
                "entry_gate_short",
                "entry_gate_weight",
                "entry_gate_weight_long",
                "entry_gate_weight_short",
            ],
            inplace=True,
            errors="ignore",
        )
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
    "EDGE_HORIZON_MIN",
    "EDGE_DD_LAMBDA",
    "EDGE_LABEL_CLIP",
    "EDGE_LABEL_SCALE",
    "ENTRY_GATE_TMAX_MIN",
    "ENTRY_GATE_TP_PCT",
    "ENTRY_GATE_SL_PCT",
    "ENTRY_GATE_SCALE",
    "ENTRY_GATE_REVERSAL_ONLY",
    "ENTRY_GATE_PRELOOKBACK_MIN",
    "ENTRY_GATE_PREMOVE_ATR_SOFT",
    "ENTRY_GATE_PREMOVE_ATR_HARD",
    "ENTRY_GATE_TP_ATR",
    "ENTRY_GATE_SL_ATR",
    "ENTRY_GATE_NEAR_EXTREMA_ATR",
    "ENTRY_GATE_TIMEOUT_RET_ATR_MIN",
    "ENTRY_GATE_ATR_SPAN",
    "ENTRY_GATE_WEIGHT_MIN",
    "ENTRY_GATE_WEIGHT_MAX",
]
