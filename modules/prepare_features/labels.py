# -*- coding: utf-8 -*-
from typing import Tuple
import os
from pathlib import Path

import numpy as np, pandas as pd

if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = Path(__file__).resolve().parents[2].parent / "cache_sniper" / "numba"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache_dir)

from numba import njit

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT


@njit(cache=True)
def _simulate_entry_contract_numba(
    close: np.ndarray,
    low: np.ndarray,
    horizon_bars: int,
    min_profit_pct: float,
    exit_ema_span: int,
    exit_ema_init_offset_pct: float,
    entry_weight_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    labels = np.zeros(n, np.uint8)
    mae = np.zeros(n, np.float32)
    exit_code = np.zeros(n, np.int8)
    exit_wait = np.zeros(n, np.int32)
    weights = np.zeros(n, np.float32)
    if horizon_bars < 0:
        horizon_bars = 0
    if exit_ema_span < 0:
        exit_ema_span = 0

    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            mae[i] = 0.0
            exit_code[i] = 0
            exit_wait[i] = 0
            continue

        worst = px0
        best_mae = 0.0
        label = 0
        code = 0
        exit_bar = i
        dipped_below_entry = False
        if horizon_bars <= 0:
            last_bar = n - 1
        else:
            last_bar = i + horizon_bars
            if last_bar >= n:
                last_bar = n - 1
        use_ema_exit = exit_ema_span > 0
        ema = px0 * (1.0 - exit_ema_init_offset_pct) if use_ema_exit else 0.0
        ema_alpha = (2.0 / (exit_ema_span + 1.0)) if use_ema_exit else 0.0
        code = -3
        exit_bar = last_bar
        for j in range(i + 1, last_bar + 1):
            lo = close[j]
            if not np.isfinite(lo):
                lo = close[j]
            if lo < worst:
                worst = lo
            cur_mae = (worst / px0) - 1.0
            if cur_mae < best_mae:
                best_mae = cur_mae
            if lo < px0:
                dipped_below_entry = True
            if use_ema_exit:
                px = close[j]
                if np.isfinite(px) and px > 0.0:
                    ema = ema + (ema_alpha * (px - ema))
                    if px < ema:
                        code = 2
                        exit_bar = j
                        break

        exit_px = close[exit_bar]
        if not np.isfinite(exit_px) or exit_px <= 0.0:
            exit_px = px0
        r = (exit_px / px0) - 1.0
        scale = float(entry_weight_alpha) if float(entry_weight_alpha) > 1e-9 else 0.01
        margin = abs(r - min_profit_pct)
        if not np.isfinite(margin):
            margin = 0.0
        w = 0.1 + 0.9 * min(1.0, float(margin) / float(scale))
        weights[i] = float(w)
        if (not dipped_below_entry) and (r >= min_profit_pct):
            label = 1
        else:
            label = 0

        labels[i] = label
        mae[i] = best_mae
        exit_code[i] = code
        exit_wait[i] = max(0, exit_bar - i)

    return labels, mae, exit_code, exit_wait, weights


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


@njit(cache=True)
def _price_pattern_labels_numba(
    close: np.ndarray,
    trend_past_bars: int,
    trend_future_bars: int,
    mr_past_bars: int,
    mr_future_bars: int,
    trend_past_ret_thr: float,
    trend_future_ret_thr: float,
    mr_past_ret_thr: float,
    mr_future_ret_thr: float,
    trend_past_eff_thr: float,
    trend_future_eff_thr: float,
    mr_past_eff_thr: float,
    mr_future_eff_thr: float,
    trend_early_bars: int,
    mr_early_bars: int,
    trend_early_adverse_max: float,
    mr_early_adverse_max: float,
    trend_opp_stop_pct: float,
    mr_opp_stop_pct: float,
    trend_end_ret_min: float,
    mr_end_ret_min: float,
    trend_hit_max_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    trend_long = np.zeros(n, np.uint8)
    trend_short = np.zeros(n, np.uint8)
    mr_long = np.zeros(n, np.uint8)
    mr_short = np.zeros(n, np.uint8)
    # rows: [trend_long, trend_short, mr_long, mr_short]
    # cols:
    # [window, past_ret, past_eff, future_eff, target_hit, hit_order, early_ok, end_ret, final,
    #  profit_core, profit_cut_past_ret, profit_cut_past_eff, profit_cut_future_eff,
    #  profit_cut_early_ok, profit_cut_end_ret, profit_cut_any, profit_cut_multi]
    diag = np.zeros((4, 17), np.int64)

    for i in range(n):
        px = close[i]
        if (not np.isfinite(px)) or px <= 0.0:
            continue

        # Trend windows
        if (trend_past_bars > 0) and (trend_future_bars > 0):
            i0 = i - trend_past_bars
            i1 = i + trend_future_bars
            if i0 >= 0 and i1 < n:
                p0 = close[i0]
                p1 = close[i1]
                if np.isfinite(p0) and p0 > 0.0 and np.isfinite(p1) and p1 > 0.0:
                    diag[0, 0] += 1
                    diag[1, 0] += 1
                    ret_p = (px / p0) - 1.0
                    ret_f = (p1 / px) - 1.0
                    path_p = 0.0
                    path_f = 0.0
                    for j in range(i0 + 1, i + 1):
                        a = close[j - 1]
                        b = close[j]
                        if np.isfinite(a) and np.isfinite(b):
                            path_p += abs(b - a)
                    for j in range(i + 1, i1 + 1):
                        a = close[j - 1]
                        b = close[j]
                        if np.isfinite(a) and np.isfinite(b):
                            path_f += abs(b - a)
                    eff_p = abs(px - p0) / path_p if path_p > 1e-12 else 0.0
                    eff_f = abs(p1 - px) / path_f if path_f > 1e-12 else 0.0
                    trend_long_past_ret_ok = ret_p >= trend_past_ret_thr
                    trend_short_past_ret_ok = ret_p <= -trend_past_ret_thr
                    trend_past_eff_ok = eff_p >= trend_past_eff_thr
                    trend_future_eff_ok = eff_f >= trend_future_eff_thr
                    if trend_long_past_ret_ok:
                        diag[0, 1] += 1
                    if trend_short_past_ret_ok:
                        diag[1, 1] += 1
                    if trend_past_eff_ok:
                        diag[0, 2] += 1
                        diag[1, 2] += 1
                    if trend_future_eff_ok:
                        diag[0, 3] += 1
                        diag[1, 3] += 1
                    # future path quality: target-hit ordering + early adverse movement
                    up_hit = -1
                    dn_hit = -1
                    worst_ret = 0.0
                    best_ret = 0.0
                    early_worst = 0.0
                    early_best = 0.0
                    early_end = i + trend_early_bars
                    if early_end > i1:
                        early_end = i1
                    for j in range(i + 1, i1 + 1):
                        pj = close[j]
                        if (not np.isfinite(pj)) or pj <= 0.0:
                            continue
                        rj = (pj / px) - 1.0
                        if rj < worst_ret:
                            worst_ret = rj
                        if rj > best_ret:
                            best_ret = rj
                        if j <= early_end:
                            if rj < early_worst:
                                early_worst = rj
                            if rj > early_best:
                                early_best = rj
                        if up_hit < 0 and rj >= trend_future_ret_thr:
                            up_hit = j
                        if dn_hit < 0 and rj <= -trend_opp_stop_pct:
                            dn_hit = j
                    trend_long_target_ok = up_hit >= 0
                    trend_long_order_ok = trend_long_target_ok and (dn_hit < 0 or up_hit <= dn_hit)
                    trend_long_hit_fast_ok = trend_long_target_ok and ((up_hit - i) <= trend_hit_max_bars)
                    trend_long_early_ok = (-early_worst) <= trend_early_adverse_max
                    trend_long_end_ok = ret_f >= trend_end_ret_min
                    if trend_long_target_ok:
                        diag[0, 4] += 1
                    if trend_long_order_ok:
                        diag[0, 5] += 1
                    if trend_long_early_ok:
                        diag[0, 6] += 1
                    if trend_long_end_ok:
                        diag[0, 7] += 1
                    trend_long_profit_core = trend_long_target_ok and trend_long_order_ok
                    if trend_long_profit_core:
                        diag[0, 9] += 1
                        fail_cnt = 0
                        if not trend_long_past_ret_ok:
                            diag[0, 10] += 1
                            fail_cnt += 1
                        if not trend_past_eff_ok:
                            diag[0, 11] += 1
                            fail_cnt += 1
                        if not trend_future_eff_ok:
                            diag[0, 12] += 1
                            fail_cnt += 1
                        if not trend_long_early_ok:
                            diag[0, 13] += 1
                            fail_cnt += 1
                        if not trend_long_end_ok:
                            diag[0, 14] += 1
                            fail_cnt += 1
                        if fail_cnt > 0:
                            diag[0, 15] += 1
                        if fail_cnt > 1:
                            diag[0, 16] += 1
                    if (
                        trend_long_past_ret_ok
                        and trend_past_eff_ok
                        and trend_future_eff_ok
                        and trend_long_target_ok
                        and trend_long_order_ok
                        and trend_long_hit_fast_ok
                        and trend_long_early_ok
                        and trend_long_end_ok
                    ):
                        trend_long[i] = 1
                        diag[0, 8] += 1
                    up_hit_s = -1
                    dn_hit_s = -1
                    early_adverse_short = 0.0
                    for j in range(i + 1, i1 + 1):
                        pj = close[j]
                        if (not np.isfinite(pj)) or pj <= 0.0:
                            continue
                        rj = (pj / px) - 1.0
                        if up_hit_s < 0 and rj <= -trend_future_ret_thr:
                            up_hit_s = j
                        if dn_hit_s < 0 and rj >= trend_opp_stop_pct:
                            dn_hit_s = j
                        if j <= early_end and rj > early_adverse_short:
                            early_adverse_short = rj
                    trend_short_target_ok = up_hit_s >= 0
                    trend_short_order_ok = trend_short_target_ok and (dn_hit_s < 0 or up_hit_s <= dn_hit_s)
                    trend_short_hit_fast_ok = trend_short_target_ok and ((up_hit_s - i) <= trend_hit_max_bars)
                    trend_short_early_ok = early_adverse_short <= trend_early_adverse_max
                    trend_short_end_ok = ret_f <= -trend_end_ret_min
                    if trend_short_target_ok:
                        diag[1, 4] += 1
                    if trend_short_order_ok:
                        diag[1, 5] += 1
                    if trend_short_early_ok:
                        diag[1, 6] += 1
                    if trend_short_end_ok:
                        diag[1, 7] += 1
                    trend_short_profit_core = trend_short_target_ok and trend_short_order_ok
                    if trend_short_profit_core:
                        diag[1, 9] += 1
                        fail_cnt = 0
                        if not trend_short_past_ret_ok:
                            diag[1, 10] += 1
                            fail_cnt += 1
                        if not trend_past_eff_ok:
                            diag[1, 11] += 1
                            fail_cnt += 1
                        if not trend_future_eff_ok:
                            diag[1, 12] += 1
                            fail_cnt += 1
                        if not trend_short_early_ok:
                            diag[1, 13] += 1
                            fail_cnt += 1
                        if not trend_short_end_ok:
                            diag[1, 14] += 1
                            fail_cnt += 1
                        if fail_cnt > 0:
                            diag[1, 15] += 1
                        if fail_cnt > 1:
                            diag[1, 16] += 1
                    if (
                        trend_short_past_ret_ok
                        and trend_past_eff_ok
                        and trend_future_eff_ok
                        and trend_short_target_ok
                        and trend_short_order_ok
                        and trend_short_hit_fast_ok
                        and trend_short_early_ok
                        and trend_short_end_ok
                    ):
                        trend_short[i] = 1
                        diag[1, 8] += 1

        # Mean reversion windows
        if (mr_past_bars > 0) and (mr_future_bars > 0):
            i0 = i - mr_past_bars
            i1 = i + mr_future_bars
            if i0 >= 0 and i1 < n:
                p0 = close[i0]
                p1 = close[i1]
                if np.isfinite(p0) and p0 > 0.0 and np.isfinite(p1) and p1 > 0.0:
                    diag[2, 0] += 1
                    diag[3, 0] += 1
                    ret_p = (px / p0) - 1.0
                    ret_f = (p1 / px) - 1.0
                    path_p = 0.0
                    path_f = 0.0
                    for j in range(i0 + 1, i + 1):
                        a = close[j - 1]
                        b = close[j]
                        if np.isfinite(a) and np.isfinite(b):
                            path_p += abs(b - a)
                    for j in range(i + 1, i1 + 1):
                        a = close[j - 1]
                        b = close[j]
                        if np.isfinite(a) and np.isfinite(b):
                            path_f += abs(b - a)
                    eff_p = abs(px - p0) / path_p if path_p > 1e-12 else 0.0
                    eff_f = abs(p1 - px) / path_f if path_f > 1e-12 else 0.0
                    mr_long_past_ret_ok = ret_p <= -mr_past_ret_thr
                    mr_short_past_ret_ok = ret_p >= mr_past_ret_thr
                    mr_past_eff_ok = eff_p >= mr_past_eff_thr
                    mr_future_eff_ok = eff_f >= mr_future_eff_thr
                    if mr_long_past_ret_ok:
                        diag[2, 1] += 1
                    if mr_short_past_ret_ok:
                        diag[3, 1] += 1
                    if mr_past_eff_ok:
                        diag[2, 2] += 1
                        diag[3, 2] += 1
                    if mr_future_eff_ok:
                        diag[2, 3] += 1
                        diag[3, 3] += 1
                    up_hit = -1
                    dn_hit = -1
                    early_worst = 0.0
                    early_best = 0.0
                    early_end = i + mr_early_bars
                    if early_end > i1:
                        early_end = i1
                    for j in range(i + 1, i1 + 1):
                        pj = close[j]
                        if (not np.isfinite(pj)) or pj <= 0.0:
                            continue
                        rj = (pj / px) - 1.0
                        if up_hit < 0 and rj >= mr_future_ret_thr:
                            up_hit = j
                        if dn_hit < 0 and rj <= -mr_opp_stop_pct:
                            dn_hit = j
                        if j <= early_end:
                            if rj < early_worst:
                                early_worst = rj
                            if rj > early_best:
                                early_best = rj
                    mr_long_target_ok = up_hit >= 0
                    mr_long_order_ok = mr_long_target_ok and (dn_hit < 0 or up_hit <= dn_hit)
                    mr_long_early_ok = (-early_worst) <= mr_early_adverse_max
                    mr_long_end_ok = ret_f >= mr_end_ret_min
                    if mr_long_target_ok:
                        diag[2, 4] += 1
                    if mr_long_order_ok:
                        diag[2, 5] += 1
                    if mr_long_early_ok:
                        diag[2, 6] += 1
                    if mr_long_end_ok:
                        diag[2, 7] += 1
                    mr_long_profit_core = mr_long_target_ok and mr_long_order_ok
                    if mr_long_profit_core:
                        diag[2, 9] += 1
                        fail_cnt = 0
                        if not mr_long_past_ret_ok:
                            diag[2, 10] += 1
                            fail_cnt += 1
                        if not mr_past_eff_ok:
                            diag[2, 11] += 1
                            fail_cnt += 1
                        if not mr_future_eff_ok:
                            diag[2, 12] += 1
                            fail_cnt += 1
                        if not mr_long_early_ok:
                            diag[2, 13] += 1
                            fail_cnt += 1
                        if not mr_long_end_ok:
                            diag[2, 14] += 1
                            fail_cnt += 1
                        if fail_cnt > 0:
                            diag[2, 15] += 1
                        if fail_cnt > 1:
                            diag[2, 16] += 1
                    if (
                        mr_long_past_ret_ok
                        and mr_past_eff_ok
                        and mr_future_eff_ok
                        and mr_long_target_ok
                        and mr_long_order_ok
                        and mr_long_early_ok
                        and mr_long_end_ok
                    ):
                        mr_long[i] = 1
                        diag[2, 8] += 1
                    up_hit_s = -1
                    dn_hit_s = -1
                    early_adverse_short = 0.0
                    for j in range(i + 1, i1 + 1):
                        pj = close[j]
                        if (not np.isfinite(pj)) or pj <= 0.0:
                            continue
                        rj = (pj / px) - 1.0
                        if up_hit_s < 0 and rj <= -mr_future_ret_thr:
                            up_hit_s = j
                        if dn_hit_s < 0 and rj >= mr_opp_stop_pct:
                            dn_hit_s = j
                        if j <= early_end and rj > early_adverse_short:
                            early_adverse_short = rj
                    mr_short_target_ok = up_hit_s >= 0
                    mr_short_order_ok = mr_short_target_ok and (dn_hit_s < 0 or up_hit_s <= dn_hit_s)
                    mr_short_early_ok = early_adverse_short <= mr_early_adverse_max
                    mr_short_end_ok = ret_f <= -mr_end_ret_min
                    if mr_short_target_ok:
                        diag[3, 4] += 1
                    if mr_short_order_ok:
                        diag[3, 5] += 1
                    if mr_short_early_ok:
                        diag[3, 6] += 1
                    if mr_short_end_ok:
                        diag[3, 7] += 1
                    mr_short_profit_core = mr_short_target_ok and mr_short_order_ok
                    if mr_short_profit_core:
                        diag[3, 9] += 1
                        fail_cnt = 0
                        if not mr_short_past_ret_ok:
                            diag[3, 10] += 1
                            fail_cnt += 1
                        if not mr_past_eff_ok:
                            diag[3, 11] += 1
                            fail_cnt += 1
                        if not mr_future_eff_ok:
                            diag[3, 12] += 1
                            fail_cnt += 1
                        if not mr_short_early_ok:
                            diag[3, 13] += 1
                            fail_cnt += 1
                        if not mr_short_end_ok:
                            diag[3, 14] += 1
                            fail_cnt += 1
                        if fail_cnt > 0:
                            diag[3, 15] += 1
                        if fail_cnt > 1:
                            diag[3, 16] += 1
                    if (
                        mr_short_past_ret_ok
                        and mr_past_eff_ok
                        and mr_future_eff_ok
                        and mr_short_target_ok
                        and mr_short_order_ok
                        and mr_short_early_ok
                        and mr_short_end_ok
                    ):
                        mr_short[i] = 1
                        diag[3, 8] += 1

    return trend_long, trend_short, mr_long, mr_short, diag


@njit(cache=True)
def _single_profit_label_both_numba(
    close: np.ndarray,
    future_end_bars: int,
    future_avg_start_bars: int,
    future_avg_split_bars: int,
    profit_thr: float,
    future_eff_thr: float,
    early_bars: int,
    early_adverse_max: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    n = close.size
    labels_long = np.zeros(n, np.uint8)
    labels_short = np.zeros(n, np.uint8)
    fut_eff = np.zeros(n, np.float32)
    mean_early = np.zeros(n, np.float32)
    mean_full = np.zeros(n, np.float32)
    profit_core_long = np.zeros(n, np.uint8)
    profit_core_short = np.zeros(n, np.uint8)
    # [window, mean_early_ok, mean_full_ok, mean_pair_ok, future_eff, early_ok, final,
    #  profit_core, cut_future_eff, cut_mean_pair, cut_early_ok, cut_any]
    diag_long = np.zeros(12, np.int64)
    diag_short = np.zeros(12, np.int64)

    # Sanitize window bounds once (not per-row).
    if future_end_bars < 1:
        future_end_bars = 1
    if future_avg_start_bars < 1:
        future_avg_start_bars = 1
    if future_avg_split_bars < future_avg_start_bars:
        future_avg_split_bars = future_avg_start_bars
    if future_avg_split_bars > future_end_bars:
        future_avg_split_bars = future_end_bars

    for i in range(n):
        px = close[i]
        if (not np.isfinite(px)) or px <= 0.0:
            continue
        future_end_idx = i + future_end_bars
        if future_end_idx >= n:
            continue
        future_end_px = close[future_end_idx]
        if (not np.isfinite(future_end_px)) or future_end_px <= 0.0:
            continue

        diag_long[0] += 1
        diag_short[0] += 1

        path_f = 0.0
        for j in range(i + 1, future_end_idx + 1):
            a = close[j - 1]
            b = close[j]
            if np.isfinite(a) and np.isfinite(b):
                path_f += abs(b - a)
        future_eff_val = abs(future_end_px - px) / path_f if path_f > 1e-12 else 0.0
        fut_eff[i] = future_eff_val
        future_eff_ok = future_eff_val >= future_eff_thr
        if future_eff_ok:
            diag_long[4] += 1
            diag_short[4] += 1

        sum_early = 0.0
        cnt_early = 0
        sum_full = 0.0
        cnt_full = 0
        early_worst = 0.0
        early_best = 0.0
        early_end = i + early_bars
        if early_end > future_end_idx:
            early_end = future_end_idx
        for j in range(i + 1, future_end_idx + 1):
            pj = close[j]
            if (not np.isfinite(pj)) or pj <= 0.0:
                continue
            rj = (pj / px) - 1.0
            rel = j - i
            if rel >= future_avg_start_bars and rel <= future_avg_split_bars:
                sum_early += rj
                cnt_early += 1
            if rel >= future_avg_start_bars and rel <= future_end_bars:
                sum_full += rj
                cnt_full += 1
            if j <= early_end:
                if rj < early_worst:
                    early_worst = rj
                if rj > early_best:
                    early_best = rj

        avg_ret_early = (sum_early / cnt_early) if cnt_early > 0 else 0.0
        avg_ret_full = (sum_full / cnt_full) if cnt_full > 0 else 0.0
        mean_early[i] = avg_ret_early
        mean_full[i] = avg_ret_full

        # LONG side
        long_mean_early_ok = avg_ret_early >= profit_thr
        long_mean_full_ok = avg_ret_full >= profit_thr
        long_mean_pair_ok = long_mean_early_ok and long_mean_full_ok
        long_early_ok = (-early_worst) <= early_adverse_max
        if long_mean_early_ok:
            diag_long[1] += 1
        if long_mean_full_ok:
            diag_long[2] += 1
        if long_mean_pair_ok:
            diag_long[3] += 1
        if long_early_ok:
            diag_long[5] += 1

        # profit_core agora = medias futuras suficientes (sem filtros de qualidade)
        pc_long = long_mean_pair_ok
        if pc_long:
            profit_core_long[i] = 1
            diag_long[7] += 1
            fail = 0
            if not future_eff_ok:
                diag_long[8] += 1
                fail += 1
            if not long_mean_pair_ok:
                diag_long[9] += 1
                fail += 1
            if not long_early_ok:
                diag_long[10] += 1
                fail += 1
            if fail > 0:
                diag_long[11] += 1
        if pc_long and future_eff_ok and long_early_ok:
            labels_long[i] = 1
            diag_long[6] += 1

        # SHORT side (exact opposite)
        short_mean_early_ok = avg_ret_early <= -profit_thr
        short_mean_full_ok = avg_ret_full <= -profit_thr
        short_mean_pair_ok = short_mean_early_ok and short_mean_full_ok
        short_early_ok = early_best <= early_adverse_max
        if short_mean_early_ok:
            diag_short[1] += 1
        if short_mean_full_ok:
            diag_short[2] += 1
        if short_mean_pair_ok:
            diag_short[3] += 1
        if short_early_ok:
            diag_short[5] += 1

        pc_short = short_mean_pair_ok
        if pc_short:
            profit_core_short[i] = 1
            diag_short[7] += 1
            fail = 0
            if not future_eff_ok:
                diag_short[8] += 1
                fail += 1
            if not short_mean_pair_ok:
                diag_short[9] += 1
                fail += 1
            if not short_early_ok:
                diag_short[10] += 1
                fail += 1
            if fail > 0:
                diag_short[11] += 1
        if pc_short and future_eff_ok and short_early_ok:
            labels_short[i] = 1
            diag_short[6] += 1

    return (
        labels_long,
        labels_short,
        fut_eff,
        mean_early,
        mean_full,
        profit_core_long,
        profit_core_short,
        diag_long,
        diag_short,
    )


def apply_trade_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
) -> pd.DataFrame:
    """Gera labels de contrato (legacy) + labels/weights de qualidade (novo)."""
    if contract is None:
        contract = DEFAULT_TRADE_CONTRACT

    candle_seconds = candle_sec or contract.timeframe_sec
    candle_seconds = max(1, int(candle_seconds))
    close = df["close"].to_numpy(np.float64, copy=False)
    low = df["low"].to_numpy(np.float64, copy=False) if "low" in df.columns else close

    # Legacy contract-based entry labels (funcionavam melhor no pipeline antigo).
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    profits = list(getattr(contract, "entry_label_min_profit_pcts", []) or [])
    if len(windows) < 1:
        windows = [360]
    if len(profits) < 1:
        profits = [0.02]
    if len(windows) != len(profits):
        raise ValueError("entry_label_windows_minutes e entry_label_min_profit_pcts devem ter o mesmo tamanho (>=1)")

    gap_next = None
    forbid_gap = bool(getattr(contract, "forbid_exit_on_gap", False))
    gap_hours = float(getattr(contract, "gap_hours_forbidden", 0.0) or 0.0)
    if forbid_gap and gap_hours > 0 and isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index)
        gth = pd.Timedelta(hours=gap_hours)
        n = len(idx)
        gap_next = np.full(n, -1, dtype=np.int32)
        next_gap = -1
        for i in range(n - 1, 0, -1):
            if idx[i] - idx[i - 1] >= gth:
                next_gap = i
            gap_next[i - 1] = next_gap

    first_suffix = ""
    for w_min, pmin in zip(windows, profits):
        hb = int(max(1, round((float(w_min) * 60.0) / float(candle_seconds))))
        entry_label, mae_pct, exit_code, exit_wait, weight = _simulate_entry_contract_numba(
            close,
            low,
            0,
            float(pmin),
            int(hb),
            float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0),
            float(getattr(contract, "entry_label_weight_alpha", 1.0) or 1.0),
        )
        if gap_next is not None:
            exit_bar = np.arange(exit_wait.size, dtype=np.int64) + exit_wait.astype(np.int64)
            hit_gap = (gap_next >= 0) & (gap_next <= exit_bar)
            if hit_gap.any():
                entry_label = entry_label.copy()
                exit_code = exit_code.copy()
                entry_label[hit_gap] = 0
                exit_code[hit_gap] = -4
        suffix = f"{int(w_min)}m"
        if not first_suffix:
            first_suffix = suffix
        df[f"sniper_entry_label_{suffix}"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
        df[f"sniper_mae_pct_{suffix}"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
        df[f"sniper_exit_code_{suffix}"] = pd.Series(exit_code.astype(np.int8), index=df.index)
        df[f"sniper_exit_wait_bars_{suffix}"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
        df[f"sniper_entry_weight_{suffix}"] = pd.Series(weight.astype(np.float32), index=df.index)

    # Future evaluation windows (minutes):
    # - avg_early: [future_avg_start_min .. future_avg_split_min]
    # - avg_full:  [future_avg_start_min .. future_window_end_min]
    future_window_end_min = _env_int("PF_LABEL_FUTURE_MIN", 120)
    future_window_end_bars = int(max(1, round((float(future_window_end_min) * 60.0) / float(candle_seconds))))
    future_avg_start_min = _env_int("PF_LABEL_FUTURE_MEAN_START_MIN", 0)
    future_avg_split_min = _env_int("PF_LABEL_FUTURE_MEAN_MID_MIN", 60)
    future_avg_start_bars = int(max(1, round((float(future_avg_start_min) * 60.0) / float(candle_seconds))))
    future_avg_split_bars = int(max(future_avg_start_bars, round((float(future_avg_split_min) * 60.0) / float(candle_seconds))))
    if future_avg_split_bars > future_window_end_bars:
        future_avg_split_bars = future_window_end_bars

    future_mean_profit_thr = float(_env_float("PF_LABEL_PROFIT_THR", 0.01))
    future_eff_min = float(_env_float("PF_LABEL_FUTURE_EFF_THR", 0.10))
    early_guard_min = _env_int("PF_LABEL_EARLY_MIN", 30)
    early_guard_bars = int(max(1, round((float(early_guard_min) * 60.0) / float(candle_seconds))))
    early_adverse_max = float(_env_float("PF_LABEL_EARLY_ADVERSE_MAX", 0.008))

    (
        labels_long,
        labels_short,
        fut_eff,
        mean_early,
        mean_full,
        profit_core_long,
        profit_core_short,
        diag_long,
        diag_short,
    ) = _single_profit_label_both_numba(
        close,
        int(future_window_end_bars),
        int(future_avg_start_bars),
        int(future_avg_split_bars),
        float(future_mean_profit_thr),
        float(future_eff_min),
        int(early_guard_bars),
        float(early_adverse_max),
    )

    def _score_col(col: str, lo: float, hi: float) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(len(df), dtype=np.float32)
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64, copy=False)
        z = (s - lo) / max(1e-9, (hi - lo))
        z = np.clip(z, 0.0, 1.0)
        z[~np.isfinite(z)] = 0.0
        return z.astype(np.float32, copy=False)

    def _avg_scores(cols: list[str], lo: float, hi: float) -> np.ndarray:
        arrs = [_score_col(c, lo, hi) for c in cols if c in df.columns]
        if not arrs:
            return np.zeros(len(df), dtype=np.float32)
        mat = np.vstack(arrs).astype(np.float32, copy=False)
        valid = np.isfinite(mat)
        den = valid.sum(axis=0).astype(np.float32)
        num = np.where(valid, mat, 0.0).sum(axis=0, dtype=np.float32)
        out = np.divide(num, np.where(den > 0, den, 1.0), dtype=np.float32)
        out[den <= 0] = 0.0
        return out.astype(np.float32, copy=False)

    w_adx = _avg_scores(["adx_15", "adx_30", "adx_120"], 12.0, 35.0)

    fut_eff_score = np.clip((fut_eff.astype(np.float64) - future_eff_min) / max(1e-9, (0.50 - future_eff_min)), 0.0, 1.0)
    fut_eff_score[~np.isfinite(fut_eff_score)] = 0.0
    # Score principal de retorno futuro: usa apenas excedente acima do threshold (nao valor bruto).
    m_early_long_score = np.clip((mean_early.astype(np.float64) - future_mean_profit_thr) / max(1e-9, future_mean_profit_thr), 0.0, 1.0)
    m_full_long_score = np.clip((mean_full.astype(np.float64) - future_mean_profit_thr) / max(1e-9, future_mean_profit_thr), 0.0, 1.0)
    timing_long = np.clip(0.60 * m_early_long_score + 0.40 * m_full_long_score, 0.0, 1.0)
    m_early_short_score = np.clip(((-mean_early.astype(np.float64)) - future_mean_profit_thr) / max(1e-9, future_mean_profit_thr), 0.0, 1.0)
    m_full_short_score = np.clip(((-mean_full.astype(np.float64)) - future_mean_profit_thr) / max(1e-9, future_mean_profit_thr), 0.0, 1.0)
    timing_short = np.clip(0.60 * m_early_short_score + 0.40 * m_full_short_score, 0.0, 1.0)

    # Weight design (long-only):
    # - pontos TRUE: premiar alta eficiencia + alto retorno futuro + ADX alto
    # - pontos FALSE: premiar baixa eficiencia + baixo retorno futuro + ADX baixo
    # Escala alvo: 1x .. 7x
    adx_support = np.clip(w_adx.astype(np.float64), 0.0, 1.0)
    adx_low = np.clip(1.0 - adx_support, 0.0, 1.0)
    bad_future_eff = np.clip(1.0 - fut_eff_score, 0.0, 1.0)

    # retorno "bom" e "ruim" (baixo) ao redor do threshold
    ret_good = np.clip(0.60 * m_early_long_score + 0.40 * m_full_long_score, 0.0, 1.0)
    ret_bad = np.clip(
        0.60 * np.clip((future_mean_profit_thr - mean_early.astype(np.float64)) / max(1e-9, future_mean_profit_thr), 0.0, 1.0)
        + 0.40 * np.clip((future_mean_profit_thr - mean_full.astype(np.float64)) / max(1e-9, future_mean_profit_thr), 0.0, 1.0),
        0.0,
        1.0,
    )

    # score de dificuldade/importancia para cada classe
    score_pos = np.clip(0.40 * fut_eff_score + 0.35 * ret_good + 0.25 * adx_support, 0.0, 1.0)
    score_neg = np.clip(0.40 * bad_future_eff + 0.35 * ret_bad + 0.25 * adx_low, 0.0, 1.0)

    # escala 1x..7x com forte seletividade (topos raros)
    def _sharp01(x: np.ndarray, start: float = 0.55, power: float = 3.2) -> np.ndarray:
        z = np.clip((x - float(start)) / max(1e-9, (1.0 - float(start))), 0.0, 1.0)
        return np.power(z, float(power))

    w_pos = 1.0 + 6.0 * _sharp01(score_pos, start=0.58, power=3.0)
    w_neg = 1.0 + 6.0 * _sharp01(score_neg, start=0.58, power=3.0)
    lbl_long_f = labels_long.astype(np.float64)
    w_long = np.where(lbl_long_f >= 0.5, w_pos, w_neg)
    w_long = np.clip(w_long, 1.0, 7.0).astype(np.float32, copy=False)

    for c in (
        "sniper_price_label",
        "sniper_price_weight",
        "sniper_price_weight_eff",
        "sniper_price_weight_adx",
        "sniper_price_weight_atr",
        "sniper_price_weight_future_eff",
        "sniper_price_weight_timing",
        "sniper_price_label_short",
        "sniper_price_weight_short",
        "sniper_price_weight_timing_short",
        "sniper_price_profit_core_short",
        "sniper_price_profit_core",
        "sniper_price_trend_long",
        "sniper_price_trend_short",
        "sniper_price_mr_long",
        "sniper_price_mr_short",
    ):
        if c in df.columns:
            try:
                del df[c]
            except Exception:
                pass

    df["sniper_price_label_long"] = pd.Series(labels_long.astype(np.uint8), index=df.index)
    df["sniper_price_weight_long"] = pd.Series(w_long, index=df.index)
    # Keep only future-eff quality component in weight diagnostics (past eff belongs to features).
    if "sniper_price_weight_eff" in df.columns:
        try:
            del df["sniper_price_weight_eff"]
        except Exception:
            pass
    df["sniper_price_weight_adx"] = pd.Series(w_adx.astype(np.float32), index=df.index)
    df["sniper_price_weight_future_eff"] = pd.Series(fut_eff_score.astype(np.float32), index=df.index)
    df["sniper_price_weight_timing_long"] = pd.Series(timing_long.astype(np.float32), index=df.index)
    df["sniper_price_profit_core_long"] = pd.Series(profit_core_long.astype(np.uint8), index=df.index)

    # Peso final de treino (1x..7x), direto (sem coluna intermediaria "hybrid"):
    # combina evidência de qualidade (score_pos/score_neg) com o peso legado do contrato.
    for w_min in windows:
        suffix = f"{int(w_min)}m"
        base_col = f"sniper_entry_weight_{suffix}"
        if base_col not in df.columns:
            continue
        base = pd.to_numeric(df[base_col], errors="coerce").to_numpy(np.float32, copy=False)
        base_norm = np.clip((base.astype(np.float64) - 0.10) / 0.90, 0.0, 1.0)
        quality = np.where(lbl_long_f >= 0.5, score_pos, score_neg)
        mix = np.clip(0.80 * quality + 0.20 * base_norm, 0.0, 1.0)
        mix_sharp = _sharp01(mix, start=0.60, power=3.2)
        w_final = (1.0 + 6.0 * mix_sharp).astype(np.float32, copy=False)
        w_final = np.clip(w_final, 1.0, 7.0).astype(np.float32, copy=False)
        df[base_col] = pd.Series(w_final, index=df.index)

    # Compatibilidade legado: primeira janela em colunas canônicas.
    if not first_suffix:
        first_suffix = f"{int(windows[0])}m"
    for c in (
        "sniper_entry_label",
        "sniper_mae_pct",
        "sniper_exit_code",
        "sniper_exit_wait_bars",
        "sniper_entry_weight",
        "sniper_entry_weight_hybrid",
    ):
        if c in df.columns:
            try:
                del df[c]
            except Exception:
                pass
    for c in list(df.columns):
        if c.startswith("sniper_entry_weight_hybrid_"):
            try:
                del df[c]
            except Exception:
                pass
    df["sniper_entry_label"] = df[f"sniper_entry_label_{first_suffix}"].astype(np.uint8)
    df["sniper_mae_pct"] = df[f"sniper_mae_pct_{first_suffix}"].astype(np.float32)
    df["sniper_exit_code"] = df[f"sniper_exit_code_{first_suffix}"].astype(np.int8)
    df["sniper_exit_wait_bars"] = df[f"sniper_exit_wait_bars_{first_suffix}"].astype(np.int32)
    df["sniper_entry_weight"] = df[f"sniper_entry_weight_{first_suffix}"].astype(np.float32)

    def _diag_dict(diag: np.ndarray) -> dict:
        return {
            "window": int(diag[0]),
            "mean_early_ok": int(diag[1]),
            "mean_full_ok": int(diag[2]),
            "mean_pair_ok": int(diag[3]),
            "future_eff": int(diag[4]),
            "early_ok": int(diag[5]),
            "final": int(diag[6]),
            "profit_core": int(diag[7]),
            "profit_cut_future_eff": int(diag[8]),
            "profit_cut_mean_pair": int(diag[9]),
            "profit_cut_early_ok": int(diag[10]),
            "profit_cut_any": int(diag[11]),
        }

    try:
        df.attrs["price_label_diag"] = {
            "single_long": _diag_dict(diag_long),
            "params": {
                "future_window_end_min": int(future_window_end_min),
                "future_avg_start_min": int(future_avg_start_min),
                "future_avg_split_min": int(future_avg_split_min),
                "future_mean_profit_thr": float(future_mean_profit_thr),
                "future_eff_min": float(future_eff_min),
                "early_guard_min": int(early_guard_min),
                "early_adverse_max": float(early_adverse_max),
            },
        }
    except Exception:
        pass
    return df
