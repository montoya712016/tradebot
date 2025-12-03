# -*- coding: utf-8 -*-
from typing import Dict
import time, warnings
import numpy as np, pandas as pd
from numba import njit
from .pf_config import (
    EMA_WINDOWS, EMA_PAIRS,
    ATR_MIN, VOL_MIN, CI_MIN, LOGRET_MIN,
    KELTNER_WIDTH_MIN, KELTNER_CENTER_MIN, KELTNER_POS_MIN, KELTNER_Z_MIN,
    RSI_PRICE_MIN, RSI_EMA_PAIRS, SLOPE_MIN, CCI_MIN, ADX_MIN, ZLOG_MIN,
    MINMAX_MIN, SLOPE_RESERR_MIN, VOL_RATIO_PAIRS, REV_WINDOWS,
    # novos
    RUN_WINDOWS_MIN, HHHL_WINDOWS_MIN, EMA_CONFIRM_SPANS_MIN, BREAK_LOOKBACK_MIN,
    SLOPE_DIFF_PAIRS_MIN, WICK_MEAN_WINDOWS_MIN,
)

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


def _segment_valid(df: pd.DataFrame):
    valid = ~(df["open"].isna() | df["high"].isna() | df["low"].isna() | df["close"].isna())
    seg_id = (valid != valid.shift()).cumsum()
    return seg_id, valid


def _minutes_to_candles_from_index(idx: pd.DatetimeIndex, minutes: int) -> int:
    if len(idx) < 2:
        return 1
    secs = float((idx[1] - idx[0]).total_seconds() or 60.0)
    return max(1, int(round((minutes * 60.0) / max(1.0, secs))))


@njit
def _ema(x, span):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    if n == 0: return out
    alpha = 2.0 / (span + 1.0)
    v = 0.0
    cnt = 0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi) or not np.isfinite(xi):
            out[i] = np.nan
            continue
        if cnt == 0:
            v = xi; cnt = 1
            out[i] = np.nan if span > 1 else v
        else:
            v = alpha * xi + (1.0 - alpha) * v
            cnt += 1
            out[i] = v if cnt >= span else np.nan
    return out


@njit
def _rolling_mean(x, win, minp):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    S = np.empty(n+1, np.float64); S[0] = 0.0
    for i in range(n): S[i+1] = S[i] + x[i]
    for i in range(n):
        s = i - win + 1
        if s < 0: s = 0
        L = i - s + 1
        if L >= minp:
            out[i] = (S[i+1] - S[s]) / L
    return out


@njit
def _rolling_sum(x, win, minp):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    S = np.empty(n+1, np.float64); S[0] = 0.0
    for i in range(n): S[i+1] = S[i] + x[i]
    for i in range(n):
        s = i - win + 1
        if s < 0: s = 0
        L = i - s + 1
        if L >= minp:
            out[i] = (S[i+1] - S[s])
    return out


@njit
def _rolling_std(x, win, minp):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    S  = np.empty(n+1, np.float64); S[0] = 0.0
    S2 = np.empty(n+1, np.float64); S2[0] = 0.0
    for i in range(n):
        xi = x[i]
        S[i+1]  = S[i]  + xi
        S2[i+1] = S2[i] + xi*xi
    for i in range(n):
        s = i - win + 1
        if s < 0: s = 0
        L = i - s + 1
        if L >= minp:
            sumx  = S[i+1]  - S[s]
            sumx2 = S2[i+1] - S2[s]
            mu = sumx / L
            var = sumx2 / L - mu*mu
            if var < 0.0: var = 0.0
            out[i] = np.sqrt(var)
    return out


@njit
def _rolling_min(x, win):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    dq = np.empty(n, np.int64); head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] >= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = x[dq[head]]
    return out


@njit
def _rolling_max(x, win):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    dq = np.empty(n, np.int64); head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] <= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = x[dq[head]]
    return out


@njit
def _true_range(high, low, close):
    n = close.size
    out = np.empty(n, np.float64)
    prev = close[0]
    for i in range(n):
        hi = high[i]; lo = low[i]; cp = close[i]
        a = hi - lo
        b = abs(hi - prev)
        c = abs(lo - prev)
        tr = a if a > b else b
        if c > tr: tr = c
        out[i] = tr
        prev = cp
    return out


@njit
def _slope_pct(logc, win):
    n = logc.size
    out = np.empty(n, np.float64); out[:] = np.nan
    for i in range(win, n):
        out[i] = (logc[i] - logc[i-win]) / win * 100.0
    return out


@njit
def _rolling_sum_absdiff_fast(x, win, minp):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    if n == 0:
        return out
    ad = np.empty(n, np.float64)
    ad[0] = 0.0
    for i in range(1, n):
        d = x[i] - x[i-1]
        if d < 0.0: d = -d
        ad[i] = d
    S = np.empty(n+1, np.float64); S[0] = 0.0
    for i in range(n): S[i+1] = S[i] + ad[i]
    for i in range(n):
        start = i - win + 1
        if start < 1: start = 1
        L = i - start + 1
        if L >= minp and i >= 1:
            out[i] = S[i+1] - S[start]
    return out


@njit
def _rolling_argmin_dist(x, win):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    dq = np.empty(n, np.int64); head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] >= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = float(i - dq[head])
    return out


@njit
def _rolling_argmax_dist(x, win):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    dq = np.empty(n, np.int64); head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] <= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = float(i - dq[head])
    return out


@njit
def _rolling_std_fast(x, win, minp):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    S = np.empty(n+1, np.float64); S[0] = 0.0
    S2 = np.empty(n+1, np.float64); S2[0] = 0.0
    for i in range(n):
        xi = x[i]
        S[i+1] = S[i] + xi
        S2[i+1] = S2[i] + xi*xi
    for i in range(n):
        s = i - win + 1
        if s < 0: s = 0
        L = i - s + 1
        if L >= minp:
            sumx = S[i+1] - S[s]
            sumx2 = S2[i+1] - S2[s]
            mu = sumx / L
            var = sumx2 / L - mu*mu
            if var < 0.0: var = 0.0
            out[i] = np.sqrt(var)
    return out


@njit
def _slope_resid_std_lstsq(logc, win):
    n = logc.size
    out = np.empty(n, np.float64); out[:] = np.nan
    if win <= 1 or n < win: return out
    W = float(win)
    S_t  = W*(W-1.0)/2.0
    S_t2 = W*(W-1.0)*(2.0*W-1.0)/6.0
    Sz = 0.0; Sz2 = 0.0; Stz = 0.0
    for u in range(win):
        z = logc[u]
        Sz += z; Sz2 += z*z; Stz += u*z
    denom = (W * S_t2 - S_t * S_t)
    if denom == 0.0: denom = 1.0
    b = (W * Stz - S_t * Sz) / denom
    a = (Sz - b * S_t) / W
    SSE = Sz2 - 2.0*a*Sz - 2.0*b*Stz + a*a*W + 2.0*a*b*S_t + b*b*S_t2
    if SSE < 0.0: SSE = 0.0
    out[win-1] = np.sqrt(SSE / W)
    for i in range(win, n):
        z_out = logc[i - win]
        z_in  = logc[i]
        Sz  += z_in - z_out
        Sz2 += z_in*z_in - z_out*z_out
        Sz_old = Sz - z_in + z_out
        Stz = (Stz - (Sz_old - z_out)) + (win-1) * z_in
        b = (W * Stz - S_t * Sz) / denom
        a = (Sz - b * S_t) / W
        SSE = Sz2 - 2.0*a*Sz - 2.0*b*Stz + a*a*W + 2.0*a*b*S_t + b*b*S_t2
        if SSE < 0.0: SSE = 0.0
        out[i] = np.sqrt(SSE / W)
    return out


@njit
def _wilder_ema(x, win):
    n = x.size
    out = np.empty(n, np.float64); out[:] = np.nan
    if n == 0 or win <= 0: return out
    alpha = 1.0 / win
    started = False
    prev = 0.0
    for i in range(n):
        xi = x[i]
        if np.isnan(xi):
            continue
        if not started:
            prev = xi
            out[i] = prev
            started = True
        else:
            prev = (1.0 - alpha) * prev + alpha * xi
            out[i] = prev
    return out


# auxiliares novos (numba opcional pode ser adicionado depois)
@njit
def _consecutive_count_nb(arr_bool: np.ndarray) -> np.ndarray:
    n = arr_bool.size
    out = np.empty(n, np.float64)
    run = 0
    for i in range(n):
        if arr_bool[i]:
            run += 1
        else:
            run = 0
        out[i] = run
    return out


@njit
def _bars_since_flip_nb(sign: np.ndarray) -> np.ndarray:
    """
    Conta barras desde o último flip de sinal em 'sign' (positivo/negativo).
    NaN/0.0 contam como continuação (sem flip).
    """
    n = sign.size
    out = np.empty(n, np.float64)
    cnt = 0
    prev = 0  # 1, -1 ou 0
    for i in range(n):
        v = sign[i]
        if np.isnan(v) or v == 0.0:
            cnt += 1
            out[i] = cnt
            continue
        s = 1 if v > 0.0 else -1
        if i == 0:
            cnt = 0
            prev = s
        else:
            if s != prev:
                cnt = 0
                prev = s
            else:
                cnt += 1
        out[i] = cnt
    return out


@njit
def _since_last_true_nb(x_bool: np.ndarray) -> np.ndarray:
    n = x_bool.size
    out = np.empty(n, np.float64)
    cnt = 0
    for i in range(n):
        if x_bool[i]:
            cnt = 0
        else:
            cnt += 1
        out[i] = cnt
    return out


def make_features(df: pd.DataFrame, flags: Dict[str, bool], *, verbose: bool = True) -> None:
    initial_time = time.time()
    eps = 1e-9
    seg_id, valid = _segment_valid(df)

    def _win_for(idx: pd.DatetimeIndex, minutes: int) -> int:
        return _minutes_to_candles_from_index(idx, minutes)

    # Pré-cria colunas
    cols: list[str] = []
    def add(prefix: str, lst):
        cols.extend(f"{prefix}{m}" for m in lst)

    feat_counts: Dict[str, int] = {}

    if flags.get("shitidx"):
        for s, l in EMA_PAIRS:
            cols.append(f"shitidx_pct_{s}_{l}")
        feat_counts["shitidx"] = len(EMA_PAIRS)

    if flags.get("atr"):
        add("atr_pct_", ATR_MIN); feat_counts["atr"] = len(ATR_MIN)

    if flags.get("rsi"):
        add("rsi_price_", RSI_PRICE_MIN)
        for span, m in RSI_EMA_PAIRS:
            cols.append(f"rsi_ema{span}_{m}")
        feat_counts["rsi"] = len(RSI_PRICE_MIN) + len(RSI_EMA_PAIRS)

    if flags.get("slope"):
        add("slope_pct_", SLOPE_MIN); feat_counts["slope"] = len(SLOPE_MIN)

    if flags.get("vol"):
        add("vol_pct_", VOL_MIN); feat_counts["vol"] = len(VOL_MIN)

    if flags.get("ci"):
        add("ci_", CI_MIN); feat_counts["ci"] = len(CI_MIN)

    if flags.get("cum_logret"):
        add("cum_ret_pct_", LOGRET_MIN); feat_counts["cum_logret"] = len(LOGRET_MIN)

    if flags.get("keltner"):
        for m in KELTNER_WIDTH_MIN:  cols.append(f"keltner_halfwidth_pct_{m}")
        for m in KELTNER_CENTER_MIN: cols.append(f"keltner_center_pct_{m}")
        for m in KELTNER_POS_MIN:    cols.append(f"keltner_pos_{m}")
        for m in KELTNER_Z_MIN:      cols.append(f"keltner_width_z_{m}")
        feat_counts["keltner"] = (
            len(KELTNER_WIDTH_MIN) + len(KELTNER_CENTER_MIN) + len(KELTNER_POS_MIN) + len(KELTNER_Z_MIN)
        )

    if flags.get("cci"):
        add("cci_", CCI_MIN); feat_counts["cci"] = len(CCI_MIN)

    if flags.get("adx"):
        add("adx_", ADX_MIN); feat_counts["adx"] = len(ADX_MIN)

    if flags.get("time_since"):
        for m in MINMAX_MIN:
            cols += [f"pct_from_min_{m}", f"pct_from_max_{m}", f"time_since_min_{m}", f"time_since_max_{m}"]
        feat_counts["time_since"] = 4 * len(MINMAX_MIN)

    if flags.get("zlog"):
        for m in ZLOG_MIN: cols.append(f"zlog_{m}m")
        feat_counts["zlog"] = len(ZLOG_MIN)

    if flags.get("slope_reserr"):
        for m in SLOPE_RESERR_MIN: cols.append(f"slope_reserr_pct_{m}")
        feat_counts["slope_reserr"] = len(SLOPE_RESERR_MIN)

    if flags.get("vol_ratio"):
        for a, b in VOL_RATIO_PAIRS: cols.append(f"vol_ratio_pct_{a}_{b}")
        feat_counts["vol_ratio"] = len(VOL_RATIO_PAIRS)

    if flags.get("regime"):
        cols += ["log_volume_ema", "liquidity_ratio"]; feat_counts["regime"] = 2

    if flags.get("liquidity"):
        cols += ["volume_to_range_ema1440"]; feat_counts["liquidity"] = 1

    if flags.get("rev_speed"):
        for m in REV_WINDOWS:
            cols.append(f"rev_speed_up_{m}"); cols.append(f"rev_speed_down_{m}")
        feat_counts["rev_speed"] = 2 * len(REV_WINDOWS)

    if flags.get("vol_z"):
        cols += ["vol_z", "signed_vol_z"]; feat_counts["vol_z"] = 2

    if flags.get("shadow"):
        cols += ["shadow_balance", "shadow_balance_raw"]; feat_counts["shadow"] = 2

    if flags.get("range_ratio"):
        cols += ["range_ratio_60_1440"]; feat_counts["range_ratio"] = 1

    # novos blocos de contexto recente
    if flags.get("runs"):
        for m in RUN_WINDOWS_MIN:
            cols += [f"run_up_cnt_{m}", f"run_dn_cnt_{m}"]
        cols += ["run_up_len", "run_dn_len"]
        feat_counts["runs"] = 2 * len(RUN_WINDOWS_MIN) + 2
    if flags.get("hh_hl"):
        for m in HHHL_WINDOWS_MIN:
            cols += [f"hh_cnt_{m}", f"hl_cnt_{m}"]
        feat_counts["hh_hl"] = 2 * len(HHHL_WINDOWS_MIN)
    if flags.get("ema_cross"):
        for s in EMA_CONFIRM_SPANS_MIN:
            cols += [f"bars_above_ema_{s}", f"bars_below_ema_{s}", f"bars_since_cross_{s}"]
        feat_counts["ema_cross"] = 3 * len(EMA_CONFIRM_SPANS_MIN)
    if flags.get("breakout"):
        for n in BREAK_LOOKBACK_MIN:
            cols += [f"break_high_{n}", f"break_low_{n}", f"bars_since_bhigh_{n}", f"bars_since_blow_{n}"]
        feat_counts["breakout"] = 4 * len(BREAK_LOOKBACK_MIN)
    if flags.get("mom_short"):
        for a, b in SLOPE_DIFF_PAIRS_MIN:
            cols.append(f"slope_diff_{a}_{b}")
        feat_counts["mom_short"] = len(SLOPE_DIFF_PAIRS_MIN)
    if flags.get("wick_stats"):
        for m in WICK_MEAN_WINDOWS_MIN:
            cols.append(f"wick_lower_mean_{m}")
        cols.append("wick_lower_streak")
        feat_counts["wick_stats"] = len(WICK_MEAN_WINDOWS_MIN) + 1

    missing = [c for c in cols if c not in df.columns]
    if missing:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for c in missing: df[c] = np.nan

    timings = {k: 0.0 for k in [
        "shitidx","atr","keltner","rsi","slope","vol","ci","cum_logret",
        "cci","adx","time_since","zlog","slope_reserr","vol_ratio","regime","liquidity","rev_speed","vol_z","shadow","range_ratio"
    ]}

    for idxs in df.groupby(seg_id).groups.values():
        if not valid.at[idxs[0]]:
            continue
        sub = df.loc[idxs]
        close  = sub["close"].to_numpy(np.float64)
        high   = sub["high"].to_numpy(np.float64)
        low    = sub["low"].to_numpy(np.float64)
        open_  = sub["open"].to_numpy(np.float64)
        logc = np.log(close + 1e-9)

        def w(m):
            return _win_for(sub.index, m)
        def MP(win):
            return win

        if flags.get("shitidx"):
            _t0 = time.perf_counter()
            ema_map = {}
            for m in EMA_WINDOWS:
                ema_map[m] = _ema(close, w(m))
            for s, l in EMA_PAIRS:
                num = ema_map[s] - ema_map[l]
                df.loc[idxs, f"shitidx_pct_{s}_{l}"] = num / (close + eps) * 100.0
            timings["shitidx"] += (time.perf_counter() - _t0)

        if flags.get("atr"):
            _t0 = time.perf_counter()
            tr = _true_range(high, low, close)
            for m in ATR_MIN:
                atr = _rolling_mean(tr, w(m), MP(w(m)))
                df.loc[idxs, f"atr_pct_{m}"] = atr / (close + eps) * 100.0
            timings["atr"] += (time.perf_counter() - _t0)

        if flags.get("keltner"):
            _t0 = time.perf_counter()
            tr = _true_range(high, low, close)
            for m in KELTNER_WIDTH_MIN:
                ema_c = _ema(close, w(m))
                atr_m = _rolling_mean(tr, w(m), MP(w(m)))
                halfw = 2.0 * atr_m
                df.loc[idxs, f"keltner_halfwidth_pct_{m}"] = halfw / (close + eps) * 100.0
            for m in KELTNER_CENTER_MIN:
                ema_c = _ema(close, w(m))
                df.loc[idxs, f"keltner_center_pct_{m}"] = (ema_c - close) / (close + eps) * 100.0
            for m in KELTNER_POS_MIN:
                ema_c = _ema(close, w(m))
                atr_m = _rolling_mean(tr, w(m), MP(w(m)))
                upper = ema_c + 2.0 * atr_m
                lower = ema_c - 2.0 * atr_m
                pos = (close - lower) / ((upper - lower) + eps)
                df.loc[idxs, f"keltner_pos_{m}"] = pos
            for m in KELTNER_Z_MIN:
                ema_c = _ema(close, w(m))
                atr_m = _rolling_mean(tr, w(m), MP(w(m)))
                width_pct = ((ema_c + 2.0*atr_m) - (ema_c - 2.0*atr_m)) / (close + eps) * 100.0
                win_z = w(int(m * 2))
                minp_z = max(win_z // 2, w(60))
                mu = _rolling_mean(width_pct, win_z, minp_z)
                sd = _rolling_std(width_pct, win_z, minp_z)
                z = (width_pct - mu) / (sd + 1e-9)
                df.loc[idxs, f"keltner_width_z_{m}"] = z
            timings["keltner"] += (time.perf_counter() - _t0)

        if flags.get("rsi"):
            _t0 = time.perf_counter()
            diff = np.empty_like(close); diff[:] = np.nan
            diff[1:] = close[1:] - close[:-1]; diff[0] = 0.0
            gain = np.where(diff > 0.0, diff, 0.0)
            loss = np.where(diff < 0.0, -diff, 0.0)
            for m in RSI_PRICE_MIN:
                g = _rolling_mean(gain, w(m), MP(w(m)))
                l = _rolling_mean(loss, w(m), MP(w(m)))
                rs = g / (l + eps)
                df.loc[idxs, f"rsi_price_{m}"] = 100.0 - 100.0 / (1.0 + rs)
            for span, m in RSI_EMA_PAIRS:
                smooth = _ema(close, w(span))
                d2 = np.empty_like(smooth); d2[:] = np.nan
                d2[1:] = smooth[1:] - smooth[:-1]; d2[0] = 0.0
                g2 = np.where(d2 > 0.0, d2, 0.0)
                l2 = np.where(d2 < 0.0, -d2, 0.0)
                rg = _rolling_mean(g2, w(m), MP(w(m)))
                rl = _rolling_mean(l2, w(m), MP(w(m)))
                rs2 = rg / (rl + eps)
                df.loc[idxs, f"rsi_ema{span}_{m}"] = 100.0 - 100.0 / (1.0 + rs2)
            timings["rsi"] += (time.perf_counter() - _t0)

        if flags.get("slope"):
            _t0 = time.perf_counter()
            for m in SLOPE_MIN:
                df.loc[idxs, f"slope_pct_{m}"] = _slope_pct(logc, w(m))
            timings["slope"] += (time.perf_counter() - _t0)

        if flags.get("vol"):
            _t0 = time.perf_counter()
            ret1 = np.empty_like(close); ret1[:] = np.nan
            ret1[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps); ret1[0] = 0.0
            for m in VOL_MIN:
                df.loc[idxs, f"vol_pct_{m}"] = _rolling_std(ret1, w(m), MP(w(m))) * 100.0
            timings["vol"] += (time.perf_counter() - _t0)

        if flags.get("ci"):
            _t0 = time.perf_counter()
            for m in CI_MIN:
                mov = _rolling_sum_absdiff_fast(close, w(m), MP(w(m)))
                hi  = _rolling_max(close, w(m))
                lo  = _rolling_min(close, w(m))
                rng = (hi - lo)
                ci  = 100.0 * np.log10(mov / (rng + eps) + eps) / np.log10(w(m))
                df.loc[idxs, f"ci_{m}"] = ci
            timings["ci"] += (time.perf_counter() - _t0)

        if flags.get("cum_logret"):
            _t0 = time.perf_counter()
            lr = np.empty_like(close); lr[:] = np.nan
            lr[1:] = np.log(close[1:] + eps) - np.log(close[:-1] + eps); lr[0] = 0.0
            for m in LOGRET_MIN:
                cs = _rolling_sum(lr, w(m), MP(w(m)))
                df.loc[idxs, f"cum_ret_pct_{m}"] = (np.exp(cs) - 1.0) * 100.0
            timings["cum_logret"] += (time.perf_counter() - _t0)

        if flags.get("cci"):
            _t0 = time.perf_counter()
            tp = (high + low + close) / 3.0
            for m in CCI_MIN:
                sma = _rolling_mean(tp, w(m), MP(w(m)))
                sd = _rolling_std_fast(tp, w(m), MP(w(m)))
                mad_approx = sd * 0.7978845608028654
                df.loc[idxs, f"cci_{m}"] = (tp - sma) / (0.015 * (mad_approx + eps))
            timings["cci"] += (time.perf_counter() - _t0)

        if flags.get("adx"):
            _t0 = time.perf_counter()
            up = np.empty_like(high); dn = np.empty_like(low)
            up[0] = 0.0; dn[0] = 0.0
            for i in range(1, high.size):
                up[i] = high[i] - high[i-1]
                dn[i] = low[i-1]  - low[i]
                if up[i] < 0.0: up[i] = 0.0
                if dn[i] < 0.0: dn[i] = 0.0
                if up[i] < dn[i]: up[i] = 0.0
                else: dn[i] = 0.0
            tr = _true_range(high, low, close)
            for m in ADX_MIN:
                sum_up = _rolling_sum(up, w(m), MP(w(m)))
                sum_dn = _rolling_sum(dn, w(m), MP(w(m)))
                sum_tr = _rolling_sum(tr, w(m), MP(w(m)))
                pdi = (sum_up / (sum_tr + eps)) * 100.0
                mdi = (sum_dn / (sum_tr + eps)) * 100.0
                dx  = (np.abs(pdi - mdi) / (pdi + mdi + eps)) * 100.0
                adx = _wilder_ema(dx, w(m))
                df.loc[idxs, f"adx_{m}"] = adx
            timings["adx"] += (time.perf_counter() - _t0)

        if flags.get("time_since"):
            _t0 = time.perf_counter()
            for m in MINMAX_MIN:
                rmin = _rolling_min(close, w(m))
                rmax = _rolling_max(close, w(m))
                df.loc[idxs, f"pct_from_min_{m}"] = (close - rmin) / (rmin + eps) * 100.0
                df.loc[idxs, f"pct_from_max_{m}"] = (rmax - close) / (rmax + eps) * 100.0
                df.loc[idxs, f"time_since_min_{m}"] = _rolling_argmin_dist(close, w(m))
                df.loc[idxs, f"time_since_max_{m}"] = _rolling_argmax_dist(close, w(m))
            timings["time_since"] += (time.perf_counter() - _t0)

        if flags.get("zlog"):
            _t0 = time.perf_counter()
            for m in ZLOG_MIN:
                mu = _rolling_mean(logc, w(m), MP(w(m)))
                sd = _rolling_std(logc, w(m), MP(w(m)))
                df.loc[idxs, f"zlog_{m}m"] = (logc - mu) / (sd + 1e-9)
            timings["zlog"] += (time.perf_counter() - _t0)

        if flags.get("slope_reserr"):
            _t0 = time.perf_counter()
            for m in SLOPE_RESERR_MIN:
                sr = _slope_resid_std_lstsq(logc, w(m))
                sr_pct = (np.exp(sr) - 1.0) * 100.0
                df.loc[idxs, f"slope_reserr_pct_{m}"] = sr_pct
            timings["slope_reserr"] += (time.perf_counter() - _t0)

        if flags.get("vol_ratio"):
            _t0 = time.perf_counter()
            ret1 = np.empty_like(close); ret1[:] = np.nan
            ret1[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps); ret1[0] = 0.0
            _cache = {}
            def _get_vol(minutes_win: int):
                if minutes_win not in _cache:
                    _cache[minutes_win] = _rolling_std(ret1, w(minutes_win), MP(w(minutes_win))) * 100.0
                return _cache[minutes_win]
            for a, b in VOL_RATIO_PAIRS:
                va = _get_vol(a); vb = _get_vol(b)
                df.loc[idxs, f"vol_ratio_pct_{a}_{b}"] = (va / (vb + 1e-9) - 1.0) * 100.0
            timings["vol_ratio"] += (time.perf_counter() - _t0)

        if flags.get("regime"):
            _t0 = time.perf_counter()
            if "volume" in sub.columns:
                vol_arr = sub["volume"].to_numpy(np.float64)
            else:
                vol_arr = np.full_like(close, np.nan)
            notional_vol = vol_arr * close
            ema_vol = _ema(notional_vol, w(1440))
            df.loc[idxs, "log_volume_ema"] = np.log1p(ema_vol)
            tr = _true_range(high, low, close)
            tr_pct = tr / (close + 1e-9)
            ema_nv_60  = _ema(notional_vol, w(60))
            ema_trp_60 = _ema(tr_pct, w(60))
            vol_liq = ema_nv_60 / (ema_trp_60 + 1e-9)
            df.loc[idxs, "liquidity_ratio"] = np.log1p(np.maximum(0.0, vol_liq))
            timings["regime"] += (time.perf_counter() - _t0)

        if flags.get("liquidity"):
            _t0 = time.perf_counter()
            if "volume" in sub.columns:
                vol_arr = sub["volume"].to_numpy(np.float64)
            else:
                vol_arr = np.full_like(close, np.nan)
            notional_vol = vol_arr * close
            range_pct = (high - low) / (close + 1e-9)
            ema_nv   = _ema(notional_vol, max(2, w(10)))
            ema_rpct = _ema(range_pct,   max(2, w(10)))
            vtr_usdt = np.log1p(ema_nv / (ema_rpct + 1e-9))
            df.loc[idxs, "volume_to_range_ema1440"] = _ema(vtr_usdt, w(1440))
            timings["liquidity"] += (time.perf_counter() - _t0)

        if flags.get("rev_speed"):
            _t0 = time.perf_counter()
            for m in REV_WINDOWS:
                rmin = _rolling_min(close, w(m))
                pct_up = (close / (rmin + 1e-9) - 1.0) * 100.0
                tmin = _rolling_argmin_dist(close, w(m))
                rs_up = pct_up / (tmin + 1.0)
                df.loc[idxs, f"rev_speed_up_{m}"] = rs_up
                rmax = _rolling_max(close, w(m))
                pct_dn = (rmax / (close + 1e-9) - 1.0) * 100.0
                tmax = _rolling_argmax_dist(close, w(m))
                rs_dn = pct_dn / (tmax + 1.0)
                df.loc[idxs, f"rev_speed_down_{m}"] = rs_dn
            timings["rev_speed"] += (time.perf_counter() - _t0)

        if flags.get("vol_z"):
            _t0 = time.perf_counter()
            if "volume" in sub.columns:
                vol = sub["volume"].to_numpy(np.float64)
                lv = np.log1p(vol)
                ema_s = _ema(lv, w(60))
                ema_l = _ema(lv, w(720))
                num = (lv - ema_s)
                denom = _ema(np.abs(lv - ema_l), w(720)) + 1e-9
                z = num / denom
                df.loc[idxs, "vol_z"] = z
                ret1_sign = np.empty_like(close); ret1_sign[:] = 0.0
                ret1_sign[1:] = np.sign((close[1:] / (close[:-1] + 1e-9)) - 1.0)
                df.loc[idxs, "signed_vol_z"] = z * ret1_sign
            timings["vol_z"] += (time.perf_counter() - _t0)

        if flags.get("shadow"):
            _t0 = time.perf_counter()
            up_wick   = (high - np.maximum(open_, close))
            low_wick  = (np.minimum(open_, close) - low)
            range_all = (high - low) + 1e-9
            sb_raw = (up_wick - low_wick) / range_all
            sb_clip = np.clip(sb_raw, -1.0, 1.0)
            sb_smooth = _ema(sb_clip, max(2, w(10)))
            df.loc[idxs, "shadow_balance_raw"] = sb_raw
            df.loc[idxs, "shadow_balance"] = sb_smooth
            timings["shadow"] += (time.perf_counter() - _t0)

        if flags.get("range_ratio"):
            _t0 = time.perf_counter()
            r_short = _rolling_max(high, w(60)) - _rolling_min(low, w(60))
            r_long  = _rolling_max(high, w(1440)) - _rolling_min(low, w(1440))
            rr = r_short / (r_long + 1e-9)
            df.loc[idxs, "range_ratio_60_1440"] = rr
            timings["range_ratio"] += (time.perf_counter() - _t0)

        # ── novos blocos ────────────────────────────────────────────────────────
        if flags.get("runs"):
            _t0 = time.perf_counter()
            cp = np.empty_like(close); cp[:] = np.nan
            cp[1:] = close[1:] - close[:-1]; cp[0] = 0.0
            upb = (cp > 0.0)
            dnb = (cp < 0.0)
            up = upb.astype(np.float64); dn = dnb.astype(np.float64)
            # contagem em janelas
            for m in RUN_WINDOWS_MIN:
                df.loc[idxs, f"run_up_cnt_{m}"] = _rolling_sum(up, w(m), 1)
                df.loc[idxs, f"run_dn_cnt_{m}"] = _rolling_sum(dn, w(m), 1)
            # runs consecutivos
            df.loc[idxs, "run_up_len"] = _consecutive_count_nb(upb)
            df.loc[idxs, "run_dn_len"] = _consecutive_count_nb(dnb)
            timings["runs"] = timings.get("runs", 0.0) + (time.perf_counter() - _t0)

        if flags.get("hh_hl"):
            _t0 = time.perf_counter()
            is_hh = (high > np.concatenate(([np.nan], high[:-1]))).astype(np.float64)
            is_hl = (low  > np.concatenate(([np.nan], low[:-1]))).astype(np.float64)
            for m in HHHL_WINDOWS_MIN:
                df.loc[idxs, f"hh_cnt_{m}"] = _rolling_sum(is_hh, w(m), 1)
                df.loc[idxs, f"hl_cnt_{m}"] = _rolling_sum(is_hl, w(m), 1)
            timings["hh_hl"] = timings.get("hh_hl", 0.0) + (time.perf_counter() - _t0)

        if flags.get("ema_cross"):
            _t0 = time.perf_counter()
            for s in EMA_CONFIRM_SPANS_MIN:
                ema_s = _ema(close, w(s))
                diff  = close - ema_s
                above_b = (diff > 0.0)
                below_b = (diff < 0.0)
                df.loc[idxs, f"bars_above_ema_{s}"] = _consecutive_count_nb(above_b)
                df.loc[idxs, f"bars_below_ema_{s}"] = _consecutive_count_nb(below_b)
                df.loc[idxs, f"bars_since_cross_{s}"] = _bars_since_flip_nb(diff)
            timings["ema_cross"] = timings.get("ema_cross", 0.0) + (time.perf_counter() - _t0)

        if flags.get("breakout"):
            _t0 = time.perf_counter()
            for n in BREAK_LOOKBACK_MIN:
                prev_max = _rolling_max(high, w(n))
                prev_min = _rolling_min(low,  w(n))
                # usar janela "passada": desloca 1 para trás para não contar a própria vela
                prev_max = np.concatenate(([np.nan], prev_max[:-1]))
                prev_min = np.concatenate(([np.nan], prev_min[:-1]))
                brk_hi_b = (close > (prev_max + 1e-9))
                brk_lo_b = (close < (prev_min - 1e-9))
                df.loc[idxs, f"break_high_{n}"] = brk_hi_b.astype(np.float64)
                df.loc[idxs, f"break_low_{n}"]  = brk_lo_b.astype(np.float64)
                # barras desde último rompimento (numba)
                df.loc[idxs, f"bars_since_bhigh_{n}"] = _since_last_true_nb(brk_hi_b)
                df.loc[idxs, f"bars_since_blow_{n}"]  = _since_last_true_nb(brk_lo_b)
            timings["breakout"] = timings.get("breakout", 0.0) + (time.perf_counter() - _t0)

        if flags.get("mom_short"):
            _t0 = time.perf_counter()
            for a, b in SLOPE_DIFF_PAIRS_MIN:
                sl_a = _slope_pct(logc, w(a))
                sl_b = _slope_pct(logc, w(b))
                df.loc[idxs, f"slope_diff_{a}_{b}"] = sl_a - sl_b
            timings["mom_short"] = timings.get("mom_short", 0.0) + (time.perf_counter() - _t0)

        if flags.get("wick_stats"):
            _t0 = time.perf_counter()
            low_wick  = (np.minimum(open_, close) - low)
            rng = (high - low) + 1e-9
            ratio = np.clip(low_wick / rng, 0.0, 1.0)
            for m in WICK_MEAN_WINDOWS_MIN:
                df.loc[idxs, f"wick_lower_mean_{m}"] = _rolling_mean(ratio, w(m), 1)
            # streak de pavios baixos "relevantes" (ratio > 0.5) consecutivos
            streak = _consecutive_count_nb((ratio > 0.5))
            df.loc[idxs, "wick_lower_streak"] = streak
            timings["wick_stats"] = timings.get("wick_stats", 0.0) + (time.perf_counter() - _t0)

    n_valid = int(valid.sum())
    dt = time.time() - initial_time
    order = [
        "shitidx","keltner","atr","rsi","slope","vol","ci","cum_logret",
        "cci","adx","time_since","zlog","slope_reserr","vol_ratio","regime","liquidity","rev_speed","vol_z","shadow","range_ratio",
        # novos
        "runs","hh_hl","ema_cross","breakout","mom_short","wick_stats"
    ]
    parts = []
    for k in order:
        if k in feat_counts and feat_counts[k] > 0 and timings.get(k, 0.0) > 0.0:
            parts.append(f"{k}:{timings[k]:.2f}s({feat_counts[k]})")
    summary_line = (f"[features] rows={n_valid:,} | cols={len(cols)} | {dt:.2f}s").replace(",", ".")
    extras = (" | ".join(parts)) if parts else ""
    print(summary_line + ("\n   • " + extras if extras else ""), flush=True)
