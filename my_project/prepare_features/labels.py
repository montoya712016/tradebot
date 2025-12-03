# -*- coding: utf-8 -*-
from typing import Tuple
import time
import numpy as np, pandas as pd
from numba import njit

# Padrões (em %), compatíveis com o monolito
DEFAULT_COST = 0.20
DEFAULT_LAMB = 2.00

def _hours_to_bars(idx: pd.DatetimeIndex, hours: float) -> int:
    if len(idx) < 2:
        return 1
    try:
        dt = float((idx[1] - idx[0]).total_seconds())
        if dt <= 0.0:
            dt = 60.0
    except Exception:
        dt = 60.0
    n = int(round(float(hours) * 3600.0 / max(1.0, dt)))
    return max(1, n)


@njit
def _label_peakstop_avg_numba(cp, csum, n_max, clip, p_tgt, max_dd, drop_conf):
    N = cp.size
    y  = np.zeros(N, np.float64)
    dd = np.zeros(N, np.float64)
    ru = np.zeros(N, np.float64)
    for i in range(N):
        px0 = cp[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            y[i] = 0.0; dd[i] = 0.0; ru[i] = 0.0
            continue
        j_end = min(N - 1, i + n_max)
        min_px = px0
        px_peak = px0
        best_ret = -1e300
        dd_at_best = 0.0
        ret_at_peak = 0.0
        dd_at_peak = 0.0
        decided = False
        ru_max = 0.0
        for j in range(i, j_end + 1):
            px = cp[j]
            if not np.isfinite(px) or px <= 0.0:
                break
            if px < min_px:
                min_px = px
            dd_curr = (min_px / px0 - 1.0) * 100.0
            ru_curr = (px / px0 - 1.0) * 100.0
            if ru_curr > ru_max:
                ru_max = ru_curr
            if dd_curr <= -max_dd:
                if best_ret == -1e300:
                    y[i] = 0.0
                else:
                    yr = best_ret
                    if yr < -clip: yr = -clip
                    elif yr > clip: yr = clip
                    y[i] = yr
                dd[i] = -max_dd
                ru[i] = ru_max
                decided = True
                break
            L = (j - i + 1)
            avg = (csum[j+1] - csum[i]) / L
            r_avg = (avg / px0 - 1.0) * 100.0
            if r_avg > best_ret:
                best_ret = r_avg
                dd_at_best = dd_curr
            if px > px_peak:
                px_peak = px
                ret_at_peak = r_avg
                dd_at_peak  = dd_curr
            if (ret_at_peak >= p_tgt) and (px_peak > 0.0):
                queda_pct = (px / px_peak - 1.0) * 100.0
                if queda_pct <= -drop_conf:
                    yr = ret_at_peak
                    if yr < -clip: yr = -clip
                    elif yr > clip: yr = clip
                    y[i] = yr
                    dd[i] = dd_at_peak
                    ru[i] = ru_max
                    decided = True
                    break
        if not decided:
            if best_ret == -1e300:
                y[i] = 0.0; dd[i] = 0.0; ru[i] = ru_max
            else:
                yr = best_ret
                if yr < -clip: yr = -clip
                elif yr > clip: yr = clip
                y[i] = yr
                dd[i] = dd_at_best
                ru[i] = ru_max
    return y, dd, ru


@njit
def _label_troughstop_avg_numba(cp, csum, n_max, clip, p_tgt, max_ru, rise_conf):
    N = cp.size
    y  = np.zeros(N, np.float64)
    ru = np.zeros(N, np.float64)
    for i in range(N):
        px0 = cp[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            y[i] = 0.0; ru[i] = 0.0
            continue
        j_end = min(N - 1, i + n_max)
        px_trough = px0
        worst_ret = 1e300
        ret_at_trough = 0.0
        decided = False
        ru_max = 0.0
        for j in range(i, j_end + 1):
            px = cp[j]
            if not np.isfinite(px) or px <= 0.0:
                break
            ru_curr = (px / px0 - 1.0) * 100.0
            if ru_curr > ru_max:
                ru_max = ru_curr
            if ru_curr >= max_ru:
                if worst_ret == 1e300:
                    y[i] = 0.0
                else:
                    yr = worst_ret
                    if yr < -clip: yr = -clip
                    elif yr > clip: yr = clip
                    y[i] = yr
                ru[i] = ru_max
                decided = True
                break
            L = (j - i + 1)
            avg = (csum[j+1] - csum[i]) / L
            r_avg = (avg / px0 - 1.0) * 100.0
            if r_avg < worst_ret:
                worst_ret = r_avg
            if px < px_trough:
                px_trough = px
                ret_at_trough = r_avg
            if (ret_at_trough <= -p_tgt) and (px_trough > 0.0):
                alta_pct = (px / px_trough - 1.0) * 100.0
                if alta_pct >= rise_conf:
                    yr = ret_at_trough
                    if yr < -clip: yr = -clip
                    elif yr > clip: yr = clip
                    y[i] = yr
                    ru[i] = ru_max
                    decided = True
                    break
        if not decided:
            if worst_ret == 1e300:
                y[i] = 0.0; ru[i] = ru_max
            else:
                yr = worst_ret
                if yr < -clip: yr = -clip
                elif yr > clip: yr = clip
                y[i] = yr
                ru[i] = ru_max
    return y, ru


def make_U(y: np.ndarray | pd.Series,
           dd: np.ndarray | pd.Series,
           *, cost: float, p_tgt: float, dmax: float, lamb: float) -> np.ndarray:
    y  = np.asarray(y,  dtype=np.float64)
    dd = np.asarray(dd, dtype=np.float64)
    dd_mag = np.maximum(0.0, -dd)  # |dd| em %
    return (y - float(cost)) - float(lamb) * dd_mag


def make_U_short(y: np.ndarray | pd.Series,
                 ru_up: np.ndarray | pd.Series,
                 *, cost: float, p_tgt: float, dmax: float, lamb: float) -> np.ndarray:
    y    = np.asarray(y,     dtype=np.float64)
    ru_u = np.asarray(ru_up, dtype=np.float64)
    ru_mag = np.maximum(0.0, ru_u)
    return ((-y) - float(cost)) - float(lamb) * ru_mag


def label_from_mode(
    df: pd.DataFrame,
    *,
    hours: float,
    clip: float,
    p_tgt: float,
    dmax: float,
    drop_confirm: float,
    verbose: bool = False,
    cost: float = DEFAULT_COST,
    lamb: float = DEFAULT_LAMB,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    t_all = time.perf_counter()
    cp = df["close"].to_numpy(np.float64)
    # horizonte em barras
    n = _hours_to_bars(df.index, float(hours))
    # prefix sum de preços
    csum = np.empty(cp.size + 1, np.float64)
    csum[0] = 0.0
    for k in range(cp.size):
        csum[k+1] = csum[k] + cp[k]

    t0_long = time.perf_counter()
    y_long_arr, dd_arr, _ru_l_arr = _label_peakstop_avg_numba(
        cp, csum, int(n), float(clip), float(p_tgt), float(dmax), float(drop_confirm)
    )
    t1_long = time.perf_counter()

    t0_short = time.perf_counter()
    y_short_arr, ru_arr = _label_troughstop_avg_numba(
        cp, csum, int(n), float(clip), float(p_tgt), float(dmax), float(drop_confirm)
    )
    t1_short = time.perf_counter()

    # séries pandas
    y_long  = pd.Series(y_long_arr,  index=df.index, name="y_long")
    dd      = pd.Series(dd_arr,      index=df.index, name="dd_pct")
    y_short = pd.Series(y_short_arr, index=df.index, name="y_short")
    ru_up   = pd.Series(ru_arr,      index=df.index, name="ru_up_pct")

    # escreve no DF
    df["y_long"]    = y_long
    df["dd_pct"]    = dd
    df["y_short"]   = y_short
    df["ru_up_pct"] = ru_up

    # utilidades contínuas
    u_long_raw  = make_U(y_long, dd, cost=cost, p_tgt=p_tgt, dmax=dmax, lamb=lamb)
    u_short_raw = make_U_short(y_short, ru_up, cost=cost, p_tgt=p_tgt, dmax=dmax, lamb=lamb)
    u_long_pos  = np.maximum(0.0, u_long_raw)
    u_short_neg = -np.maximum(0.0, u_short_raw)
    df["U_compra"] = u_long_pos
    df["U_venda"]  = u_short_neg
    df["U_total"]  = u_long_pos + u_short_neg
    df["y"] = df["y_long"]

    total = time.perf_counter() - t_all
    if verbose:
        print((f"[label   ] rows={len(df):,} | n_bars={n} | clip={float(clip):.2f} | {total:.2f}s").replace(",","."), flush=True)
        parts = (
            f"long={t1_long - t0_long:.2f}s | short={t1_short - t0_short:.2f}s"
        )
        print(f"   • {parts}", flush=True)

    return y_long, dd, y_short
