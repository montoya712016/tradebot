# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np, pandas as pd
from numba import njit
from .pf_config import (
    PIVOT_MIN_MOVE_PCT, PIVOT_MIN_BARS_MIN,
    REV_LOOKAHEAD_MIN, REV_MIN_UP_PCT, IMPULSE_MAX_1BAR_PCT,
    PIVOT_LOCAL_WIN_MIN, PIVOT_LOCAL_MERGE_TOL_MIN, PIVOT_INCLUDE_LOCAL,
    RULE_LOOKBACK_MIN, RULE_DROP_PCT, RULE_RISE_PCT, RULE_U_HI, RULE_U_LO,
    RULE_MAX_DIST_FROM_EXTREME_PCT,
)
@njit(cache=True)
def _compute_drop_rise_nb(close: np.ndarray, win: int, drop_pct: float, rise_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    n = close.size
    drop_mask = np.zeros(n, dtype=np.bool_)
    rise_mask = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return drop_mask, rise_mask

    rmax = _rolling_max(close, win)
    rmin = _rolling_min(close, win)
    prev_max = np.empty(n, np.float64)
    prev_min = np.empty(n, np.float64)
    prev_max[0] = close[0]
    prev_min[0] = close[0]
    for i in range(1, n):
        prev_max[i] = rmax[i-1]
        prev_min[i] = rmin[i-1]

    for i in range(1, n):
        mx = prev_max[i]
        mn = prev_min[i]
        if mx > 0.0:
            drop_mask[i] = ((close[i] / (mx + 1e-9)) - 1.0) <= (-float(drop_pct))
        if mn > 0.0:
            rise_mask[i] = ((close[i] / (mn + 1e-9)) - 1.0) >= float(rise_pct)
    return drop_mask, rise_mask



def _minutes_to_candles_from_index(idx: pd.DatetimeIndex, minutes: int) -> int:
    if len(idx) < 2:
        return 1
    secs = float((idx[1] - idx[0]).total_seconds() or 60.0)
    return max(1, int(round((minutes * 60.0) / max(1.0, secs))))


@njit
def _compute_pivots_and_reversals(close: np.ndarray,
                                  min_move: float,
                                  min_bars: int,
                                  look_fwd: int,
                                  min_up: float,
                                  max_1bar: float) -> Tuple[np.ndarray, np.ndarray]:
    n = close.size
    piv_kind = np.zeros(n, np.int8)
    rev_buy  = np.zeros(n, np.uint8)
    if n == 0:
        return piv_kind, rev_buy

    last_pivot = 0
    trend = 0
    cand_high = 0
    cand_low  = 0

    for t in range(1, n):
        if close[t] > close[cand_high]:
            cand_high = t
        if close[t] < close[cand_low]:
            cand_low = t

        up_move = (close[cand_high] / close[last_pivot]) - 1.0
        if (trend <= 0) and (cand_high - last_pivot >= min_bars) and (up_move >= min_move):
            piv_kind[cand_high] = 1
            last_pivot = cand_high
            trend = +1
            cand_low = cand_high
            continue

        down_move = 1.0 - (close[cand_low] / close[last_pivot])
        if (trend >= 0) and (cand_low - last_pivot >= min_bars) and (down_move >= min_move):
            piv_kind[cand_low] = -1
            # confirmação de reversão (buy)
            i = cand_low
            j_end = i + look_fwd
            if j_end >= n:
                j_end = n - 1
            max_ret = 0.0
            max_step = 0.0
            prev = close[i]
            for j in range(i + 1, j_end + 1):
                r = (close[j] / close[i]) - 1.0
                if r > max_ret:
                    max_ret = r
                step = (close[j] / prev) - 1.0
                if step < 0.0:
                    step = -step
                if step > max_step:
                    max_step = step
                prev = close[j]
            if (max_ret >= min_up) and (max_step <= max_1bar):
                rev_buy[i] = 1

            last_pivot = cand_low
            trend = -1
            cand_high = cand_low
            continue

    return piv_kind, rev_buy


@njit
def _local_extrema(close: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    n = close.size
    trough = np.zeros(n, np.uint8)
    peak   = np.zeros(n, np.uint8)
    if n == 0 or win <= 0:
        return trough, peak
    for i in range(n):
        l = i - win
        if l < 0:
            l = 0
        r = i + win
        if r >= n:
            r = n - 1
        v = close[i]
        is_min = 1
        is_max = 1
        for k in range(l, r + 1):
            if close[k] < v:
                is_max = 0
            if close[k] > v:
                is_min = 0
            if is_min == 0 and is_max == 0:
                break
        if is_min == 1:
            trough[i] = 1
        if is_max == 1:
            peak[i] = 1
    return trough, peak


@njit
def _thin_marks(marks: np.ndarray, min_dist: int) -> np.ndarray:
    n = marks.size
    out = np.zeros(n, np.uint8)
    if n == 0 or min_dist <= 0:
        return marks.copy()
    last = -min_dist - 1
    for i in range(n):
        if marks[i] != 0:
            if i - last > min_dist:
                out[i] = 1
                last = i
    return out


def apply_pivots(df: pd.DataFrame,
                 *,
                 min_move_pct: float = PIVOT_MIN_MOVE_PCT,
                 min_bars_min: int = PIVOT_MIN_BARS_MIN,
                 lookahead_min: int = REV_LOOKAHEAD_MIN,
                 min_up_pct: float = REV_MIN_UP_PCT,
                 max_1bar_pct: float = IMPULSE_MAX_1BAR_PCT,
                 local_win_min: int = PIVOT_LOCAL_WIN_MIN,
                 local_merge_tol_min: int = PIVOT_LOCAL_MERGE_TOL_MIN,
                 include_local: bool = PIVOT_INCLUDE_LOCAL) -> None:
    """
    Escreve colunas:
      - pivot_trough (uint8)
      - pivot_peak (uint8)
      - rev_buy_candidate (uint8)
    """
    close = df["close"].to_numpy(np.float64)
    min_bars = _minutes_to_candles_from_index(df.index, int(min_bars_min))
    look_fwd = _minutes_to_candles_from_index(df.index, int(lookahead_min))
    piv_kind, rev_buy = _compute_pivots_and_reversals(
        close,
        float(min_move_pct),
        int(min_bars),
        int(look_fwd),
        float(min_up_pct),
        float(max_1bar_pct),
    )
    p_tr = (piv_kind == -1).astype(np.uint8)
    p_pk = (piv_kind ==  1).astype(np.uint8)
    if include_local:
        # Extremos locais (modo permissivo)
        local_win = _minutes_to_candles_from_index(df.index, int(local_win_min))
        loc_tr, loc_pk = _local_extrema(close, int(local_win))
        # União zigzag + local
        p_tr = (p_tr | loc_tr)
        p_pk = (p_pk | loc_pk)
        # Afina marcas muito próximas
        merge_tol = _minutes_to_candles_from_index(df.index, int(local_merge_tol_min))
        if merge_tol > 0:
            p_tr = _thin_marks(p_tr, int(merge_tol))
            p_pk = _thin_marks(p_pk, int(merge_tol))
    df["pivot_trough"] = p_tr
    df["pivot_peak"]   = p_pk
    df["rev_buy_candidate"] = rev_buy.astype(np.uint8)


@njit
def _rolling_max(x: np.ndarray, win: int) -> np.ndarray:
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
def _rolling_min(x: np.ndarray, win: int) -> np.ndarray:
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


def apply_rule_reversals(
    df: pd.DataFrame,
    *,
    lookback_min: int = RULE_LOOKBACK_MIN,
    drop_pct: float = RULE_DROP_PCT,
    rise_pct: float = RULE_RISE_PCT,
    u_hi: float = RULE_U_HI,
    u_lo: float = RULE_U_LO,
    max_distance_from_extreme_pct: float = RULE_MAX_DIST_FROM_EXTREME_PCT,  # Máximo 1.5% de distância do extremo
) -> None:
    """
    Marca REVERSÕES EM V com verificação de ORDEM TEMPORAL:
    
    Para ser um FUNDO em V (buy):
      1. O MÁXIMO da janela veio ANTES do MÍNIMO (preço CAIU)
      2. Queda significativa: (max - min) / max >= drop_pct
      3. Preço atual MUITO perto do mínimo: (close - min) / min <= 1.5%
      4. U_total > u_hi
    
    Para ser um PICO em V (short):
      1. O MÍNIMO da janela veio ANTES do MÁXIMO (preço SUBIU)
      2. Alta significativa: (max - min) / min >= rise_pct
      3. Preço atual MUITO perto do máximo: (max - close) / max <= 1.5%
      4. U_total < u_lo
    
    Isso garante reversões em V REAIS, não "explosões do nada"!
    """
    close = df["close"].to_numpy(np.float64)
    n = len(close)
    
    u_tot = df.get("U_total")
    if u_tot is None:
        if ("U_compra" in df.columns) and ("U_venda" in df.columns):
            u_tot = (df["U_compra"] + df["U_venda"]).astype(np.float64)
        else:
            u_tot = pd.Series(np.zeros_like(close), index=df.index)
    u_arr = u_tot.to_numpy(np.float64)

    win = _minutes_to_candles_from_index(df.index, int(lookback_min))
    if win < 1:
        win = 1

    drop_mask, rise_mask = _compute_drop_rise_nb(close, win, float(drop_pct), float(rise_pct))
    
    rev_buy_rule = np.zeros(n, dtype=bool)
    peak_rule = np.zeros(n, dtype=bool)
    
    for i in range(win, n):
        start = i - win
        end = i + 1
        
        # Encontra índices do máximo e mínimo na janela
        window = close[start:end]
        idx_max_local = int(np.argmax(window))
        idx_min_local = int(np.argmin(window))
        
        val_max = window[idx_max_local]
        val_min = window[idx_min_local]
        
        if val_min <= 0 or val_max <= 0:
            continue
        
        # Queda e alta em %
        drop_pct_actual = (val_max - val_min) / val_max
        rise_pct_actual = (val_max - val_min) / val_min
        
        # Distância do preço atual até os extremos
        distance_from_min = (close[i] - val_min) / val_min
        distance_from_max = (val_max - close[i]) / val_max
        
        # FUNDO em V: 
        # 1. Máximo veio ANTES do mínimo (idx_max_local < idx_min_local)
        # 2. Queda significativa
        # 3. Preço atual muito perto do mínimo
        # 4. U positivo
        if (idx_max_local < idx_min_local and 
            drop_pct_actual >= float(drop_pct) and 
            distance_from_min <= float(max_distance_from_extreme_pct) and
            u_arr[i] > float(u_hi) and
            drop_mask[i]):
            rev_buy_rule[i] = True
        
        # PICO em V:
        # 1. Mínimo veio ANTES do máximo (idx_min_local < idx_max_local)
        # 2. Alta significativa
        # 3. Preço atual muito perto do máximo
        # 4. U negativo
        if (idx_min_local < idx_max_local and 
            rise_pct_actual >= float(rise_pct) and 
            distance_from_max <= float(max_distance_from_extreme_pct) and
            u_arr[i] < float(u_lo) and
            rise_mask[i]):
            peak_rule[i] = True

    df["rev_buy_rule"] = rev_buy_rule.astype(np.uint8)
    df["peak_rule"]    = peak_rule.astype(np.uint8)
    
    # Debug info
    n_buy = int(rev_buy_rule.sum())
    n_peak = int(peak_rule.sum())
    if n_buy > 0 or n_peak > 0:
        print(f"[pivots] rev_buy_rule={n_buy} | peak_rule={n_peak} | "
              f"drop>={drop_pct:.1%}, max_dist<={max_distance_from_extreme_pct:.1%}", flush=True)


__all__ = ["apply_rule_reversals"]


__all__ = ["apply_pivots"]


