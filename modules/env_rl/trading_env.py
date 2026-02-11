# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import os
import numpy as np
import pandas as pd

from .action_space import (
    ACTION_HOLD,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_LONG_BIG,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_OPEN_SHORT_BIG,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)
from .reward import RewardConfig

njit = None  # type: ignore
_HAS_NUMBA = False
_NUMBA_READY = False
_FUTURE_REGRET_NUMBA = None
_STATE_VEC_NUMBA = None
_STEP_CORE_NUMBA = None
_NUMBA_WARMED = False


def _ensure_numba() -> bool:
    global njit, _HAS_NUMBA, _NUMBA_READY
    if _NUMBA_READY:
        return _HAS_NUMBA
    _NUMBA_READY = True
    if os.getenv("SNIPER_DISABLE_NUMBA_RL", "").strip().lower() in {"1", "true", "yes"}:
        _HAS_NUMBA = False
        return False
    try:
        from numba import njit as _njit  # type: ignore

        njit = _njit
        _HAS_NUMBA = True
    except Exception:
        njit = None  # type: ignore
        _HAS_NUMBA = False
    return _HAS_NUMBA


def _future_regret_arrays_py(close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    w = int(max(1, window))
    n = int(close.shape[0])
    fut_min = np.zeros(n, dtype=np.float32)
    fut_max = np.zeros(n, dtype=np.float32)
    for i in range(n):
        px = float(close[i])
        if px <= 0.0 or (not np.isfinite(px)):
            continue
        j0 = i + 1
        j1 = min(n, i + 1 + w)
        if j0 >= j1:
            continue
        win = close[j0:j1]
        rets = (win / px) - 1.0
        fut_min[i] = float(np.nanmin(rets)) if rets.size else 0.0
        fut_max[i] = float(np.nanmax(rets)) if rets.size else 0.0
    return fut_min, fut_max


def _future_regret_arrays_numba(close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    global _FUTURE_REGRET_NUMBA
    if (not _ensure_numba()) or (njit is None):
        return _future_regret_arrays_py(close, window)
    if _FUTURE_REGRET_NUMBA is None:
        @njit(cache=True)
        def _impl(close_arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
            n = close_arr.shape[0]
            fut_min = np.zeros(n, dtype=np.float32)
            fut_max = np.zeros(n, dtype=np.float32)
            for i in range(n):
                px = close_arr[i]
                if (px <= 0.0) or (not np.isfinite(px)):
                    continue
                j0 = i + 1
                j1 = i + 1 + w
                if j1 > n:
                    j1 = n
                if j0 >= j1:
                    continue
                mn = 1e30
                mx = -1e30
                for j in range(j0, j1):
                    v = (close_arr[j] / px) - 1.0
                    if not np.isfinite(v):
                        continue
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                if mn <= 1e29:
                    fut_min[i] = np.float32(mn)
                    fut_max[i] = np.float32(mx)
            return fut_min, fut_max

        _FUTURE_REGRET_NUMBA = _impl
    try:
        return _FUTURE_REGRET_NUMBA(np.asarray(close, dtype=np.float64), int(max(1, window)))
    except Exception:
        return _future_regret_arrays_py(close, window)


def _state_vec_py(
    t: int,
    position_side: int,
    position_size: float,
    entry_price: float,
    time_in_trade: int,
    cooldown_left: int,
    cooldown_bars: int,
    trade_peak_pnl: float,
    close: np.ndarray,
    mu_long_norm: np.ndarray,
    mu_short_norm: np.ndarray,
    edge_norm: np.ndarray,
    strength_norm: np.ndarray,
    uncertainty_norm: np.ndarray,
    vol_short_norm: np.ndarray,
    vol_long_norm: np.ndarray,
    trend_strength_norm: np.ndarray,
    shock_flag: np.ndarray,
) -> np.ndarray:
    pos_flat = 1.0 if position_side == 0 else 0.0
    pos_long = 1.0 if position_side > 0 else 0.0
    pos_short = 1.0 if position_side < 0 else 0.0
    if position_side == 0 or entry_price <= 0.0:
        unr = 0.0
    else:
        px = float(close[t])
        unr = 0.0 if px <= 0.0 else ((px / float(entry_price)) - 1.0) * float(position_side) * float(position_size)
    denom = max(1e-4, abs(float(vol_long_norm[t])) + 1e-4)
    unr_norm = float(np.clip(unr / denom, -8.0, 8.0))
    dd_trade = max(0.0, float(trade_peak_pnl - unr)) if position_side != 0 else 0.0
    t_norm = float(np.clip(np.log1p(max(0, int(time_in_trade))) / np.log1p(720.0), 0.0, 1.0))
    cd_norm = float(np.clip(float(cooldown_left) / max(1.0, float(cooldown_bars)), 0.0, 1.0))
    return np.asarray(
        [
            float(mu_long_norm[t]),
            float(mu_short_norm[t]),
            float(edge_norm[t]),
            float(strength_norm[t]),
            float(uncertainty_norm[t]),
            float(vol_short_norm[t]),
            float(vol_long_norm[t]),
            float(trend_strength_norm[t]),
            float(shock_flag[t]),
            pos_flat,
            pos_long,
            pos_short,
            float(position_size),
            t_norm,
            unr_norm,
            float(np.clip(dd_trade, 0.0, 8.0)),
            cd_norm,
        ],
        dtype=np.float32,
    )


def _state_vec_numba(
    t: int,
    position_side: int,
    position_size: float,
    entry_price: float,
    time_in_trade: int,
    cooldown_left: int,
    cooldown_bars: int,
    trade_peak_pnl: float,
    close: np.ndarray,
    mu_long_norm: np.ndarray,
    mu_short_norm: np.ndarray,
    edge_norm: np.ndarray,
    strength_norm: np.ndarray,
    uncertainty_norm: np.ndarray,
    vol_short_norm: np.ndarray,
    vol_long_norm: np.ndarray,
    trend_strength_norm: np.ndarray,
    shock_flag: np.ndarray,
) -> np.ndarray:
    global _STATE_VEC_NUMBA
    if (not _ensure_numba()) or (njit is None):
        return _state_vec_py(
            t,
            position_side,
            position_size,
            entry_price,
            time_in_trade,
            cooldown_left,
            cooldown_bars,
            trade_peak_pnl,
            close,
            mu_long_norm,
            mu_short_norm,
            edge_norm,
            strength_norm,
            uncertainty_norm,
            vol_short_norm,
            vol_long_norm,
            trend_strength_norm,
            shock_flag,
        )
    if _STATE_VEC_NUMBA is None:
        @njit(cache=True)
        def _impl(
            t_idx: int,
            pos_side: int,
            pos_size: float,
            ent_px: float,
            t_trade: int,
            cd_left: int,
            cd_bars: int,
            tr_peak: float,
            close_arr: np.ndarray,
            mu_l: np.ndarray,
            mu_s: np.ndarray,
            edge_arr: np.ndarray,
            strength_arr: np.ndarray,
            uncertainty_arr: np.ndarray,
            vol_s: np.ndarray,
            vol_l: np.ndarray,
            trend_arr: np.ndarray,
            shock_arr: np.ndarray,
        ) -> np.ndarray:
            out = np.zeros(17, dtype=np.float32)
            if pos_side == 0:
                out[9] = np.float32(1.0)
            elif pos_side > 0:
                out[10] = np.float32(1.0)
            else:
                out[11] = np.float32(1.0)
            unr = 0.0
            if (pos_side != 0) and (ent_px > 0.0):
                px = close_arr[t_idx]
                if px > 0.0:
                    unr = ((px / ent_px) - 1.0) * float(pos_side) * pos_size
            denom = np.abs(float(vol_l[t_idx])) + 1e-4
            if denom < 1e-4:
                denom = 1e-4
            unr_norm = unr / denom
            if unr_norm > 8.0:
                unr_norm = 8.0
            elif unr_norm < -8.0:
                unr_norm = -8.0
            dd_trade = 0.0
            if pos_side != 0:
                dd_trade = tr_peak - unr
                if dd_trade < 0.0:
                    dd_trade = 0.0
            t_norm = np.log1p(float(max(0, t_trade))) / np.log1p(720.0)
            if t_norm > 1.0:
                t_norm = 1.0
            if t_norm < 0.0:
                t_norm = 0.0
            cd_den = float(max(1, cd_bars))
            cd_norm = float(cd_left) / cd_den
            if cd_norm > 1.0:
                cd_norm = 1.0
            if cd_norm < 0.0:
                cd_norm = 0.0

            out[0] = np.float32(mu_l[t_idx])
            out[1] = np.float32(mu_s[t_idx])
            out[2] = np.float32(edge_arr[t_idx])
            out[3] = np.float32(strength_arr[t_idx])
            out[4] = np.float32(uncertainty_arr[t_idx])
            out[5] = np.float32(vol_s[t_idx])
            out[6] = np.float32(vol_l[t_idx])
            out[7] = np.float32(trend_arr[t_idx])
            out[8] = np.float32(shock_arr[t_idx])
            out[12] = np.float32(pos_size)
            out[13] = np.float32(t_norm)
            out[14] = np.float32(unr_norm)
            if dd_trade > 8.0:
                dd_trade = 8.0
            out[15] = np.float32(dd_trade)
            out[16] = np.float32(cd_norm)
            return out

        _STATE_VEC_NUMBA = _impl
    try:
        return _STATE_VEC_NUMBA(
            int(t),
            int(position_side),
            float(position_size),
            float(entry_price),
            int(time_in_trade),
            int(cooldown_left),
            int(cooldown_bars),
            float(trade_peak_pnl),
            close,
            mu_long_norm,
            mu_short_norm,
            edge_norm,
            strength_norm,
            uncertainty_norm,
            vol_short_norm,
            vol_long_norm,
            trend_strength_norm,
            shock_flag,
        )
    except Exception:
        return _state_vec_py(
            t,
            position_side,
            position_size,
            entry_price,
            time_in_trade,
            cooldown_left,
            cooldown_bars,
            trade_peak_pnl,
            close,
            mu_long_norm,
            mu_short_norm,
            edge_norm,
            strength_norm,
            uncertainty_norm,
            vol_short_norm,
            vol_long_norm,
            trend_strength_norm,
            shock_flag,
        )


def _step_core_py(
    action: int,
    t: int,
    n: int,
    prev_side: int,
    prev_size: float,
    entry_price_prev: float,
    time_in_trade_prev: int,
    cooldown_left_prev: int,
    equity_prev: float,
    equity_peak_prev: float,
    max_dd_prev: float,
    prev_dd_prev: float,
    trade_peak_prev: float,
    bars_since_last_exit_prev: int,
    use_signal_gate: bool,
    small_size: float,
    big_size: float,
    edge_thr: float,
    strength_thr: float,
    min_reentry_gap_bars: int,
    min_hold_bars: int,
    max_hold_bars: int,
    cooldown_bars: int,
    fee_rate: float,
    slippage_rate: float,
    dd_penalty: float,
    dd_level_penalty: float,
    dd_soft_limit: float,
    dd_excess_penalty: float,
    dd_hard_limit: float,
    dd_hard_penalty: float,
    hold_bar_penalty: float,
    hold_soft_bars: int,
    hold_excess_penalty: float,
    hold_regret_penalty: float,
    stagnation_bars: int,
    stagnation_ret_epsilon: float,
    stagnation_penalty: float,
    reverse_penalty: float,
    entry_penalty: float,
    weak_entry_penalty: float,
    turnover_penalty: float,
    regret_penalty: float,
    idle_penalty: float,
    close: np.ndarray,
    fwd_ret: np.ndarray,
    edge_norm: np.ndarray,
    strength_norm: np.ndarray,
    future_min_ret: np.ndarray,
    future_max_ret: np.ndarray,
) -> tuple:
    a = int(action)
    if a == 0:
        target_side, target_size = int(prev_side), float(prev_size)
    elif a == 1:
        target_side, target_size = 1, float(small_size)
    elif a == 2:
        target_side, target_size = 1, float(big_size)
    elif a == 3:
        target_side, target_size = -1, float(small_size)
    elif a == 4:
        target_side, target_size = -1, float(big_size)
    elif a == 5:
        if prev_side > 0:
            target_side, target_size = 0, 0.0
        else:
            target_side, target_size = int(prev_side), float(prev_size)
    elif a == 6:
        if prev_side < 0:
            target_side, target_size = 0, 0.0
        else:
            target_side, target_size = int(prev_side), float(prev_size)
    else:
        target_side, target_size = int(prev_side), float(prev_size)

    edge_now = float(edge_norm[t])
    strength_now = float(strength_norm[t])
    # gate supervisionado so para novas entradas (nao forca fechamento intratrade)
    if use_signal_gate and prev_side == 0 and target_side != 0:
        if strength_now < float(strength_thr):
            target_side, target_size = 0, 0.0
        elif edge_now >= float(edge_thr):
            if target_side < 0:
                target_side, target_size = 0, 0.0
        elif edge_now <= -float(edge_thr):
            if target_side > 0:
                target_side, target_size = 0, 0.0
        else:
            target_side, target_size = 0, 0.0

    if prev_side == 0 and target_side != 0 and bars_since_last_exit_prev < int(max(0, min_reentry_gap_bars)):
        target_side, target_size = 0, 0.0
    if prev_side != 0 and time_in_trade_prev < int(max(0, min_hold_bars)):
        # Enforce true minimum hold: no size/side/close changes before min_hold_bars.
        if target_side != prev_side or abs(target_size - prev_size) > 1e-12:
            target_side, target_size = prev_side, prev_size

    forced_max_hold_close = 0
    if prev_side != 0 and int(max_hold_bars) > 0 and int(time_in_trade_prev) >= int(max_hold_bars):
        target_side, target_size = 0, 0.0
        forced_max_hold_close = 1

    # Hard DD lock: apos ultrapassar o limite, nao permite manter/abrir risco no episodio.
    if float(dd_hard_limit) < 1.0 and float(max_dd_prev) >= float(dd_hard_limit):
        target_side, target_size = 0, 0.0

    changed = (target_side != prev_side) or (abs(target_size - prev_size) > 1e-12)
    is_reverse = bool(prev_side != 0 and target_side != 0 and target_side != prev_side)
    if cooldown_left_prev > 0 and changed:
        is_entry_or_reverse = bool(
            (prev_side == 0 and target_side != 0)
            or (prev_side != 0 and target_side != 0 and target_side != prev_side)
        )
        if is_entry_or_reverse:
            target_side, target_size = prev_side, prev_size
            changed = False

    prev_exp = float(prev_side) * float(prev_size)
    new_exp = float(target_side) * float(target_size)
    turn = float(abs(new_exp - prev_exp))
    tx_cost = float(turn * (float(fee_rate) + float(slippage_rate)))

    close_prev_trade = int(prev_side != 0 and (target_side != prev_side or target_side == 0))
    open_new_trade = int(target_side != 0 and (prev_side == 0 or target_side != prev_side))
    position_side = int(target_side)
    position_size = float(target_size)

    if open_new_trade == 1:
        entry_price = float(close[t])
    elif target_side == 0:
        entry_price = 0.0
    else:
        entry_price = float(entry_price_prev)

    step_ret = float(fwd_ret[t])
    if not np.isfinite(step_ret):
        step_ret = 0.0
    pnl_step = float(float(position_side) * float(position_size) * step_ret)
    equity = float(equity_prev * (1.0 + pnl_step - tx_cost))
    equity_peak = float(max(float(equity_peak_prev), equity))
    dd = float((equity_peak - equity) / max(1e-12, equity_peak))
    max_dd = float(max(float(max_dd_prev), dd))
    dd_increase = float(max(0.0, dd - float(prev_dd_prev)))
    hard_dd_triggered = bool(
        (float(dd_hard_limit) < 1.0) and (float(max_dd_prev) < float(dd_hard_limit)) and (dd >= float(dd_hard_limit))
    )

    unr_now = 0.0
    if position_side != 0:
        base_tit = 0 if open_new_trade == 1 else int(time_in_trade_prev)
        time_in_trade = int(base_tit + 1)
        next_idx = t + 1 if (t + 1) < n else t
        if entry_price > 0.0 and close[next_idx] > 0.0:
            unr_now = ((float(close[next_idx]) / float(entry_price)) - 1.0) * float(position_side) * float(position_size)
        else:
            unr_now = 0.0
        base_peak = 0.0 if open_new_trade == 1 else float(trade_peak_prev)
        trade_peak_pnl = float(max(base_peak, unr_now))
    else:
        time_in_trade = 0
        trade_peak_pnl = 0.0

    regret = 0.0
    if position_side != 0 and (prev_side == 0 or prev_side != position_side):
        fut_min = float(future_min_ret[t])
        fut_max = float(future_max_ret[t])
        if position_side > 0:
            regret = max(0.0, (-fut_min) - max(0.0, fut_max))
        else:
            regret = max(0.0, fut_max - max(0.0, -fut_min))
        regret = float(regret * position_size)

    delta_equity = float(equity - equity_prev)
    was_idle = bool(position_side == 0 and strength_now > 1.0)
    idle_pen = float(idle_penalty) if was_idle else 0.0
    dd_level_pen = float(dd_level_penalty) * float(dd)
    dd_excess = max(0.0, float(dd) - float(dd_soft_limit))
    dd_excess_pen = float(dd_excess_penalty) * float(dd_excess * dd_excess)
    dd_hard_pen = float(dd_hard_penalty) if hard_dd_triggered else 0.0
    hold_bar_pen = float(hold_bar_penalty) * float(max(0.0, position_size)) if position_side != 0 else 0.0
    hold_soft = int(max(1, hold_soft_bars))
    hold_excess = max(0.0, float(time_in_trade) - float(hold_soft)) / float(hold_soft) if position_side != 0 else 0.0
    hold_excess_pen = float(hold_excess_penalty) * float(hold_excess * hold_excess)
    hold_regret = max(0.0, float(trade_peak_pnl - unr_now)) if position_side != 0 else 0.0
    hold_regret_pen = float(hold_regret_penalty) * float(hold_regret)
    stagnation_pen = 0.0
    if position_side != 0 and int(time_in_trade) > int(max(1, stagnation_bars)):
        if abs(float(step_ret)) <= float(max(0.0, stagnation_ret_epsilon)):
            st_norm = float(int(time_in_trade) - int(stagnation_bars)) / float(max(1, int(stagnation_bars)))
            stagnation_pen = float(stagnation_penalty) * float(max(0.0, st_norm))
    entry_pen = 0.0
    if open_new_trade == 1:
        edge_abs = float(abs(edge_now))
        edge_ref = float(max(1e-6, abs(edge_thr)))
        weak_ratio = float(max(0.0, (edge_ref - edge_abs) / edge_ref))
        entry_pen = float(entry_penalty) + float(weak_entry_penalty) * weak_ratio
    reward = float(
        delta_equity
        - float(turnover_penalty) * turn
        - float(dd_penalty) * dd_increase
        - dd_level_pen
        - dd_excess_pen
        - dd_hard_pen
        - hold_bar_pen
        - hold_excess_pen
        - hold_regret_pen
        - stagnation_pen
        - (float(reverse_penalty) if is_reverse else 0.0)
        - entry_pen
        - float(regret_penalty) * regret
        - idle_pen
    )

    if changed:
        cooldown_left = int(max(0, cooldown_bars))
    elif cooldown_left_prev > 0:
        cooldown_left = int(cooldown_left_prev - 1)
    else:
        cooldown_left = 0

    bars_since_last_exit = int(bars_since_last_exit_prev + 1) if position_side == 0 else 0
    done = bool((t + 1) >= (n - 1))
    return (
        position_side,
        position_size,
        entry_price,
        time_in_trade,
        cooldown_left,
        equity,
        equity_peak,
        max_dd,
        dd,
        dd_increase,
        trade_peak_pnl,
        reward,
        turn,
        tx_cost,
        pnl_step,
        regret,
        int(hard_dd_triggered),
        hold_bar_pen,
        hold_excess_pen,
        hold_regret_pen,
        stagnation_pen,
        float(reverse_penalty) if is_reverse else 0.0,
        entry_pen,
        int(forced_max_hold_close),
        bars_since_last_exit,
        int(close_prev_trade),
        int(open_new_trade),
        int(changed),
        bool(done),
    )


def _step_core_numba(*args):
    global _STEP_CORE_NUMBA
    if (not _ensure_numba()) or (njit is None):
        return _step_core_py(*args)
    if _STEP_CORE_NUMBA is None:
        @njit(cache=True)
        def _impl(
            action: int,
            t: int,
            n: int,
            prev_side: int,
            prev_size: float,
            entry_price_prev: float,
            time_in_trade_prev: int,
            cooldown_left_prev: int,
            equity_prev: float,
            equity_peak_prev: float,
            max_dd_prev: float,
            prev_dd_prev: float,
            trade_peak_prev: float,
            bars_since_last_exit_prev: int,
            use_signal_gate: bool,
            small_size: float,
            big_size: float,
            edge_thr: float,
            strength_thr: float,
            min_reentry_gap_bars: int,
            min_hold_bars: int,
            max_hold_bars: int,
            cooldown_bars: int,
            fee_rate: float,
            slippage_rate: float,
            dd_penalty: float,
            dd_level_penalty: float,
            dd_soft_limit: float,
            dd_excess_penalty: float,
            dd_hard_limit: float,
            dd_hard_penalty: float,
            hold_bar_penalty: float,
            hold_soft_bars: int,
            hold_excess_penalty: float,
            hold_regret_penalty: float,
            stagnation_bars: int,
            stagnation_ret_epsilon: float,
            stagnation_penalty: float,
            reverse_penalty: float,
            entry_penalty: float,
            weak_entry_penalty: float,
            turnover_penalty: float,
            regret_penalty: float,
            idle_penalty: float,
            close: np.ndarray,
            fwd_ret: np.ndarray,
            edge_norm: np.ndarray,
            strength_norm: np.ndarray,
            future_min_ret: np.ndarray,
            future_max_ret: np.ndarray,
        ):
            # action -> position
            if action == 0:
                target_side, target_size = int(prev_side), float(prev_size)
            elif action == 1:
                target_side, target_size = 1, small_size
            elif action == 2:
                target_side, target_size = 1, big_size
            elif action == 3:
                target_side, target_size = -1, small_size
            elif action == 4:
                target_side, target_size = -1, big_size
            elif action == 5:
                if prev_side > 0:
                    target_side, target_size = 0, 0.0
                else:
                    target_side, target_size = int(prev_side), float(prev_size)
            elif action == 6:
                if prev_side < 0:
                    target_side, target_size = 0, 0.0
                else:
                    target_side, target_size = int(prev_side), float(prev_size)
            else:
                target_side, target_size = int(prev_side), float(prev_size)

            edge_now = float(edge_norm[t])
            strength_now = float(strength_norm[t])
            # gate supervisionado apenas na entrada quando flat
            if use_signal_gate and prev_side == 0 and target_side != 0:
                if strength_now < strength_thr:
                    target_side, target_size = 0, 0.0
                elif edge_now >= edge_thr:
                    if target_side < 0:
                        target_side, target_size = 0, 0.0
                elif edge_now <= -edge_thr:
                    if target_side > 0:
                        target_side, target_size = 0, 0.0
                else:
                    target_side, target_size = 0, 0.0

            if prev_side == 0 and target_side != 0 and bars_since_last_exit_prev < max(0, min_reentry_gap_bars):
                target_side, target_size = 0, 0.0
            if prev_side != 0 and time_in_trade_prev < max(0, min_hold_bars):
                # Enforce true minimum hold: no size/side/close changes before min_hold_bars.
                if (target_side != prev_side) or (np.abs(target_size - prev_size) > 1e-12):
                    target_side, target_size = prev_side, prev_size

            forced_max_hold_close = 0
            if prev_side != 0 and max_hold_bars > 0 and time_in_trade_prev >= max_hold_bars:
                target_side, target_size = 0, 0.0
                forced_max_hold_close = 1

            # Hard DD lock: apos ultrapassar o limite, nao permite manter/abrir risco no episodio.
            if (dd_hard_limit < 1.0) and (max_dd_prev >= dd_hard_limit):
                target_side, target_size = 0, 0.0

            changed = (target_side != prev_side) or (np.abs(target_size - prev_size) > 1e-12)
            is_reverse = (prev_side != 0 and target_side != 0 and target_side != prev_side)
            if cooldown_left_prev > 0 and changed:
                is_entry_or_reverse = (
                    (prev_side == 0 and target_side != 0)
                    or (prev_side != 0 and target_side != 0 and target_side != prev_side)
                )
                if is_entry_or_reverse:
                    target_side, target_size = prev_side, prev_size
                    changed = False

            prev_exp = float(prev_side) * float(prev_size)
            new_exp = float(target_side) * float(target_size)
            turn = np.abs(new_exp - prev_exp)
            tx_cost = turn * (fee_rate + slippage_rate)

            close_prev_trade = 1 if (prev_side != 0 and (target_side != prev_side or target_side == 0)) else 0
            open_new_trade = 1 if (target_side != 0 and (prev_side == 0 or target_side != prev_side)) else 0

            position_side = int(target_side)
            position_size = float(target_size)
            if open_new_trade == 1:
                entry_price = float(close[t])
            elif target_side == 0:
                entry_price = 0.0
            else:
                entry_price = float(entry_price_prev)

            step_ret = float(fwd_ret[t])
            if not np.isfinite(step_ret):
                step_ret = 0.0
            pnl_step = float(float(position_side) * float(position_size) * step_ret)
            equity = float(equity_prev * (1.0 + pnl_step - tx_cost))
            equity_peak = float(equity_peak_prev if equity_peak_prev > equity else equity)
            dd = float((equity_peak - equity) / max(1e-12, equity_peak))
            max_dd = float(max_dd_prev if max_dd_prev > dd else dd)
            dd_increase = float(dd - prev_dd_prev if dd > prev_dd_prev else 0.0)
            hard_dd_triggered = (dd_hard_limit < 1.0) and (max_dd_prev < dd_hard_limit) and (dd >= dd_hard_limit)

            unr_now = 0.0
            if position_side != 0:
                base_tit = 0 if open_new_trade == 1 else int(time_in_trade_prev)
                time_in_trade = int(base_tit + 1)
                next_idx = t + 1 if (t + 1) < n else t
                if entry_price > 0.0 and close[next_idx] > 0.0:
                    unr_now = ((float(close[next_idx]) / float(entry_price)) - 1.0) * float(position_side) * float(position_size)
                else:
                    unr_now = 0.0
                base_peak = 0.0 if open_new_trade == 1 else float(trade_peak_prev)
                trade_peak_pnl = float(base_peak if base_peak > unr_now else unr_now)
            else:
                time_in_trade = 0
                trade_peak_pnl = 0.0

            regret = 0.0
            if position_side != 0 and (prev_side == 0 or prev_side != position_side):
                fut_min = float(future_min_ret[t])
                fut_max = float(future_max_ret[t])
                if position_side > 0:
                    regret = (-fut_min) - (fut_max if fut_max > 0.0 else 0.0)
                else:
                    regret = fut_max - ((-fut_min) if (-fut_min) > 0.0 else 0.0)
                if regret < 0.0:
                    regret = 0.0
                regret = float(regret * position_size)

            delta_equity = float(equity - equity_prev)
            idle_pen = float(idle_penalty) if (position_side == 0 and strength_now > 1.0) else 0.0
            dd_level_pen = dd_level_penalty * dd
            dd_excess = dd - dd_soft_limit
            if dd_excess < 0.0:
                dd_excess = 0.0
            dd_excess_pen = dd_excess_penalty * (dd_excess * dd_excess)
            dd_hard_pen = dd_hard_penalty if hard_dd_triggered else 0.0
            hold_bar_pen = hold_bar_penalty * (position_size if position_size > 0.0 else 0.0) if position_side != 0 else 0.0
            hold_soft = max(1, hold_soft_bars)
            hold_excess = ((time_in_trade - hold_soft) / float(hold_soft)) if (position_side != 0 and time_in_trade > hold_soft) else 0.0
            hold_excess_pen = hold_excess_penalty * (hold_excess * hold_excess)
            hold_regret = (trade_peak_pnl - unr_now) if position_side != 0 else 0.0
            if hold_regret < 0.0:
                hold_regret = 0.0
            hold_regret_pen = hold_regret_penalty * hold_regret
            stagnation_pen = 0.0
            if position_side != 0 and time_in_trade > max(1, stagnation_bars):
                if np.abs(step_ret) <= max(0.0, stagnation_ret_epsilon):
                    st_norm = float(time_in_trade - stagnation_bars) / float(max(1, stagnation_bars))
                    if st_norm < 0.0:
                        st_norm = 0.0
                    stagnation_pen = stagnation_penalty * st_norm
            entry_pen = 0.0
            if open_new_trade == 1:
                edge_abs = np.abs(edge_now)
                edge_ref = np.abs(edge_thr)
                if edge_ref < 1e-6:
                    edge_ref = 1e-6
                weak_ratio = (edge_ref - edge_abs) / edge_ref
                if weak_ratio < 0.0:
                    weak_ratio = 0.0
                entry_pen = entry_penalty + (weak_entry_penalty * weak_ratio)
            reward = float(
                delta_equity
                - turnover_penalty * turn
                - dd_penalty * dd_increase
                - dd_level_pen
                - dd_excess_pen
                - dd_hard_pen
                - hold_bar_pen
                - hold_excess_pen
                - hold_regret_pen
                - stagnation_pen
                - (reverse_penalty if is_reverse else 0.0)
                - entry_pen
                - regret_penalty * regret
                - idle_pen
            )

            if changed:
                cooldown_left = int(max(0, cooldown_bars))
            elif cooldown_left_prev > 0:
                cooldown_left = int(cooldown_left_prev - 1)
            else:
                cooldown_left = 0

            bars_since_last_exit = int(bars_since_last_exit_prev + 1) if position_side == 0 else 0
            done = (t + 1) >= (n - 1)
            return (
                position_side,
                position_size,
                entry_price,
                time_in_trade,
                cooldown_left,
                equity,
                equity_peak,
                max_dd,
                dd,
                dd_increase,
                trade_peak_pnl,
                reward,
                turn,
                tx_cost,
                pnl_step,
                regret,
                1 if hard_dd_triggered else 0,
                hold_bar_pen,
                hold_excess_pen,
                hold_regret_pen,
                stagnation_pen,
                (reverse_penalty if is_reverse else 0.0),
                entry_pen,
                forced_max_hold_close,
                bars_since_last_exit,
                close_prev_trade,
                open_new_trade,
                1 if changed else 0,
                done,
            )

        _STEP_CORE_NUMBA = _impl
    try:
        return _STEP_CORE_NUMBA(*args)
    except Exception:
        return _step_core_py(*args)


def _warmup_numba_kernels() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    if not _ensure_numba():
        return
    try:
        close = np.asarray([1.0, 1.01, 1.02], dtype=np.float64)
        fwd = np.asarray([0.01, 0.009, 0.0], dtype=np.float64)
        zf = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        edge = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        strength = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        fut_min = np.asarray([-0.01, -0.01, 0.0], dtype=np.float32)
        fut_max = np.asarray([0.02, 0.02, 0.0], dtype=np.float32)
        _ = _state_vec_numba(0, 0, 0.0, 0.0, 0, 0, 1, 0.0, close, zf, zf, edge, strength, zf, zf, zf, zf, zf)
        _ = _step_core_numba(
            0,
            0,
            3,
            0,
            0.0,
            0.0,
            0,
            0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0,
            True,
            0.5,
            1.0,
            0.2,
            0.2,
            0,
            0,
            0,
            1,
            0.0005,
            0.0001,
            0.1,
            0.0,
            0.15,
            0.0,
            1.0,
            0.0,
            0.0,
            360,
            0.0,
            0.0,
            240,
            0.00005,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0005,
            0.02,
            0.0,
            close,
            fwd,
            edge,
            strength,
            fut_min,
            fut_max,
        )
        _NUMBA_WARMED = True
    except Exception:
        _NUMBA_WARMED = False


@dataclass
class TradingEnvConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    small_size: float = 0.50
    big_size: float = 1.00
    cooldown_bars: int = 3
    regret_window: int = 10
    min_reentry_gap_bars: int = 0
    min_hold_bars: int = 0
    max_hold_bars: int = 0
    use_signal_gate: bool = True
    edge_entry_threshold: float = 0.2
    strength_entry_threshold: float = 0.2
    force_close_on_adverse: bool = True
    edge_exit_threshold: float = 0.18
    strength_exit_threshold: float = 0.08
    allow_direct_flip: bool = True
    allow_scale_in: bool = False
    # Feature switches no estado (mantem state_dim fixo por compatibilidade).
    # Se False, o slot correspondente eh zerado.
    state_use_uncertainty: bool = False
    state_use_vol_long: bool = False
    state_use_shock: bool = False
    # Contexto temporal do estado (janela curta/longa em barras de 1m).
    state_use_window_context: bool = True
    state_window_short_bars: int = 30
    state_window_long_bars: int = 30
    reward: RewardConfig = field(default_factory=RewardConfig)


class HybridTradingEnv:
    """
    Ambiente discreto para camada RL sobre sinais supervisionados.

    Requer colunas no dataframe:
    - close
    - fwd_ret_1
    - mu_long_norm, mu_short_norm, edge_norm, strength_norm, uncertainty_norm
    - vol_short_norm, vol_long_norm, trend_strength_norm, shock_flag
    Obs: algumas features podem ser zeradas via TradingEnvConfig.state_use_*.
    """

    def __init__(self, df: pd.DataFrame, cfg: TradingEnvConfig | None = None):
        if df.empty:
            raise ValueError("df vazio para o ambiente RL")
        _warmup_numba_kernels()
        self.df = df.sort_index().copy()
        self.cfg = cfg or TradingEnvConfig()

        self._required_defaults = {
            "close": 0.0,
            "fwd_ret_1": 0.0,
            "mu_long_norm": 0.0,
            "mu_short_norm": 0.0,
            "edge_norm": 0.0,
            "strength_norm": 0.0,
            "uncertainty_norm": 0.0,
            "vol_short_norm": 0.0,
            "vol_long_norm": 0.0,
            "trend_strength_norm": 0.0,
            "shock_flag": 0.0,
        }
        for col, d in self._required_defaults.items():
            if col not in self.df.columns:
                self.df[col] = d
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.n = len(self.df)
        self.close = self.df["close"].to_numpy(dtype=np.float64, copy=False)
        self.fwd_ret = self.df["fwd_ret_1"].to_numpy(dtype=np.float64, copy=False)
        self.mu_long_norm = self.df["mu_long_norm"].to_numpy(dtype=np.float32, copy=False)
        self.mu_short_norm = self.df["mu_short_norm"].to_numpy(dtype=np.float32, copy=False)
        self.edge_norm = self.df["edge_norm"].to_numpy(dtype=np.float32, copy=False)
        self.strength_norm = self.df["strength_norm"].to_numpy(dtype=np.float32, copy=False)
        self.uncertainty_norm = self.df["uncertainty_norm"].to_numpy(dtype=np.float32, copy=False)
        self.vol_short_norm = self.df["vol_short_norm"].to_numpy(dtype=np.float32, copy=False)
        self.vol_long_norm = self.df["vol_long_norm"].to_numpy(dtype=np.float32, copy=False)
        self.trend_strength_norm = self.df["trend_strength_norm"].to_numpy(dtype=np.float32, copy=False)
        self.shock_flag = self.df["shock_flag"].to_numpy(dtype=np.float32, copy=False)
        if not bool(self.cfg.state_use_uncertainty):
            self.uncertainty_norm = np.zeros(self.n, dtype=np.float32)
        if not bool(self.cfg.state_use_vol_long):
            self.vol_long_norm = np.zeros(self.n, dtype=np.float32)
        if not bool(self.cfg.state_use_shock):
            self.shock_flag = np.zeros(self.n, dtype=np.float32)
        self.future_min_ret, self.future_max_ret = self._build_future_regret_arrays(int(self.cfg.regret_window))
        self._window_state_feats = self._build_window_state_features(int(self.cfg.state_window_short_bars))
        self.reset()

    def _build_future_regret_arrays(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        return _future_regret_arrays_numba(self.close, int(window))

    @property
    def state_dim(self) -> int:
        # estado minimo: mu_long, mu_short + estado da operacao
        base = 9
        if bool(self.cfg.state_use_window_context):
            return int(base + self._window_state_feats.shape[1])
        return int(base)

    def _build_window_state_features(self, w_short: int) -> np.ndarray:
        w = int(max(1, w_short))
        n = int(self.n)
        out = np.zeros((n, 2 * w), dtype=np.float32)
        mu_l = self.mu_long_norm.astype(np.float32, copy=False)
        mu_s = self.mu_short_norm.astype(np.float32, copy=False)
        for lag in range(w):
            if lag == 0:
                out[:, lag] = mu_l
                out[:, w + lag] = mu_s
            else:
                out[lag:, lag] = mu_l[:-lag]
                out[lag:, w + lag] = mu_s[:-lag]
        return out

    def reset(self, *, start_idx: int = 0) -> np.ndarray:
        self.t = int(max(0, min(start_idx, self.n - 2)))
        self.position_side = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.time_in_trade = 0
        self.cooldown_left = 0

        self.equity = 1.0
        self.equity_peak = 1.0
        self.max_dd = 0.0
        self.prev_dd = 0.0

        self.trade_peak_pnl = 0.0
        self.trade_returns: list[float] = []
        self.trade_holds_bars: list[int] = []
        self.trade_holds_long_bars: list[int] = []
        self.trade_holds_short_bars: list[int] = []
        self.trade_peak_returns: list[float] = []
        self.trade_givebacks: list[float] = []
        self.trade_efficiencies: list[float] = []
        self.entry_indices: list[int] = []
        self.current_entry_t: int = -1
        self.bars_since_last_exit: int = int(max(0, self.cfg.min_reentry_gap_bars))
        self.turnover_history: list[float] = []
        self.reward_history: list[float] = []
        self.equity_curve: list[float] = [1.0]
        return self._state()

    def _unrealized(self, t: int) -> float:
        if self.position_side == 0 or self.entry_price <= 0.0:
            return 0.0
        px = float(self.close[t])
        if px <= 0.0:
            return 0.0
        r = ((px / self.entry_price) - 1.0) * float(self.position_side) * float(self.position_size)
        return float(r)

    def _close_trade(self, t: int, side: int, size: float) -> None:
        if side == 0 or self.entry_price <= 0.0:
            return
        px = float(self.close[t])
        if px <= 0.0:
            return
        r = ((px / self.entry_price) - 1.0) * float(side) * float(size)
        self.trade_returns.append(float(r))
        peak = float(max(0.0, self.trade_peak_pnl))
        giveback = float(max(0.0, peak - float(r))) if peak > 0.0 else 0.0
        eff = float(r / peak) if peak > 1e-12 else 0.0
        self.trade_peak_returns.append(float(peak))
        self.trade_givebacks.append(float(giveback))
        self.trade_efficiencies.append(float(np.clip(eff, -3.0, 3.0)))
        if self.current_entry_t >= 0:
            hold = int(max(1, int(t) - int(self.current_entry_t)))
            self.trade_holds_bars.append(hold)
            if int(side) > 0:
                self.trade_holds_long_bars.append(hold)
            elif int(side) < 0:
                self.trade_holds_short_bars.append(hold)
        self.current_entry_t = -1

    def _state(self) -> np.ndarray:
        base_state = _state_vec_numba(
            int(self.t),
            int(self.position_side),
            float(self.position_size),
            float(self.entry_price),
            int(self.time_in_trade),
            int(self.cooldown_left),
            int(self.cfg.cooldown_bars),
            float(self.trade_peak_pnl),
            self.close,
            self.mu_long_norm,
            self.mu_short_norm,
            self.edge_norm,
            self.strength_norm,
            self.uncertainty_norm,
            self.vol_short_norm,
            self.vol_long_norm,
            self.trend_strength_norm,
            self.shock_flag,
        )
        # estado minimo: [mu_long, mu_short, pos_flat, pos_long, pos_short, t_norm, unr_norm, dd_trade, cooldown]
        base_state = base_state[[0, 1, 9, 10, 11, 13, 14, 15, 16]]
        if not bool(self.cfg.state_use_window_context):
            return base_state
        t = int(max(0, min(self.t, self.n - 1)))
        ctx = self._window_state_feats[t]
        return np.concatenate((base_state, ctx), axis=0).astype(np.float32, copy=False)

    def valid_actions(self) -> list[int]:
        """
        Mascara de acoes validas no estado atual para reduzir exploracao inviavel.
        """
        out = [ACTION_HOLD]
        t = int(self.t)
        side = int(self.position_side)
        can_adjust = bool(int(self.time_in_trade) >= int(max(0, self.cfg.min_hold_bars)))

        if side == 0:
            if self.cooldown_left > 0:
                return out
            if int(self.bars_since_last_exit) < int(max(0, self.cfg.min_reentry_gap_bars)):
                return out
            if not bool(self.cfg.use_signal_gate):
                out.extend([ACTION_OPEN_LONG_SMALL, ACTION_OPEN_SHORT_SMALL])
                return out
            strength_now = float(self.strength_norm[t])
            edge_now = float(self.edge_norm[t])
            if strength_now < float(self.cfg.strength_entry_threshold):
                return out
            if edge_now >= float(self.cfg.edge_entry_threshold):
                out.extend([ACTION_OPEN_LONG_SMALL])
                return out
            if edge_now <= -float(self.cfg.edge_entry_threshold):
                out.extend([ACTION_OPEN_SHORT_SMALL])
                return out
            return out

        if side > 0:
            if not can_adjust:
                return out
            edge_now = float(self.edge_norm[t])
            strength_now = float(self.strength_norm[t])
            flip_edge_thr = float(1.25 * float(self.cfg.edge_entry_threshold))
            flip_strength_thr = float(1.05 * float(self.cfg.strength_entry_threshold))
            flip_ok = bool(
                self.cfg.allow_direct_flip
                and (
                    (not bool(self.cfg.use_signal_gate))
                    or (
                        strength_now >= flip_strength_thr
                        and edge_now <= -flip_edge_thr
                    )
                )
            )
            adverse = False
            if bool(self.cfg.force_close_on_adverse):
                edge_thr = float(max(0.0, self.cfg.edge_exit_threshold))
                strength_weak = float(max(self.cfg.strength_exit_threshold, 0.85 * self.cfg.strength_entry_threshold))
                adverse = bool(
                    (edge_now <= -max(0.25, 1.6 * edge_thr))
                    or ((edge_now <= -edge_thr) and (strength_now <= strength_weak))
                )
                if adverse:
                    out_adv = [ACTION_HOLD, ACTION_CLOSE_LONG]
                    if flip_ok:
                        out_adv.append(ACTION_OPEN_SHORT_SMALL)
                    return out_adv
            if int(self.cfg.max_hold_bars) > 0 and int(self.time_in_trade) >= int(self.cfg.max_hold_bars):
                return [ACTION_CLOSE_LONG]
            out.append(ACTION_CLOSE_LONG)
            if flip_ok:
                out.extend([ACTION_OPEN_SHORT_SMALL])
            if bool(self.cfg.allow_scale_in) and (not adverse):
                out.extend([ACTION_OPEN_LONG_SMALL])
            return out

        if not can_adjust:
            return out
        edge_now = float(self.edge_norm[t])
        strength_now = float(self.strength_norm[t])
        flip_edge_thr = float(1.25 * float(self.cfg.edge_entry_threshold))
        flip_strength_thr = float(1.05 * float(self.cfg.strength_entry_threshold))
        flip_ok = bool(
            self.cfg.allow_direct_flip
            and (
                (not bool(self.cfg.use_signal_gate))
                or (
                    strength_now >= flip_strength_thr
                    and edge_now >= flip_edge_thr
                )
            )
        )
        adverse = False
        if bool(self.cfg.force_close_on_adverse):
            edge_thr = float(max(0.0, self.cfg.edge_exit_threshold))
            strength_weak = float(max(self.cfg.strength_exit_threshold, 0.85 * self.cfg.strength_entry_threshold))
            adverse = bool(
                (edge_now >= max(0.25, 1.6 * edge_thr))
                or ((edge_now >= edge_thr) and (strength_now <= strength_weak))
            )
            if adverse:
                out_adv = [ACTION_HOLD, ACTION_CLOSE_SHORT]
                if flip_ok:
                    out_adv.append(ACTION_OPEN_LONG_SMALL)
                return out_adv
        if int(self.cfg.max_hold_bars) > 0 and int(self.time_in_trade) >= int(self.cfg.max_hold_bars):
            return [ACTION_CLOSE_SHORT]
        out.append(ACTION_CLOSE_SHORT)
        if flip_ok:
            out.extend([ACTION_OPEN_LONG_SMALL])
        if bool(self.cfg.allow_scale_in) and (not adverse):
            out.extend([ACTION_OPEN_SHORT_SMALL])
        return out

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.t >= self.n - 1:
            t_last = int(max(0, min(self.t, self.n - 1)))
            return self._state(), 0.0, True, {
                "done_reason": "eod",
                "t_before": int(self.t),
                "t_after": int(self.t),
                "event_ts": pd.to_datetime(self.df.index[t_last]),
                "close_prev_trade": 0,
                "open_new_trade": 0,
                "forced_eod_close": 0,
            }
        t_before = int(self.t)
        prev_side = int(self.position_side)
        prev_size = float(self.position_size)
        prev_entry_price = float(self.entry_price)
        out = _step_core_numba(
            int(action),
            int(self.t),
            int(self.n),
            int(prev_side),
            float(prev_size),
            float(prev_entry_price),
            int(self.time_in_trade),
            int(self.cooldown_left),
            float(self.equity),
            float(self.equity_peak),
            float(self.max_dd),
            float(self.prev_dd),
            float(self.trade_peak_pnl),
            int(self.bars_since_last_exit),
            bool(self.cfg.use_signal_gate),
            float(self.cfg.small_size),
            float(self.cfg.big_size),
            float(self.cfg.edge_entry_threshold),
            float(self.cfg.strength_entry_threshold),
            int(self.cfg.min_reentry_gap_bars),
            int(self.cfg.min_hold_bars),
            int(self.cfg.max_hold_bars),
            int(self.cfg.cooldown_bars),
            float(self.cfg.fee_rate),
            float(self.cfg.slippage_rate),
            float(self.cfg.reward.dd_penalty),
            float(self.cfg.reward.dd_level_penalty),
            float(self.cfg.reward.dd_soft_limit),
            float(self.cfg.reward.dd_excess_penalty),
            float(self.cfg.reward.dd_hard_limit),
            float(self.cfg.reward.dd_hard_penalty),
            float(self.cfg.reward.hold_bar_penalty),
            int(self.cfg.reward.hold_soft_bars),
            float(self.cfg.reward.hold_excess_penalty),
            float(self.cfg.reward.hold_regret_penalty),
            int(self.cfg.reward.stagnation_bars),
            float(self.cfg.reward.stagnation_ret_epsilon),
            float(self.cfg.reward.stagnation_penalty),
            float(self.cfg.reward.reverse_penalty),
            float(self.cfg.reward.entry_penalty),
            float(self.cfg.reward.weak_entry_penalty),
            float(self.cfg.reward.turnover_penalty),
            float(self.cfg.reward.regret_penalty),
            float(self.cfg.reward.idle_penalty),
            self.close,
            self.fwd_ret,
            self.edge_norm,
            self.strength_norm,
            self.future_min_ret,
            self.future_max_ret,
        )
        (
            new_position_side,
            new_position_size,
            new_entry_price,
            new_time_in_trade,
            new_cooldown_left,
            new_equity,
            new_equity_peak,
            new_max_dd,
            dd,
            dd_increase,
            new_trade_peak_pnl,
            reward,
            turn,
            tx_cost,
            pnl_step,
            regret,
            hard_dd_triggered,
            hold_bar_pen,
            hold_excess_pen,
            hold_regret_pen,
            stagnation_pen,
            reverse_pen,
            entry_pen,
            forced_max_hold_close,
            new_bars_since_last_exit,
            close_prev_trade,
            open_new_trade,
            changed,
            done,
        ) = out

        if int(close_prev_trade) == 1:
            self.entry_price = float(prev_entry_price)
            self._close_trade(self.t, prev_side, prev_size)

        self.position_side = int(new_position_side)
        self.position_size = float(new_position_size)
        self.entry_price = float(new_entry_price)
        self.time_in_trade = int(new_time_in_trade)
        self.cooldown_left = int(new_cooldown_left)
        self.equity = float(new_equity)
        self.equity_peak = float(new_equity_peak)
        self.max_dd = float(new_max_dd)
        self.trade_peak_pnl = float(new_trade_peak_pnl)
        self.bars_since_last_exit = int(new_bars_since_last_exit)

        if int(open_new_trade) == 1:
            self.entry_indices.append(int(self.t))
            self.current_entry_t = int(self.t)
            self.bars_since_last_exit = 0
        elif int(self.position_side) == 0:
            self.current_entry_t = -1

        self.turnover_history.append(float(turn))
        self.reward_history.append(float(reward))
        self.equity_curve.append(float(self.equity))

        self.prev_dd = float(dd)

        self.t = int(self.t + 1)
        done = bool(done or (self.t >= (self.n - 1)))
        forced_eod_close = 0
        forced_exit_ts = None
        forced_exit_price = 0.0
        if done and self.position_side != 0:
            forced_eod_close = 1
            t_exit = int(max(0, min(self.t, self.n - 1)))
            forced_exit_ts = pd.to_datetime(self.df.index[t_exit])
            forced_exit_price = float(self.close[t_exit]) if t_exit < len(self.close) else 0.0
            self._close_trade(self.t, self.position_side, self.position_size)
            self.position_side = 0
            self.position_size = 0.0
            self.entry_price = 0.0
            self.time_in_trade = 0
            self.trade_peak_pnl = 0.0
            self.current_entry_t = -1
            self.bars_since_last_exit = int(self.bars_since_last_exit + 1)

        components = {
            "delta_equity": float(self.equity_curve[-1] - self.equity_curve[-2]) if len(self.equity_curve) >= 2 else 0.0,
            "turnover_pen": float(self.cfg.reward.turnover_penalty) * float(turn),
            "dd_pen": float(self.cfg.reward.dd_penalty) * float(dd_increase),
            "dd_level_pen": float(self.cfg.reward.dd_level_penalty) * float(dd),
            "dd_excess_pen": float(self.cfg.reward.dd_excess_penalty) * float(max(0.0, dd - float(self.cfg.reward.dd_soft_limit)) ** 2),
            "dd_hard_pen": float(self.cfg.reward.dd_hard_penalty) if int(hard_dd_triggered) == 1 else 0.0,
            "hold_bar_pen": float(hold_bar_pen),
            "hold_excess_pen": float(hold_excess_pen),
            "hold_regret_pen": float(hold_regret_pen),
            "stagnation_pen": float(stagnation_pen),
            "reverse_pen": float(reverse_pen),
            "entry_pen": float(entry_pen),
            "regret_pen": float(self.cfg.reward.regret_penalty) * float(regret),
            "idle_pen": float(self.cfg.reward.idle_penalty) if (self.position_side == 0 and float(self.strength_norm[max(0, self.t - 1)]) > 1.0) else 0.0,
        }
        info = {
            "t_before": int(t_before),
            "t_after": int(self.t),
            "event_ts": pd.to_datetime(self.df.index[min(t_before, self.n - 1)]),
            "close_prev_trade": int(close_prev_trade),
            "open_new_trade": int(open_new_trade),
            "changed": int(changed),
            "hard_dd_triggered": int(hard_dd_triggered),
            "forced_max_hold_close": int(forced_max_hold_close),
            "forced_eod_close": int(forced_eod_close),
            "forced_exit_ts": forced_exit_ts,
            "forced_exit_price": float(forced_exit_price),
            "prev_position_side": int(prev_side),
            "prev_position_size": float(prev_size),
            "new_position_side": int(self.position_side),
            "new_position_size": float(self.position_size),
            "entry_price_before": float(prev_entry_price),
            "entry_price_after": float(self.entry_price),
            "turnover": float(turn),
            "transaction_cost": float(tx_cost),
            "pnl_step": float(pnl_step),
            "equity": float(self.equity),
            "drawdown": float(dd),
            "regret": float(regret),
            "reward_components": components,
        }
        return self._state(), float(reward), bool(done), info

    def summary(self) -> dict[str, float]:
        tr = np.asarray(self.trade_returns, dtype=np.float64)
        wins = tr[tr > 0]
        losses = -tr[tr <= 0]
        pf = float(wins.sum() / max(1e-12, losses.sum())) if losses.size else float("inf")
        win_rate = float(wins.size / max(1, tr.size))
        avg_turn = float(np.mean(self.turnover_history)) if self.turnover_history else 0.0
        if len(self.df.index) >= 2:
            dt_min = float((pd.to_datetime(self.df.index[1]) - pd.to_datetime(self.df.index[0])).total_seconds() / 60.0)
            if not np.isfinite(dt_min) or dt_min <= 0.0:
                dt_min = 1.0
        else:
            dt_min = 1.0
        total_days = float(max(1e-9, (len(self.df) * dt_min) / (60.0 * 24.0)))
        trades = float(tr.size)
        trades_per_day = float(trades / total_days)
        trades_per_week = float(trades_per_day * 7.0)
        avg_hold_bars = float(np.mean(self.trade_holds_bars)) if self.trade_holds_bars else 0.0
        avg_hold_hours = float(avg_hold_bars * (dt_min / 60.0))
        avg_hold_long_bars = float(np.mean(self.trade_holds_long_bars)) if self.trade_holds_long_bars else 0.0
        avg_hold_short_bars = float(np.mean(self.trade_holds_short_bars)) if self.trade_holds_short_bars else 0.0
        avg_hold_long_hours = float(avg_hold_long_bars * (dt_min / 60.0))
        avg_hold_short_hours = float(avg_hold_short_bars * (dt_min / 60.0))
        n_long = float(len(self.trade_holds_long_bars))
        n_short = float(len(self.trade_holds_short_bars))
        avg_peak_ret = float(np.mean(self.trade_peak_returns)) if self.trade_peak_returns else 0.0
        avg_giveback = float(np.mean(self.trade_givebacks)) if self.trade_givebacks else 0.0
        trade_eff_mean = float(np.mean(self.trade_efficiencies)) if self.trade_efficiencies else 0.0
        return {
            "equity_end": float(self.equity),
            "ret_total": float(self.equity - 1.0),
            "max_dd": float(self.max_dd),
            "trades": trades,
            "trades_per_day": trades_per_day,
            "trades_per_week": trades_per_week,
            "avg_hold_bars": avg_hold_bars,
            "avg_hold_hours": avg_hold_hours,
            "trades_long": n_long,
            "trades_short": n_short,
            "avg_hold_long_bars": avg_hold_long_bars,
            "avg_hold_short_bars": avg_hold_short_bars,
            "avg_hold_long_hours": avg_hold_long_hours,
            "avg_hold_short_hours": avg_hold_short_hours,
            "avg_peak_ret": avg_peak_ret,
            "avg_giveback": avg_giveback,
            "trade_efficiency_mean": trade_eff_mean,
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "avg_turnover": float(avg_turn),
        }
