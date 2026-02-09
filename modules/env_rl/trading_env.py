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
    cooldown_bars: int,
    fee_rate: float,
    slippage_rate: float,
    dd_penalty: float,
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
        close_req = bool((a == 5 and prev_side > 0) or (a == 6 and prev_side < 0))
        if (not close_req) and (target_side != prev_side or abs(target_size - prev_size) > 1e-12):
            target_side, target_size = prev_side, prev_size

    changed = (target_side != prev_side) or (abs(target_size - prev_size) > 1e-12)
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
    open_new_trade = int(prev_side == 0 and target_side != 0)
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

    if position_side != 0:
        base_tit = 0 if open_new_trade == 1 else int(time_in_trade_prev)
        time_in_trade = int(base_tit + 1)
        next_idx = t + 1 if (t + 1) < n else t
        if entry_price > 0.0 and close[next_idx] > 0.0:
            unr = ((float(close[next_idx]) / float(entry_price)) - 1.0) * float(position_side) * float(position_size)
        else:
            unr = 0.0
        base_peak = 0.0 if open_new_trade == 1 else float(trade_peak_prev)
        trade_peak_pnl = float(max(base_peak, unr))
    else:
        time_in_trade = 0
        trade_peak_pnl = 0.0

    regret = 0.0
    if prev_side == 0 and position_side != 0:
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
    reward = float(delta_equity - float(turnover_penalty) * turn - float(dd_penalty) * dd_increase - float(regret_penalty) * regret - idle_pen)

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
            cooldown_bars: int,
            fee_rate: float,
            slippage_rate: float,
            dd_penalty: float,
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
                close_req = (action == 5 and prev_side > 0) or (action == 6 and prev_side < 0)
                if (not close_req) and ((target_side != prev_side) or (np.abs(target_size - prev_size) > 1e-12)):
                    target_side, target_size = prev_side, prev_size

            changed = (target_side != prev_side) or (np.abs(target_size - prev_size) > 1e-12)
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
            open_new_trade = 1 if (prev_side == 0 and target_side != 0) else 0

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

            if position_side != 0:
                base_tit = 0 if open_new_trade == 1 else int(time_in_trade_prev)
                time_in_trade = int(base_tit + 1)
                next_idx = t + 1 if (t + 1) < n else t
                if entry_price > 0.0 and close[next_idx] > 0.0:
                    unr = ((float(close[next_idx]) / float(entry_price)) - 1.0) * float(position_side) * float(position_size)
                else:
                    unr = 0.0
                base_peak = 0.0 if open_new_trade == 1 else float(trade_peak_prev)
                trade_peak_pnl = float(base_peak if base_peak > unr else unr)
            else:
                time_in_trade = 0
                trade_peak_pnl = 0.0

            regret = 0.0
            if prev_side == 0 and position_side != 0:
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
            reward = float(delta_equity - turnover_penalty * turn - dd_penalty * dd_increase - regret_penalty * regret - idle_pen)

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
            1,
            0.0005,
            0.0001,
            0.1,
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
    use_signal_gate: bool = True
    edge_entry_threshold: float = 0.2
    strength_entry_threshold: float = 0.2
    reward: RewardConfig = field(default_factory=RewardConfig)


class HybridTradingEnv:
    """
    Ambiente discreto para camada RL sobre sinais supervisionados.

    Requer colunas no dataframe:
    - close
    - fwd_ret_1
    - mu_long_norm, mu_short_norm, edge_norm, strength_norm, uncertainty_norm
    - vol_short_norm, vol_long_norm, trend_strength_norm, shock_flag
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

        self.n = len(self.df)
        self.future_min_ret, self.future_max_ret = self._build_future_regret_arrays(int(self.cfg.regret_window))
        self.reset()

    def _build_future_regret_arrays(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        return _future_regret_arrays_numba(self.close, int(window))

    @property
    def state_dim(self) -> int:
        return 17

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
        if self.current_entry_t >= 0:
            hold = int(max(1, int(t) - int(self.current_entry_t)))
            self.trade_holds_bars.append(hold)
        self.current_entry_t = -1

    def _state(self) -> np.ndarray:
        return _state_vec_numba(
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
                out.extend([ACTION_OPEN_LONG_SMALL, ACTION_OPEN_LONG_BIG, ACTION_OPEN_SHORT_SMALL, ACTION_OPEN_SHORT_BIG])
                return out
            strength_now = float(self.strength_norm[t])
            edge_now = float(self.edge_norm[t])
            if strength_now < float(self.cfg.strength_entry_threshold):
                return out
            if edge_now >= float(self.cfg.edge_entry_threshold):
                out.extend([ACTION_OPEN_LONG_SMALL, ACTION_OPEN_LONG_BIG])
                return out
            if edge_now <= -float(self.cfg.edge_entry_threshold):
                out.extend([ACTION_OPEN_SHORT_SMALL, ACTION_OPEN_SHORT_BIG])
                return out
            return out

        if side > 0:
            out.append(ACTION_CLOSE_LONG)
            if can_adjust:
                out.extend([ACTION_OPEN_LONG_SMALL, ACTION_OPEN_LONG_BIG])
            return out

        out.append(ACTION_CLOSE_SHORT)
        if can_adjust:
            out.extend([ACTION_OPEN_SHORT_SMALL, ACTION_OPEN_SHORT_BIG])
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
            int(self.cfg.cooldown_bars),
            float(self.cfg.fee_rate),
            float(self.cfg.slippage_rate),
            float(self.cfg.reward.dd_penalty),
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
        return {
            "equity_end": float(self.equity),
            "ret_total": float(self.equity - 1.0),
            "max_dd": float(self.max_dd),
            "trades": trades,
            "trades_per_day": trades_per_day,
            "trades_per_week": trades_per_week,
            "avg_hold_bars": avg_hold_bars,
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "avg_turnover": float(avg_turn),
        }
