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
    danger_label: np.ndarray,
    horizon_bars: int,
    min_hold_bars: int,
    min_profit_pct: float,
    sl_pct: float,
    time_per_bar_hours: float,
    exit_score_threshold: float,
    exit_score_w_time: float,
    exit_score_w_pnl: float,
    exit_score_w_dd: float,
    exit_score_w_danger: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    labels = np.zeros(n, np.uint8)
    mae = np.zeros(n, np.float32)
    exit_code = np.zeros(n, np.int8)
    exit_wait = np.zeros(n, np.int32)
    if horizon_bars < 1:
        horizon_bars = 1

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
        last_bar = i + horizon_bars
        if last_bar >= n:
            last_bar = n - 1
        min_bar = i + min_hold_bars
        if min_bar > last_bar:
            labels[i] = 0
            mae[i] = best_mae
            exit_code[i] = -4
            exit_wait[i] = max(0, last_bar - i)
            continue

        sl_price = px0 * (1.0 - sl_pct)
        code = -3
        exit_bar = last_bar
        for j in range(i + 1, last_bar + 1):
            lo = low[j]
            if not np.isfinite(lo):
                lo = close[j]
            if lo < worst:
                worst = lo
            cur_mae = (worst / px0) - 1.0
            if cur_mae < best_mae:
                best_mae = cur_mae
            if lo <= sl_price:
                code = -1
                exit_bar = j
                break
            if j >= min_bar:
                px = close[j]
                if not np.isfinite(px) or px <= 0.0:
                    continue
                pnl_pct = ((px / px0) - 1.0) * 100.0
                if pnl_pct < 0.0:
                    pnl_pct = 0.0
                dd_pct = ((px0 - lo) / px0) * 100.0 if lo < px0 else 0.0
                t_hours = float(j - i) * time_per_bar_hours
                score = (exit_score_w_time * t_hours) + (exit_score_w_pnl * pnl_pct) + (exit_score_w_dd * dd_pct)
                if danger_label[j] != 0:
                    score += exit_score_w_danger
                if score >= exit_score_threshold:
                    code = 1
                    exit_bar = j
                    break

        exit_px = close[exit_bar]
        if not np.isfinite(exit_px) or exit_px <= 0.0:
            exit_px = px0
        if code == -1:
            exit_px = sl_price
        r = (exit_px / px0) - 1.0
        if r >= min_profit_pct:
            label = 1
        else:
            label = 0

        labels[i] = label
        mae[i] = best_mae
        exit_code[i] = code
        exit_wait[i] = max(0, exit_bar - i)

    return labels, mae, exit_code, exit_wait


@njit(cache=True)
def _simulate_danger_label_numba(
    close: np.ndarray,
    low: np.ndarray,
    horizon_bars: int,
    drop_pct: float,
    fast_bars: int,
    critical_drop_pct: float,
) -> np.ndarray:
    """
    Danger label (Sniper):
    - Intenção: servir como um filtro de entrada (evitar entrar antes de queda forte e rápida).
    - Label=1 se o MÍNIMO da janela futura (até `fast_bars`) cair >= `critical_drop_pct`.
    """
    n = close.size
    labels = np.zeros(n, np.uint8)
    if horizon_bars < 1:
        horizon_bars = 1
    if fast_bars < 1:
        fast_bars = 1
    if not np.isfinite(critical_drop_pct) or critical_drop_pct <= 0.0:
        critical_drop_pct = drop_pct
    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue

        last_bar = i + horizon_bars
        if last_bar >= n:
            last_bar = n - 1
        fast_last = i + fast_bars
        if fast_last > last_bar:
            fast_last = last_bar
        drop_price = px0 * (1.0 - critical_drop_pct)
        triggered = 0
        min_low = px0
        for j in range(i + 1, fast_last + 1):
            lo = low[j]
            if not np.isfinite(lo):
                lo = close[j]
            if lo < min_low:
                min_low = lo
        if min_low <= drop_price:
            triggered = 1
        labels[i] = triggered

    return labels


def apply_trade_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
) -> pd.DataFrame:
    """
    Gera labels binarios baseados no contrato fixo (sniper).
    Entry_label usa a mesma logica de exit (score) para evitar entradas perdedoras.

    Novas colunas:
        - sniper_entry_label (uint8)
        - sniper_mae_pct (float32)
        - sniper_exit_code (int8)
        - sniper_exit_wait_bars (int32)
        - sniper_danger_label (uint8)
    """
    if contract is None:
        contract = DEFAULT_TRADE_CONTRACT

    candle_seconds = candle_sec or contract.timeframe_sec
    candle_seconds = max(1, int(candle_seconds))

    close = df["close"].to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)

    horizon_bars = contract.entry_horizon_bars(candle_seconds)
    danger_bars = contract.danger_horizon_bars(candle_seconds)
    min_hold_bars = contract.min_hold_bars(candle_seconds)
    min_profit_pct = float(max(contract.entry_min_profit_pct, contract.sl_pct))

    danger_label = _simulate_danger_label_numba(
        close,
        low,
        int(danger_bars),
        float(contract.danger_drop_pct),
        int(max(1, round((float(getattr(contract, "danger_fast_minutes", 60.0)) * 60.0) / float(candle_seconds)))),
        float(getattr(contract, "danger_drop_pct_critical", contract.danger_drop_pct)),
    )
    entry_label, mae_pct, exit_code, exit_wait = _simulate_entry_contract_numba(
        close,
        low,
        danger_label,
        int(horizon_bars),
        int(min_hold_bars),
        float(min_profit_pct),
        float(contract.sl_pct),
        float(candle_seconds) / 3600.0,
        float(getattr(contract, "exit_score_threshold", 10.0)),
        float(getattr(contract, "exit_score_w_time", 1.0)),
        float(getattr(contract, "exit_score_w_pnl", 1.0)),
        float(getattr(contract, "exit_score_w_dd", 2.0)),
        float(getattr(contract, "exit_score_w_danger", 4.0)),
    )

    df["sniper_entry_label"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
    df["sniper_mae_pct"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
    df["sniper_exit_code"] = pd.Series(exit_code.astype(np.int8), index=df.index)
    df["sniper_exit_wait_bars"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
    df["sniper_danger_label"] = pd.Series(danger_label.astype(np.uint8), index=df.index)
    return df
