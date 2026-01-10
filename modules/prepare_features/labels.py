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
    min_hold_bars: int,
    min_profit_pct: float,
    sl_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    labels = np.zeros(n, np.uint8)
    mae = np.zeros(n, np.float32)
    exit_code = np.zeros(n, np.int8)
    exit_wait = np.zeros(n, np.int32)
    if horizon_bars < 1:
        horizon_bars = 1
    min_profit_mult = 1.0 + min_profit_pct

    # prefix sum para media futura
    cs = np.empty(n + 1, np.float64)
    cs[0] = 0.0
    for i in range(n):
        v = close[i]
        if not np.isfinite(v):
            v = 0.0
        cs[i + 1] = cs[i] + v

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
        sl_hit = False
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
                sl_hit = True
                exit_bar = j
                break

        if sl_hit:
            label = 0
            code = -1
        else:
            count = last_bar - i
            if count <= 0:
                label = 0
                code = -3
            else:
                mean_future = (cs[last_bar + 1] - cs[i + 1]) / float(count)
                if mean_future >= (px0 * min_profit_mult):
                    label = 1
                    code = 1
                else:
                    label = 0
                    code = -3

        labels[i] = label
        mae[i] = best_mae
        exit_code[i] = code
        exit_wait[i] = max(0, exit_bar - i)

    return labels, mae, exit_code, exit_wait


@njit(cache=True)
def _simulate_danger_label_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon_bars: int,
    drop_pct: float,
    recovery_pct: float,
    stabilize_recovery_pct: float,
    stabilize_bars: int,
) -> np.ndarray:
    """
    Danger label (Sniper):
    - Intenção: servir como um filtro de entrada (evitar entrar antes de queda rápida).
    - Label=1 se, a partir do tempo i, ocorrer uma queda de pelo menos `drop_pct`
      dentro de `horizon_bars`.
    """
    n = close.size
    labels = np.zeros(n, np.uint8)
    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue

        drop_price = px0 * (1.0 - drop_pct)
        last_bar = i + horizon_bars
        if last_bar >= n:
            last_bar = n - 1
        triggered = 0
        for j in range(i + 1, last_bar + 1):
            lo = low[j]
            if not np.isfinite(lo):
                lo = close[j]
            if lo <= drop_price:
                triggered = 1
                break
        labels[i] = triggered

    return labels


def apply_trade_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
) -> pd.DataFrame:
    """
    Gera labels binários baseados no contrato fixo (sniper).
    Entry_label considera lucro minimo na media futura (sem TP de saida).

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
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)

    timeout_bars = contract.timeout_bars(candle_seconds)
    horizon_bars = contract.entry_horizon_bars(candle_seconds)
    danger_bars = contract.danger_horizon_bars(candle_seconds)
    min_hold_bars = contract.min_hold_bars(candle_seconds)
    dd_limit = contract.dd_intermediate_limit_pct or contract.sl_pct
    min_profit_pct = float(max(contract.entry_min_profit_pct, contract.sl_pct))

    entry_label, mae_pct, exit_code, exit_wait = _simulate_entry_contract_numba(
        close,
        low,
        int(horizon_bars),
        int(min_hold_bars),
        float(min_profit_pct),
        float(contract.sl_pct),
    )
    danger_label = _simulate_danger_label_numba(
        close,
        high,
        low,
        int(danger_bars),
        float(contract.danger_drop_pct),
        float(contract.danger_recovery_pct),
        float(getattr(contract, "danger_stabilize_recovery_pct", 0.01)),
        int(getattr(contract, "danger_stabilize_bars", 30)),
    )

    df["sniper_entry_label"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
    df["sniper_mae_pct"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
    df["sniper_exit_code"] = pd.Series(exit_code.astype(np.int8), index=df.index)
    df["sniper_exit_wait_bars"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
    df["sniper_danger_label"] = pd.Series(danger_label.astype(np.uint8), index=df.index)
    return df
