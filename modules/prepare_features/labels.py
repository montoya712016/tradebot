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


def apply_trade_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
) -> pd.DataFrame:
    """
    Gera labels binarios baseados no contrato fixo (sniper).
    Entry_label usa a mesma logica de exit (EMA) e exige que o preco nunca caia abaixo do entry.

    Novas colunas:
        - sniper_entry_label (uint8)  [janela unica para compat]
        - sniper_mae_pct (float32)
        - sniper_exit_code (int8)
        - sniper_exit_wait_bars (int32)
        - sniper_entry_label_{Xm} (uint8) para cada janela (ex.: 30m/120m/360m)
        - sniper_entry_weight_{Xm} (float32) para cada janela
        - sniper_mae_pct_{Xm} (float32)
        - sniper_exit_code_{Xm} (int8)
        - sniper_exit_wait_bars_{Xm} (int32)
        - sniper_entry_weight_{Xm} (float32)
    """
    if contract is None:
        contract = DEFAULT_TRADE_CONTRACT

    candle_seconds = candle_sec or contract.timeframe_sec
    candle_seconds = max(1, int(candle_seconds))

    close = df["close"].to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)

    # labels (ema exit com span ~ janela)
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    profits = list(getattr(contract, "entry_label_min_profit_pcts", []) or [])
    # VALIDATE_ENTRY_WINDOWS
    if len(windows) < 1 or len(profits) < 1 or len(windows) != len(profits):
        raise ValueError("entry_label_windows_minutes e entry_label_min_profit_pcts devem ter o mesmo tamanho (>=1)")
    labels_by_win = []
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
        suffix = f"{int(w_min)}m"
        df[f"sniper_entry_label_{suffix}"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
        df[f"sniper_mae_pct_{suffix}"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
        df[f"sniper_exit_code_{suffix}"] = pd.Series(exit_code.astype(np.int8), index=df.index)
        df[f"sniper_exit_wait_bars_{suffix}"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
        df[f"sniper_entry_weight_{suffix}"] = pd.Series(weight.astype(np.float32), index=df.index)
        labels_by_win.append((suffix, entry_label, mae_pct, exit_code, exit_wait, weight))

    # compat: usa a primeira janela para colunas legadas
    suffix, entry_label, mae_pct, exit_code, exit_wait, weight = labels_by_win[0]
    df["sniper_entry_label"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
    df["sniper_mae_pct"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
    df["sniper_exit_code"] = pd.Series(exit_code.astype(np.int8), index=df.index)
    df["sniper_exit_wait_bars"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
    if f"sniper_entry_weight_{suffix}" in df.columns:
        df["sniper_entry_weight"] = df[f"sniper_entry_weight_{suffix}"].astype(np.float32)
    return df
