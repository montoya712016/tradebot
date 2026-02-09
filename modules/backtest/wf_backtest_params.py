# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestOptimizedRanges:
    # parametros otimizados pelo wf_random_loop
    # regressao: threshold em retorno previsto (ex.: 0.002 = 0.2%)
    tau_entry_range: tuple[float, float, float] = (0.001, 0.02, 0.001)

    # backtest config (varied per backtest)
    max_positions_range: tuple[int, int, int] = (15, 30, 3)
    total_exposure_range: tuple[float, float, float] = (0.50, 1.00, 0.10)
    max_trade_exposure_range: tuple[float, float, float] = (0.05, 0.10, 0.01)
    min_trade_exposure_range: tuple[float, float, float] = (0.02, 0.05, 0.01)
    exit_confirm_bars_range: tuple[int, int, int] = (1, 3, 1)
    universe_history_days_range: tuple[int, int, int] = (365, 730, 90)
    universe_min_pf_range: tuple[float, float, float] = (1.0, 1.20, 0.05)
    universe_min_win_range: tuple[float, float, float] = (0.30, 0.45, 0.05)
    universe_max_dd_range: tuple[float, float, float] = (0.70, 1.00, 0.10)


__all__ = [
    "BacktestOptimizedRanges",
]
