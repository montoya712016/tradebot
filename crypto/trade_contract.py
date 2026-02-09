# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass(frozen=True)
class TradeContract:
    """
    Define um contrato fixo de trade usado tanto na geração de labels quanto nas simulações.
    """

    timeframe_sec: int = 60
    entry_label_windows_minutes: Tuple[int, ...] = (360,)
    exit_ema_span: int = 360
    exit_ema_init_offset_pct: float = 0.01
    fee_pct_per_side: float = 0.0005
    slippage_pct: float = 0.0
    max_adds: int = 0
    add_spacing_pct: float = 0.0
    add_sizing: Tuple[float, ...] = field(default_factory=lambda: (1.0,))
    risk_max_cycle_pct: float = 0.0
    dd_intermediate_limit_pct: float = 0.0
    danger_drop_pct: float = 0.0
    danger_recovery_pct: float = 0.0
    danger_timeout_hours: float = 0.0
    danger_fast_minutes: float = 0.0
    danger_drop_pct_critical: float = 0.0
    danger_stabilize_recovery_pct: float = 0.0
    danger_stabilize_bars: int = 0
    # Stocks-only rule (ignored in crypto defaults)
    forbid_exit_on_gap: bool = False
    gap_hours_forbidden: float = 0.0

    def danger_horizon_bars(self, candle_sec: int) -> int:
        hours = self.danger_timeout_hours
        if hours <= 0:
            return 0
        bars = int(round((hours * 3600.0) / float(max(1, candle_sec))))
        return max(1, bars)


DEFAULT_TRADE_CONTRACT = TradeContract()

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "exit_ema_span_from_window",
]


def exit_ema_span_from_window(contract: TradeContract, candle_sec: int = 60) -> int:
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    if not windows:
        return int(getattr(contract, "exit_ema_span", 0) or 0)
    candle_sec = int(max(1, candle_sec))
    w_min = float(windows[0])
    return int(max(1, round((w_min * 60.0) / float(candle_sec))))
