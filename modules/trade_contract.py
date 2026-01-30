# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shim para compatibilidade: importa de core.contracts.
"""

from core.contracts import TradeContract, DEFAULT_TRADE_CONTRACT
from core.contracts.base import exit_ema_span_from_window

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "exit_ema_span_from_window",
]
