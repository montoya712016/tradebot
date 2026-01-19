# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shim para compatibilidade: usa o contrato crypto como default.
"""

from crypto.trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "exit_ema_span_from_window",
]
