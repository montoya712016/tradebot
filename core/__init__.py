# -*- coding: utf-8 -*-
"""
Core module - Componentes compartilhados do Tradebot.
"""

from core.contracts import TradeContract, DEFAULT_TRADE_CONTRACT
from core.executors import PaperExecutor

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT", 
    "PaperExecutor",
]
