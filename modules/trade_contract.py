# -*- coding: utf-8 -*-
"""
Re-export shim: mantém compatibilidade com `from trade_contract import ...`
quando `modules/` está no sys.path.  O módulo real vive em config/trade_contract.py.
"""
from config.trade_contract import *  # noqa: F401,F403
from config.trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # noqa: F401
