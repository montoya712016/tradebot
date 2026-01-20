# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shim para compatibilidade: seleciona contrato de crypto/stocks.
"""

from pathlib import Path
import os
import sys


def _ensure_repo_on_sys_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _load_contract():
    mode = os.getenv("TRADE_CONTRACT_MODE", "").strip().lower()
    if mode in ("stocks", "stock"):
        candidates = ("stocks.trade_contract", "crypto.trade_contract")
    elif mode in ("crypto", "coin", "coins"):
        candidates = ("crypto.trade_contract", "stocks.trade_contract")
    else:
        candidates = ("crypto.trade_contract", "stocks.trade_contract")

    last_exc: Exception | None = None
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["TradeContract", "DEFAULT_TRADE_CONTRACT", "exit_ema_span_from_window"])
            return m.TradeContract, m.DEFAULT_TRADE_CONTRACT, m.exit_ema_span_from_window
        except Exception as exc:
            last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError("No trade contract module available")


_ensure_repo_on_sys_path()
TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window = _load_contract()

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "exit_ema_span_from_window",
]
