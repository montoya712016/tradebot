# -*- coding: utf-8 -*-
"""
Live Executor - Execução real na Binance.
"""
from __future__ import annotations

class LiveExecutor:
    """
    Executor real (Binance). Depende de credenciais via env.
    """

    def __init__(self, notify: bool = False):
        try:
            from crypto.trading_client import BinanceTrader  # lazy import
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"BinanceTrader indisponível: {e}")
        self.trader = BinanceTrader()
        self.notify = bool(notify)

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        res = self.trader.market_buy_usdt(symbol, float(notional_usd), margin=False, leverage=1.0, notify=self.notify)
        return {"ok": True, "trade": res.__dict__}

    def short(self, symbol: str, price: float, notional_usd: float) -> dict:
        res = self.trader.market_short_usdt(symbol, float(notional_usd), notify=self.notify)
        return {"ok": True, "trade": res.__dict__}
