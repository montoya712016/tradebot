from __future__ import annotations

from dataclasses import dataclass

from modules.data_providers.binance.trading_client import BinanceTrader, TradingConfig


@dataclass
class BinanceService:
    trader: BinanceTrader

    @classmethod
    def from_env(cls) -> "BinanceService":
        return cls(trader=BinanceTrader(TradingConfig.from_env()))

    def buy(self, symbol: str, notional_usd: float, *, notify: bool = False) -> dict:
        res = self.trader.market_buy_usdt(symbol, float(notional_usd), margin=False, leverage=1.0, notify=bool(notify))
        return {"ok": True, "trade": res.__dict__}

    def short(self, symbol: str, notional_usd: float, *, notify: bool = False) -> dict:
        res = self.trader.market_short_usdt(symbol, float(notional_usd), notify=bool(notify))
        return {"ok": True, "trade": res.__dict__}

    def close_short(self, symbol: str, *, notify: bool = False) -> dict:
        res = self.trader.close_short(symbol, notify=bool(notify))
        return {"ok": True, "trade": res.__dict__}
