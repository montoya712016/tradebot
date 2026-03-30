from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.executors.paper import PaperExecutor

from .binance import BinanceService


@runtime_checkable
class BaseExecutor(Protocol):
    def buy(self, symbol: str, price: float, notional_usd: float) -> dict: ...
    def sell(self, symbol: str, price: float, fraction: float = 1.0) -> dict: ...


@dataclass
class PaperTradingExecutor:
    initial_cash: float
    fee_rate: float = 0.0

    def __post_init__(self) -> None:
        self.paper = PaperExecutor(float(self.initial_cash), fee_rate=float(self.fee_rate))

    @property
    def cash(self) -> float:
        return float(self.paper.cash)

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        return self.paper.buy(symbol, price, notional_usd)

    def sell(self, symbol: str, price: float, fraction: float = 1.0) -> dict:
        return self.paper.close(symbol, price)


@dataclass
class BinanceSpotExecutor:
    api_key: str = ""
    api_secret: str = ""
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.service = BinanceService.from_env()

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        return self.service.buy(symbol, notional_usd, notify=not self.dry_run)

    def sell(self, symbol: str, price: float, fraction: float = 1.0) -> dict:
        raise NotImplementedError("Binance spot sell adapter ainda não foi extraído para realtime/bot/services.")


@dataclass
class BinanceFuturesExecutor:
    api_key: str = ""
    api_secret: str = ""
    dry_run: bool = False
    leverage: int = 1

    def __post_init__(self) -> None:
        self.service = BinanceService.from_env()

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        return self.service.buy(symbol, notional_usd, notify=not self.dry_run)

    def sell(self, symbol: str, price: float, fraction: float = 1.0) -> dict:
        raise NotImplementedError("Binance futures sell adapter ainda não foi extraído para realtime/bot/services.")
