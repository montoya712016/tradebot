# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Client utilitário para operações básicas na Binance (spot/margin), com funções:
- carregar preços e calcular equity total em USDT
- market buy (spot ou margin com empréstimo opcional)
- market short (margin) e encerramento de short (compra + repay)

Credenciais:
- BINANCE_API_KEY / BINANCE_API_SECRET (env)
- Para testnet, configure BINANCE_TESTNET=1
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Dict, List, Optional, Tuple
import os
import time

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

try:
    from modules.utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore
except Exception:
    try:
        from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore
    except Exception:  # pragma: no cover
        _pushover_load_default = None
        _pushover_send = None

# precisão decente para quantização
getcontext().prec = 28


class TradingError(RuntimeError):
    pass


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    avg_price: float
    quote_spent: float
    order_id: str
    raw: dict
    pnl_usd: float | None = None
    pnl_pct: float | None = None


@dataclass
class TradingConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    # limites de segurança
    max_notional_usd: float = 10_000.0
    max_retries: int = 3
    retry_sleep_sec: float = 2.0

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            api_key=os.getenv("BINANCE_API_KEY", "").strip(),
            api_secret=os.getenv("BINANCE_API_SECRET", "").strip(),
            testnet=os.getenv("BINANCE_TESTNET", "0").strip().lower() in {"1", "true", "yes", "on"},
            max_notional_usd=float(os.getenv("BINANCE_MAX_NOTIONAL_USD", "10000") or 10_000.0),
            max_retries=int(os.getenv("BINANCE_MAX_RETRIES", "3") or 3),
            retry_sleep_sec=float(os.getenv("BINANCE_RETRY_SLEEP_SEC", "2.0") or 2.0),
        )


class BinanceTrader:
    def __init__(self, cfg: Optional[TradingConfig] = None):
        self.cfg = cfg or TradingConfig.from_env()
        if not self.cfg.api_key or not self.cfg.api_secret:
            raise TradingError("API key/secret ausentes (defina BINANCE_API_KEY/BINANCE_API_SECRET)")
        self.client = Client(self.cfg.api_key, self.cfg.api_secret, testnet=bool(self.cfg.testnet))

    # ───────────────────────── Utils ─────────────────────────
    def _sleep(self, t: float) -> None:
        time.sleep(max(0.0, float(t)))

    def _lot_size_info(self, symbol_info: dict) -> Tuple[Decimal, Decimal, Decimal]:
        lot = next((f for f in symbol_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
        if not lot:
            raise TradingError(f"LOT_SIZE não encontrado para {symbol_info.get('symbol')}")
        return (
            Decimal(lot["stepSize"]),
            Decimal(lot["minQty"]),
            Decimal(lot["maxQty"]),
        )

    def _quantize_qty(self, qty: Decimal, step: Decimal) -> Decimal:
        return qty.quantize(step, rounding=ROUND_DOWN)

    def _fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        tickers = self.client.get_all_tickers()
        price_map = {t["symbol"].upper(): float(t["price"]) for t in tickers}
        return {s.upper(): float(price_map.get(s.upper(), 0.0)) for s in symbols}

    def _symbol_info(self, symbol: str) -> dict:
        return self.client.get_symbol_info(symbol)

    # ───────────────────── Equity / posições ─────────────────────
    def total_equity_usdt(self) -> float:
        """
        Soma spot + margem (net de empréstimos) usando preços de mercado.
        """
        spot = self.client.get_account()
        margin = self.client.get_margin_account()
        tickers = self._fetch_prices([b["asset"] + "USDT" for b in spot.get("balances", []) if b["asset"] not in {"USDT", "BUSD", "USD"}])

        total = 0.0
        # Spot
        for b in spot.get("balances", []):
            asset = b["asset"]
            free = float(b["free"] or 0)
            locked = float(b["locked"] or 0)
            amt = free + locked
            if amt <= 0:
                continue
            if asset in {"USD", "USDT", "BUSD"}:
                total += amt
            else:
                px = tickers.get(asset + "USDT", 0.0)
                total += amt * px

        # Margin (net de empréstimos)
        for a in margin.get("userAssets", []):
            asset = a["asset"]
            net = float(a.get("netAsset", 0.0) or 0.0)
            borrowed = float(a.get("borrowed", 0.0) or 0.0) + float(a.get("interest", 0.0) or 0.0)
            free = float(a.get("free", 0.0) or 0.0)
            locked = float(a.get("locked", 0.0) or 0.0)
            amt = free + locked
            if asset in {"USD", "USDT", "BUSD"}:
                total += net  # netAsset já considera empréstimos
            else:
                px = tickers.get(asset + "USDT", 0.0)
                total += net * px
                # se houver empréstimo líquido (borrowed > 0), subtrai notional
                if borrowed > 0:
                    total -= borrowed * px

        return float(total)

    def positions_snapshot(self) -> List[dict]:
        """
        Retorna uma visão simples: símbolo, lado (LONG/SHORT), qty, mark_price, notional_usd.
        Não inclui preço de entrada (API spot não fornece); isso deve vir do log de execuções.
        """
        margin = self.client.get_margin_account()
        tickers = self._fetch_prices([t["symbol"] for t in self.client.get_all_tickers()])

        out: List[dict] = []
        for a in margin.get("userAssets", []):
            asset = a["asset"]
            if asset in {"USDT", "BUSD", "USD"}:
                continue
            free = float(a.get("free", 0.0) or 0.0)
            locked = float(a.get("locked", 0.0) or 0.0)
            net = float(a.get("netAsset", 0.0) or 0.0)
            borrowed = float(a.get("borrowed", 0.0) or 0.0) + float(a.get("interest", 0.0) or 0.0)
            qty = free + locked
            if abs(qty) < 1e-9 and borrowed <= 0:
                continue
            sym = asset.upper() + "USDT"
            px = float(tickers.get(sym, 0.0))
            if px <= 0:
                continue
            side = "LONG" if net >= 0 else "SHORT"
            notional = abs(net) * px
            out.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": float(net),
                    "entry_price": 0.0,  # desconhecido na API spot; precisa de log externo
                    "mark_price": px,
                    "notional_usd": notional,
                    "pnl_usd": None,
                    "pnl_pct": None,
                }
            )
        return out

    # ───────────────────── Execução: BUY / SHORT ─────────────────────
    def _ensure_notional(self, symbol: str, notional_usd: float) -> float:
        if notional_usd <= 0:
            raise TradingError("notional_usd deve ser > 0")
        if notional_usd > float(self.cfg.max_notional_usd):
            raise TradingError(f"notional_usd {notional_usd} excede limite {self.cfg.max_notional_usd}")
        return float(notional_usd)

    def market_buy_usdt(
        self,
        symbol: str,
        notional_usd: float,
        *,
        margin: bool = False,
        leverage: float = 1.0,
        notify: bool = False,
    ) -> OrderResult:
        """
        Compra a mercado usando um valor em USDT.
        - margin=True: usa conta de margem; se leverage>1, toma empréstimo de USDT proporcional.
        """
        notional_usd = self._ensure_notional(symbol, notional_usd)
        info = self._symbol_info(symbol)
        step, min_qty, max_qty = self._lot_size_info(info)
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        price = Decimal(ticker["price"])
        qty = Decimal(str(notional_usd)) / price
        if leverage > 1.0 and margin:
            borrow = Decimal(str(notional_usd)) * (Decimal(str(leverage)) - Decimal("1"))
            self._retry(lambda: self.client.create_margin_loan(asset="USDT", amount=str(borrow)))
        qty_q = self._quantize_qty(qty, step)
        if qty_q < min_qty:
            raise TradingError(f"qty ajustada {qty_q} < minQty {min_qty}")
        if qty_q > max_qty:
            qty_q = max_qty
        side = Client.SIDE_BUY
        order_fn = self.client.create_margin_order if margin else self.client.order_market
        order = self._retry(lambda: order_fn(symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, quantity=str(qty_q)))
        filled_qty, avg_price = _filled_avg(order)
        res = OrderResult(
            symbol=symbol,
            side="BUY",
            qty=float(filled_qty),
            avg_price=float(avg_price),
            quote_spent=float(filled_qty * avg_price),
            order_id=str(order.get("orderId", "")),
            raw=order,
        )
        if notify:
            self.send_pushover_trade(self.format_trade_msg(res))
        return res

    def market_short_usdt(self, symbol: str, notional_usd: float, *, notify: bool = False) -> OrderResult:
        """
        Short simples em margem (spot margin):
        - toma empréstimo do ativo base
        - vende a mercado o notional solicitado
        """
        notional_usd = self._ensure_notional(symbol, notional_usd)
        info = self._symbol_info(symbol)
        step, min_qty, max_qty = self._lot_size_info(info)
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        price = Decimal(ticker["price"])
        qty = Decimal(str(notional_usd)) / price
        qty_q = self._quantize_qty(qty, step)
        if qty_q < min_qty:
            raise TradingError(f"qty ajustada {qty_q} < minQty {min_qty}")
        if qty_q > max_qty:
            qty_q = max_qty
        base = symbol.replace("USDT", "")
        self._retry(lambda: self.client.create_margin_loan(asset=base, amount=str(qty_q)))
        order = self._retry(
            lambda: self.client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(qty_q),
            )
        )
        filled_qty, avg_price = _filled_avg(order)
        res = OrderResult(
            symbol=symbol,
            side="SELL",
            qty=float(filled_qty),
            avg_price=float(avg_price),
            quote_spent=float(filled_qty * avg_price),
            order_id=str(order.get("orderId", "")),
            raw=order,
        )
        if notify:
            self.send_pushover_trade(self.format_trade_msg(res))
        return res

    def close_short(self, symbol: str, *, notify: bool = False) -> OrderResult:
        """
        Fecha short cross margin comprando a quantidade devida e pagando empréstimo.
        """
        base = symbol.replace("USDT", "")
        margin = self.client.get_margin_account()
        target_qty = Decimal("0")
        for a in margin.get("userAssets", []):
            if a.get("asset", "").upper() == base.upper():
                target_qty = Decimal(str(a.get("borrowed", 0) or 0)) + Decimal(str(a.get("interest", 0) or 0))
                break
        if target_qty <= 0:
            raise TradingError(f"Nenhum empréstimo ativo para {symbol}")
        info = self._symbol_info(symbol)
        step, min_qty, max_qty = self._lot_size_info(info)
        qty_q = self._quantize_qty(target_qty, step)
        if qty_q < min_qty:
            raise TradingError(f"qty ajustada {qty_q} < minQty {min_qty}")
        if qty_q > max_qty:
            qty_q = max_qty

        order = self._retry(
            lambda: self.client.create_margin_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(qty_q),
            )
        )
        filled_qty, avg_price = _filled_avg(order)
        # repagamento
        try:
            self.client.repay_margin_loan(asset=base, amount=str(qty_q))
        except Exception:
            pass
        res = OrderResult(
            symbol=symbol,
            side="BUY",
            qty=float(filled_qty),
            avg_price=float(avg_price),
            quote_spent=float(filled_qty * avg_price),
            order_id=str(order.get("orderId", "")),
            raw=order,
        )
        if notify:
            self.send_pushover_trade(self.format_trade_msg(res))
        return res

    # ───────────────────── Retry helper ─────────────────────
    def _retry(self, fn):
        last_err: Exception | None = None
        for _ in range(int(max(1, self.cfg.max_retries))):
            try:
                return fn()
            except (BinanceAPIException, BinanceOrderException) as e:
                last_err = e
                self._sleep(self.cfg.retry_sleep_sec)
                continue
            except Exception as e:  # pragma: no cover - apenas defensive
                last_err = e
                self._sleep(self.cfg.retry_sleep_sec)
                continue
        raise TradingError(f"falha após retries: {last_err}")

    # ───────────────────── Pushover helpers ─────────────────────
    def send_pushover_trade(self, msg: str, *, url: str | None = None, url_title: str | None = None) -> None:
        if _pushover_load_default is None or _pushover_send is None:
            return
        cfg = _pushover_load_default(
            user_env="PUSHOVER_USER_KEY",
            token_env="PUSHOVER_TOKEN_TRADE",
            token_name_fallback="PUSHOVER_TOKEN_TRADE",
            title="Tradebot",
            priority=0,
        )
        if cfg is None:
            return
        try:
            _pushover_send(msg, cfg=cfg, url=url, url_title=url_title)
        except Exception:
            pass

    def format_trade_msg(self, order: OrderResult) -> str:
        side = order.side.upper()
        base = (
            f"{side} {order.symbol} qty={order.qty:.6f} "
            f"avg={order.avg_price:.4f} notional={order.quote_spent:.2f}"
        )
        if order.pnl_usd is not None:
            pnl = f" pnl={order.pnl_usd:+.2f} ({(order.pnl_pct or 0):+.2f}%)"
            base += pnl
        return base


def _filled_avg(order: dict) -> Tuple[Decimal, Decimal]:
    fills = order.get("fills") or []
    if not fills:
        qty = Decimal(str(order.get("executedQty", "0") or "0"))
        price = Decimal(str(order.get("price", "0") or "0"))
        return qty, price
    qty_sum = Decimal("0")
    cost = Decimal("0")
    for f in fills:
        q = Decimal(str(f.get("qty", "0") or "0"))
        p = Decimal(str(f.get("price", "0") or "0"))
        qty_sum += q
        cost += q * p
    avg = cost / qty_sum if qty_sum > 0 else Decimal("0")
    return qty_sum, avg


__all__ = ["TradingConfig", "BinanceTrader", "TradingError", "OrderResult"]
