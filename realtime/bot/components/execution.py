from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Any, Optional
from realtime.bot.settings import LiveSettings

# Executor imports
from modules.realtime.realtime_dashboard_ngrok_monolith import (
    BaseExecutor,
    PaperExecutor,
    BinanceExecutor,
    BinanceFuturesExecutor,
)

log = logging.getLogger("realtime.components.execution")

class ExecutionManager:
    """
    Handles trade execution, open positions tracking, and order management.
    """
    def __init__(self, settings: LiveSettings, state_manager=None):
        self.settings = settings
        self.state_manager = state_manager
        self.open_positions: Dict[str, Any] = {}
        self.trades: list = []
        self.trade_lock = threading.RLock()
        
        # Initialize Executor
        self.executor = self._init_executor()

    def _init_executor(self) -> BaseExecutor:
        mode = str(self.settings.trade_mode or "paper").lower()
        log.info("[exec] Iniciando executor modo=%s", mode)
        if mode == "binance_spot":
            return BinanceExecutor(
                api_key=self.settings.binance_api_key,
                api_secret=self.settings.binance_api_secret,
                dry_run=False,
            )
        elif mode == "binance_futures":
            return BinanceFuturesExecutor(
                api_key=self.settings.binance_api_key,
                api_secret=self.settings.binance_api_secret,
                dry_run=False,
                leverage=int(self.settings.leverage or 1),
            )
        else:
            return PaperExecutor(initial_cash=float(self.settings.paper_balance or 10_000.0))

    def load_state(self, positions: Dict, trades: list):
        """Loads positions and trades from persistence."""
        with self.trade_lock:
            self.open_positions = positions.copy()
            
            # Filter legacy trades: remove standalone entries where side is SELL (since we now merge them)
            cleaned_trades = []
            seen_ids = set() # Optional: if trades had IDs, we could dedupe.
            
            for t in trades:
                # If it's a legacy "SELL" record (no entry info, just exit), skip it
                # Modern records use side='BUY' with status='CLOSED'
                s = str(t.get("side") or t.get("action") or "BUY").upper()
                if s == "SELL":
                    continue
                
                # Also dedupe by symbol+timestamp if needed, but for now just filter side
                cleaned_trades.append(t)
                
            self.trades = cleaned_trades[:]
            log.info("[exec] State loaded: %d positions, %d trades (filtered from %d)", len(self.open_positions), len(self.trades), len(trades))

    def current_equity(self, price_hint: Optional[Dict[str, float]] = None) -> float:
        """Calculates current equity = cash + unrealized_pnl."""
        cash = float(getattr(self.executor, "cash", 0.0))
        unrealized = 0.0
        with self.trade_lock:
            for sym, pos in self.open_positions.items():
                qty = float(pos.get("qty", 0.0))
                if qty == 0:
                    continue
                # Use current price if available, else entry price
                current_price = 0.0
                if price_hint and sym in price_hint:
                    current_price = price_hint[sym]
                else:
                    current_price = float(pos.get("entry_price", 0.0)) # Fallback
                
                # Simple value calculation (long only for now as per original code logic usually)
                # But original code had PnL logic.
                val = qty * current_price
                unrealized += (val - (qty * float(pos.get("entry_price", 0.0))))
        
        return cash + unrealized

    def handle_buy(self, decision: dict, price: float) -> Optional[dict]:
        """Executes a BUY order with risk checks."""
        with self.trade_lock:
            symbol = decision["symbol"]
            price = float(price or 0.0)
            if price <= 0:
                return None

            if self.open_positions.get(symbol):
                log.info("[trade] %s já em posição, ignorando novo BUY", symbol)
                return None

            # Risk Checks
            used_exposure = sum(float(pos.get("weight", 0.0)) for pos in self.open_positions.values())
            open_count = len(self.open_positions)
            
            if int(self.settings.max_positions) > 0 and open_count >= int(self.settings.max_positions):
                log.info("[trade] limite de posições atingido (%s)", self.settings.max_positions)
                return None
            
            remaining = float(self.settings.total_exposure) - float(used_exposure)
            if remaining <= 1e-9:
                log.info("[trade] orçamento de exposição esgotado (%.3f)", used_exposure)
                return None

            desired = float(self.settings.total_exposure) / float(max(1, open_count + 1))
            weight = float(min(float(self.settings.max_trade_exposure), remaining, desired))
            
            if weight < float(self.settings.min_trade_exposure):
                log.info("[trade] peso %.4f abaixo do mínimo %.4f", weight, self.settings.min_trade_exposure)
                return None

            # Sizing
            equity = self.current_equity(price_hint={symbol: price})
            notional = float(equity * weight) if equity > 0 else float(self.settings.trade_notional_usd)
            
            # Paper trading constraint
            available_cash = float(getattr(self.executor, "cash", 0.0))
            if isinstance(self.executor, PaperExecutor):
                notional = min(notional, available_cash)

            if notional <= 0:
                log.info("[trade] notional inválido (eq=%.2f weight=%.4f cash=%.2f)", equity, weight, available_cash)
                return None

            actual_weight = weight
            if equity > 0:
                actual_weight = min(weight, notional / max(equity, 1e-9))

            # Execute
            mode = str(self.settings.trade_mode or "paper").lower()
            try:
                res = self.executor.buy(symbol, price, notional)
                log.info("[%s] BUY %s notional=%.2f price=%.4f", mode, symbol, notional, price)
                
                self._record_trade(symbol, price, notional, mode, side="BUY", weight=actual_weight, equity=equity)
                return res
            except Exception as e:
                log.error("[%s][err] buy %s: %s", mode, symbol, e)
                return None

    def handle_sell(self, symbol: str, price: float, reason: str = "signal") -> Optional[dict]:
        """Executes a SELL order."""
        with self.trade_lock:
            price = float(price or 0.0)
            if price <= 0:
                return None
            
            pos = self.open_positions.get(symbol)
            if not pos:
                return None

            mode = str(self.settings.trade_mode or "paper").lower()
            try:
                res = self.executor.sell(symbol, price, 1.0) # Sell 100%
                log.info("[%s] SELL %s price=%.4f reason=%s", mode, symbol, price, reason)
                
                self._close_trade_record(symbol, price, mode)
                self.open_positions.pop(symbol, None)
                return res
            except Exception as e:
                log.error("[%s][err] sell %s: %s", mode, symbol, e)
                return None

    def _record_trade(self, symbol: str, price: float, notional: float, mode: str, side: str, weight: float, equity: float):
        """Internal record keeping."""
        ts = time.time()
        
        # Add to open positions
        self.open_positions[symbol] = {
            "entry_price": price,
            "entry_ts": ts,
            "weight": weight,
            "equity_at_entry": equity,
            "qty": notional / (price if price > 0 else 1.0),
            "side": side
        }
        
        # Add to history (incomplete trade)
        trade = {
            "symbol": symbol,
            "side": side,
            "entry_price": price, # Store explicit entry_price
            "price": price,       # Store generic price for compatibility
            "qty": notional / (price if price > 0 else 1.0),
            "entry_ts": ts,
            "entry_date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)),
            "status": "OPEN",
            "mode": mode
        }
        self.trades.append(trade)
        
        # Trim history
        if len(self.trades) > 2000:
            self.trades = self.trades[-2000:]

    def _close_trade_record(self, symbol: str, price: float, mode: str):
        """Updates trade history with exit info."""
        # Find the open trade in history
        for t in reversed(self.trades):
            if t["symbol"] == symbol and t.get("status") == "OPEN":
                t["status"] = "CLOSED"
                t["exit_price"] = price
                t["exit_ts"] = time.time()
                t["exit_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t["exit_ts"]))
                
                entry = float(t.get("entry_price", 0) or t.get("price", 0))
                qty = float(t.get("qty", 0))
                
                # Calc PnL
                if entry > 0:
                    raw_diff = (price - entry)
                    t["pnl_usd"] = raw_diff * qty
                    t["pnl_pct"] = (raw_diff / entry) * 100.0
                
                break
