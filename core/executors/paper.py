# -*- coding: utf-8 -*-
"""
Paper Executor - Executor de simulação para paper trading.

Não envia ordens reais, apenas atualiza PnL virtual.
Útil para backtesting e simulação em tempo real sem risco.
"""
from __future__ import annotations

from typing import Dict, List


class PaperExecutor:
    """
    Executor de simulação: não envia ordens reais, apenas atualiza PnL virtual.
    
    Mantém track de:
    - cash: Saldo em USD
    - positions: Posições abertas por símbolo
    - trades: Histórico de trades executados
    """

    def __init__(self, cash_usd: float, fee_rate: float = 0.0):
        """
        Inicializa o executor de paper trading.
        
        Args:
            cash_usd: Saldo inicial em USD
            fee_rate: Taxa por lado (ex: 0.0005 = 0.05%)
        """
        self.cash = float(cash_usd)
        self.fee_rate = float(fee_rate)
        self.positions: Dict[str, Dict[str, float]] = {}  # sym -> {qty, avg}
        self.trades: List[dict] = []

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        """
        Executa uma compra (abre/adiciona posição long).
        
        Args:
            symbol: Símbolo do ativo
            price: Preço de execução
            notional_usd: Valor em USD a comprar
            
        Returns:
            Dict com status da operação e detalhes do trade
        """
        qty = float(notional_usd) / float(price)
        fee = float(notional_usd) * self.fee_rate
        total_cost = float(notional_usd) + fee
        
        if self.cash < total_cost:
            return {"ok": False, "reason": "no_cash"}
            
        self.cash -= total_cost
        pos = self.positions.get(symbol, {"qty": 0.0, "avg": 0.0})
        old_qty, old_avg = float(pos["qty"]), float(pos["avg"])
        new_qty = old_qty + qty
        new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty if new_qty > 0 else price
        self.positions[symbol] = {"qty": new_qty, "avg": new_avg}
        
        tr = {
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "price": price,
            "notional": notional_usd,
            "fee": fee,
        }
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def close(self, symbol: str, price: float) -> dict:
        """
        Fecha uma posição existente.
        
        Args:
            symbol: Símbolo do ativo
            price: Preço de execução
            
        Returns:
            Dict com status da operação e PnL realizado
        """
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "reason": "no_position"}
            
        qty = float(pos.get("qty", 0.0))
        avg = float(pos.get("avg", 0.0))
        gross_proceeds = qty * price
        fee = gross_proceeds * self.fee_rate
        pnl = gross_proceeds - fee - (qty * avg)
        
        self.cash += gross_proceeds - fee
        self.positions.pop(symbol, None)
        
        tr = {
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": price,
            "notional": gross_proceeds,
            "fee": fee,
            "pnl": pnl,
        }
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def has_position(self, symbol: str) -> bool:
        """Verifica se há posição aberta no símbolo."""
        return symbol in self.positions and abs(float(self.positions[symbol].get("qty", 0.0))) > 0

    def short(self, symbol: str, price: float, notional_usd: float) -> dict:
        """Abre uma posição short (venda a descoberto)."""
        qty = float(notional_usd) / float(price)
        self.cash += float(notional_usd)
        pos = self.positions.get(symbol, {"qty": 0.0, "avg": 0.0})
        old_qty = float(pos["qty"])
        new_qty = old_qty - qty
        self.positions[symbol] = {"qty": new_qty, "avg": price}
        
        tr = {"symbol": symbol, "side": "SELL", "qty": qty, "price": price, "notional": notional_usd}
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def equity(self, price_map: Dict[str, float]) -> float:
        """
        Calcula o equity total (cash + valor das posições).
        
        Args:
            price_map: Mapa símbolo -> preço atual
            
        Returns:
            Equity total em USD
        """
        eq = self.cash
        for sym, pos in self.positions.items():
            px = float(price_map.get(sym, 0.0) or 0.0)
            eq += float(pos["qty"]) * px
        return eq


__all__ = ["PaperExecutor"]
