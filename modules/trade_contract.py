# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass(frozen=True)
class TradeContract:
    """
    Define um contrato fixo de trade usado tanto na geração de labels quanto nas simulações.

    Todos os valores percentuais são fornecidos em frações (0.05 = 5%).
    """

    timeframe_sec: int = 60
    # TP usado para exit hard/funcoes de exit (nao define entry).
    tp_min_pct: float = 0.04
    sl_pct: float = 0.03
    timeout_hours: float = 6.0
    # Minimo de duracao do trade (para entry label valido).
    min_hold_minutes: float = 30.0
    # Entry label: horizonte de media futura (ex.: 2h) e lucro minimo.
    entry_horizon_hours: float = 2.0
    entry_min_profit_pct: float = 0.03
    fee_pct_per_side: float = 0.0005
    slippage_pct: float = 0.0005
    max_adds: int = 1
    add_spacing_pct: float = 0.01
    add_sizing: Tuple[float, ...] = field(default_factory=lambda: (1.0, 0.5))
    risk_max_cycle_pct: float = 0.06
    # IMPORTANTE: para "não comprar antes do mínimo", exigimos MAE pequeno.
    # Isso força exemplos onde a entrada acontece perto do fundo (ou após estabilização).
    dd_intermediate_limit_pct: float = 0.005
    # Danger deve ser um filtro *útil* (não raríssimo):
    # - drop_pct muito alto (ex.: 10% em 2h) vira evento raro e o modelo não aprende.
    # Defaults abaixo miram "queda relevante em poucas horas" (~1%+ de positivos em alts),
    # mantendo estabilidade para não bloquear tudo.
    # Queda "relevante" em poucas horas: use algo na faixa 3%..6%.
    # 3% tende a disparar muito em alts (muito conservador); 5% é um bom ponto de partida.
    danger_drop_pct: float = 0.04
    danger_recovery_pct: float = 0.02
    danger_timeout_hours: float = 6.0
    # Danger "reset" (pós-fundo): se após uma queda o preço estabilizar, queremos danger=0 mais cedo.
    # A estabilização é detectada por:
    # - recuperação mínima a partir do mínimo local, OU
    # - ficar N barras sem fazer novo mínimo
    danger_stabilize_recovery_pct: float = 0.01
    danger_stabilize_bars: int = 30

    def timeout_bars(self, candle_sec: int) -> int:
        if candle_sec <= 0:
            candle_sec = self.timeframe_sec
        bars = int(round((self.timeout_hours * 3600.0) / float(candle_sec)))
        return max(1, bars)

    def danger_horizon_bars(self, candle_sec: int) -> int:
        hours = self.danger_timeout_hours or self.timeout_hours
        bars = int(round((hours * 3600.0) / float(max(1, candle_sec))))
        return max(1, bars)

    def min_hold_bars(self, candle_sec: int) -> int:
        mins = float(self.min_hold_minutes or 0.0)
        bars = int(round((mins * 60.0) / float(max(1, candle_sec))))
        return max(0, bars)

    def entry_horizon_bars(self, candle_sec: int) -> int:
        hours = float(self.entry_horizon_hours or 0.0)
        bars = int(round((hours * 3600.0) / float(max(1, candle_sec))))
        return max(1, bars)


DEFAULT_TRADE_CONTRACT = TradeContract()

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
]
