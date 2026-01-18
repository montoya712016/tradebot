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
    # Heurísticas removidas (mantidas apenas por compatibilidade; não usadas).
    tp_min_pct: float = 0.0
    sl_pct: float = 0.0
    timeout_hours: float = 0.0
    min_hold_minutes: float = 0.0
    # Entry labels (janela unica em minutos) e lucro minimo correspondente.
    # Ajuste manual aqui para calibrar o label.
    entry_label_windows_minutes: Tuple[int, ...] = (360,)
    entry_label_min_profit_pcts: Tuple[float, ...] = (0.02,)
    # Peso dos labels: escala da margem (abs(retorno - min_profit_pct))
    entry_label_weight_alpha: float = 0.01
    # Exit EMA: se span > 0, fecha em cross/close < EMA
    # Default acompanha a janela unica do label.
    exit_ema_span: int = 360
    exit_ema_init_offset_pct: float = 0.01
    fee_pct_per_side: float = 0.0005
    slippage_pct: float = 0.0
    max_adds: int = 0
    add_spacing_pct: float = 0.0
    add_sizing: Tuple[float, ...] = field(default_factory=lambda: (1.0,))
    risk_max_cycle_pct: float = 0.0
    dd_intermediate_limit_pct: float = 0.0
    # Danger deve ser um filtro *útil* (não raríssimo):
    # - drop_pct muito alto (ex.: 10% em 2h) vira evento raro e o modelo não aprende.
    # Defaults abaixo miram "queda relevante em poucas horas" (~1%+ de positivos em alts),
    # mantendo estabilidade para não bloquear tudo.
    # Queda "relevante" em poucas horas: use algo na faixa 3%..6%.
    # 3% tende a disparar muito em alts (muito conservador); 5% é um bom ponto de partida.
    danger_drop_pct: float = 0.0
    danger_recovery_pct: float = 0.0
    danger_timeout_hours: float = 0.0
    danger_fast_minutes: float = 0.0
    danger_drop_pct_critical: float = 0.0
    danger_stabilize_recovery_pct: float = 0.0
    danger_stabilize_bars: int = 0

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


DEFAULT_TRADE_CONTRACT = TradeContract()

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "exit_ema_span_from_window",
]


def exit_ema_span_from_window(contract: TradeContract, candle_sec: int = 60) -> int:
    """
    Deriva o span da EMA de exit a partir da primeira janela (em minutos).
    Fallback: usa exit_ema_span do contrato.
    """
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    if not windows:
        return int(getattr(contract, "exit_ema_span", 0) or 0)
    candle_sec = int(max(1, candle_sec))
    w_min = float(windows[0])
    return int(max(1, round((w_min * 60.0) / float(candle_sec))))
