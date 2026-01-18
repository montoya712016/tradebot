# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Central de thresholds manuais (usados nos backtests/simulacoes).

Objetivo:
- Permitir que você altere os thresholds (ex.: tau_entry=0.8)
  em UM lugar e isso reflita em todas as simulações.
- Sem depender de variáveis de ambiente.

Observação:
- O treino não calcula thresholds; eles são definidos aqui.
- Isso altera APENAS a simulação/execução (decisão de operar).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdOverrides:
    tau_entry: float | None = 0.75
    # danger desativado por enquanto (mantemos 1.0 para nÃ£o bloquear)
    tau_danger: float | None = 1.0
    tau_exit: float | None = 0.85


DEFAULT_THRESHOLD_OVERRIDES = ThresholdOverrides()
