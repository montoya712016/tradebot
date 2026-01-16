# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Central de thresholds manuais (usados nos backtests/simulacoes).

Objetivo:
- Permitir que você altere os thresholds (ex.: tau_entry=0.8, tau_danger=0.4)
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
    tau_danger: float | None = 0.75
    tau_exit: float | None = 0.85


DEFAULT_THRESHOLD_OVERRIDES = ThresholdOverrides()
