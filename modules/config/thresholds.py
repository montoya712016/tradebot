# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Central de thresholds manuais (usados nos backtests/simulacoes).

Objetivo:
- Permitir que você altere os thresholds (ex.: tau_entry=0.002)
  em UM lugar e isso reflita em todas as simulações.
- Sem depender de variáveis de ambiente.

Observação:
- O treino não calcula thresholds; eles são definidos aqui.
- Isso altera APENAS a simulação/execução (decisão de operar).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdOverrides:
    # Regressao: threshold em retorno previsto (ex.: 0.002 = 0.2%)
    tau_entry: float | None = 0.002

    # Fallback para campos legados (danger/exit) que não são mais usados.
    def __getattr__(self, name: str):
        if name in {"tau_danger", "tau_exit", "tau_danger_add"}:
            return 1.0
        raise AttributeError(name)

DEFAULT_THRESHOLD_OVERRIDES = ThresholdOverrides()
