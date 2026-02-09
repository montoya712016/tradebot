# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainOptimizedContractRanges:
    # parametros otimizados pelo wf_random_loop (contract/labels)
    entry_window: tuple[int, int, int] = (120, 720, 60)
    # Exit EMA acompanha a janela (em minutos) quando usada no loop random.
    exit_ema_span: tuple[int, int, int] = (120, 720, 60)
    exit_ema_init_offset_pct: tuple[float, float, float] = (0.0, 0.02, 0.002)


__all__ = [
    "TrainOptimizedContractRanges",
]
