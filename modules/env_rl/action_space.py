# -*- coding: utf-8 -*-
from __future__ import annotations

ACTION_HOLD = 0
ACTION_OPEN_LONG_SMALL = 1
ACTION_OPEN_LONG_BIG = 2
ACTION_OPEN_SHORT_SMALL = 3
ACTION_OPEN_SHORT_BIG = 4
ACTION_CLOSE_LONG = 5
ACTION_CLOSE_SHORT = 6

# aliases canonicos (modo simplificado em 5 acoes efetivas)
ACTION_OPEN_LONG = ACTION_OPEN_LONG_SMALL
ACTION_OPEN_SHORT = ACTION_OPEN_SHORT_SMALL

# aliases legados
ACTION_FLAT = ACTION_HOLD
ACTION_LONG_SMALL = ACTION_OPEN_LONG_SMALL
ACTION_LONG_BIG = ACTION_OPEN_LONG_BIG
ACTION_SHORT_SMALL = ACTION_OPEN_SHORT_SMALL
ACTION_SHORT_BIG = ACTION_OPEN_SHORT_BIG

ALL_ACTIONS = (
    ACTION_HOLD,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_LONG_BIG,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_OPEN_SHORT_BIG,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)


def action_to_position(action: int, small_size: float, big_size: float) -> tuple[int, float]:
    """
    Mapeamento legado de acao para alvo de posicao.
    Acoes de fechamento/hold retornam flat para compatibilidade.
    """
    a = int(action)
    if a == ACTION_OPEN_LONG_SMALL:
        return 1, float(small_size)
    if a == ACTION_OPEN_LONG_BIG:
        return 1, float(big_size)
    if a == ACTION_OPEN_SHORT_SMALL:
        return -1, float(small_size)
    if a == ACTION_OPEN_SHORT_BIG:
        return -1, float(big_size)
    return 0, 0.0
