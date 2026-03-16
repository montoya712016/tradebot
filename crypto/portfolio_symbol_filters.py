# -*- coding: utf-8 -*-
from __future__ import annotations

# Denylist opcional derivada de experimentos do portfolio no wf_022:
# símbolos com contribuição negativa recorrente nas versões com correlação
# habilitada e amostra mínima de trades.
DEFAULT_PORTFOLIO_BAD_SYMBOLS: tuple[str, ...] = (
    "AMPUSDT",
    "ARKUSDT",
    "ASTRUSDT",
    "CVXUSDT",
    "DEXEUSDT",
    "GNOUSDT",
    "INJUSDT",
    "JUPUSDT",
    "MOVEUSDT",
    "PROMUSDT",
    "SAHARAUSDT",
    "SEIUSDT",
    "TONUSDT",
    "WUSDT",
)


__all__ = ["DEFAULT_PORTFOLIO_BAD_SYMBOLS"]
