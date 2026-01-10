### backtest/ - simuladores e entrypoints

Core:
- `sniper_simulator.py`: backtest single-symbol (ciclo stateful) com Entry/Danger/Exit.
- `sniper_portfolio.py`: backtest de portfolio multi-cripto (carteira unica).
- `sniper_walkforward.py`: utilitarios walk-forward (carregar wf_*, scores, simulacao).

Entrypoints principais:
- `python modules/backtest/single_symbol.py`
- `python modules/backtest/portfolio.py`
- `python modules/backtest/wf_portfolio.py`

Core de simulacao:
- `single_symbol.py`, `portfolio.py`
