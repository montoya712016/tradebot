# modules/backtest

Backtesting utilities for supervised and hybrid supervised+RL workflows.

Core files
----------
- `single_symbol.py` - single-symbol walk-forward backtest.
- `single_symbol_rl.py` - single-symbol backtest/plot for trained RL policy (trades, Q-values, regressors, equity).
- `sniper_walkforward.py` - load WF models, generate scores, and simulate cycles.
- `sniper_portfolio.py` - multi-symbol portfolio simulation.
- `wf_portfolio.py` - batch walk-forward portfolio runner.
- `wf_backtest_sweep.py` - batch parameter sweeps for WF runs.
- `backtest_supervised_only.py` - baseline backtest from exported supervised signals.
- `backtest_supervised_plus_rl.py` - backtest using a trained RL policy on top of supervised signals.
