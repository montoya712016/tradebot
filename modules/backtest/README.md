# modules/backtest

Backtesting utilities for both single symbols and portfolios.

## Key pieces
- `portfolio_backtester.py` — multi‑symbol, capital‑aware executor; supports fee, slippage, position sizing, and walk‑forward style evaluation.
- `single_symbol_bt.py` — fast single‑symbol simulation for debugging labels/thresholds.
- `plot_equity.py` — helpers to plot equity curves and drawdowns.
- `perf_metrics.py` — CAGR, max DD, Sharpe, hit‑rate, turnover, exposure stats.

## Typical use
```bash
python modules/backtest/single_symbol_bt.py --symbol BTCUSDT --threshold 0.7
python modules/backtest/portfolio_backtester.py --config configs/bt_crypto.yml
```

Outputs (CSV/PNG) are written beside the config or to `data/generated/*` depending on the caller.
