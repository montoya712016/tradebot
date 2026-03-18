# modules/backtest

Backtesting utilities for portfolio simulation and single-symbol debugging.

## Key Pieces
- `sniper_portfolio.py` — multi-symbol, capital-aware executor; supports fees, slippage, position sizing, concurrency limits, and walk-forward evaluations directly from disk models.
- `single_symbol.py` — fast single-symbol simulation for deep-dive debugging of labels/thresholds. Generates rich Plotly HTML reports.
- `sniper_walkforward.py` — loads walk-forward models and predicts probability scores dynamically across time boundaries.

## Typical Use
Use the `scripts/` interface to interact with this module:

```bash
# Run a portfolio backtest across all loaded assets
python scripts/backtest.py
```

Outputs (CSV metrics) and dashboard plots are written to `data/generated/`.
