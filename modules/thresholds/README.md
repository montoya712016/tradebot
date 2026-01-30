# modules/thresholds

Utilities to search and evaluate entry/exit thresholds.

## Pieces
- `grid_search.py` — brute-force sweep over tau_entry/exit spans on validation sets.
- `score_curves.py` — plots PnL vs threshold; helps choose operating point.
- `utils.py` — shared helpers for percentile cuts, smoothing, and balancing long/short.

Often invoked indirectly by training sweeps (see `data/generated/wf_backtest_sweep` outputs).
