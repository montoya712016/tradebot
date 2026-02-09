# modules/thresholds

Utilities to search and evaluate entry thresholds (regression scores).

## Pieces
- `grid_search.py` — brute-force sweep over tau_entry/exit spans on validation sets.
- `score_curves.py` — plots PnL vs threshold; helps choose operating point.
- `utils.py` — shared helpers for percentile cuts, smoothing, and score balancing.

Often invoked indirectly by training sweeps (see `data/generated/wf_backtest_sweep` outputs).
