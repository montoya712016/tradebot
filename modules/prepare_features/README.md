# modules/prepare_features

Feature engineering and labeling pipeline for the Sniper Trading System.

## Main Scripts / Modules
- `prepare_features.py` — computes price/volume features, EMAs, ATR, ranges.
- `labels.py` — generates forward-return labels and meta info (hit windows).
- `sniper_dataset.py` — writes training matrices to Parquet/Pickle for model fitting.
- `refresh_sniper_labels_in_cache.py` — rapidly rewrites target labels for hyperparameter exploration without recalculating base OHLC features.

## Usage Environment
All interactions with this module should be done via the unified CLI in `scripts/`:

```bash
# To build caches (downloads OHLC and generates parquets)
python scripts/data_sync.py

# To refresh labels rapidly during threshold tuning
python scripts/refresh_labels.py
```

The outputs land in `d:\astra\cache_sniper\` for training and backtests.
