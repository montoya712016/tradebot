# modules/prepare_features

Feature engineering and regression-label pipeline for both crypto and stocks.

## Main scripts / modules
- `features.py` — computes price/volume features, EMAs, ATR, ranges.
- `labels.py` — timing‑adjusted regression labels (entry return + weights).
- `prepare_features.py` — main entrypoint (features + labels + optional plot).
- `refresh_sniper_labels_in_cache.py` — recompute timing labels inside cached feature files.

## Usage examples
```bash
python crypto/prepare_features_crypto.py      # uses these helpers
python stocks/prepare_features_stocks.py      # same idea for equities
```

The outputs land in MySQL feature tables and `data/generated/` for training/backtests.
