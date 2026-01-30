# modules/prepare_features

Feature engineering and labeling pipeline for both crypto and stocks.

## Main scripts / modules
- `feature_builder.py` — computes price/volume features, EMAs, ATR, ranges.
- `label_maker.py` — generates forward‑return labels and meta info (hit windows).
- `dataset_export.py` — writes training matrices to Parquet/CSV for model fitting.
- `db_ops.py` — fetch/store OHLCV windows in MySQL; chunked to avoid RAM spikes.

## Usage examples
```bash
python crypto/prepare_features_crypto.py      # uses these helpers
python stocks/prepare_features_stocks.py      # same idea for equities
```

The outputs land in MySQL feature tables and `data/generated/` for training/backtests.
