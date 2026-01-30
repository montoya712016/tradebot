# modules/train

Model training helpers (primarily XGBoost) shared across crypto and stock workflows.

## Modules
- `xgb_trainer.py` — fit/eval/persist XGBoost models; supports GPU/CPU and early stopping.
- `cv_split.py` — time-series aware cross‑validation splits.
- `sweep_runner.py` — lightweight hyperparam sweep harness (used in `data/generated/wf_backtest_sweep`).
- `feature_select.py` — optional feature filtering / importance reports.

## Typical call
```python
from modules.train.xgb_trainer import train_model
model, metrics = train_model(df_features, cfg)
```

Models are saved under `models_sniper/<asset_class>/...` and later loaded by the realtime bots.
