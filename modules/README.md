# modules/

Shared building blocks used across training, backtesting, realtime bots, and sweep tooling.

## Subpackages
- `backtest/` — portfolio and single‑symbol backtest engines plus plotting aids.
- `config/` — typed configuration loaders and defaults shared by scripts.
- `prepare_features/` — feature engineering, labeling, and dataset assembly.
- `realtime/` — websocket ingestion, decision loop helpers, dashboard server.
- `thresholds/` — utilities to search/score entry/exit thresholds.
- `train/` — model training helpers (XGBoost wrappers, CV, sweep support).
- `utils/` — generic helpers: MySQL, caching, date/time, logging, etc.

Each subfolder has its own README with entry points and key functions.
