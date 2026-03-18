# modules/

Shared building blocks used across feature prep, training, backtesting, and the real-time bot.

## Subpackages
- `backtest/` — portfolio and single-symbol backtest engines.
- `config/` — structural definitions and dynamic loading of configs (e.g., `trade_contract.py`, `symbols.py`).
- `data_providers/` — modular integrations for external data sources (e.g., `binance/`).
- `prepare_features/` — feature engineering, labeling, and dataset assembly mechanisms.
- `plotting/` — tools to generate HTML visualizations and equity curves.
- `realtime/` — dashboard state manager and API.
- `thresholds/` — utilities to tune and search optimal threshold hyperparameters.
- `train/` — XGBoost wrappers, walk-forward orchestrators, and exploration logic.
- `utils/` — generic helpers: logging, parallelization, math, etc.

Each subfolder contains the core business logic. Entry points are located at the root `scripts/` directory.
