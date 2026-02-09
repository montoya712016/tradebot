# modules/

Shared building blocks across training, backtesting, realtime execution, and the hybrid RL pipeline.

Subpackages
-----------
- `prepare_features/` - feature engineering, labeling, cache refresh.
- `train/` - walk-forward supervised training.
- `backtest/` - single-symbol and portfolio backtesting.
- `thresholds/` - threshold search and scoring utilities.
- `realtime/` - realtime dashboard server and assets.
- `supervised/` - long/short signal export for RL.
- `env_rl/` - RL trading environment, action space, reward/cost components.
- `rl/` - RL walk-forward training, evaluation, and orchestration.
- `utils/` - shared helpers (paths, notifications, progress, etc.).
- `config/` - symbols/threshold defaults and local secrets loader.
