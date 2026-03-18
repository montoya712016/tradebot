# modules/train

Model training logic and walk-forward handlers.

## Modules
- `sniper_trainer.py` — fit/eval/persist XGBoost models; handles train/val splitting and early stopping.
- `train_sniper_wf.py` — walk-forward orchestration. Progresses through time, training distinct models on sliding periods.
- `wf_portfolio_explorer.py` — hyperparameter sweep harness that orchestrates dataset prep, model training, and portfolio backtesting systematically.

## Entry Points
This module shouldn't be executed directly. Use the unified CLI:

```bash
# Standard Walk-Forward Training
python scripts/train.py

# Hyperparameter Exploration
python scripts/explore.py
```

Models are saved externally to `d:\astra\models_sniper\models\` and loaded by the backtest suites and the real-time bot.
