# modules/config

Centralized configuration handlers and typed definitions.

## Contents
- `trade_contract.py` — definitions and validation for trading parameters (SL/TP ratios, spans, offsets).
- `symbols.py` — logic to load active universes of symbols.

## Pattern
Configurations are shared by training, backtests, and the realtime bot for absolute consistency. The modules here provide programmatic defaults that scripts can override via environment variables.
