# modules/config

Centralized config handling.

## Contents
- `base_config.py` — common dataclasses (paths, DB, model, dashboard, ngrok).
- `loaders.py` — YAML/JSON loaders with validation and env‑var expansion.
- `defaults.py` — sane defaults used by scripts when flags are omitted.

## Pattern
```python
from modules.config.loaders import load_config
cfg = load_config("configs/realtime_crypto.yml")
```

Configs are shared by training, backtests, and the realtime bot for consistency.
