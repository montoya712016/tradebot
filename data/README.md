# data/

Runtime state, metrics, and generated artifacts live here. Nothing is versioned by default; many files are gitignored in practice.

## Root files
- `equity_history_live.json` — rolling equity curve (cash + positions) produced by the realtime bot for the dashboard.
- `state_live.json` — latest bot state snapshot (positions, open orders, last decisions).
- `sysmon.jsonl` — system monitor samples (CPU, RAM, GPU) appended by the realtime bot.

## Generated artifacts
- `generated/` — sweep outputs, plots, and backtest logs. See sub‑READMEs for details.

Most scripts write here automatically; clean up old runs to save disk space.
