# modules/utils

Grab‑bag of helpers used everywhere.

## Highlights
- `db.py` — MySQL connectors, query helpers, retries.
- `time_utils.py` — timezone-safe conversions, bar alignment, timestamp rounding.
- `logging_utils.py` — structured logging setup used by CLI scripts.
- `math_utils.py` — small numeric helpers (z‑scores, rolling ops).
- `plot_utils.py` — shared color palettes and matplotlib helpers.
- `sysinfo.py` — GPU/CPU/RAM probes for dashboard/system monitor.

Import sparingly to avoid circular deps; most scripts import only what they need.
