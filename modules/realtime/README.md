# realtime/

Helpers used by the live real-time bot (via `scripts/bot_live.py`) and its dashboards.

## Components
- `bot/` — logical components of the Sniper trading bot (decision loop, settings loader, etc.).
- `market_data/` — fetching active streaming data, rolling OHLC windows, REST backfill logic for gaps.
- `dashboard_server.py` — Flask server exposing the React-lite UI and state APIs (`/api/ohlc_window`, `/api/state`, etc.).
- `realtime_dashboard_ngrok_monolith.py` — orchestrates the Flask UI to be exposed securely over ngrok to the public internet.

## Execution
Real-time processes should be launched via the central unified scripts:
```bash
# To run the live (or paper) trading bot
python scripts/bot_live.py

# To host the dashboard monitoring it
python scripts/bot_dashboard.py
```
