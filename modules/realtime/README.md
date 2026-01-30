# modules/realtime

Helpers used by `crypto/realtime_sniper_live.py` and related dashboards.

## Components
- `ws_client.py` — Binance (and compatible) websocket multiplexing with backpressure control.
- `decision_loop.py` — scoring queue, throttling, exposure limits, paper/live order router.
- `dashboard_ngrok_monolith.py` — Flask/Werkzeug server that serves the React-lite dashboard and exposes JSON APIs (`/api/ohlc_window`, `/api/state`, etc.).
- `sysmon.py` — system metrics sampler (CPU, RAM, GPU) exposed to the dashboard.
- `state_sync.py` — persistence of equity, positions, and last decisions to JSON for UI refresh.

## Notes
- Designed to run multi-threaded; avoid blocking the websocket thread.
- Dashboard polls JSON endpoints; keep responses small (windowed OHLC, not full history).
- Ngrok support is optional but enabled by default for remote access.
