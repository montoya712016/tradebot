crypto/ - crypto trading stack
==============================

Core entrypoint
---------------
- `realtime_sniper_live.py` - production bot (paper/live).  
  - Loads walk‑forward models from `d:\astra\models_sniper\crypto\wf_002` (path configurable).  
  - Async backfill of 1m OHLC from MySQL before scoring.  
  - Builds features with `modules.prepare_features` (same pipeline used in training).  
  - Scores in batches (thread pool) and applies decisions via `trade_contract.py` (limits, fees, exposure).  
  - Streams prices for ~256 symbols via Binance WS; pushes state to the dashboard (`modules/realtime/dashboard_server.py`).  
  - Optional ngrok custom domain (e.g., `https://astra-assistent.ngrok.app`).

Supporting scripts
------------------
- `realtime_bot.py` - shared engine (backfill, REST queue, decision workers, dashboard integration).  
- `trading_client.py` - thin REST/WS client for Binance (quotes, positions, orders).  
- `trade_contract.py` - contract with fee/slippage/size rules reused by backtests and live bot.  
- `prepare_features_crypto.py` - quick run to refresh the crypto feature cache.  
- `train_sniper_wf.py` - walk‑forward training for crypto (uses `modules/train`).  
- `refresh_sniper_labels.py` - recompute labels in cached features.  
- `backtest_single_symbol.py` - quick single‑symbol backtest (uses `modules/backtest`).

Subfolder `binance/`
--------------------
- `discover_binance_symbols.py` - builds an eligible universe by market cap / liquidity.  
- `download_to_mysql.py` - ingests 1m OHLC into MySQL.  
- `README.md` - short usage notes.

Prereqs
-------
- MySQL with 1m OHLC (see scripts above).  
- Walk‑forward models compatible with the current feature pipeline (produced in `modules/train`).  
- Python 3.12 + deps (xgboost, pandas, numpy, websocket-client, requests, mysqlclient or mysql-connector, etc.).

Typical command (paper)
-----------------------
```bash
python crypto/realtime_sniper_live.py
```

Debug tip
---------
To debug the dashboard locally, enable `DEBUG` in `modules/realtime/dashboard_server.py` and run `modules/realtime/run_dashboard.py`.
