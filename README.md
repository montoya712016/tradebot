Sniper Tradebot — repository map and quickstart
==============================================

This repo hosts a full walk‑forward trading stack: data ingestion, feature prep, model training, backtesting, threshold tuning, and a realtime bot + dashboard. Models and large caches live outside the repo (ex.: `d:\astra\models_sniper\`).

Top‑level folders
-----------------
- `crypto/` - production crypto bot (`realtime_sniper_live.py`) and Binance helpers.
- `stocks/` - scripts for the equities/Tiingo pipeline.
- `modules/` - shared libraries (prepare_features, train, backtest, realtime dashboard, thresholds, utils, config, plotting).
- `realtime_bot_antigo/` - legacy 1s bot kept for reference only.
- `data/` - runtime artefacts (equity history, sysmon, live state, sweep outputs).
- Aux files: `state.json`, `tmp_state.json`, `tmp_ohlc.json`, `contar_linhas.py`.

End‑to‑end flow (quick map)
---------------------------
1) Discover & download data  
   - Crypto: `crypto/binance/discover_binance_symbols.py`, `crypto/binance/download_to_mysql.py`  
   - Stocks: `stocks/tiingo/build_universe.py`, `stocks/tiingo/download_to_mysql.py`
2) Prepare features & cache  
   - `modules/prepare_features/prepare_features.py` (configs in `pf_config.py`)
3) Train walk‑forward models  
   - `modules/train/train_sniper_wf.py` or loops `wf_random_loop*.py`
4) Backtest / sweep / portfolio  
   - `modules/backtest/single_symbol.py`, `modules/backtest/portfolio.py`, `modules/backtest/wf_backtest_sweep.py`
5) Tune thresholds  
   - `modules/thresholds/optimize_thresholds_wf_ga.py`
6) Run realtime + dashboard  
   - Bot: `python crypto/realtime_sniper_live.py` (paper/live)  
   - Dashboard: `modules/realtime/run_dashboard.py` or `realtime_dashboard_ngrok_monolith.py`

Dashboards
----------
- Realtime (production): `modules/realtime` (templates + static). Server ingests state pushed by the bot (`dashboard_server.py`).
- WF monitor: `modules/train/wf_dashboard_*`.

Where to read next
------------------
- `crypto/README.md`
- `stocks/README.md`
- `modules/README.md` (and sub‑readmes inside each module)
- `realtime_bot_antigo/README.md`
- `data/README.md`

Conventions and paths
---------------------
- Data and runtime artefacts: `data/`
- Models: `d:\astra\models_sniper\...`
- Market‑cap seeds: `modules/top_market_cap.txt`

Tip
---
For notifications see `modules/utils/pushover_notify.py`. For counting lines quickly, run `python contar_linhas.py`.
