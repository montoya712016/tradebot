Sniper Tradebot - Repository Map
================================

This repo contains a full walk-forward trading stack: data ingestion, feature engineering, model training, backtesting, threshold tuning, realtime execution, and a hybrid supervised+RL pipeline.

Top-level folders
-----------------
- `crypto/` - crypto entrypoints and Binance helpers.
- `stocks/` - equities/Tiingo pipeline.
- `modules/` - shared libraries (`prepare_features`, `train`, `backtest`, `thresholds`, `realtime`, `supervised`, `env_rl`, `rl`, `utils`).
- `realtime/` - realtime bot runtime (`realtime.bot.sniper`) and market-data adapters.
- `data/` - runtime artifacts (equity history, sysmon, live state, reports).

End-to-end flow
---------------
1) Discover and download data
   - Crypto: `crypto/binance/discover_binance_symbols.py`, `crypto/binance/download_to_mysql.py`
   - Stocks: `stocks/tiingo/build_universe.py`, `stocks/tiingo/download_to_mysql.py`
2) Prepare features and labels
   - `modules/prepare_features/prepare_features.py`
3) Train walk-forward supervised models
   - `modules/train/train_sniper_wf.py` (or wrappers in `crypto/` and `stocks/`)
4) Backtest and portfolio runs
   - `modules/backtest/single_symbol.py`, `modules/backtest/wf_portfolio.py`, `modules/backtest/wf_backtest_sweep.py`
5) Threshold tuning
   - `modules/thresholds/optimize_thresholds_wf_ga.py`
6) Hybrid pipeline (supervised + RL)
   - `crypto/train_hybrid_wf.py` (or `modules/rl/run_hybrid_wf_pipeline.py`)
7) Realtime run
   - `python crypto/realtime_sniper_live.py`

Where to read next
------------------
- `crypto/README.md`
- `stocks/README.md`
- `modules/README.md`
- `data/README.md`

Conventions
-----------
- Runtime artifacts: `data/`
- Models: `d:/astra/models_sniper/...`
- Market-cap seed universe: `modules/top_market_cap.txt`
