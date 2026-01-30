# Stocks workflows

Utilities to train, backtest and paper‑trade the stock (“stocks”) version of the sniper pipeline. All scripts are CLI oriented; run from repo root.

## Prerequisites
- Python 3.12 environment with project dependencies installed.
- MySQL reachable via env vars used elsewhere in the repo.
- Tiingo API token in `TIINGO_API_KEY`.

## Core scripts
- `train_sniper_wf.py` — walk‑forward training for the stock universe.
- `prepare_features_stocks.py` — build feature tables in MySQL from raw bars.
- `backtest_single_symbol.py` — quick single‑symbol backtest for diagnostics.
- `trade_contract.py` — lightweight live/paper executor for one contract.
- `plot_symbol.py` — render recent candles + signals for a given ticker.
- `label_stats.py` — inspect label distributions to spot drift.
- `refresh_sniper_labels.py` — recompute labels in bulk after schema changes.

## Tiingo data helpers (`stocks/tiingo/`)
- `build_universe.py` — create the list of tradable tickers (writes to MySQL).
- `download_to_mysql.py` — ingest OHLCV history from Tiingo into MySQL.
- `tiingo_info.py` — sanity checks and metadata dumps for Tiingo symbols.

## Typical flows
1. Build universe & download: `python stocks/tiingo/build_universe.py && python stocks/tiingo/download_to_mysql.py`.
2. Features: `python stocks/prepare_features_stocks.py`.
3. Train: `python stocks/train_sniper_wf.py`.
4. Backtest spot checks: `python stocks/backtest_single_symbol.py --symbol AAPL`.
5. (Optional) Run contract paper bot: `python stocks/trade_contract.py`.

Logs and outputs follow the same conventions as the crypto workflows (see root README).
