# data_providers/binance

Integrations and ingestion tools for Binance.

## Overview
This module contains the logic for interacting directly with Binance endpoints. It handles fetching symbol lists, downloading bulk OHLC data into MySQL, and provides the trading client used by the live executor.

## Usage
Data ingestion is orchestrated via the central cli script:
```bash
python scripts/data_sync.py
```
This script internally consumes `download_to_mysql.py` and `discover_binance_symbols.py`.
