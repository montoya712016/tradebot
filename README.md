### tradebot - Sniper tradebot (walk-forward)

Este repositorio agora usa apenas `modules/`, organizado pelo fluxo principal:

1) Buscar dados da Binance
2) Preparar features e salvar cache
3) Treinar modelos
4) Backtest
5) Otimizar thresholds
6) Operar em tempo real (dashboard)

### Entrypoints (por fluxo)

1) Binance
- `python modules/binance/discover_binance_symbols.py`
- `python modules/binance/download_to_mysql.py`

2) Features + cache
- `python modules/prepare_features/prepare_features.py`
- `python modules/prepare_features/refresh_sniper_labels_in_cache.py`

3) Treino
- `python modules/train/train_sniper_wf.py`

4) Backtest
- `python modules/backtest/single_symbol.py`
- `python modules/backtest/portfolio.py`
- `python modules/backtest/wf_portfolio.py`

5) Thresholds
- `python modules/thresholds/optimize_thresholds_wf_ga.py`

6) Realtime
- `python modules/realtime/realtime_dashboard_ngrok_monolith.py`
- `python modules/realtime/run_dashboard.py --demo`

7) WF Monitor
- `python modules/train/wf_dashboard_ngrok_monolith.py`
- `python modules/train/run_wf_dashboard.py`

### Observacoes

- `models_sniper/` fica fora do repo (em `d:\astra\models_sniper`).
- Saidas geradas por scripts (CSVs/plots/caches de GA/WF) vao para `data/generated/` na raiz do repo.
