Sniper Tradebot — Repository Map & Quickstart
==============================================

Este repositório está organizado seguindo uma arquitetura modular e profissional, separando claramente a lógica de negócio dos pontos de entrada de execução. A estrutura é dividida em cinco pilares principais: `scripts/` contém todos os comandos unificados para operação do sistema (treino, backtest, bot live); `modules/` concentra a lógica compartilhada de domínio (limpeza de dados, engenharia de features, modelos); `core/` define os contratos base e classes fundamentais; `realtime/` gerencia o bot de execução em tempo real e o dashboard; e `data/` centraliza todos os artefatos de execução, logs e estados temporários, mantendo a raiz do projeto limpa e organizada.

This repository hosts a full walk-forward trading stack: data ingestion, feature prep, model training, backtesting, threshold tuning, and a real-time bot with a live dashboard. Models and large caches live outside the repo (e.g., `d:\astra\models_sniper\`).

Top-Level Folders
-----------------
- `scripts/` - Unified CLI entry points for all core operations (training, backtesting, live bot, exploration).
- `modules/` - Shared business logic and domain code (backtest, config, data_providers, prepare_features, train, etc.).
- `core/` - Foundational contracts and base executor classes.
- `realtime/` - Real-time bot logic, market data ingestion, and dashboard UI.
- `data/` - Runtime artifacts (equity history, live state, generated plots, run logs).

End-to-End Flow (Quick Map)
---------------------------
All primary operations are executed via the `scripts/` directory.

1) Data Sync & Cache Building  
   - `python scripts/data_sync.py` (Downloads OHLC data for top symbols to MySQL and builds parquet caches)

2) Target Labels Generation
   - `python scripts/refresh_labels.py` (Applies the trade contract to generate entry/exit labels for training)

3) Train Walk-Forward Models  
   - `python scripts/train.py` (Executes the full dataset assembly and walk-forward training pipeline)

4) Backtest & Exploration  
   - `python scripts/backtest.py` (Runs portfolio-level backtesting across available assets)
   - `python scripts/explore.py` (Explores hyperparameter combinations across walk-forward models)
   - `python scripts/resume_explore.py` (Auto-resumes the latest exploration run)

5) Live Bot + Dashboard Execution
   - `python scripts/bot_live.py` (Runs the production trading bot in paper/live mode)
   - `python scripts/bot_dashboard.py` (Runs the real-time web UI and ngrok tunnel for remote access)

Dashboards
----------
- **Real-time Live Bot**: Hosted via `scripts/bot_dashboard.py`. Features an interactive UI showing equity history, open positions, recent trades, and latency metrics.
- **Walk-Forward Explorer**: Hosted automatically during `scripts/explore_monolith.py`. Shows backtest equity curves and metric tables.

Conventions and Paths
---------------------
- **Logs & State**: All runtime states (`state.json`), intermediate cache configs, and logs are kept strictly in `data/`.
- **Models**: Saved externally to `d:\astra\models_sniper\models\`.
- **Feature Parquets**: Saved externally to `D:\astra\cache_sniper\`.

Pro-Tips
--------
- Use `python scripts/contar_linhas.py` for a quick breakdown of repository size and code statistics.
- Environment variables override default configurations. See module definitions for specific `SNIPER_*` keys.
