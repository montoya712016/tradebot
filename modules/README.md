# modules/

Pacote central de lógica compartilhada do tradebot.

## Subpacotes
- `backtest/` - simulação single-symbol e portfolio, além das rotinas usadas no OOS walk-forward.
- `config/` - contratos de trade, universos de símbolos e defaults estruturais.
- `data_providers/` - integrações externas, principalmente Binance.
- `prepare_features/` - build de features, labels e datasets.
- `plotting/` - geração de gráficos e relatórios HTML.
- `realtime/` - dashboard Flask e componentes de suporte ao bot live.
- `thresholds/` - utilidades auxiliares para curvas e estudos de limiar.
- `train/` - treino walk-forward e o explorer fair.
- `utils/` - helpers genéricos de path, paralelização, logs e monitoramento.

Os entrypoints públicos ficam em `scripts/`.
