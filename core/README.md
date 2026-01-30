# Core Module

Módulo central compartilhado por todos os componentes do Tradebot.

## Estrutura

```
core/
├── contracts/     # Definições de contratos de trading
├── executors/     # Executores de ordens (paper, live)
├── models/        # Wrappers de modelos ML
└── utils/         # Utilitários compartilhados
```

## Uso

```python
from core.contracts import TradeContract, DEFAULT_TRADE_CONTRACT
from core.executors import PaperExecutor, LiveExecutor
from core.utils import notify
```
