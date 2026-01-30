# Realtime Trading Module

Módulo responsável pela execução do bot em tempo real.

## Estrutura

- `bot/`: Lógica de decisão e controle do bot
- `market_data/`: Ingestão de dados (WebSocket, REST) e janelas deslizantes
- `scoring/`: Lógica de inferência e cálculo de scores
- `dashboard/`: Interface web de monitoramento

## Execução

```python
from realtime.bot.sniper import LiveDecisionBot
from realtime.bot.settings import LiveSettings

settings = LiveSettings()
bot = LiveDecisionBot(settings)
bot.run()
```
