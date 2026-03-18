# Realtime Trading Module

Módulo responsável pela execução do bot em tempo real.

## Estrutura

- `bot/`: Lógica de decisão e controle do bot
- `market_data/`: Ingestão de dados (WebSocket, REST) e janelas deslizantes
- `scoring/`: Lógica de inferência e cálculo de scores
- `dashboard/`: Interface web de monitoramento

## Execução

Este módulo não deve ser executado diretamente. Utilize os scripts unificados na raiz do projeto:

```bash
# Para rodar o bot (paper ou live)
python scripts/bot_live.py

# Para rodar o dashboard de monitoramento
python scripts/bot_dashboard.py
```
