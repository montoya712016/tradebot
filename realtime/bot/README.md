# realtime/bot

Núcleo do bot live/paper.

## Estrutura
- `sniper.py` - loop principal do bot.
- `settings.py` - configuração operacional.
- `components/` - partes do bot ainda acopladas ao runtime principal.
- `services/` - integrações externas e serviços operacionais do bot.

## services/
- `pushover.py` - carregamento de credenciais e envio de notificações.
- `dashboard.py` - subida do dashboard e abertura opcional de túnel ngrok.
- `binance.py` - operações encapsuladas via `BinanceTrader`.
- `executors.py` - adaptadores de execução paper/Binance usados pelo bot.

## Dashboard
Para a documentação consolidada do dashboard, login, cadastro, papéis e base local de usuários:
- [../../modules/realtime/DASHBOARD.md](../../modules/realtime/DASHBOARD.md)

O objetivo dessa pasta é reduzir o acoplamento do loop principal com detalhes de infraestrutura.
