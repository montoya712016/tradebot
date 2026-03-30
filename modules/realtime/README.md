# modules/realtime

Suporte ao bot live/paper e ao dashboard realtime.

## Componentes
- `dashboard_server.py` - servidor Flask com APIs e UI leve de monitoramento.
- `dashboard_state.py` - estado compartilhado do dashboard.
- `realtime_dashboard_ngrok_monolith.py` - helpers internos de ngrok e execução ainda usados por partes do runtime.

## Entry points públicos
```bash
python scripts/bot_live.py
python scripts/bot_dashboard.py
```

O fluxo suportado hoje passa pelos scripts em `scripts/`, não pela execução direta de módulos internos.
