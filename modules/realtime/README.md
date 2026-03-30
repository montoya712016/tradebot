# modules/realtime

Infra de runtime realtime do projeto:
- dashboard web do bot
- autenticação compartilhada dos dashboards
- estado e APIs do dashboard
- utilitários ainda usados pelo bot live

## Documentação
- [DASHBOARD.md](DASHBOARD.md) - visão consolidada dos dashboards, autenticação, cadastro, papéis e base local de usuários.

## Componentes
- `dashboard_server.py` - servidor Flask do dashboard realtime do bot.
- `dashboard_state.py` - estado compartilhado do dashboard do bot.
- `auth.py` - autenticação local compartilhada entre dashboard do bot e fair explore.
- `templates/` - login, cadastro, administração de usuários e shell visual compartilhado.
- `static/` - assets visuais compartilhados, incluindo `astra_shared.css`.
- `realtime_dashboard_ngrok_monolith.py` - legado ainda referenciado por partes do runtime; não é o entrypoint principal.

## Entry points públicos
```bash
python scripts/bot_live.py
python scripts/bot_dashboard.py
python scripts/fair_dashboard.py
```

O fluxo suportado hoje passa pelos scripts em `scripts/`, não pela execução direta de módulos internos.
