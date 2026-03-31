# modules/realtime

Infra de runtime realtime do projeto:
- dashboard web do bot
- autenticacao compartilhada dos dashboards
- estado e APIs do dashboard
- utilitarios ainda usados pelo bot live

## Documentacao
- [DASHBOARD.md](DASHBOARD.md) - visao consolidada dos dashboards, autenticacao, cadastro, papeis e base local de usuarios.

## Componentes
- `dashboard_server.py` - servidor Flask do dashboard realtime do bot.
- `dashboard_state.py` - estado compartilhado do dashboard do bot.
- `auth.py` - autenticacao local compartilhada entre dashboard do bot e fair explore.
- `site_oos_assets.py` - loader dos artefatos OOS do `fair_wf_explore_v5` para a landing publica.
- `templates/` - login, cadastro, administracao de usuarios e shell visual compartilhado.
- `static/` - assets visuais compartilhados, incluindo `astra_shared.css`.
- a landing publica compartilhada usa o simbolo SVG da Astra, carrega metricas reais do `fair_wf_explore_v5/robustness_report`, embute um grafico Plotly da curva OOS stitched e prioriza uma mensagem comercial centrada em robustez e drawdown controlado
- `realtime_dashboard_ngrok_monolith.py` - legado ainda referenciado por partes do runtime; nao e o entrypoint principal.

## Entry points publicos
```bash
python scripts/bot_live.py
python scripts/bot_dashboard.py
python scripts/fair_dashboard.py
```

O fluxo suportado hoje passa pelos scripts em `scripts/`, nao pela execucao direta de modulos internos.
