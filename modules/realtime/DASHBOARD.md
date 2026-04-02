# Dashboard Guide

Visao consolidada do sistema de dashboards do Tradebot.

## Superficies atuais

### 1. Dashboard do bot
- entrypoint: `python scripts/bot_dashboard.py`
- app Flask: `modules/realtime/dashboard_server.py`
- templates: `modules/realtime/templates/`
- assets: `modules/realtime/static/`
- fluxo web:
  - `/` = landing publica da Astra/Tradebot
  - `/login` = login
  - `/dashboard` = dashboard realtime protegido

### 2. Fair Explore dashboard
- entrypoint: `python scripts/fair_dashboard.py`
- app Flask inline: `scripts/fair_dashboard.py`
- reutiliza os mesmos templates e assets compartilhados do realtime
- fluxo web:
  - `/` = landing publica da Astra/Tradebot
  - `/login` = login
  - `/dashboard` = dashboard do explore protegido
  - `/assistant` = painel administrativo para controle remoto do Codex local

## Identidade visual

Os dois dashboards compartilham a mesma base visual:
- branding `Astra Tradebot`
- navbar glass
- cards e hero shell no mesmo estilo
- paleta, spacing e atmosfera visual compartilhados
- CSS base comum em `modules/realtime/static/astra_shared.css`
- landing publica compartilhada em `modules/realtime/templates/landing.html`
- superficies protegidas com o simbolo SVG da Astra no navbar

Tambem foi consolidado um ajuste de escala e responsividade:
- tipografia e paddings mais compactos
- cards e badges em escala menos inflada
- melhor leitura em laptop e desktop sem desperdiçar altura
- comportamento mobile revisado para login, cadastro, admin, dashboard e `/assistant`
- navegação lateral do dashboard realtime escondida em mobile, mantendo as tabs principais no topo
- no `/assistant`, conversa e composer aparecem antes do histórico em telas pequenas

- na landing mobile, CTAs deixam de ocupar 100% da largura sem necessidade
- no login mobile, o card volta a ficar centrado verticalmente dentro da viewport util
- no Fair Explore mobile, a tabela principal fica contida em wrapper com scroll horizontal; as metricas seguem visiveis e o campo `Trial` foi compactado para `label/model/bt`

O conteudo e layout mudam por dashboard, mas a identidade visual e a mesma.

## Autenticacao

O login foi movido para dentro da aplicacao.

Hoje existe:
- landing publica institucional antes do login
- tela oficial de login
- tela oficial de cadastro
- sessao por cookie
- logout pela propria UI
- protecao de rotas no backend

Arquivos principais:
- `modules/realtime/auth.py`
- `modules/realtime/templates/login.html`
- `modules/realtime/templates/register.html`
- `modules/realtime/templates/admin_users.html`

## Base local de usuarios

Os usuarios ficam em:
- `local/dashboard_users.json`

Cada usuario pode ter:
- `full_name`
- `cpf`
- `phone`
- `email`
- `username`
- `password_hash`
- `enabled`
- `is_admin`
- `is_owner`
- `created_at`
- `approved_at`
- `notes`

As senhas sao salvas com hash, nao em texto puro.

## Cadastro e aprovacao

Fluxo atual:
1. o usuario abre `/login`
2. clica em `Criar conta`
3. preenche o cadastro
4. a conta entra na base local com `enabled=false`
5. um admin/owner aprova ou bloqueia pela UI de usuarios

Sem aprovacao, o login informa que a conta ainda nao foi liberada.

## Papeis

### User
- entra no dashboard se `enabled=true`
- nao administra outros usuarios

### Admin
- ve a tela `Usuarios`
- pode liberar, bloquear e excluir usuarios comuns
- pode promover/remover admins respeitando as protecoes do sistema
- no Fair Explore, tambem acessa `/assistant`

### Owner
- nivel maximo
- pode gerenciar owners
- nenhum admin comum pode agir contra uma conta owner
- o owner nao pode remover o proprio papel de owner
- o sistema nao permite remover o ultimo owner

## Admin panel

Rota:
- `/admin/users`

A tela mostra:
- total de contas
- pendentes
- liberadas
- admins
- owners
- lista completa de usuarios
- status e papel
- acoes de liberar, bloquear, excluir, promover admin e promover owner

Somente admins enxergam o botao `Usuarios`.
As rotas tambem validam permissao no backend.

## Seguranca pratica

Abrir DevTools no navegador nao concede acesso extra por si so.

O importante e que:
- os dados administrativos nao sejam enviados para usuarios comuns
- as rotas de admin validem o papel no servidor

Protecoes principais hoje:
- paginas privadas exigem sessao valida
- `/admin/users` exige admin
- acoes sensiveis exigem admin
- acoes contra owner exigem owner
- `/assistant` exige admin

## Bootstrap do primeiro usuario

O modo padrao atual nao cria conta bootstrap automaticamente.

O primeiro acesso administrativo pode exigir uma liberacao inicial manual no JSON local:
- marcar `enabled=true`
- marcar `is_admin=true`
- marcar `is_owner=true`

Depois disso, os proximos usuarios podem ser aprovados pela interface.

## Endpoints relevantes

### Comuns aos dashboards
- `/`
- `/login`
- `/register`
- `/dashboard`
- `POST /logout`
- `/admin/users`

### Dashboard do bot
- `/api/state`
- `POST /api/update`
- `/api/health`
- `/api/config`
- `/api/system`
- `/api/ohlc_window`

### Fair Explore
- `/api/data`
- `/artifact/<path>`

### Assistant remoto do Fair Explore
- `GET /assistant`
- `GET /api/assistant/capabilities`
- `GET /api/assistant/conversations`
- `GET /api/assistant/conversations/<conversation_id>`
- `GET /api/assistant/jobs`
- `POST /api/assistant/jobs`
- `GET /api/assistant/jobs/<job_id>`
- `GET /api/assistant/jobs/<job_id>/log`
- `POST /api/assistant/jobs/<job_id>/cancel`

## Ngrok

O Fair Explore pode abrir ngrok pelo proprio script.

O basic auth do ngrok deixou de ser a camada principal de login. A autenticacao vive dentro da UI do dashboard.

## Arquivos principais

### Dashboard do bot
- `scripts/bot_dashboard.py`
- `modules/realtime/dashboard_server.py`
- `modules/realtime/dashboard_state.py`

### Fair Explore
- `scripts/fair_dashboard.py`

### Assistant remoto
- `modules/realtime/remote_control.py`
- `modules/realtime/templates/fair_assistant.html`

### Auth e UI compartilhados
- `modules/realtime/auth.py`
- `modules/realtime/site_oos_assets.py`
- `modules/realtime/templates/base.html`
- `modules/realtime/templates/login.html`
- `modules/realtime/templates/register.html`
- `modules/realtime/templates/admin_users.html`
- `modules/realtime/static/astra_shared.css`

## Landing publica

As duas landings publicas usam artefatos reais do OOS walk-forward `v5`:
- metricas carregadas de `data/generated/fair_wf_explore_v5/robustness_report`
- curva stitched OOS embutida em Plotly para uso visual no site
- mesma imagem e mesmo bloco de metricas tanto no realtime quanto no Fair Explore

## Observacoes

- O dashboard do bot e o Fair Explore ja estao no mesmo padrao visual.
- O painel de usuarios hoje e compartilhado entre os dois.
- O `/assistant` agora funciona como uma superficie de chat para o Codex local via `codex app-server`: mensagem do usuario, resposta final do Codex e atividade operacional separada.
- As respostas finais do `/assistant` agora renderizam markdown com tipografia dedicada; referencias locais do repositório deixam de aparecer como markdown cru.
- O historico do `/assistant` agora e por conversa; com uma thread selecionada, o proximo envio continua a mesma sessao via `threadId` persistido no `codex app-server`.
- No desktop, o `/assistant` agora usa layout de app: sidebar de historico na esquerda em altura integral, conversa com scroll proprio e composer fixado no rodape do painel principal.
- O contexto operacional do ultimo turno foi movido para uma sidebar na direita; o centro do `/assistant` fica mais proximo de um chat, com conversa e composer.
- O composer do `/assistant` foi reduzido ao essencial: campo de mensagem com auto-expansao ate algumas linhas, selects compactos sem legendas redundantes e menos ruido visual persistente.
- Os painéis rolaveis do `/assistant` usam scrollbar vertical padronizada; o historico deixa de exibir scroll horizontal.
- O runner do `/assistant` agora fala com o `codex app-server`, o mesmo stack local usado pela extensao OpenAI no VS Code, em vez de depender de `codex exec --json`.
- O remote expõe controle de `model`, `reasoning_effort` e `access_mode`, incluindo execucao sem sandbox por `danger-full-access`.
- O runner do remote injeta `rg.exe` e `git.exe` no `PATH` do subprocesso para aproximar o ambiente do dashboard ao ambiente do terminal/VS Code.
- Jobs do `/assistant` agora fazem reconciliacao de estado: se um job marcado como `running` perder o processo vivo ou ficar sem atividade nova por tempo demais, ele sai automaticamente de `running` e vira falha stale, evitando travas falsas na UI.
- O assistant controla o CLI local, nao a interface do VS Code.
- O runtime do bot ainda tem partes legadas no backend que merecem refatoracao futura, mas a camada de dashboard/auth ja foi centralizada.
