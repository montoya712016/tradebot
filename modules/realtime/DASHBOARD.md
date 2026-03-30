# Dashboard Guide

Visão consolidada do sistema de dashboards do Tradebot.

## Superfícies atuais

### 1. Dashboard do bot
- entrypoint: `python scripts/bot_dashboard.py`
- app Flask: `modules/realtime/dashboard_server.py`
- templates: `modules/realtime/templates/`
- assets: `modules/realtime/static/`
- fluxo web:
  - `/` = landing pública da Astra/Tradebot
  - `/login` = login
  - `/dashboard` = dashboard realtime protegido

### 2. Fair Explore dashboard
- entrypoint: `python scripts/fair_dashboard.py`
- app Flask inline: `scripts/fair_dashboard.py`
- reutiliza os mesmos templates e assets compartilhados do realtime
- fluxo web:
  - `/` = landing pública da Astra/Tradebot
  - `/login` = login
  - `/dashboard` = dashboard do explore protegido

## Identidade visual

Os dois dashboards agora compartilham a mesma base visual:
- branding `Astra Tradebot`
- navbar glass
- cards e hero shell no mesmo estilo
- paleta, spacing e atmosfera visual compartilhados
- CSS base comum em `modules/realtime/static/astra_shared.css`
- landing pública compartilhada em `modules/realtime/templates/landing.html`

O conteúdo e layout mudam por dashboard, mas a identidade visual é a mesma.

## Autenticação

O login foi movido para dentro da aplicação.

### O que existe hoje
- landing pública institucional antes do login
- tela oficial de login
- tela oficial de cadastro
- sessão por cookie
- logout pela própria UI
- proteção de rotas no backend

Arquivos principais:
- `modules/realtime/auth.py`
- `modules/realtime/templates/login.html`
- `modules/realtime/templates/register.html`
- `modules/realtime/templates/admin_users.html`

## Base local de usuários

Os usuários ficam em arquivo local:
- `local/dashboard_users.json`

Esse arquivo não deve ir para o GitHub e é ignorado pelo repositório.

Cada usuário pode ter:
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

As senhas são salvas com hash, não em texto puro.

## Cadastro e aprovação

Fluxo atual:
1. o usuário abre `/login`
2. clica em `Criar conta`
3. preenche o cadastro
4. a conta entra na base local com `enabled=false`
5. um admin/owner aprova ou bloqueia pela UI de usuários

Sem aprovação, o login mostra que a conta foi cadastrada mas ainda não foi liberada.

## Papéis

Hoje existem 3 níveis:

### User
- consegue entrar no dashboard se `enabled=true`
- não administra outros usuários

### Admin
- vê a tela `Usuários`
- pode liberar, bloquear e excluir usuários comuns
- pode promover/remover admins, respeitando as proteções do sistema

### Owner
- nível máximo
- pode gerenciar owners
- nenhum admin comum pode agir contra uma conta owner
- o owner não pode remover o próprio papel de owner
- o sistema não permite remover o último owner

## Admin panel

Rota:
- `/admin/users`

A tela mostra:
- total de contas
- pendentes
- liberadas
- admins
- owners
- lista completa de usuários
- status e papel
- ações de liberar, bloquear, excluir, promover admin e promover owner

Somente admins enxergam o botão `Usuários`.
As rotas também validam permissão no backend, então esconder o botão não é a única proteção.

## Segurança prática

Abrir DevTools no navegador não concede acesso extra por si só.

O importante é que:
- os dados administrativos não sejam enviados para usuários comuns
- as rotas de admin validem o papel no servidor

Hoje a proteção principal está no backend:
- páginas privadas exigem sessão válida
- `/admin/users` exige admin
- ações sensíveis exigem admin
- ações contra owner exigem owner

## Bootstrap do primeiro usuário

O modo padrão atual não cria conta bootstrap automaticamente.

Então o primeiro acesso administrativo pode exigir uma liberação inicial manual no JSON local:
- marcar `enabled=true`
- marcar `is_admin=true`
- marcar `is_owner=true`

Depois disso, os próximos usuários já podem ser aprovados pela interface.

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

## Ngrok

O Fair Explore pode abrir ngrok pelo próprio script.

O `basic auth` do ngrok deixou de ser a camada principal de login. A autenticação agora vive dentro da UI do dashboard.

## Arquivos principais

### Dashboard do bot
- `scripts/bot_dashboard.py`
- `modules/realtime/dashboard_server.py`
- `modules/realtime/dashboard_state.py`

### Fair Explore
- `scripts/fair_dashboard.py`

### Auth e UI compartilhados
- `modules/realtime/auth.py`
- `modules/realtime/templates/base.html`
- `modules/realtime/templates/login.html`
- `modules/realtime/templates/register.html`
- `modules/realtime/templates/admin_users.html`
- `modules/realtime/static/astra_shared.css`

## Observações

- O dashboard do bot e o Fair Explore já estão no mesmo padrão visual.
- O painel de usuários hoje é compartilhado entre os dois.
- O runtime do bot ainda tem partes legadas no backend que merecem refatoração futura, mas a camada de dashboard/auth já foi centralizada.
