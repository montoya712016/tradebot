# AGENTS.md

Guia operacional curto para agentes trabalhando neste repositório.

## Escopo

Estas regras valem para mudanças de código, scripts, dashboards, workflow, backtests, explore e runtime live.

## Princípios

- Preserve a arquitetura atual sempre que possível.
- Prefira mudanças pequenas, explícitas e verificáveis.
- Não mantenha código legado sem motivo real.
- Se algo ficou obsoleto, remova em vez de esconder.
- Se uma decisão de produto/processo mudou, a documentação deve mudar junto.

## Disciplina de documentação

Toda mudança relevante de código deve revisar a documentação afetada no mesmo trabalho.

Considere `relevante` qualquer mudança que altere:
- fluxo operacional
- entrypoints
- parâmetros principais
- defaults
- política de risco
- auth, cadastro, papéis ou permissões
- UX de dashboards
- estrutura de pastas e responsabilidades de módulos

Arquivos a revisar conforme o impacto:
- `README.md`
- `WORKFLOW.md`
- READMEs dentro de `modules/`
- READMEs dentro de `realtime/`
- docs específicas como `modules/realtime/DASHBOARD.md`

Se a mudança não exigir atualização documental, isso deve ser uma decisão consciente, não omissão.

## Expectativa padrão

Ao finalizar uma tarefa, o agente deve verificar:
- o código compila/roda no nível razoável para a mudança
- a documentação principal continua correta
- defaults e exemplos ainda refletem o estado real do repositório

## Dashboards

Mudanças de dashboard devem preservar a identidade visual compartilhada do projeto.

Ao alterar:
- login
- cadastro
- auth
- papéis
- painel de usuários
- assets visuais compartilhados

também revisar:
- `modules/realtime/DASHBOARD.md`
- `modules/realtime/README.md`
- `realtime/bot/README.md` se afetar o bot

## Cleanup

Scripts temporários, experimentais ou substituídos não devem permanecer no branch principal sem justificativa.

Se um código foi claramente substituído:
- remova-o
- ajuste imports/docs
- deixe o Git como fonte de recuperação histórica

## Regra prática final

Código e documentação devem envelhecer juntos.

Se o código mudou de forma que um humano poderia se confundir ao ler os READMEs, os READMEs precisam ser atualizados no mesmo trabalho.
