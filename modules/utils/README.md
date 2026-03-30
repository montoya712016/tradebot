# modules/utils

Helpers compartilhados do repositório.

## Destaques
- `paths.py` - resolução de roots de modelo, cache e artefatos.
- `guarded_runner.py` - paralelização com proteção por RAM/recursos.
- `progress.py` - barras e progresso em linha.
- `notify.py` / `pushover_notify.py` - notificações.
- `sysinfo.py` - métricas de CPU/RAM/GPU.

Este pacote concentra utilidades operacionais; os módulos de domínio devem depender dele apenas no que for realmente necessário.
