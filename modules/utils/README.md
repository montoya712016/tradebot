# modules/utils

Helpers operacionais compartilhados do repositório.

## O que fica aqui
- `paths.py`
  Resolve roots de modelos, caches e artefatos.
- `resource_sizing.py`
  Escolha automática do `max_workers` inicial por tipo de workload e reaproveitamento de telemetria de runs anteriores.
- `guarded_runner.py`
  Wrapper padrão para rotinas paralelas com proteção de RAM e thermal guard.
- `adaptive_parallel.py`
  Executor em threads com cap adaptativo de concorrência baseado em memória.
- `progress.py`
  Barras de progresso e printers de linha única para terminal/log.
- `thermal_guard.py`
  Pausa ou aborta trabalho quando a máquina entra em zona térmica ruim.

## Padrão de paralelização

O padrão atual do projeto é em duas camadas:

1. escolher um `max_workers` inicial automaticamente
2. deixar o `GuardedRunner` reduzir a concorrência efetiva se a RAM apertar

Isso evita dois extremos:
- defaults fixos do tipo `4/8/16` sem contexto da máquina
- explosão de threads só porque o host tem muitos núcleos

### 1. Sizing automático

Use `resource_sizing.py` para decidir o `max_workers` inicial.

API principal:
- `host_resources()`
- `recommend_workers(kind, requested=0)`
- `apply_env_worker_default(env_name, kind, default=0)`
- `record_workload_observation(...)`
- `telemetry_file_path()`

Workloads padronizados hoje:
- `ohlc_1m`
- `ohlc_5m`
- `feature_cache`
- `feature_cache_global`
- `dataset`
- `labels_refresh`
- `explore`

Regra prática:
- entrypoints devem chamar `apply_env_worker_default(...)` cedo
- módulos internos usam esse valor como fallback quando a env ainda não foi setada
- override manual por env continua válido e tem precedência

### Telemetria persistida

O sizing agora grava observações por host e por workload em:

- [parallel_runtime_telemetry.json](/d:/astra/tradebot/modules/utils/parallel_runtime_telemetry.json)

Esse arquivo:
- fica versionado no repositório
- não entra em `.gitignore`
- acumula histórico útil para máquinas diferentes

O que é gravado por sample:
- `workers`
- `duration_s`
- `peak_process_mb`
- `estimated_per_worker_mb`
- `peak_used_pct`
- `min_available_mb`
- `seconds_per_worker`
- `seconds_per_unit`

Uso no sizing:
- o perfil base continua existindo no código
- se já houver histórico suficiente para aquele host/workload, o `resource_sizing.py` sobe o `per_worker_mb` efetivo de forma conservadora
- isso reduz a chance de repetir paralelismo agressivo demais em runs futuros

Exemplo:

```python
from utils.resource_sizing import apply_env_worker_default

workers = apply_env_worker_default("SNIPER_CACHE_WORKERS", "feature_cache")
```

### 2. Guard de RAM

Use `GuardedRunner` para execução paralela longa.

API principal:
- `GuardedRunner.make_policy()`
- `GuardedRunner.adaptive_map(...)`

O `GuardedRunner` encapsula:
- `ThermalGuard`
- `AdaptiveParallelPolicy`
- classificação padronizada de erros de guard

Parâmetros importantes do policy:
- `max_ram_pct`
- `min_free_mb`
- `per_worker_mem_mb`
- `critical_ram_pct`
- `critical_min_free_mb`
- `min_workers`

O comportamento esperado é:
- começar com um `max_workers` razoável
- reduzir submissão se a memória disponível cair
- abortar só em zona crítica, se configurado para isso
- e, a cada run, registrar mais evidência para calibrar o sizing inicial futuro

## Prefixos de env

Cada workload longo tem um prefixo de env próprio para o `GuardedRunner`.

- `SNIPER_CACHE_*`
  build de feature cache
- `SNIPER_DATASET_*`
  montagem de dataset de treino
- `SNIPER_LABELS_*`
  refresh de labels no cache
- `OHLC_CACHE_*`
  build de OHLC base

Essas envs controlam a política adaptativa, não a heurística inicial de sizing.

Exemplos:
- `SNIPER_CACHE_RAM_PCT`
- `SNIPER_CACHE_MIN_FREE_MB`
- `SNIPER_CACHE_PER_WORKER_MB`
- `OHLC_CACHE_CRITICAL_RAM_PCT`

## Padrão de progresso

O repositório usa dois níveis:

### `progress(...)`

Wrapper simples:
- usa `tqdm` se disponível
- cai para ETA textual se não houver `tqdm`

Bom para loops lineares e curtos.

### `LineProgressPrinter`

Padrão recomendado para processos longos e logs contínuos.

Uso típico:

```python
from utils.progress import LineProgressPrinter

p = LineProgressPrinter(prefix="cache", total=100)
p.update(17, current="BTCUSDT")
p.close()
```

Boas práticas:
- use `prefix` curto e consistente
- em TTY ele atualiza em uma linha
- fora de TTY ele vira logging periódico
- chame `close()` no final

## Convenções do repositório

- entrypoints escolhem workers automáticos no começo da execução
- módulos internos nunca devem assumir `16` como padrão universal
- overrides manuais via env continuam permitidos
- workloads memory-bound devem usar `GuardedRunner`, não `ThreadPoolExecutor` cru
- progresso novo deve preferir `LineProgressPrinter` em vez de prints soltos

## Quando não usar isso

- loops muito pequenos e triviais
- código onde o custo dominante não é CPU/I/O e o overhead de threads não compensa
- fluxos onde a ordem estrita de execução é necessária
