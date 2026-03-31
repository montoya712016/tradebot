# Fair Explore Workflow

Este documento descreve o workflow atual de exploração e validação fair usado no repositório.

## 1. Exploração por steps independentes

```powershell
python scripts/run_independent_step_explores.py
```

O orquestrador roda cada step histórico de forma independente e gravável:

- steps: `1440, 1260, 1080, 900, 720, 540, 360`
- root padrão: `data/generated/fair_wf_explore_v6/`
- resume granular por `refresh`, `train` e `backtest`
- dashboard automático com acompanhamento em tempo real
- prewarm obrigatório por step antes do primeiro refresh
- build global de OHLC antes de qualquer step
- build global do cache de features do preset ativo antes de qualquer step
- sizing automático de workers por workload antes de cada fase principal

## 2. Como a v6 busca parâmetros

Na `v6`, o explore continua com risco fixo, mas abre um espaço pequeno de treino/calibração e usa um preset de features reduzido.

Preset de features:
- `core80`
- cache de features separado por preset quando não for `full`
- prewarm integrado de features/OHLC por padrão
- pass separado de OHLC opcional por `WF_EXPLORE_PREWARM_SEPARATE_OHLC=1`

Parâmetros de edge otimizados:
- `label_profit_thr`
- `exit_ema_span_min`
- `exit_ema_init_offset_pct`
- `entry_ratio_neg_per_pos`
- `calib_tail_blend`
- `calib_tail_boost`
- `tau_entry`

Espaço estruturado de treino:
- `entry_ratio_neg_per_pos = (4.0, 6.0, 8.0)`
- `calib_tail_blend = (0.60, 0.70, 0.80)`
- `calib_tail_boost = (1.75, 2.25, 2.75)`
- `top_metric_qs` em poucos presets curtos
- `top_metric_min_count = (48, 64)`

Parâmetros de risco fixos:
- `total_exposure = 1.00`
- `max_trade_exposure = 0.10`
- `min_trade_exposure = 0.02`

`max_positions` saiu do explore novo. O cap simultâneo agora é implícito: a carteira para de abrir novas posições quando o orçamento de exposição já foi consumido.

Plano por step:
- `56` refreshes
- `2` retrains por refresh
- `21` backtests por retrain

Total por step:
- `56` refreshes
- `112` treinos
- `2352` backtests

A busca continua determinística e espalhada. Os refreshes percorrem um subconjunto bem distribuído do grid completo de labels/contrato, os retrains cobrem um grid pequeno e uniforme de presets de treino, e cada retrain varre `tau` em `0.70..0.90` (passo `0.01`).

Ordem operacional padrão do script:
1. build global do cache OHLC base para o universo inteiro
2. build global do cache OHLC `5m` para o universo inteiro
3. build global do cache de features do preset ativo (`features_pf_5m_core80`)
4. prewarm integrado do cache de features dos steps (`tail_1260d..tail_180d`)
5. opcionalmente, pass separado de OHLC por step se `WF_EXPLORE_PREWARM_SEPARATE_OHLC=1`
6. refreshes de labels
7. retrains
8. backtests

Concorrência:
- os entrypoints não assumem mais `4/8/16` como padrão fixo
- cada fase escolhe `max_workers` automaticamente com base em CPU e RAM
- o `GuardedRunner` continua reduzindo a concorrência efetiva se a memória apertar
- o `run_independent_step_explores.py` registra telemetria de workers, duração e pico de memória por workload em `modules/utils/parallel_runtime_telemetry.json`

## 3. Validação OOS walk-forward

Depois de concluir os steps desejados:

```powershell
python scripts/run_oos_walkforward.py
```

Esse script:
- lê os pools já explorados
- seleciona representantes por step
- costura os segmentos OOS em uma curva única

Seleção padrão atual no OOS:
- `WF_OOS_SCORE_MODE=survival` por padrão
- `WF_OOS_SELECTOR=family_causal` por padrão
- o score base de seleção passa a penalizar mais cauda esquerda e exposição
- a seleção agora monta shortlist por família, aplica continuidade causal entre steps e escolhe a variante mais prudente dentro da família
- o score legado ainda pode ser reativado com `WF_OOS_SCORE_MODE=legacy`
- a leitura de `explore_runs.csv` no OOS usa `duckdb` em vez de `pandas.read_csv`

Saída padrão:
- `data/generated/fair_wf_explore_v6/robustness_report/`

## 4. Observações importantes

- O step `180d` foi removido do explore atual porque ele não tem mais uma perna OOS comparável dentro dessa grade.
- O script de OOS ainda suporta usar `180d` se existir um step concluído manualmente, mas isso não faz parte do workflow padrão da `v6`.
- Se for preciso reiniciar o processo, o orquestrador continua de onde parou pelo `explore_runs.csv`.
- O `duckdb` passou a ser usado para ler `explore_runs.csv` no orchestrator, no explorer e no OOS. Isso ajuda no overhead de CSV, mas não troca o gargalo principal do pipeline, que continua sendo refresh de features, montagem do dataset e treino.
