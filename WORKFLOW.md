# Fair Explore Workflow

Este documento descreve o workflow atual de exploração e validação fair usado no repositório.

## 1. Exploração por steps independentes

```powershell
python scripts/run_independent_step_explores.py
```

O orquestrador roda cada step histórico de forma independente e gravável:

- steps: `1440, 1260, 1080, 900, 720, 540, 360`
- root padrão: `data/generated/fair_wf_explore_v5/`
- resume granular por `refresh`, `train` e `backtest`
- dashboard automático com acompanhamento em tempo real

## 2. Como a v5 busca parâmetros

Na `v5`, o explore não tenta mais otimizar treino e risco ao mesmo tempo.

Parâmetros de edge otimizados:
- `label_profit_thr`
- `exit_ema_span_min`
- `exit_ema_init_offset_pct`
- `tau_entry`

Parâmetros de treino fixos:
- `entry_ratio_neg_per_pos = 6.0`
- `calib_tail_blend = 0.70`
- `calib_tail_boost = 2.25`

Parâmetros de risco fixos:
- `max_positions = 10`
- `total_exposure = 1.00`
- `max_trade_exposure = 0.10`

Plano por step:
- `49` refreshes
- `1` retrain por refresh
- `26` backtests por refresh

Total por step:
- `49` refreshes
- `49` treinos
- `1274` backtests

A busca deixou de ser puramente aleatória. Os refreshes percorrem um subconjunto determinístico e espalhado do grid completo de labels/contrato, enquanto cada refresh varre todo o grid de `tau` (`0.70..0.95`, passo `0.01`).

## 3. Validação OOS walk-forward

Depois de concluir os steps desejados:

```powershell
python scripts/run_oos_walkforward.py
```

Esse script:
- lê os pools já explorados
- seleciona representantes por step
- costura os segmentos OOS em uma curva única

Saída padrão:
- `data/generated/fair_wf_explore_v5/robustness_report/`

## 4. Observações importantes

- O step `180d` foi removido do explore atual porque ele não tem mais uma perna OOS comparável dentro dessa grade.
- O script de OOS ainda suporta usar `180d` se existir um step concluído manualmente, mas isso não faz parte do workflow padrão da `v5`.
- Se for preciso reiniciar o processo, o orquestrador continua de onde parou pelo `explore_runs.csv`.
