# modules/train

Treino walk-forward e explorer fair.

## Peças principais
- `sniper_trainer.py` - fit/eval/persistência dos modelos.
- `train_sniper_wf.py` - orquestra treino walk-forward por períodos.
- `wf_portfolio_explorer.py` - explorer fair atual.

## Explorer fair atual
O `wf_portfolio_explorer.py` da `v6` trabalha assim:

- usa preset de features `core80`
- grava cache de features em diretório segregado por preset quando não for `full`
- constrói primeiro um cache global de OHLC `5m`
- depois constrói um cache global do preset ativo (`features_pf_5m_core80`) para os backtests OOS do explore
- dimensiona workers automaticamente por fase:
  - `ohlc_1m`
  - `ohlc_5m`
  - `feature_cache_global`
  - `feature_cache`
  - `dataset`
  - `labels_refresh`
- faz prewarm por step antes do primeiro refresh:
  - por padrão, o bootstrap de features já popula o cache OHLC base
  - o pass separado de OHLC é opcional
  - a triagem de elegibilidade do step usa primeiro os metadados do cache `ohlc_5m` e só faz fallback para load real nos casos limítrofes
- o score dos backtests do explore ficou mais simples e mais causal:
  - base = `log1p(retorno)` penalizado por `max_dd` absoluto
  - atividade medida por `clusters` de entrada, não por contagem bruta de trades
  - cauda medida por `worst_rolling_90d` e pelo `worst_trade` efetivo no portfólio
  - o CSV do explore agora grava `worst_trade` e `worst_trade_raw`
- no workflow padrão do orquestrador, os caches OHLC globais (`1m` e `5m`) são construídos antes de qualquer prewarm de step
- os refreshes/treinos usam o cache recortado do step, mas o prepare/backtest usa o cache global do preset para conseguir enxergar a janela OOS seguinte
- dentro de cada label, o trainer agora monta um `full_pool` base com o maior `neg_per_pos` do explore e reutiliza esse pool entre os retrains do mesmo label, reduzindo rebuild redundante do dataset
- otimiza `7` parâmetros entre edge e treino leve:
  - `label_profit_thr`
  - `exit_ema_span_min`
  - `exit_ema_init_offset_pct`
  - `entry_ratio_neg_per_pos`
  - `calib_tail_blend`
  - `calib_tail_boost`
  - `tau_entry`
- fixa risco em:
  - `total_exposure = 1.00`
  - `max_trade_exposure = 0.10`
  - `min_trade_exposure = 0.02`

`max_positions` não faz mais parte do explore novo. O limite simultâneo passa a ser imposto indiretamente pelo orçamento de exposição.

Plano padrão:
- `56` refreshes
- `2` retrains por refresh
- `21` backtests por retrain

Os refreshes cobrem um subconjunto determinístico e bem distribuído do grid de contratos, os retrains percorrem um grid pequeno e uniforme de presets de treino/calibração, e os backtests varrem `tau` de `0.70` a `0.90` em passo `0.01`.

## Padrão operacional de workers

- defaults de workers agora vêm de `modules/utils/resource_sizing.py`
- overrides via env continuam possíveis
- o `GuardedRunner` segue sendo a segunda camada de proteção quando o workload fica memory-bound
- o orchestrator grava observações por host/workload em `modules/utils/parallel_runtime_telemetry.json`

## Entry points
```bash
python scripts/train.py
python scripts/explore.py
python scripts/run_independent_step_explores.py
```
