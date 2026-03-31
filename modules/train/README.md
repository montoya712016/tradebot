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

## Entry points
```bash
python scripts/train.py
python scripts/explore.py
python scripts/run_independent_step_explores.py
```
