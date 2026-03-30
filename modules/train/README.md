# modules/train

Treino walk-forward e explorer fair.

## Peças principais
- `sniper_trainer.py` - fit/eval/persistência dos modelos.
- `train_sniper_wf.py` - orquestra treino walk-forward por períodos.
- `wf_portfolio_explorer.py` - explorer fair atual.

## Explorer fair atual
O `wf_portfolio_explorer.py` da `v5` trabalha assim:

- otimiza só `4` parâmetros de edge:
  - `label_profit_thr`
  - `exit_ema_span_min`
  - `exit_ema_init_offset_pct`
  - `tau_entry`
- fixa treino em:
  - `entry_ratio_neg_per_pos = 6.0`
  - `calib_tail_blend = 0.70`
  - `calib_tail_boost = 2.25`
- fixa risco em:
  - `max_positions = 10`
  - `total_exposure = 1.00`
  - `max_trade_exposure = 0.10`

Plano padrão:
- `49` refreshes
- `1` retrain por refresh
- `26` backtests por refresh

Os refreshes cobrem um subconjunto determinístico e bem distribuído do grid de contratos, e os backtests varrem `tau` de `0.70` a `0.95` em passo `0.01`.

## Entry points
```bash
python scripts/train.py
python scripts/explore.py
python scripts/run_independent_step_explores.py
```
