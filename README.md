Sniper Tradebot
===============

Repositório do pipeline completo de crypto walk-forward:

1. baixar OHLC e manter caches
2. preparar features e labels
3. treinar modelos walk-forward
4. explorar parâmetros de edge por step histórico
5. validar com OOS walk-forward
6. operar no bot live/paper com dashboard

Estrutura
---------
- `scripts/` - entrypoints oficiais.
- `modules/` - lógica compartilhada de dados, treino, backtest e realtime.
- `realtime/` - implementação do bot live/paper.
- `core/` - contratos e executores base.
- `data/` - artefatos gerados, relatórios, estado e logs locais.

Fluxo principal
---------------
1. **Sincronizar dados / preencher caches**
   - `python scripts/data_sync.py`

2. **Refresh de labels**
   - `python scripts/refresh_labels.py`

3. **Treino walk-forward**
   - `python scripts/train.py`

4. **Exploração fair por steps**
   - `python scripts/run_independent_step_explores.py`
   - dashboard automático: `python scripts/fair_dashboard.py`

5. **Validação OOS walk-forward**
   - `python scripts/run_oos_walkforward.py`

6. **Bot live/paper e dashboard**
   - `python scripts/bot_live.py`
   - `python scripts/bot_dashboard.py`

Fair Explore v5
---------------
O explore atual foi simplificado para separar melhor edge de treino e risco.

- Steps explorados: `1440, 1260, 1080, 900, 720, 540, 360`
- Segmento-base de validação: `180` dias
- Root padrão: `data/generated/fair_wf_explore_v5/`

Parâmetros otimizados no explore:
- `label_profit_thr`
- `exit_ema_span_min`
- `exit_ema_init_offset_pct`
- `tau_entry`

Parâmetros fixos de treino:
- `entry_ratio_neg_per_pos = 6.0`
- `calib_tail_blend = 0.70`
- `calib_tail_boost = 2.25`

Política fixa de risco no explore:
- `max_positions = 10`
- `total_exposure = 1.00`
- `max_trade_exposure = 0.10`

Busca atual:
- `49` refreshes por step
- `1` retrain por refresh
- `26` backtests por refresh
- `1` geração por step

Os `49` refreshes não são mais uma busca puramente aleatória. O explorer usa uma cobertura determinística e bem distribuída do grid de contratos/labels, e para cada refresh varre `tau` de `0.70` a `0.95` em passo `0.01`.

Convensões de storage
---------------------
- modelos: `D:\astra\models_sniper\crypto\`
- caches de features / labels: `D:\astra\cache_sniper\`
- artefatos gerados do repositório: `data/generated/`

Observações
-----------
- `scripts/backtest.py`, `scripts/train.py` e `scripts/refresh_labels.py` são entrypoints úteis, mas hoje carregam presets opinativos; o workflow fair principal é dado por `run_independent_step_explores.py` + `run_oos_walkforward.py`.
- Scripts legados e utilitários ad hoc antigos foram removidos do branch atual. Se algo precisar voltar, recupere via Git.
