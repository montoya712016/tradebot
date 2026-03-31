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
   - padrão atual: `WF_OOS_SCORE_MODE=survival`
   - seletor atual: `WF_OOS_SELECTOR=family_causal`
   - legado opcional: `WF_OOS_SCORE_MODE=legacy`

6. **Bot live/paper e dashboard**
   - `python scripts/bot_live.py`
   - `python scripts/bot_dashboard.py`

Fair Explore v6
---------------
O explore atual continua separando risco estrutural do resto, mas soltou um pouco o treino em relaÃ§Ã£o Ã  `v5` e passou a usar um preset de features reduzido.

- Steps explorados: `1440, 1260, 1080, 900, 720, 540, 360`
- Segmento-base de validação: `180` dias
- Root padrão: `data/generated/fair_wf_explore_v6/`
- Preset de features padrÃ£o: `core80`

Parâmetros otimizados no explore:
- `label_profit_thr`
- `exit_ema_span_min`
- `exit_ema_init_offset_pct`
- `entry_ratio_neg_per_pos`
- `calib_tail_blend`
- `calib_tail_boost`
- `tau_entry`

O `v6` nÃ£o voltou ao `v4`: risco segue fixo, mas o treino volta a explorar um espaÃ§o pequeno e estruturado:
- `entry_ratio_neg_per_pos = (4.0, 6.0, 8.0)`
- `calib_tail_blend = (0.60, 0.70, 0.80)`
- `calib_tail_boost = (1.75, 2.25, 2.75)`
- `top_metric_qs` em alguns presets curtos
- `top_metric_min_count = (48, 64)`

Política fixa de risco no explore:
- `total_exposure = 1.00`
- `max_trade_exposure = 0.10`
- `min_trade_exposure = 0.02`

`max_positions` deixou de fazer parte do explore novo. O limite prático de posições simultâneas passa a vir do próprio orçamento de exposição (`total_exposure / max_trade_exposure`).

Busca atual:
- `56` refreshes por step
- `2` retrains por refresh
- `21` backtests por retrain
- `1` geração por step

Os refreshes continuam cobrindo um subconjunto determinÃ­stico e bem distribuÃ­do do grid de contratos/labels. No `v6`, os retrains tambÃ©m percorrem um grid pequeno e uniforme de presets de treino/calibraÃ§Ã£o, enquanto os backtests varrem `tau` de `0.70` a `0.90` em passo `0.01`.

Convensões de storage
---------------------
- modelos: `D:\astra\models_sniper\crypto\`
- caches de features / labels: `D:\astra\cache_sniper\`
- artefatos gerados do repositório: `data/generated/`

Observações
-----------
- `scripts/backtest.py`, `scripts/train.py` e `scripts/refresh_labels.py` são entrypoints úteis, mas hoje carregam presets opinativos; o workflow fair principal é dado por `run_independent_step_explores.py` + `run_oos_walkforward.py`.
- o `duckdb` foi adotado para leitura de `explore_runs.csv` e acompanhamento do progresso do orchestrator/explorer; isso reduz overhead de CSV, mas o hot path ainda continua sendo refresh de features, montagem do dataset e treino XGBoost.
- no OOS walk-forward, a seleção de representante por step usa por padrão um `survival_score` mais duro contra cauda esquerda e excesso de exposição, monta shortlist por família, usa continuidade causal entre steps e depois escolhe a variante mais prudente dentro da família; o score antigo continua acessível por `WF_OOS_SCORE_MODE=legacy`.
- Scripts legados e utilitários ad hoc antigos foram removidos do branch atual. Se algo precisar voltar, recupere via Git.
