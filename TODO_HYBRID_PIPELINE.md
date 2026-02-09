# TODOs - Hibrido Regressao Long/Short + RL (Walk-Forward)

## Regras obrigatorias de validacao temporal (correlacao cross-asset)
- [x] Split global por tempo: a mesma faixa de timestamp entra inteira em `train` ou em `valid` para todos os ativos.
- [x] Sem vazamento temporal entre treino e validacao no WF (`train_end < valid_start` + embargo opcional).
- [ ] Adicionar `purged split` por barras para labels com horizonte longo (aprimoramento).
- [ ] Adicionar testes automatizados de leakage por fold.

## Correcao de base (estabilidade do repositorio)
- [x] Corrigir `PeriodModel` em `modules/backtest/sniper_walkforward.py` (erro de dataclass com ordem de defaults).
- [x] Compatibilizar imports de contrato de trade (`modules.trade_contract`) em backtest/simulador.
- [x] Adicionar `_apply_calibration` no simulador para compatibilidade com realtime.
- [x] Corrigir `ModelBundle` para incluir `calib`.
- [x] Corrigir loader de modelos no realtime para estrutura atual (`entry_models_long`, `entry_calibration_map_long`, `tau_entry_long_map`).
- [x] Remover segredos hardcoded de `modules/config/secrets.py` (migrado para variaveis de ambiente).
- [x] Remover senha hardcoded no `main()` do realtime (usa `SNIPER_DB_*` / `SNIPER_RUN_DIR` / `SNIPER_TAU_ENTRY`).

## Camada supervisionada padronizada (input do RL)
- [x] Criar exportador de sinais long/short: `modules/supervised/inference_long_short.py`.
- [x] Exportar `mu_long`, `mu_short`, `edge`, `strength`, `uncertainty`.
- [x] Exportar versoes normalizadas para RL (`*_norm`).
- [x] Exportar regime enxuto (`vol_short`, `vol_long`, `trend_strength`, `shock_flag`) + normalizados.
- [x] Exportar `fwd_ret_1` para reward incremental.
- [ ] Adicionar proxy de incerteza por erro rolling out-of-sample do regressor (hoje: dispersao entre janelas).

## Camada RL (decisao/controle)
- [x] Criar espaco de acoes discreto inicial: `{flat, long_small, long_big, short_small, short_big}`.
- [x] Criar ambiente RL: `modules/env_rl/trading_env.py`.
- [x] Incluir custos realistas (fee/slippage), turnover e cooldown.
- [x] Incluir reward composto: `delta_equity - custos - dd_penalty - turnover_penalty - regret_penalty`.
- [x] Incluir penalizacao de early-entry/regret (janela curta futura no ambiente offline).
- [ ] Adicionar opcao de latencia e fill delay no ambiente.
- [ ] Adicionar constraints por ativo (ex.: no flip na mesma barra, hard limits configuraveis por simbolo).

## Treino e avaliacao RL em walk-forward
- [x] Implementar treino DQN: `modules/rl/train_rl.py`.
- [x] Implementar folds temporais globais: `modules/rl/walkforward.py`.
- [x] Treinar/validar por fold com resumo versionado em disco.
- [x] Implementar avaliacao RL vs baseline supervisionado: `modules/rl/evaluate_rl.py`.
- [x] Implementar inferencia da policy salva: `modules/rl/policy_inference.py`.
- [ ] Adicionar metricas Calmar/Sortino e tempo de recuperacao de DD nos relatorios.
- [ ] Adicionar tuning de hiperparametros RL por janela longa (evitar overfit de fold unico).

## Backtests e orquestracao fim-a-fim
- [x] Backtest supervisionado puro: `modules/backtest/backtest_supervised_only.py`.
- [x] Backtest supervisionado + RL: `modules/backtest/backtest_supervised_plus_rl.py`.
- [x] Orquestrador completo: `modules/rl/run_hybrid_wf_pipeline.py`.
- [x] Gerar resumo final do pipeline em JSON.
- [ ] Integrar policy RL no realtime (modo shadow primeiro, depois modo decisorio).

## Operacao segura antes de capital real
- [ ] Rodar shadow/paper com logs de componentes de reward por trade.
- [ ] Comparar RL vs baseline por pelo menos 3-6 meses OOS rolling.
- [ ] Definir gatilhos de rollback (degradacao de PF, aumento de DD, aumento de turnover).
- [ ] Checklist de reproducibilidade (seed, versao de modelo, hash de dados, hash de codigo).

