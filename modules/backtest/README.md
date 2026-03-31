# modules/backtest

Motores de backtest e simulação usados pelo repositório.

## Peças principais
- `sniper_portfolio.py` - executor multi-ativo com sizing, fees, slippage e limites de exposição.
- `portfolio.py` - helpers para preparar dados, rodar backtests de portfolio e gerar curva/equity.
- `single_symbol.py` - debug detalhado por ativo.
- `sniper_walkforward.py` - carrega bundles walk-forward e aplica score por período.

## Fluxo atual
- backtest rápido/manual: `python scripts/backtest.py`
- OOS walk-forward oficial: `python scripts/run_oos_walkforward.py`

O script `run_oos_walkforward.py` é o entrypoint principal para validar os pools do fair explore.
Por padrão, ele agora lê `explore_runs.csv` via `duckdb`, usa `WF_OOS_SCORE_MODE=survival` e `WF_OOS_SELECTOR=family_causal` para selecionar o representante por step.
Nesse modo, o OOS monta shortlist por família de parâmetros, aplica continuidade causal entre steps e escolhe a variante mais prudente dentro da família.
Para reproduzir o score antigo, use `WF_OOS_SCORE_MODE=legacy`.
