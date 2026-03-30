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
