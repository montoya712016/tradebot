# modules/config

Configurações estruturais compartilhadas entre treino, backtest e bot live.

## Conteúdo
- `trade_contract.py` - contrato de trade, janelas, spans e helpers de timeframe.
- `symbols.py` - carregamento e seleção do universo de símbolos.
- `secrets.py` - segredos locais opcionais, quando presentes.

Esses módulos definem defaults coerentes; os scripts podem sobrescrever detalhes via ambiente quando necessário.
