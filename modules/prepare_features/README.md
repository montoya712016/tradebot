# modules/prepare_features

Pipeline de preparação de dados para treino e backtest.

## Peças principais
- `data.py` - leitura de OHLC, caches e acesso a parquet/pickle.
- `prepare_features.py` - cálculo de features.
- `labels.py` - geração de labels e metadados forward-looking a partir do contrato.
- `refresh_sniper_labels_in_cache.py` - refresh rápido de labels reaproveitando features já prontas.
- `sniper_dataset.py` - montagem dos datasets supervisionados.

## Como isso entra no workflow
- `scripts/data_sync.py` preenche OHLC/caches.
- `scripts/refresh_labels.py` regrava labels para um contrato específico.
- `scripts/train.py` e `scripts/explore.py` usam esse pacote para montar datasets de treino.

No fair explore `v5`, cada refresh muda apenas o contrato/labeling (`label_profit_thr`, `exit_ema_span_min`, `exit_ema_init_offset_pct`), e o sweep de `tau` fica para os backtests.
