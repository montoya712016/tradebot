### prepare_features/ - features + labels + plots (core)

Este modulo e responsavel por:
- carregar series OHLC (MySQL)
- calcular features (blocos em `features.py`)
- gerar labels (Sniper) em `labels.py`
- plotar/inspecionar em `plotting.py`

Entrypoints:
- `python modules/prepare_features/prepare_features.py`
- `python modules/prepare_features/refresh_sniper_labels_in_cache.py`

Notas:
- `prepare_features.py` (neste modulo) e o orquestrador interno.
- Flags de features ficam em `prepare_features.prepare_features.FLAGS` (e helpers `build_flags`).
