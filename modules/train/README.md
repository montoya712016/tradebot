### train/ - treino/modelos (core)

Este modulo contem a logica de treino do Sniper:
- `sniper_dataflow.py`: cache de features/labels por simbolo e carregamento de OHLC (MySQL).
- `sniper_trainer.py`: treino de modelos (XGBoost) e geracao de artefatos `wf_*`.
- `train_sniper_wf.py`: pipeline de treino walk-forward.

Entrypoint recomendado:
- `python modules/train/train_sniper_wf.py`
