"""
DEPRECATED:
O entrypoint de treino foi movido para `train/train_sniper_wf.py`
com parâmetros definidos em código (sem ENV).

Este módulo fica como compatibilidade para quem ainda importa `train.train_sniper`.
"""

from __future__ import annotations


def main() -> None:
    from train.train_sniper_wf import main as _main

    _main()


if __name__ == "__main__":
    main()


