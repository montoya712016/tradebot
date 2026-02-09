# -*- coding: utf-8 -*-
__all__ = ["train_sniper_models", "TrainConfig"]


def __getattr__(name: str):
    if name in ("train_sniper_models", "TrainConfig"):
        from .sniper_trainer import train_sniper_models, TrainConfig

        return train_sniper_models if name == "train_sniper_models" else TrainConfig
    raise AttributeError(name)


