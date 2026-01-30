# -*- coding: utf-8 -*-
"""
Model Bundle - Empacotamento de modelos ML.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import xgboost as xgb


@dataclass
class ModelBundle:
    """
    Pacote contendo modelo treinado e metadados necessários para inferência.
    """
    model: xgb.Booster
    feature_cols: List[str]
    calib: Dict[str, Any]
    tau_entry: float
    predictor: str = "cpu_predictor"
