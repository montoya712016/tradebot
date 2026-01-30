# -*- coding: utf-8 -*-
"""
Bot Utils - Funções auxiliares para o bot.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from modules.prepare_features import pf_config
from modules.prepare_features.prepare_features import FEATURE_KEYS, build_flags


def max_pf_window_minutes() -> int:
    """Retorna a maior janela de tempo necessária para calcular features."""
    values: List[int] = []
    for name in dir(pf_config):
        if not name.endswith("_MIN") and not name.endswith("_WINDOWS"):
            continue
        val = getattr(pf_config, name)
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, (list, tuple)) and item:
                    values.append(int(max(item)))
                else:
                    values.append(int(item))
    return max(values) if values else 1440


def window_days_for_minutes(minutes: int) -> int:
    """Calcula quantos dias carregar para cobrir X minutos."""
    return int(max(1, int(np.ceil(minutes / 1440.0)) + 1))


def feature_window_minutes() -> int:
    """Quantidade de minutos necessária para calcular todas as features."""
    return max_pf_window_minutes()


def make_feature_flags() -> Dict[str, bool]:
    """Cria flags de features habilitadas."""
    return build_flags(enable=FEATURE_KEYS, label=False)


def safe_last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Retorna a última linha do DataFrame de forma segura."""
    if df is None or df.empty:
        return None
    try:
        return df.iloc[-1]
    except Exception:
        return None
