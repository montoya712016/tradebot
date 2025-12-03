# -*- coding: utf-8 -*-
"""
Configuração mínima para treinamento dos modelos de ENTRADA (buy/short).
Foco: orquestração de blocos por corte de cauda e pesos por U.
"""
from __future__ import annotations
import os

# Blocos: remover cauda final de X dias (T-X) — usado para simular walk-forward
# Walk-forward: períodos T-X usados para TREINO.
# Validação OoS: sempre 90 dias APÓS o cutoff (X+90), logo não treinamos 0d.
OFFSETS_DAYS: list[int] = [90, 180, 270, 360, 450, 540, 630, 720]

# Balanceamento por lado (1:ratio)
RATIO_NEG_PER_POS: int = 3  # pode usar 4 se positivo for muito escasso

# Pesos por utilidade (U) — aplicados só no dataset de ENTRADA
# Normalização: clip(|U|, 0, U_WEIGHT_MAX) / U_WEIGHT_MAX
U_WEIGHT_MAX: float = 5.0
# Reforço em positivos com U alto
ALPHA_POS: float = 0.5
# Penalização em negativos com U “ruim” para o lado (ex.: U<0 para buy)
BETA_NEG: float = 1.0
# Alívio em negativos “aceitáveis” (ex.: buy com U>0): fator multiplicativo ~ (1 - gamma * U_pos/Umax), min 0.5
GAMMA_EASE: float = 0.5

# ===== Hiperparâmetros de treino (binário buy/short) =====
def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else int(default)
    except Exception:
        return int(default)

def _float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else float(default)
    except Exception:
        return float(default)

ENTRY_XGB_ROUNDS: int = _int_env("ENTRY_XGB_ROUNDS", 6000)
ENTRY_XGB_EARLY: int  = _int_env("ENTRY_XGB_EARLY",  400)
ENTRY_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': _float_env('ENTRY_XGB_ETA', 0.03),
    'max_depth': _int_env('ENTRY_XGB_MAX_DEPTH', 11),
    'subsample': _float_env('ENTRY_XGB_SUBSAMPLE', 0.95),
    'colsample_bytree': _float_env('ENTRY_XGB_COLS', 0.95),
    'min_child_weight': _float_env('ENTRY_XGB_MIN_CHILD', 1.0),
    'lambda': _float_env('ENTRY_XGB_LAMBDA', 1.0),
    'alpha': _float_env('ENTRY_XGB_ALPHA', 0.0),
    'max_bin': _int_env('ENTRY_XGB_MAX_BIN', 512),
    # tree_method/device são definidos no trainer (GPU/CPU)
}

# ===== Multiprocessamento/Watchdog de RAM =====
# Ativa processamento paralelo por símbolo com controle dinâmico pela RAM
MP_USE: bool = (os.getenv("MP_USE", "1") != "0")
# Máximo de processos simultâneos (hard cap)
MP_MAX_PROCS: int = _int_env("MP_MAX_PROCS", 8)
# Reserva de RAM a manter livre (GB) para evitar OOM
MP_RESERVE_GB: float = _float_env("MP_RESERVE_GB", 6.0)
# Livre mínima para permitir spawn de novo processo (GB)
MP_MIN_FREE_GB: float = _float_env("MP_MIN_FREE_GB", 3.0)
# Fator de segurança sobre pico medido por processo
MP_SAFETY_FACTOR: float = _float_env("MP_SAFETY_FACTOR", 1.30)


