# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import os
from pathlib import Path

import numpy as np
import pandas as pd

if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = Path(__file__).resolve().parents[2].parent / "cache_sniper" / "numba"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache_dir)

try:
    from .labels import apply_trade_contract_labels
except Exception:
    from prepare_features.labels import apply_trade_contract_labels  # type: ignore[import]

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT


@dataclass
class SniperDataset:
    entry: pd.DataFrame
    add: pd.DataFrame
    danger: pd.DataFrame
    exit: pd.DataFrame
    meta: Dict[str, float | int]


def _infer_candle_sec(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 2:
        return 60
    try:
        dt = float((idx[1] - idx[0]).total_seconds())
        if not np.isfinite(dt) or dt <= 0.0:
            return 60
        return int(round(dt))
    except Exception:
        return 60


def _ensure_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract,
    candle_sec: int,
    entry_label_col: str = "sniper_entry_label",
    entry_weight_col: str = "",
    exit_code_col: str = "",
    exit_target_col: str = "",
    exit_weight_col: str = "",
) -> None:
    # Se faltar label/weight solicitado, gera labels do contrato.
    need_rebuild = str(entry_label_col) not in df.columns
    if str(entry_weight_col):
        need_rebuild = need_rebuild or (str(entry_weight_col) not in df.columns)
    if str(exit_target_col):
        need_rebuild = need_rebuild or (str(exit_target_col) not in df.columns)
    if str(exit_weight_col):
        need_rebuild = need_rebuild or (str(exit_weight_col) not in df.columns)
    if need_rebuild:
        apply_trade_contract_labels(df, contract=contract, candle_sec=candle_sec)


def build_sniper_datasets(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
    entry_label_col: str = "sniper_entry_label",
    entry_weight_col: str = "",
    exit_code_col: str = "",
    exit_target_col: str = "sniper_exit_span_target",
    exit_weight_col: str = "sniper_exit_span_weight",
    max_add_starts: int = 20_000,
    max_exit_starts: int = 8_000,
    exit_neg_frac: float = 0.35,
    exit_stride_bars: int = 15,
    exit_lookahead_bars: int | None = None,
    # Margem do exit: em 1m, 0.2% costuma ser pequeno demais e gera label_exit quase morto.
    exit_margin_pct: float = 0.006,
    seed: int = 42,
    enable_exit_dataset: bool = True,
) -> SniperDataset:
    """
    Constrói quatro dataframes:
        - entry: todas as barras com o label de entry informado (ex.: sniper_entry_label ou sniper_entry_label_360m)
        - add: vazio (adds removidos)
        - danger: vazio (danger removido)
        - exit: barras com target de span EMA (regressão)
    Cada dataframe já inclui colunas de estado (cycle_*).
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    candle_sec = int(candle_sec or _infer_candle_sec(df.index))
    _ensure_contract_labels(
        df,
        contract=contract,
        candle_sec=candle_sec,
        entry_label_col=str(entry_label_col),
        entry_weight_col=str(entry_weight_col),
        exit_code_col=str(exit_code_col),
        exit_target_col=str(exit_target_col),
        exit_weight_col=str(exit_weight_col),
    )

    entry_mask = df[str(entry_label_col)].notna()
    entry_df = df.loc[entry_mask].copy()
    entry_df["ts"] = entry_df.index
    entry_df["cycle_is_add"] = 0
    entry_df["cycle_num_adds"] = 0
    entry_df["cycle_time_in_trade"] = 0
    entry_df["cycle_dd_pct"] = 0.0
    entry_df["cycle_avg_entry_price"] = entry_df["close"]
    entry_df["cycle_last_fill_price"] = entry_df["close"]
    entry_df["label_entry"] = pd.to_numeric(entry_df[str(entry_label_col)], errors="coerce").astype(np.float32)

    empty_cols = list(df.columns) + [
        "ts",
        "cycle_is_add",
        "cycle_num_adds",
        "cycle_time_in_trade",
        "cycle_dd_pct",
        "cycle_avg_entry_price",
        "cycle_last_fill_price",
    ]
    add_df = pd.DataFrame(columns=empty_cols + ["label_entry"])
    danger_df = pd.DataFrame(columns=empty_cols + ["label_danger"])
    tgt_col = str(exit_target_col or "").strip()
    w_col = str(exit_weight_col or "").strip()
    if not bool(enable_exit_dataset):
        tgt_col = ""
        w_col = ""
    else:
        if (not tgt_col) or (tgt_col not in df.columns):
            tgt_col = "sniper_exit_span_target" if "sniper_exit_span_target" in df.columns else ""
        if (not w_col) or (w_col not in df.columns):
            w_col = "sniper_exit_span_weight" if "sniper_exit_span_weight" in df.columns else ""
    if tgt_col:
        tgt = pd.to_numeric(df[tgt_col], errors="coerce")
        exit_mask = tgt.notna() & np.isfinite(tgt.to_numpy(dtype=np.float64, copy=False)) & (tgt > 0.0)
        exit_df = df.loc[exit_mask].copy()
        exit_df["ts"] = exit_df.index
        exit_df["cycle_is_add"] = 0
        exit_df["cycle_num_adds"] = 0
        exit_df["cycle_time_in_trade"] = 0
        exit_df["cycle_dd_pct"] = 0.0
        exit_df["cycle_avg_entry_price"] = exit_df["close"]
        exit_df["cycle_last_fill_price"] = exit_df["close"]
        exit_df["label_exit"] = pd.to_numeric(exit_df[tgt_col], errors="coerce").astype(np.float32)
        if w_col and (w_col in exit_df.columns):
            exit_df["label_exit_weight"] = pd.to_numeric(exit_df[w_col], errors="coerce").fillna(1.0).clip(lower=1.0).astype(np.float32)
        else:
            exit_df["label_exit_weight"] = np.float32(1.0)
    else:
        exit_df = pd.DataFrame(columns=empty_cols + ["label_exit", "label_exit_weight"])

    meta = dict(
        candle_sec=candle_sec,
    )
    return SniperDataset(entry=entry_df, add=add_df, danger=danger_df, exit=exit_df, meta=meta)


def warmup_sniper_dataset_numba() -> None:
    """Warmup vazio: kernels antigos de add/exit foram removidos."""
    return


__all__ = [
    "SniperDataset",
    "build_sniper_datasets",
    "warmup_sniper_dataset_numba",
]
