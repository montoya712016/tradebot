# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def build_time_folds(
    timestamps: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    train_days: int,
    valid_days: int,
    step_days: int,
    embargo_minutes: int = 0,
    max_folds: int = 0,
) -> list[FoldSpec]:
    ts = pd.to_datetime(pd.Index(timestamps)).sort_values().unique()
    if len(ts) == 0:
        return []
    train_days = int(max(1, train_days))
    valid_days = int(max(1, valid_days))
    step_days = int(max(1, step_days))
    emb_min = int(max(0, embargo_minutes))

    t_min = pd.Timestamp(ts.min())
    t_max = pd.Timestamp(ts.max())
    delta_train = pd.Timedelta(days=train_days)
    delta_valid = pd.Timedelta(days=valid_days)
    delta_step = pd.Timedelta(days=step_days)
    delta_emb = pd.Timedelta(minutes=emb_min)

    out: list[FoldSpec] = []
    fold = 0
    anchor = t_min
    while True:
        train_start = pd.Timestamp(anchor)
        train_end = train_start + delta_train
        valid_start = train_end + delta_emb
        valid_end = valid_start + delta_valid
        if valid_end > t_max:
            break
        out.append(
            FoldSpec(
                fold_id=int(fold),
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
        fold += 1
        if max_folds and max_folds > 0 and fold >= int(max_folds):
            break
        anchor = anchor + delta_step
    return out


def split_by_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.to_datetime(df.index)
    m_train = (idx >= fold.train_start) & (idx < fold.train_end)
    m_valid = (idx >= fold.valid_start) & (idx < fold.valid_end)
    return df.loc[m_train].copy(), df.loc[m_valid].copy()

