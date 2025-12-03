# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable
import numpy as np

from .config import OFFSETS_DAYS, RATIO_NEG_PER_POS
from .dataflow import prepare_block, prepare_block_multi

# fallback de imports para defaults do PF
try:
    from ..prepare_features.prepare_features import DEFAULT_SYMBOL, DEFAULT_DAYS, FLAGS as PF_FLAGS
except Exception:
    from my_project.prepare_features.prepare_features import DEFAULT_SYMBOL, DEFAULT_DAYS, FLAGS as PF_FLAGS


def run_entry_training_blocks(
    *,
    symbol: str | None = None,
    total_days: int | None = None,
    offsets_days: Iterable[int] = OFFSETS_DAYS,
):
    """Gera datasets por blocos T-X para buy/short (sem treinar ainda).
    Retorna uma lista de dicionários com shapes e proporções para inspeção.
    """
    sym = symbol or DEFAULT_SYMBOL
    days = int(total_days if total_days is not None else DEFAULT_DAYS)

    out = []
    for tail in list(offsets_days):
        res = prepare_block(sym, total_days=days, remove_tail_days=int(tail), flags=PF_FLAGS)
        info = dict(
            symbol=sym,
            remove_tail_days=int(tail),
            long=dict(n_pos=int((res["long"]["y"]==1).sum()), n_neg=int((res["long"]["y"]==0).sum()), ratio=float((res["long"]["y"]==0).sum()/max(1,(res["long"]["y"]==1).sum()))),
            short=dict(n_pos=int((res["short"]["y"]==1).sum()), n_neg=int((res["short"]["y"]==0).sum()), ratio=float((res["short"]["y"]==0).sum()/max(1,(res["short"]["y"]==1).sum()))),
            feat_cols=len(res["feature_cols"]),
        )
        print((
            f"[train-block] T-{tail}d | long 1:{info['long']['ratio']:.2f} (pos={info['long']['n_pos']:,}, neg={info['long']['n_neg']:,}) | "
            f"short 1:{info['short']['ratio']:.2f} (pos={info['short']['n_pos']:,}, neg={info['short']['n_neg']:,}) | "
            f"features={info['feat_cols']}"
        ).replace(',', '.'), flush=True)
        out.append((res, info))
    return out


if __name__ == "__main__":
    run_entry_training_blocks()


