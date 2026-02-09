# -*- coding: utf-8 -*-
from __future__ import annotations


def turnover_amount(prev_side: int, prev_size: float, new_side: int, new_size: float) -> float:
    prev_exp = float(prev_side) * float(prev_size)
    new_exp = float(new_side) * float(new_size)
    return float(abs(new_exp - prev_exp))


def transaction_cost(
    prev_side: int,
    prev_size: float,
    new_side: int,
    new_size: float,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> float:
    turn = turnover_amount(prev_side, prev_size, new_side, new_size)
    return float(turn * (float(fee_rate) + float(slippage_rate)))

