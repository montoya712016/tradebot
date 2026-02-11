# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    dd_penalty: float = 0.05
    dd_level_penalty: float = 0.0
    dd_soft_limit: float = 0.15
    dd_excess_penalty: float = 0.0
    dd_hard_limit: float = 1.0
    dd_hard_penalty: float = 0.0
    hold_bar_penalty: float = 0.0
    hold_soft_bars: int = 360
    hold_excess_penalty: float = 0.0
    hold_regret_penalty: float = 0.0
    stagnation_bars: int = 240
    stagnation_ret_epsilon: float = 0.00005
    stagnation_penalty: float = 0.0
    reverse_penalty: float = 0.0
    entry_penalty: float = 0.0
    weak_entry_penalty: float = 0.0
    turnover_penalty: float = 0.0002
    regret_penalty: float = 0.02
    idle_penalty: float = 0.0


def build_reward(
    *,
    delta_equity: float,
    turnover: float,
    dd_level: float,
    dd_increase: float,
    regret: float,
    time_in_trade: int,
    position_size: float,
    hold_regret: float,
    hard_dd_triggered: bool,
    was_idle: bool,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    r_turn = float(cfg.turnover_penalty) * float(turnover)
    r_dd = float(cfg.dd_penalty) * float(dd_increase)
    r_dd_level = float(cfg.dd_level_penalty) * float(dd_level)
    dd_excess = max(0.0, float(dd_level) - float(cfg.dd_soft_limit))
    r_dd_excess = float(cfg.dd_excess_penalty) * float(dd_excess * dd_excess)
    r_dd_hard = float(cfg.dd_hard_penalty) if bool(hard_dd_triggered) else 0.0
    r_hold_bar = float(cfg.hold_bar_penalty) * float(max(0.0, float(position_size)))
    hold_soft = int(max(1, cfg.hold_soft_bars))
    hold_excess = max(0.0, float(time_in_trade) - float(hold_soft))
    hold_excess_norm = hold_excess / float(hold_soft)
    r_hold_excess = float(cfg.hold_excess_penalty) * float(hold_excess_norm * hold_excess_norm)
    r_hold_regret = float(cfg.hold_regret_penalty) * float(max(0.0, hold_regret))
    stagnation_pen = 0.0
    if int(time_in_trade) > int(max(1, cfg.stagnation_bars)) and abs(float(delta_equity)) <= float(cfg.stagnation_ret_epsilon):
        st_norm = float(int(time_in_trade) - int(cfg.stagnation_bars)) / float(max(1, int(cfg.stagnation_bars)))
        stagnation_pen = float(cfg.stagnation_penalty) * float(max(0.0, st_norm))
    r_regret = float(cfg.regret_penalty) * float(regret)
    r_idle = float(cfg.idle_penalty) if was_idle else 0.0
    r_reverse = 0.0
    reward = float(
        delta_equity
        - r_turn
        - r_dd
        - r_dd_level
        - r_dd_excess
        - r_dd_hard
        - r_hold_bar
        - r_hold_excess
        - r_hold_regret
        - stagnation_pen
        - r_reverse
        - r_regret
        - r_idle
    )
    components = {
        "delta_equity": float(delta_equity),
        "turnover_pen": float(r_turn),
        "dd_pen": float(r_dd),
        "dd_level_pen": float(r_dd_level),
        "dd_excess_pen": float(r_dd_excess),
        "dd_hard_pen": float(r_dd_hard),
        "hold_bar_pen": float(r_hold_bar),
        "hold_excess_pen": float(r_hold_excess),
        "hold_regret_pen": float(r_hold_regret),
        "stagnation_pen": float(stagnation_pen),
        "reverse_pen": float(r_reverse),
        "regret_pen": float(r_regret),
        "idle_pen": float(r_idle),
    }
    return reward, components
