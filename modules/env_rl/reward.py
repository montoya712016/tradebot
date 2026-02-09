# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    dd_penalty: float = 0.05
    turnover_penalty: float = 0.0002
    regret_penalty: float = 0.02
    idle_penalty: float = 0.0


def build_reward(
    *,
    delta_equity: float,
    turnover: float,
    dd_increase: float,
    regret: float,
    was_idle: bool,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    r_turn = float(cfg.turnover_penalty) * float(turnover)
    r_dd = float(cfg.dd_penalty) * float(dd_increase)
    r_regret = float(cfg.regret_penalty) * float(regret)
    r_idle = float(cfg.idle_penalty) if was_idle else 0.0
    reward = float(delta_equity - r_turn - r_dd - r_regret - r_idle)
    components = {
        "delta_equity": float(delta_equity),
        "turnover_pen": float(r_turn),
        "dd_pen": float(r_dd),
        "regret_pen": float(r_regret),
        "idle_pen": float(r_idle),
    }
    return reward, components

