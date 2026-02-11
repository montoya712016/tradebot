# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            repo_root = p.parent
            for cand in (repo_root, p):
                sp = str(cand)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from env_rl import HybridTradingEnv, TradingEnvConfig  # type: ignore
from rl.policy_inference import load_policy, choose_action  # type: ignore


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


def _agg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    arr = lambda k: np.asarray([x.get(k, 0.0) for x in rows], dtype=np.float64)
    return {
        "symbols": float(len(rows)),
        "ret_total_mean": float(np.nanmean(arr("ret_total"))),
        "max_dd_mean": float(np.nanmean(arr("max_dd"))),
        "profit_factor_median": float(np.nanmedian(arr("profit_factor"))),
        "win_rate_mean": float(np.nanmean(arr("win_rate"))),
        "trades_total": float(np.nansum(arr("trades"))),
    }


def _resolve_ckpt(checkpoint: str | None, run_dir: str | None, fold_id: int | None) -> Path:
    if checkpoint:
        p = Path(checkpoint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint nao encontrado: {p}")
        return p
    if not run_dir:
        raise RuntimeError("Informe --checkpoint ou --run-dir")
    rd = Path(run_dir).expanduser().resolve()
    if not rd.exists():
        raise FileNotFoundError(f"run_dir nao encontrado: {rd}")
    if fold_id is not None and fold_id >= 0:
        p = rd / f"fold_{int(fold_id):03d}" / "policy_dqn.pt"
        if not p.exists():
            raise FileNotFoundError(f"checkpoint do fold nao encontrado: {p}")
        return p
    folds = sorted([p for p in rd.glob("fold_*") if p.is_dir()])
    if not folds:
        raise RuntimeError(f"Nenhum fold_* encontrado em {rd}")
    return folds[-1] / "policy_dqn.pt"


def _load_ckpt_env_cfg(checkpoint: Path) -> dict:
    try:
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location="cpu")
        env_cfg = ckpt.get("env_cfg", {}) if isinstance(ckpt, dict) else {}
        return env_cfg if isinstance(env_cfg, dict) else {}
    except Exception:
        return {}


def run(
    *,
    signals_path: str,
    checkpoint: str | None,
    run_dir: str | None,
    fold_id: int | None,
    out_json: str,
    min_rows_symbol: int,
    fee_rate: float,
    slippage_rate: float,
    small_size: float,
    big_size: float,
    cooldown_bars: int,
    min_reentry_gap_bars: int,
    min_hold_bars: int,
    max_hold_bars: int,
    edge_entry_threshold: float,
    strength_entry_threshold: float,
    dd_penalty: float = 0.05,
    dd_level_penalty: float = 0.0,
    dd_soft_limit: float = 0.15,
    dd_excess_penalty: float = 0.0,
    dd_hard_limit: float = 1.0,
    dd_hard_penalty: float = 0.0,
    hold_bar_penalty: float = 0.0,
    hold_soft_bars: int = 360,
    hold_excess_penalty: float = 0.0,
    hold_regret_penalty: float = 0.0,
    turnover_penalty: float = 0.0002,
    regret_penalty: float = 0.02,
    idle_penalty: float = 0.0,
    device: str = "cuda",
) -> Path:
    ckpt = _resolve_ckpt(checkpoint, run_dir, fold_id)
    policy = load_policy(ckpt, device=device)
    ckpt_env = _load_ckpt_env_cfg(ckpt)
    rew = ckpt_env.get("reward", {}) if isinstance(ckpt_env.get("reward", {}), dict) else {}

    df = pd.read_parquet(Path(signals_path).expanduser().resolve())
    df.index = pd.to_datetime(df.index)
    env_cfg = TradingEnvConfig(
        fee_rate=float(ckpt_env.get("fee_rate", fee_rate)),
        slippage_rate=float(ckpt_env.get("slippage_rate", slippage_rate)),
        small_size=float(ckpt_env.get("small_size", small_size)),
        big_size=float(ckpt_env.get("big_size", big_size)),
        cooldown_bars=int(ckpt_env.get("cooldown_bars", cooldown_bars)),
        min_reentry_gap_bars=int(ckpt_env.get("min_reentry_gap_bars", min_reentry_gap_bars)),
        min_hold_bars=int(ckpt_env.get("min_hold_bars", min_hold_bars)),
        max_hold_bars=int(ckpt_env.get("max_hold_bars", max_hold_bars)),
        use_signal_gate=bool(ckpt_env.get("use_signal_gate", True)),
        edge_entry_threshold=float(ckpt_env.get("edge_entry_threshold", edge_entry_threshold)),
        strength_entry_threshold=float(ckpt_env.get("strength_entry_threshold", strength_entry_threshold)),
        force_close_on_adverse=bool(ckpt_env.get("force_close_on_adverse", True)),
        edge_exit_threshold=float(ckpt_env.get("edge_exit_threshold", 0.08)),
        strength_exit_threshold=float(ckpt_env.get("strength_exit_threshold", 0.25)),
        allow_direct_flip=bool(ckpt_env.get("allow_direct_flip", True)),
        allow_scale_in=bool(ckpt_env.get("allow_scale_in", False)),
        state_use_uncertainty=bool(ckpt_env.get("state_use_uncertainty", True)),
        state_use_vol_long=bool(ckpt_env.get("state_use_vol_long", True)),
        state_use_shock=bool(ckpt_env.get("state_use_shock", True)),
        state_use_window_context=bool(ckpt_env.get("state_use_window_context", True)),
        state_window_short_bars=int(ckpt_env.get("state_window_short_bars", 30)),
        state_window_long_bars=int(ckpt_env.get("state_window_long_bars", 60)),
    )
    env_cfg.reward.dd_penalty = float(rew.get("dd_penalty", dd_penalty))
    env_cfg.reward.dd_level_penalty = float(rew.get("dd_level_penalty", dd_level_penalty))
    env_cfg.reward.dd_soft_limit = float(rew.get("dd_soft_limit", dd_soft_limit))
    env_cfg.reward.dd_excess_penalty = float(rew.get("dd_excess_penalty", dd_excess_penalty))
    env_cfg.reward.dd_hard_limit = float(rew.get("dd_hard_limit", dd_hard_limit))
    env_cfg.reward.dd_hard_penalty = float(rew.get("dd_hard_penalty", dd_hard_penalty))
    env_cfg.reward.hold_bar_penalty = float(rew.get("hold_bar_penalty", hold_bar_penalty))
    env_cfg.reward.hold_soft_bars = int(rew.get("hold_soft_bars", hold_soft_bars))
    env_cfg.reward.hold_excess_penalty = float(rew.get("hold_excess_penalty", hold_excess_penalty))
    env_cfg.reward.hold_regret_penalty = float(rew.get("hold_regret_penalty", hold_regret_penalty))
    env_cfg.reward.stagnation_bars = int(rew.get("stagnation_bars", 180))
    env_cfg.reward.stagnation_ret_epsilon = float(rew.get("stagnation_ret_epsilon", 0.00003))
    env_cfg.reward.stagnation_penalty = float(rew.get("stagnation_penalty", 0.0))
    env_cfg.reward.reverse_penalty = float(rew.get("reverse_penalty", 0.0))
    env_cfg.reward.entry_penalty = float(rew.get("entry_penalty", 0.0))
    env_cfg.reward.weak_entry_penalty = float(rew.get("weak_entry_penalty", 0.0))
    env_cfg.reward.turnover_penalty = float(rew.get("turnover_penalty", turnover_penalty))
    env_cfg.reward.regret_penalty = float(rew.get("regret_penalty", regret_penalty))
    env_cfg.reward.idle_penalty = float(rew.get("idle_penalty", idle_penalty))
    rows: list[dict[str, float]] = []
    for _sym, sdf in df.groupby("symbol", sort=True):
        sdf2 = sdf.sort_index().copy()
        if len(sdf2) < int(min_rows_symbol):
            continue
        env = HybridTradingEnv(sdf2, env_cfg)
        state = env.reset()
        done = False
        while not done:
            a = choose_action(policy, state, valid_actions=env.valid_actions())
            state, _r, done, _info = env.step(a)
        rows.append(env.summary())

    agg = _agg(rows)
    agg["checkpoint"] = str(ckpt)
    out = Path(out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _parse_args(argv: Iterable[str] | None = None):
    storage_root = _default_storage_root()
    ap = argparse.ArgumentParser(description="Backtest supervisionado + policy RL")
    ap.add_argument("--signals", default=str(storage_root / "supervised_signals.parquet"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--fold-id", type=int, default=-1)
    ap.add_argument("--out", default=str(storage_root / "reports" / "backtest_supervised_plus_rl.json"))
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--small-size", type=float, default=0.5)
    ap.add_argument("--big-size", type=float, default=1.0)
    ap.add_argument("--cooldown-bars", type=int, default=3)
    ap.add_argument("--min-reentry-gap-bars", type=int, default=0)
    ap.add_argument("--min-hold-bars", type=int, default=0)
    ap.add_argument("--max-hold-bars", type=int, default=0)
    ap.add_argument("--edge-entry-threshold", type=float, default=0.2)
    ap.add_argument("--strength-entry-threshold", type=float, default=0.2)
    ap.add_argument("--dd-penalty", type=float, default=0.05)
    ap.add_argument("--dd-level-penalty", type=float, default=0.0)
    ap.add_argument("--dd-soft-limit", type=float, default=0.15)
    ap.add_argument("--dd-excess-penalty", type=float, default=0.0)
    ap.add_argument("--dd-hard-limit", type=float, default=1.0)
    ap.add_argument("--dd-hard-penalty", type=float, default=0.0)
    ap.add_argument("--hold-bar-penalty", type=float, default=0.0)
    ap.add_argument("--hold-soft-bars", type=int, default=360)
    ap.add_argument("--hold-excess-penalty", type=float, default=0.0)
    ap.add_argument("--hold-regret-penalty", type=float, default=0.0)
    ap.add_argument("--turnover-penalty", type=float, default=0.0002)
    ap.add_argument("--regret-penalty", type=float, default=0.02)
    ap.add_argument("--idle-penalty", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    ns = _parse_args(argv)
    out = run(
        signals_path=str(ns.signals),
        checkpoint=(None if ns.checkpoint in (None, "", "none") else str(ns.checkpoint)),
        run_dir=(None if ns.run_dir in (None, "", "none") else str(ns.run_dir)),
        fold_id=(None if int(ns.fold_id) < 0 else int(ns.fold_id)),
        out_json=str(ns.out),
        min_rows_symbol=int(ns.min_rows_symbol),
        fee_rate=float(ns.fee_rate),
        slippage_rate=float(ns.slippage_rate),
        small_size=float(ns.small_size),
        big_size=float(ns.big_size),
        cooldown_bars=int(ns.cooldown_bars),
        min_reentry_gap_bars=int(ns.min_reentry_gap_bars),
        min_hold_bars=int(ns.min_hold_bars),
        max_hold_bars=int(ns.max_hold_bars),
        edge_entry_threshold=float(ns.edge_entry_threshold),
        strength_entry_threshold=float(ns.strength_entry_threshold),
        dd_penalty=float(ns.dd_penalty),
        dd_level_penalty=float(ns.dd_level_penalty),
        dd_soft_limit=float(ns.dd_soft_limit),
        dd_excess_penalty=float(ns.dd_excess_penalty),
        dd_hard_limit=float(ns.dd_hard_limit),
        dd_hard_penalty=float(ns.dd_hard_penalty),
        hold_bar_penalty=float(ns.hold_bar_penalty),
        hold_soft_bars=int(ns.hold_soft_bars),
        hold_excess_penalty=float(ns.hold_excess_penalty),
        hold_regret_penalty=float(ns.hold_regret_penalty),
        turnover_penalty=float(ns.turnover_penalty),
        regret_penalty=float(ns.regret_penalty),
        idle_penalty=float(ns.idle_penalty),
        device=str(ns.device),
    )
    print(f"[backtest] salvo: {out}")


if __name__ == "__main__":
    main()
