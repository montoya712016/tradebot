# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Iterable

import numpy as np
import pandas as pd


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
from env_rl.action_space import (  # type: ignore
    ACTION_HOLD,
    ACTION_OPEN_LONG_BIG,
    ACTION_OPEN_SHORT_BIG,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


def _heuristic_action(env: HybridTradingEnv, t: int, edge_thr: float, strength_thr: float) -> int:
    edge = float(env.edge_norm[t])
    strength = float(env.strength_norm[t])
    mu_l = float(env.mu_long_norm[t])
    mu_s = float(env.mu_short_norm[t])
    side = int(env.position_side)
    if side > 0:
        if strength < float(strength_thr) or edge < 0.0:
            return ACTION_CLOSE_LONG
        if edge > float(edge_thr):
            return ACTION_OPEN_LONG_BIG
        if mu_l >= mu_s:
            return ACTION_OPEN_LONG_SMALL
        return ACTION_HOLD
    if side < 0:
        if strength < float(strength_thr) or edge > 0.0:
            return ACTION_CLOSE_SHORT
        if edge < -float(edge_thr):
            return ACTION_OPEN_SHORT_BIG
        if mu_s > mu_l:
            return ACTION_OPEN_SHORT_SMALL
        return ACTION_HOLD
    if strength < float(strength_thr):
        return ACTION_HOLD
    if edge > float(edge_thr):
        return ACTION_OPEN_LONG_BIG
    if edge < -float(edge_thr):
        return ACTION_OPEN_SHORT_BIG
    return ACTION_OPEN_LONG_SMALL if mu_l >= mu_s else ACTION_OPEN_SHORT_SMALL


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


def run(
    *,
    signals_path: str,
    out_json: str,
    min_rows_symbol: int,
    edge_threshold: float,
    strength_threshold: float,
    fee_rate: float,
    slippage_rate: float,
    small_size: float,
    big_size: float,
    cooldown_bars: int,
    min_reentry_gap_bars: int,
    min_hold_bars: int,
    edge_entry_threshold: float,
    strength_entry_threshold: float,
) -> Path:
    df = pd.read_parquet(Path(signals_path).expanduser().resolve())
    df.index = pd.to_datetime(df.index)
    env_cfg = TradingEnvConfig(
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        small_size=float(small_size),
        big_size=float(big_size),
        cooldown_bars=int(cooldown_bars),
        min_reentry_gap_bars=int(min_reentry_gap_bars),
        min_hold_bars=int(min_hold_bars),
        use_signal_gate=True,
        edge_entry_threshold=float(edge_entry_threshold),
        strength_entry_threshold=float(strength_entry_threshold),
    )
    rows: list[dict[str, float]] = []
    for _sym, sdf in df.groupby("symbol", sort=True):
        sdf2 = sdf.sort_index().copy()
        if len(sdf2) < int(min_rows_symbol):
            continue
        env = HybridTradingEnv(sdf2, env_cfg)
        _s = env.reset()
        done = False
        while not done:
            a = _heuristic_action(env, env.t, float(edge_threshold), float(strength_threshold))
            valid = env.valid_actions()
            if valid and a not in valid:
                a = int(valid[0])
            _s, _r, done, _info = env.step(a)
        rows.append(env.summary())
    agg = _agg(rows)
    out = Path(out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _parse_args(argv: Iterable[str] | None = None):
    storage_root = _default_storage_root()
    ap = argparse.ArgumentParser(description="Backtest baseline supervisionado (sem RL)")
    ap.add_argument("--signals", default=str(storage_root / "supervised_signals.parquet"))
    ap.add_argument("--out", default=str(storage_root / "reports" / "backtest_supervised_only.json"))
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--edge-threshold", type=float, default=0.2)
    ap.add_argument("--strength-threshold", type=float, default=0.2)
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--small-size", type=float, default=0.5)
    ap.add_argument("--big-size", type=float, default=1.0)
    ap.add_argument("--cooldown-bars", type=int, default=3)
    ap.add_argument("--min-reentry-gap-bars", type=int, default=0)
    ap.add_argument("--min-hold-bars", type=int, default=0)
    ap.add_argument("--edge-entry-threshold", type=float, default=0.2)
    ap.add_argument("--strength-entry-threshold", type=float, default=0.2)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    ns = _parse_args(argv)
    out = run(
        signals_path=str(ns.signals),
        out_json=str(ns.out),
        min_rows_symbol=int(ns.min_rows_symbol),
        edge_threshold=float(ns.edge_threshold),
        strength_threshold=float(ns.strength_threshold),
        fee_rate=float(ns.fee_rate),
        slippage_rate=float(ns.slippage_rate),
        small_size=float(ns.small_size),
        big_size=float(ns.big_size),
        cooldown_bars=int(ns.cooldown_bars),
        min_reentry_gap_bars=int(ns.min_reentry_gap_bars),
        min_hold_bars=int(ns.min_hold_bars),
        edge_entry_threshold=float(ns.edge_entry_threshold),
        strength_entry_threshold=float(ns.strength_entry_threshold),
    )
    print(f"[backtest] salvo: {out}")


if __name__ == "__main__":
    main()
