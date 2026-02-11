# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Callable, Iterable

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
from env_rl.action_space import (  # type: ignore
    ACTION_HOLD,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)
from rl.train_rl import QNet  # type: ignore


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


@dataclass
class EvaluateConfig:
    signals_path: str = str(_default_storage_root() / "supervised_signals.parquet")
    run_dir: str = str(_default_storage_root() / "rl_runs" / "default")
    min_rows_symbol: int = 800
    edge_threshold: float = 0.2
    strength_threshold: float = 0.2
    device: str = "cuda"


def _pick_device(pref: str) -> torch.device:
    if str(pref).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_signals(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"signals_path nao encontrado: {p}")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _split_symbols(df: pd.DataFrame, min_rows: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, sdf in df.groupby("symbol", sort=True):
        sdf2 = sdf.sort_index().copy()
        if len(sdf2) < int(min_rows):
            continue
        out[str(sym)] = sdf2
    return out


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "ret_total_mean": 0.0,
            "max_dd_mean": 0.0,
            "profit_factor_median": 0.0,
            "win_rate_mean": 0.0,
            "trades_total": 0.0,
            "trades_per_day_mean": 0.0,
            "trades_per_week_mean": 0.0,
            "avg_hold_bars_mean": 0.0,
            "avg_hold_hours_mean": 0.0,
            "avg_hold_long_hours_mean": 0.0,
            "avg_hold_short_hours_mean": 0.0,
            "trades_long_total": 0.0,
            "trades_short_total": 0.0,
            "avg_peak_ret_mean": 0.0,
            "avg_giveback_mean": 0.0,
            "trade_efficiency_mean": 0.0,
            "avg_turnover_mean": 0.0,
        }
    arr = lambda k: np.asarray([x.get(k, 0.0) for x in rows], dtype=np.float64)
    return {
        "ret_total_mean": float(np.nanmean(arr("ret_total"))),
        "max_dd_mean": float(np.nanmean(arr("max_dd"))),
        "profit_factor_median": float(np.nanmedian(arr("profit_factor"))),
        "win_rate_mean": float(np.nanmean(arr("win_rate"))),
        "trades_total": float(np.nansum(arr("trades"))),
        "trades_per_day_mean": float(np.nanmean(arr("trades_per_day"))),
        "trades_per_week_mean": float(np.nanmean(arr("trades_per_week"))),
        "avg_hold_bars_mean": float(np.nanmean(arr("avg_hold_bars"))),
        "avg_hold_hours_mean": float(np.nanmean(arr("avg_hold_hours"))),
        "avg_hold_long_hours_mean": float(np.nanmean(arr("avg_hold_long_hours"))),
        "avg_hold_short_hours_mean": float(np.nanmean(arr("avg_hold_short_hours"))),
        "trades_long_total": float(np.nansum(arr("trades_long"))),
        "trades_short_total": float(np.nansum(arr("trades_short"))),
        "avg_peak_ret_mean": float(np.nanmean(arr("avg_peak_ret"))),
        "avg_giveback_mean": float(np.nanmean(arr("avg_giveback"))),
        "trade_efficiency_mean": float(np.nanmean(arr("trade_efficiency_mean"))),
        "avg_turnover_mean": float(np.nanmean(arr("avg_turnover"))),
    }


def _run_policy_frames(
    frames: dict[str, pd.DataFrame],
    env_cfg: TradingEnvConfig,
    act_fn: Callable[[np.ndarray, int, HybridTradingEnv], int],
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for _sym, sdf in frames.items():
        env = HybridTradingEnv(sdf, env_cfg)
        s = env.reset()
        done = False
        while not done:
            a = int(act_fn(s, env.t, env))
            valid = env.valid_actions()
            if valid and a not in valid:
                a = int(valid[0])
            s, _r, done, _info = env.step(a)
        rows.append(env.summary())
    return _aggregate(rows)


def _heuristic_action(state: np.ndarray, t: int, env: HybridTradingEnv, edge_thr: float, strength_thr: float) -> int:
    edge = float(env.edge_norm[t])
    strength = float(env.strength_norm[t])
    mu_l = float(env.mu_long_norm[t])
    mu_s = float(env.mu_short_norm[t])
    side = int(env.position_side)
    if side > 0:
        if strength < float(strength_thr) or edge < 0.0:
            return ACTION_CLOSE_LONG
        if edge > float(edge_thr):
            return ACTION_OPEN_LONG_SMALL
        if mu_l >= mu_s:
            return ACTION_OPEN_LONG_SMALL
        return ACTION_HOLD
    if side < 0:
        if strength < float(strength_thr) or edge > 0.0:
            return ACTION_CLOSE_SHORT
        if edge < -float(edge_thr):
            return ACTION_OPEN_SHORT_SMALL
        if mu_s > mu_l:
            return ACTION_OPEN_SHORT_SMALL
        return ACTION_HOLD
    if strength < float(strength_thr):
        return ACTION_HOLD
    if edge > float(edge_thr):
        return ACTION_OPEN_LONG_SMALL
    if edge < -float(edge_thr):
        return ACTION_OPEN_SHORT_SMALL
    return ACTION_OPEN_LONG_SMALL if mu_l >= mu_s else ACTION_OPEN_SHORT_SMALL


def evaluate_rl_run(cfg: EvaluateConfig) -> Path:
    t_all = time.perf_counter()
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir nao encontrado: {run_dir}")
    df = _load_signals(cfg.signals_path)

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"config.json ausente em {run_dir}")
    train_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    env_cfg = TradingEnvConfig(
        fee_rate=float(train_cfg.get("fee_rate", 0.0005)),
        slippage_rate=float(train_cfg.get("slippage_rate", 0.0001)),
        small_size=float(train_cfg.get("small_size", 0.5)),
        big_size=float(train_cfg.get("big_size", 1.0)),
        cooldown_bars=int(train_cfg.get("cooldown_bars", 3)),
        regret_window=int(train_cfg.get("regret_window", 10)),
        use_signal_gate=bool(train_cfg.get("use_signal_gate", True)),
        edge_entry_threshold=float(train_cfg.get("edge_entry_threshold", 0.2)),
        strength_entry_threshold=float(train_cfg.get("strength_entry_threshold", 0.2)),
        force_close_on_adverse=bool(train_cfg.get("force_close_on_adverse", True)),
        edge_exit_threshold=float(train_cfg.get("edge_exit_threshold", 0.08)),
        strength_exit_threshold=float(train_cfg.get("strength_exit_threshold", 0.25)),
        allow_direct_flip=bool(train_cfg.get("allow_direct_flip", True)),
        allow_scale_in=bool(train_cfg.get("allow_scale_in", False)),
        min_reentry_gap_bars=int(train_cfg.get("min_reentry_gap_bars", 0)),
        min_hold_bars=int(train_cfg.get("min_hold_bars", 0)),
        max_hold_bars=int(train_cfg.get("max_hold_bars", 0)),
        state_use_uncertainty=bool(train_cfg.get("state_use_uncertainty", True)),
        state_use_vol_long=bool(train_cfg.get("state_use_vol_long", True)),
        state_use_shock=bool(train_cfg.get("state_use_shock", True)),
        state_use_window_context=bool(train_cfg.get("state_use_window_context", True)),
        state_window_short_bars=int(train_cfg.get("state_window_short_bars", 30)),
        state_window_long_bars=int(train_cfg.get("state_window_long_bars", 60)),
    )
    env_cfg.reward.dd_penalty = float(train_cfg.get("dd_penalty", 0.05))
    env_cfg.reward.dd_level_penalty = float(train_cfg.get("dd_level_penalty", 0.0))
    env_cfg.reward.dd_soft_limit = float(train_cfg.get("dd_soft_limit", 0.15))
    env_cfg.reward.dd_excess_penalty = float(train_cfg.get("dd_excess_penalty", 0.0))
    env_cfg.reward.dd_hard_limit = float(train_cfg.get("dd_hard_limit", 1.0))
    env_cfg.reward.dd_hard_penalty = float(train_cfg.get("dd_hard_penalty", 0.0))
    env_cfg.reward.hold_bar_penalty = float(train_cfg.get("hold_bar_penalty", 0.0))
    env_cfg.reward.hold_soft_bars = int(train_cfg.get("hold_soft_bars", 180))
    env_cfg.reward.hold_excess_penalty = float(train_cfg.get("hold_excess_penalty", 0.0))
    env_cfg.reward.hold_regret_penalty = float(train_cfg.get("hold_regret_penalty", 0.0))
    env_cfg.reward.stagnation_bars = int(train_cfg.get("stagnation_bars", 180))
    env_cfg.reward.stagnation_ret_epsilon = float(train_cfg.get("stagnation_ret_epsilon", 0.00003))
    env_cfg.reward.stagnation_penalty = float(train_cfg.get("stagnation_penalty", 0.0))
    env_cfg.reward.reverse_penalty = float(train_cfg.get("reverse_penalty", 0.0))
    env_cfg.reward.entry_penalty = float(train_cfg.get("entry_penalty", 0.0))
    env_cfg.reward.weak_entry_penalty = float(train_cfg.get("weak_entry_penalty", 0.0))
    env_cfg.reward.turnover_penalty = float(train_cfg.get("turnover_penalty", 0.0002))
    env_cfg.reward.regret_penalty = float(train_cfg.get("regret_penalty", 0.02))
    env_cfg.reward.idle_penalty = float(train_cfg.get("idle_penalty", 0.0))

    device = _pick_device(cfg.device)
    rows_eval: list[dict[str, float]] = []
    for fold_dir in sorted([p for p in run_dir.glob("fold_*") if p.is_dir()]):
        t_fold = time.perf_counter()
        summary_path = fold_dir / "summary.json"
        model_path = fold_dir / "policy_dqn.pt"
        if not summary_path.exists() or not model_path.exists():
            continue
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        vs = pd.to_datetime(summ["valid_start"])
        ve = pd.to_datetime(summ["valid_end"])
        m_valid = (df.index >= vs) & (df.index < ve)
        df_valid = df.loc[m_valid].copy()
        frames = _split_symbols(df_valid, int(cfg.min_rows_symbol))
        if not frames:
            continue

        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)
        q = QNet(int(ckpt["state_dim"]), int(ckpt["n_actions"]), int(ckpt["hidden_dim"])).to(device)
        q.load_state_dict(ckpt["state_dict"])
        q.eval()
        state_buf = torch.empty((1, int(ckpt["state_dim"])), dtype=torch.float32, device=device)
        idx_cache: dict[tuple[int, ...], torch.Tensor] = {}

        def _act_rl(state: np.ndarray, _t: int, _env: HybridTradingEnv) -> int:
            with torch.inference_mode():
                s_np = state.astype(np.float32, copy=False)
                state_buf[0].copy_(torch.from_numpy(s_np), non_blocking=True)
                qs = q(state_buf)
                valid = _env.valid_actions()
                if valid:
                    key = tuple(int(a) for a in valid)
                    idx = idx_cache.get(key)
                    if idx is None:
                        idx = torch.as_tensor(key, dtype=torch.long, device=device)
                        idx_cache[key] = idx
                    qv = qs.index_select(1, idx)
                    j = int(torch.argmax(qv, dim=1).item())
                    return int(valid[j])
                return int(torch.argmax(qs, dim=1).item())

        t0 = time.perf_counter()
        rl_metrics = _run_policy_frames(frames, env_cfg, _act_rl)
        dt_rl = time.perf_counter() - t0
        t0 = time.perf_counter()
        sup_metrics = _run_policy_frames(
            frames,
            env_cfg,
            lambda s, t, env: _heuristic_action(s, t, env, cfg.edge_threshold, cfg.strength_threshold),
        )
        dt_sup = time.perf_counter() - t0
        row = {
            "fold_id": int(summ["fold_id"]),
            "valid_start": str(vs),
            "valid_end": str(ve),
            "symbols": int(len(frames)),
            "rl_ret_total_mean": float(rl_metrics["ret_total_mean"]),
            "rl_max_dd_mean": float(rl_metrics["max_dd_mean"]),
            "rl_pf_median": float(rl_metrics["profit_factor_median"]),
            "rl_trades_total": float(rl_metrics["trades_total"]),
            "rl_trades_long_total": float(rl_metrics["trades_long_total"]),
            "rl_trades_short_total": float(rl_metrics["trades_short_total"]),
            "rl_trades_per_week_mean": float(rl_metrics["trades_per_week_mean"]),
            "rl_avg_hold_bars_mean": float(rl_metrics["avg_hold_bars_mean"]),
            "rl_avg_hold_hours_mean": float(rl_metrics["avg_hold_hours_mean"]),
            "rl_avg_hold_long_hours_mean": float(rl_metrics["avg_hold_long_hours_mean"]),
            "rl_avg_hold_short_hours_mean": float(rl_metrics["avg_hold_short_hours_mean"]),
            "rl_avg_peak_ret_mean": float(rl_metrics["avg_peak_ret_mean"]),
            "rl_avg_giveback_mean": float(rl_metrics["avg_giveback_mean"]),
            "rl_trade_efficiency_mean": float(rl_metrics["trade_efficiency_mean"]),
            "sup_ret_total_mean": float(sup_metrics["ret_total_mean"]),
            "sup_max_dd_mean": float(sup_metrics["max_dd_mean"]),
            "sup_pf_median": float(sup_metrics["profit_factor_median"]),
            "sup_trades_total": float(sup_metrics["trades_total"]),
            "sup_trades_long_total": float(sup_metrics["trades_long_total"]),
            "sup_trades_short_total": float(sup_metrics["trades_short_total"]),
            "sup_trades_per_week_mean": float(sup_metrics["trades_per_week_mean"]),
            "sup_avg_hold_bars_mean": float(sup_metrics["avg_hold_bars_mean"]),
            "sup_avg_hold_hours_mean": float(sup_metrics["avg_hold_hours_mean"]),
            "sup_avg_hold_long_hours_mean": float(sup_metrics["avg_hold_long_hours_mean"]),
            "sup_avg_hold_short_hours_mean": float(sup_metrics["avg_hold_short_hours_mean"]),
            "sup_avg_peak_ret_mean": float(sup_metrics["avg_peak_ret_mean"]),
            "sup_avg_giveback_mean": float(sup_metrics["avg_giveback_mean"]),
            "sup_trade_efficiency_mean": float(sup_metrics["trade_efficiency_mean"]),
            "time_rl_eval_s": float(dt_rl),
            "time_sup_eval_s": float(dt_sup),
            "time_fold_total_s": float(time.perf_counter() - t_fold),
        }
        rows_eval.append(row)
        print(
            f"[eval][fold {row['fold_id']}] RL ret={row['rl_ret_total_mean']:+.3%} dd={row['rl_max_dd_mean']:.2%} pf={row['rl_pf_median']:.3f} | "
            f"trades/wk={row['rl_trades_per_week_mean']:.3f} hold(h)={row['rl_avg_hold_hours_mean']:.2f} "
            f"(L={row['rl_avg_hold_long_hours_mean']:.2f} S={row['rl_avg_hold_short_hours_mean']:.2f}) "
            f"eff={row['rl_trade_efficiency_mean']:+.3f} giveback={row['rl_avg_giveback_mean']:.4f} | "
            f"SUP ret={row['sup_ret_total_mean']:+.3%} dd={row['sup_max_dd_mean']:.2%} pf={row['sup_pf_median']:.3f} "
            f"trades/wk={row['sup_trades_per_week_mean']:.3f} hold(h)={row['sup_avg_hold_hours_mean']:.2f} "
            f"(L={row['sup_avg_hold_long_hours_mean']:.2f} S={row['sup_avg_hold_short_hours_mean']:.2f}) "
            f"eff={row['sup_trade_efficiency_mean']:+.3f} giveback={row['sup_avg_giveback_mean']:.4f} | "
            f"time(rl/sup/fold)={dt_rl:.2f}/{dt_sup:.2f}/{row['time_fold_total_s']:.2f}s",
            flush=True,
        )

    if not rows_eval:
        raise RuntimeError("Nenhum fold avaliado")
    out_df = pd.DataFrame(rows_eval).sort_values("fold_id")
    out_json = run_dir / "evaluation_compare.json"
    out_csv = run_dir / "evaluation_compare.csv"
    out_json.write_text(out_df.to_json(orient="records", indent=2), encoding="utf-8")
    out_df.to_csv(out_csv, index=False)
    print(f"[eval][time] total={time.perf_counter() - t_all:.2f}s", flush=True)
    return out_csv


def _parse_args(argv: Iterable[str] | None = None) -> EvaluateConfig:
    storage_root = _default_storage_root()
    ap = argparse.ArgumentParser(description="Avalia RL vs baseline supervisionado")
    ap.add_argument("--signals", default=str(storage_root / "supervised_signals.parquet"))
    ap.add_argument("--run-dir", default=str(storage_root / "rl_runs" / "default"))
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--edge-threshold", type=float, default=0.2)
    ap.add_argument("--strength-threshold", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ns = ap.parse_args(list(argv) if argv is not None else None)
    return EvaluateConfig(
        signals_path=str(ns.signals),
        run_dir=str(ns.run_dir),
        min_rows_symbol=int(ns.min_rows_symbol),
        edge_threshold=float(ns.edge_threshold),
        strength_threshold=float(ns.strength_threshold),
        device=str(ns.device),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = evaluate_rl_run(cfg)
    print(f"[eval] salvo: {out}")


if __name__ == "__main__":
    main()
