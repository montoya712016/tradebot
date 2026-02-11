# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest single-symbol para policy RL treinada no pipeline hibrido.

Saidas:
- plot HTML com trades + Q-values + regressores + acoes + equity
- timeline parquet/csv com todas as series por candle
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import time
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
from env_rl.action_space import (  # type: ignore
    ACTION_HOLD,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_LONG_BIG,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_OPEN_SHORT_BIG,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)
from env_rl.reward import RewardConfig  # type: ignore
from plotting.plotting import plot_backtest_single_rl  # type: ignore
from rl.policy_inference import LoadedPolicy, load_policy  # type: ignore
from utils.progress import ProgressPrinter  # type: ignore


@dataclass
class RLTrade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    size: float
    ret: float


@dataclass
class SingleSymbolRLDemoSettings:
    symbol: str = "STXUSDT"
    days: int = 365
    signals_path: str | None = None
    run_dir: str | None = None
    checkpoint: str | None = None
    fold_id: int = -1
    device: str = "auto"
    plot_out: str | None = None
    timeline_out: str | None = None
    show_plot: bool = True
    save_plot: bool = True
    save_timeline: bool = True
    plot_candles: bool = True


def _default_hybrid_root() -> Path:
    try:
        from utils.paths import models_root  # type: ignore

        return models_root() / "hybrid_rl"
    except Exception:
        here = Path(__file__).resolve()
        repo_root = here.parents[2]
        return repo_root.parent / "models_sniper" / "hybrid_rl"


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if not s:
        s = "STXUSDT"
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"run_dir nao encontrado: {p}")
        return p
    root = _default_hybrid_root()
    p_default = root / "rl_runs" / "default"
    if p_default.exists():
        return p_default
    base = root / "rl_runs"
    if not base.exists():
        raise FileNotFoundError(f"Diretorio de runs RL nao encontrado: {base}")
    cands: list[Path] = []
    for p in base.glob("*"):
        if not p.is_dir():
            continue
        # formatos suportados:
        # 1) .../rl_runs/<run>/policy_dqn.pt
        # 2) .../rl_runs/<run>/fold_000/policy_dqn.pt
        # 3) .../rl_runs/<loop>/iter_0001_xxx/fold_000/policy_dqn.pt
        if (p / "policy_dqn.pt").exists():
            cands.append(p)
            continue
        fold_ckpts = list(p.glob("fold_*/policy_dqn.pt"))
        if fold_ckpts:
            cands.append(p)
            continue
        iter_ckpts = list(p.glob("iter_*/fold_*/policy_dqn.pt"))
        if iter_ckpts:
            # escolhe o iter mais recente com checkpoint
            iter_dirs = sorted(
                {ck.parent.parent for ck in iter_ckpts},
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if iter_dirs:
                cands.append(iter_dirs[0])
    if not cands:
        raise FileNotFoundError(f"Nenhum run_dir encontrado em {base}")
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_checkpoint(checkpoint: str | None, run_dir: Path, fold_id: int) -> Path:
    if checkpoint:
        p = Path(checkpoint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint nao encontrado: {p}")
        return p
    if int(fold_id) >= 0:
        p = run_dir / f"fold_{int(fold_id):03d}" / "policy_dqn.pt"
        if not p.exists():
            raise FileNotFoundError(f"checkpoint do fold nao encontrado: {p}")
        return p
    p_direct = run_dir / "policy_dqn.pt"
    if p_direct.exists():
        return p_direct
    folds = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    if not folds:
        raise RuntimeError(f"Nenhum fold_* encontrado em {run_dir}")
    p = folds[-1] / "policy_dqn.pt"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint nao encontrado: {p}")
    return p


def _load_train_cfg(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    if not p.exists() and run_dir.name.startswith("fold_"):
        p = run_dir.parent / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_signals_path(signals_path: str | None, run_dir: Path) -> Path:
    # prioridade 1: caminho explicito
    if signals_path:
        p = Path(signals_path).expanduser().resolve()
        if p.exists():
            return p

    hybrid_root = _default_hybrid_root()
    repo_root = Path(__file__).resolve().parents[2]
    cfg = _load_train_cfg(run_dir)

    cand_order: list[Path] = []
    cfg_sig = str(cfg.get("signals_path", "") or "").strip()
    if cfg_sig:
        cand_order.append(Path(cfg_sig).expanduser().resolve())

    cand_order.extend(
        [
            (hybrid_root / "supervised_signals.parquet").resolve(),
            (hybrid_root / "supervised_signals_smoke.parquet").resolve(),
            (run_dir / "supervised_signals.parquet").resolve(),
            (run_dir.parent / "supervised_signals.parquet").resolve(),
            (repo_root / "data" / "generated" / "hybrid" / "supervised_signals.parquet").resolve(),
            (repo_root / "data" / "generated" / "hybrid" / "supervised_signals_smoke.parquet").resolve(),
        ]
    )

    for p in cand_order:
        if p.exists():
            return p

    # fallback: varre por qualquer supervised_signals*.parquet nas pastas usuais
    discovered: list[Path] = []
    for root in (
        hybrid_root,
        hybrid_root.parent,
        (repo_root / "data" / "generated" / "hybrid").resolve(),
    ):
        if not root.exists():
            continue
        try:
            discovered.extend([x.resolve() for x in root.glob("supervised_signals*.parquet") if x.is_file()])
        except Exception:
            pass
    if discovered:
        discovered.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return discovered[0]

    raise FileNotFoundError(
        "signals nao encontrado (tentado: config.json, models_sniper/hybrid_rl, run_dir e data/generated/hybrid)"
    )


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


def _build_env_cfg(*, train_cfg: dict, ckpt_env_cfg: dict) -> TradingEnvConfig:
    src = ckpt_env_cfg if ckpt_env_cfg else train_cfg
    rew = src.get("reward", {}) if isinstance(src.get("reward", {}), dict) else {}
    max_hold_raw = int(src.get("max_hold_bars", train_cfg.get("max_hold_bars", 720)))
    # fallback seguro para checkpoints legados sem limite de hold
    max_hold_eff = int(max_hold_raw if max_hold_raw > 0 else 720)
    return TradingEnvConfig(
        fee_rate=float(src.get("fee_rate", train_cfg.get("fee_rate", 0.0005))),
        slippage_rate=float(src.get("slippage_rate", train_cfg.get("slippage_rate", 0.0001))),
        small_size=float(src.get("small_size", train_cfg.get("small_size", 0.2))),
        big_size=float(src.get("big_size", train_cfg.get("big_size", 0.5))),
        cooldown_bars=int(src.get("cooldown_bars", train_cfg.get("cooldown_bars", 8))),
        regret_window=int(src.get("regret_window", train_cfg.get("regret_window", 10))),
        min_reentry_gap_bars=int(src.get("min_reentry_gap_bars", train_cfg.get("min_reentry_gap_bars", 0))),
        min_hold_bars=int(src.get("min_hold_bars", train_cfg.get("min_hold_bars", 0))),
        max_hold_bars=int(max_hold_eff),
        use_signal_gate=bool(src.get("use_signal_gate", train_cfg.get("use_signal_gate", True))),
        edge_entry_threshold=float(src.get("edge_entry_threshold", train_cfg.get("edge_entry_threshold", 0.2))),
        strength_entry_threshold=float(src.get("strength_entry_threshold", train_cfg.get("strength_entry_threshold", 0.2))),
        force_close_on_adverse=bool(src.get("force_close_on_adverse", train_cfg.get("force_close_on_adverse", True))),
        edge_exit_threshold=float(src.get("edge_exit_threshold", train_cfg.get("edge_exit_threshold", 0.08))),
        strength_exit_threshold=float(src.get("strength_exit_threshold", train_cfg.get("strength_exit_threshold", 0.25))),
        allow_direct_flip=bool(src.get("allow_direct_flip", train_cfg.get("allow_direct_flip", True))),
        allow_scale_in=bool(src.get("allow_scale_in", train_cfg.get("allow_scale_in", False))),
        state_use_uncertainty=bool(src.get("state_use_uncertainty", train_cfg.get("state_use_uncertainty", True))),
        state_use_vol_long=bool(src.get("state_use_vol_long", train_cfg.get("state_use_vol_long", True))),
        state_use_shock=bool(src.get("state_use_shock", train_cfg.get("state_use_shock", True))),
        state_use_window_context=bool(src.get("state_use_window_context", train_cfg.get("state_use_window_context", True))),
        state_window_short_bars=int(src.get("state_window_short_bars", train_cfg.get("state_window_short_bars", 30))),
        state_window_long_bars=int(src.get("state_window_long_bars", train_cfg.get("state_window_long_bars", 60))),
        reward=RewardConfig(
            dd_penalty=float(rew.get("dd_penalty", train_cfg.get("dd_penalty", 0.05))),
            dd_level_penalty=float(rew.get("dd_level_penalty", train_cfg.get("dd_level_penalty", 0.0))),
            dd_soft_limit=float(rew.get("dd_soft_limit", train_cfg.get("dd_soft_limit", 0.15))),
            dd_excess_penalty=float(rew.get("dd_excess_penalty", train_cfg.get("dd_excess_penalty", 0.0))),
            dd_hard_limit=float(rew.get("dd_hard_limit", train_cfg.get("dd_hard_limit", 1.0))),
            dd_hard_penalty=float(rew.get("dd_hard_penalty", train_cfg.get("dd_hard_penalty", 0.0))),
            hold_bar_penalty=float(rew.get("hold_bar_penalty", train_cfg.get("hold_bar_penalty", 0.0))),
            hold_soft_bars=int(rew.get("hold_soft_bars", train_cfg.get("hold_soft_bars", 180))),
            hold_excess_penalty=float(rew.get("hold_excess_penalty", train_cfg.get("hold_excess_penalty", 0.0))),
            hold_regret_penalty=float(rew.get("hold_regret_penalty", train_cfg.get("hold_regret_penalty", 0.0))),
            stagnation_bars=int(rew.get("stagnation_bars", train_cfg.get("stagnation_bars", 180))),
            stagnation_ret_epsilon=float(rew.get("stagnation_ret_epsilon", train_cfg.get("stagnation_ret_epsilon", 0.00003))),
            stagnation_penalty=float(rew.get("stagnation_penalty", train_cfg.get("stagnation_penalty", 0.0))),
            reverse_penalty=float(rew.get("reverse_penalty", train_cfg.get("reverse_penalty", 0.0))),
            entry_penalty=float(rew.get("entry_penalty", train_cfg.get("entry_penalty", 0.0))),
            weak_entry_penalty=float(rew.get("weak_entry_penalty", train_cfg.get("weak_entry_penalty", 0.0))),
            turnover_penalty=float(rew.get("turnover_penalty", train_cfg.get("turnover_penalty", 0.0002))),
            regret_penalty=float(rew.get("regret_penalty", train_cfg.get("regret_penalty", 0.02))),
            idle_penalty=float(rew.get("idle_penalty", train_cfg.get("idle_penalty", 0.0))),
        ),
    )


def _policy_action_with_q(policy: LoadedPolicy, state: np.ndarray, valid_actions: list[int]) -> tuple[int, np.ndarray]:
    s_np = state.astype(np.float32, copy=False)
    d = int(s_np.shape[0])
    if policy.state_buf is None or int(policy.state_buf.shape[1]) != d:
        policy.state_buf = torch.empty((1, d), dtype=torch.float32, device=policy.device)
    with torch.inference_mode():
        policy.state_buf[0].copy_(torch.from_numpy(s_np), non_blocking=True)
        q = policy.model(policy.state_buf).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if valid_actions:
        idx = np.asarray(valid_actions, dtype=np.int64)
        j = int(np.argmax(q[idx]))
        return int(valid_actions[j]), np.asarray(q, dtype=np.float32)
    return int(np.argmax(q)), np.asarray(q, dtype=np.float32)


def _ffill_nan(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.size == 0:
        return out
    s = pd.Series(out)
    s = s.ffill().bfill().fillna(float(fill_value))
    return s.to_numpy(dtype=np.float32, copy=False)


def _action_name(a: float) -> str:
    x = int(round(float(a)))
    m = {
        ACTION_HOLD: "hold",
        ACTION_OPEN_LONG_SMALL: "entry_long",
        ACTION_OPEN_LONG_BIG: "entry_long",
        ACTION_OPEN_SHORT_SMALL: "entry_short",
        ACTION_OPEN_SHORT_BIG: "entry_short",
        ACTION_CLOSE_LONG: "close_long",
        ACTION_CLOSE_SHORT: "close_short",
    }
    return m.get(x, f"action_{x}")


def _save_timeline(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.strip().lower()
    if ext == ".csv":
        df.to_csv(out_path, index=True)
        return
    if ext in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=True)
        return
    # fallback parquet
    df.to_parquet(out_path.with_suffix(".parquet"), index=True)


def run(settings: SingleSymbolRLDemoSettings | None = None) -> dict[str, str | float]:
    t0 = time.perf_counter()
    s = settings or SingleSymbolRLDemoSettings()
    symbol = _normalize_symbol(s.symbol)
    hybrid_root = _default_hybrid_root()
    run_dir = _resolve_run_dir(s.run_dir)
    ckpt = _resolve_checkpoint(s.checkpoint, run_dir, int(s.fold_id))
    signals_path = _resolve_signals_path(s.signals_path, run_dir)

    plot_path = (
        Path(s.plot_out).expanduser().resolve()
        if s.plot_out
        else (hybrid_root / "reports" / f"single_symbol_{symbol.lower()}_rl.html").resolve()
    )
    timeline_path = (
        Path(s.timeline_out).expanduser().resolve()
        if s.timeline_out
        else (hybrid_root / "reports" / f"single_symbol_{symbol.lower()}_rl_timeline.parquet").resolve()
    )

    print(
        f"[rl-single] setup symbol={symbol} days={int(s.days)} "
        f"run_dir={run_dir} checkpoint={ckpt.name} signals={signals_path.name}",
        flush=True,
    )
    df_all = pd.read_parquet(signals_path)
    df_all.index = pd.to_datetime(df_all.index)
    if "symbol" not in df_all.columns:
        raise RuntimeError("signals parquet sem coluna 'symbol'")
    df_sym = df_all[df_all["symbol"].astype(str).str.upper() == symbol].sort_index().copy()
    if df_sym.empty:
        raise RuntimeError(f"symbol {symbol} ausente em {signals_path}")
    if int(s.days) > 0:
        end_ts = pd.to_datetime(df_sym.index.max())
        start_ts = end_ts - pd.Timedelta(days=int(s.days))
        df_sym = df_sym[df_sym.index >= start_ts].copy()
    if len(df_sym) < 200:
        raise RuntimeError(f"poucos pontos para {symbol}: rows={len(df_sym)}")

    for col in ("open", "high", "low"):
        if col not in df_sym.columns:
            df_sym[col] = df_sym["close"].astype(np.float64)
    df_sym = df_sym.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    env_cfg = _build_env_cfg(train_cfg=_load_train_cfg(run_dir), ckpt_env_cfg=_load_ckpt_env_cfg(ckpt))
    mh = int(env_cfg.min_hold_bars)
    xh = int(env_cfg.max_hold_bars)
    print(
        f"[rl-single] env gap={int(env_cfg.min_reentry_gap_bars)} bars "
        f"min_hold={mh} bars ({mh/60.0:.2f}h) max_hold={xh} bars ({xh/60.0:.2f}h) "
        f"gate(edge={float(env_cfg.edge_entry_threshold):.2f},strength={float(env_cfg.strength_entry_threshold):.2f}) "
        f"force_close={int(bool(env_cfg.force_close_on_adverse))} "
        f"exit(edge={float(env_cfg.edge_exit_threshold):.2f},strength={float(env_cfg.strength_exit_threshold):.2f}) "
        f"flip={int(bool(env_cfg.allow_direct_flip))} "
        f"scale_in={int(bool(env_cfg.allow_scale_in))} "
        f"state(use_unc={int(bool(env_cfg.state_use_uncertainty))},"
        f"use_vlong={int(bool(env_cfg.state_use_vol_long))},"
        f"use_shock={int(bool(env_cfg.state_use_shock))},"
        f"use_ctx={int(bool(env_cfg.state_use_window_context))},"
        f"ctx={int(env_cfg.state_window_short_bars)}/{int(env_cfg.state_window_long_bars)})",
        flush=True,
    )
    policy = load_policy(ckpt, device=str(s.device))
    print(f"[rl-single] policy_device={policy.device}", flush=True)
    env = HybridTradingEnv(df_sym, env_cfg)

    n = len(df_sym)
    idx = pd.to_datetime(df_sym.index)
    action_id = np.full(n, np.nan, dtype=np.float32)
    action_q = np.full(n, np.nan, dtype=np.float32)
    reward_step = np.zeros(n, dtype=np.float32)
    exposure = np.zeros(n, dtype=np.float32)

    q_hold = np.full(n, np.nan, dtype=np.float32)
    q_long_small = np.full(n, np.nan, dtype=np.float32)
    q_long_big = np.full(n, np.nan, dtype=np.float32)
    q_short_small = np.full(n, np.nan, dtype=np.float32)
    q_short_big = np.full(n, np.nan, dtype=np.float32)
    q_close_long = np.full(n, np.nan, dtype=np.float32)
    q_close_short = np.full(n, np.nan, dtype=np.float32)
    q_long_best = np.full(n, np.nan, dtype=np.float32)
    q_short_best = np.full(n, np.nan, dtype=np.float32)
    q_close_best = np.full(n, np.nan, dtype=np.float32)

    trades: list[RLTrade] = []
    open_trade: dict | None = None
    state = env.reset()
    done = False
    progress_total = max(1, int(n - 1))
    progress = ProgressPrinter(prefix="[rl-single][pred]", total=progress_total, print_every_s=2.0)
    update_every = 256
    progress.update(0, suffix=f"t={int(env.t)+1}/{progress_total} trades=0")
    while not done:
        t = int(env.t)
        valid = env.valid_actions()
        a, qv = _policy_action_with_q(policy, state, valid)

        action_id[t] = float(a)
        action_q[t] = float(qv[a]) if 0 <= a < len(qv) else np.nan
        q_hold[t] = float(qv[ACTION_HOLD])
        q_long_small[t] = float(qv[ACTION_OPEN_LONG_SMALL])
        q_long_big[t] = float(qv[ACTION_OPEN_LONG_BIG])
        q_short_small[t] = float(qv[ACTION_OPEN_SHORT_SMALL])
        q_short_big[t] = float(qv[ACTION_OPEN_SHORT_BIG])
        q_close_long[t] = float(qv[ACTION_CLOSE_LONG])
        q_close_short[t] = float(qv[ACTION_CLOSE_SHORT])
        q_long_best[t] = float(max(qv[ACTION_OPEN_LONG_SMALL], qv[ACTION_OPEN_LONG_BIG]))
        q_short_best[t] = float(max(qv[ACTION_OPEN_SHORT_SMALL], qv[ACTION_OPEN_SHORT_BIG]))
        q_close_best[t] = float(max(qv[ACTION_CLOSE_LONG], qv[ACTION_CLOSE_SHORT]))

        state, reward, done, info = env.step(a)
        reward_step[t] = float(reward)
        exposure[t] = float(info.get("new_position_side", 0.0)) * float(info.get("new_position_size", 0.0))
        t_after = int(info.get("t_after", int(env.t)))
        if done or ((t_after % update_every) == 0):
            progress.update(min(progress_total, max(0, t_after)), suffix=f"t={min(progress_total, max(0, t_after))}/{progress_total} trades={len(trades)}")

        event_ts = pd.to_datetime(info.get("event_ts", idx[min(t, n - 1)]))
        px_event = float(df_sym.iloc[min(t, n - 1)]["close"])

        if int(info.get("close_prev_trade", 0)) == 1 and open_trade is not None:
            side_sign = 1.0 if str(open_trade["side"]) == "long" else -1.0
            ret = ((px_event / float(open_trade["entry_price"])) - 1.0) * side_sign * float(open_trade["size"])
            trades.append(
                RLTrade(
                    entry_ts=pd.to_datetime(open_trade["entry_ts"]),
                    exit_ts=event_ts,
                    side=str(open_trade["side"]),
                    entry_price=float(open_trade["entry_price"]),
                    exit_price=float(px_event),
                    size=float(open_trade["size"]),
                    ret=float(ret),
                )
            )
            open_trade = None

        if int(info.get("open_new_trade", 0)) == 1:
            side_new = int(info.get("new_position_side", 0))
            if side_new != 0:
                open_trade = {
                    "entry_ts": event_ts,
                    "entry_price": float(px_event),
                    "side": "long" if side_new > 0 else "short",
                    "size": float(info.get("new_position_size", 0.0)),
                }

        if int(info.get("forced_eod_close", 0)) == 1 and open_trade is not None:
            exit_ts = pd.to_datetime(info.get("forced_exit_ts", idx[-1]))
            exit_px = float(info.get("forced_exit_price", float(df_sym.iloc[-1]["close"])))
            side_sign = 1.0 if str(open_trade["side"]) == "long" else -1.0
            ret = ((exit_px / float(open_trade["entry_price"])) - 1.0) * side_sign * float(open_trade["size"])
            trades.append(
                RLTrade(
                    entry_ts=pd.to_datetime(open_trade["entry_ts"]),
                    exit_ts=exit_ts,
                    side=str(open_trade["side"]),
                    entry_price=float(open_trade["entry_price"]),
                    exit_price=float(exit_px),
                    size=float(open_trade["size"]),
                    ret=float(ret),
                )
            )
            open_trade = None
    progress.update(progress_total, suffix=f"t={progress_total}/{progress_total} trades={len(trades)}")
    progress.close()

    if open_trade is not None:
        exit_ts = pd.to_datetime(idx[-1])
        exit_px = float(df_sym.iloc[-1]["close"])
        side_sign = 1.0 if str(open_trade["side"]) == "long" else -1.0
        ret = ((exit_px / float(open_trade["entry_price"])) - 1.0) * side_sign * float(open_trade["size"])
        trades.append(
            RLTrade(
                entry_ts=pd.to_datetime(open_trade["entry_ts"]),
                exit_ts=exit_ts,
                side=str(open_trade["side"]),
                entry_price=float(open_trade["entry_price"]),
                exit_price=float(exit_px),
                size=float(open_trade["size"]),
                ret=float(ret),
            )
        )

    action_id = _ffill_nan(action_id, fill_value=float(ACTION_HOLD))
    action_q = _ffill_nan(action_q, fill_value=0.0)
    q_hold = _ffill_nan(q_hold, fill_value=0.0)
    q_long_small = _ffill_nan(q_long_small, fill_value=0.0)
    q_long_big = _ffill_nan(q_long_big, fill_value=0.0)
    q_short_small = _ffill_nan(q_short_small, fill_value=0.0)
    q_short_big = _ffill_nan(q_short_big, fill_value=0.0)
    q_close_long = _ffill_nan(q_close_long, fill_value=0.0)
    q_close_short = _ffill_nan(q_close_short, fill_value=0.0)
    q_long_best = _ffill_nan(q_long_best, fill_value=0.0)
    q_short_best = _ffill_nan(q_short_best, fill_value=0.0)
    q_close_best = _ffill_nan(q_close_best, fill_value=0.0)

    if n >= 2:
        exposure[-1] = 0.0
    equity = np.asarray(env.equity_curve, dtype=np.float64)
    if len(equity) < n:
        pad = np.full(n - len(equity), equity[-1] if len(equity) else 1.0, dtype=np.float64)
        equity = np.concatenate([equity, pad], axis=0)
    elif len(equity) > n:
        equity = equity[:n]

    reg_map = {
        "mu_long_norm": np.asarray(df_sym.get("mu_long_norm", np.zeros(n)), dtype=np.float32),
        "mu_short_norm": np.asarray(df_sym.get("mu_short_norm", np.zeros(n)), dtype=np.float32),
        "edge_norm": np.asarray(df_sym.get("edge_norm", np.zeros(n)), dtype=np.float32),
        "strength_norm": np.asarray(df_sym.get("strength_norm", np.zeros(n)), dtype=np.float32),
    }
    q_map = {
        "hold": q_hold,
        "long_best": q_long_best,
        "short_best": q_short_best,
        "close_best": q_close_best,
        "chosen": action_q,
    }

    if bool(s.save_plot):
        plot_backtest_single_rl(
            df_sym,
            trades=trades,
            equity=equity,
            action_id=action_id,
            exposure=exposure,
            rl_q_map=q_map,
            regressor_map=reg_map,
            title=f"{symbol} | RL single-symbol | rows={n} | trades={len(trades)}",
            save_path=str(plot_path),
            show=bool(s.show_plot),
            plot_candles=bool(s.plot_candles),
        )

    timeline = df_sym.copy()
    timeline["equity"] = np.asarray(equity, dtype=np.float64)
    timeline["action_id"] = action_id.astype(np.int16)
    timeline["action_name"] = [str(_action_name(v)) for v in action_id]
    timeline["action_q"] = action_q
    timeline["exposure"] = exposure.astype(np.float32)
    timeline["reward_step"] = reward_step.astype(np.float32)
    timeline["q_hold"] = q_hold
    timeline["q_long_small"] = q_long_small
    timeline["q_long_big"] = q_long_big
    timeline["q_short_small"] = q_short_small
    timeline["q_short_big"] = q_short_big
    timeline["q_close_long"] = q_close_long
    timeline["q_close_short"] = q_close_short
    timeline["q_long_best"] = q_long_best
    timeline["q_short_best"] = q_short_best
    timeline["q_close_best"] = q_close_best

    if bool(s.save_timeline):
        _save_timeline(timeline, timeline_path)

    m = env.summary()
    hold_hours_arr = np.asarray(
        [max(0.0, float((pd.to_datetime(t.exit_ts) - pd.to_datetime(t.entry_ts)).total_seconds()) / 3600.0) for t in trades],
        dtype=np.float64,
    )
    avg_trade_hours = float(np.nanmean(hold_hours_arr)) if hold_hours_arr.size else 0.0
    med_trade_hours = float(np.nanmedian(hold_hours_arr)) if hold_hours_arr.size else 0.0
    p90_trade_hours = float(np.nanpercentile(hold_hours_arr, 90.0)) if hold_hours_arr.size else 0.0
    max_trade_hours = float(np.nanmax(hold_hours_arr)) if hold_hours_arr.size else 0.0
    out = {
        "symbol": symbol,
        "signals_path": str(signals_path),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "plot_path": str(plot_path) if bool(s.save_plot) else "",
        "timeline_path": str(timeline_path) if bool(s.save_timeline) else "",
        "rows": float(n),
        "trades": float(m.get("trades", 0.0)),
        "ret_total": float(m.get("ret_total", 0.0)),
        "max_dd": float(m.get("max_dd", 0.0)),
        "profit_factor": float(m.get("profit_factor", 0.0)),
        "win_rate": float(m.get("win_rate", 0.0)),
        "trades_per_week": float(m.get("trades_per_week", 0.0)),
        "avg_hold_hours": float(m.get("avg_hold_hours", 0.0)),
        "avg_hold_long_hours": float(m.get("avg_hold_long_hours", 0.0)),
        "avg_hold_short_hours": float(m.get("avg_hold_short_hours", 0.0)),
        "avg_peak_ret": float(m.get("avg_peak_ret", 0.0)),
        "avg_giveback": float(m.get("avg_giveback", 0.0)),
        "trade_efficiency_mean": float(m.get("trade_efficiency_mean", 0.0)),
        "min_reentry_gap_bars": float(env_cfg.min_reentry_gap_bars),
        "min_hold_bars": float(env_cfg.min_hold_bars),
        "max_hold_bars": float(env_cfg.max_hold_bars),
        "avg_trade_hours": float(avg_trade_hours),
        "median_trade_hours": float(med_trade_hours),
        "p90_trade_hours": float(p90_trade_hours),
        "max_trade_hours": float(max_trade_hours),
    }
    print(
        f"[rl-single] sym={symbol} rows={n} trades={int(m.get('trades', 0.0))} "
        f"ret={float(m.get('ret_total', 0.0)):+.2%} dd={float(m.get('max_dd', 0.0)):.2%} "
        f"pf={float(m.get('profit_factor', 0.0)):.3f} trades/wk={float(m.get('trades_per_week', 0.0)):.3f} "
        f"hold(L/S)={float(m.get('avg_hold_long_hours', 0.0)):.2f}/{float(m.get('avg_hold_short_hours', 0.0)):.2f}h "
        f"eff={float(m.get('trade_efficiency_mean', 0.0)):+.3f} giveback={float(m.get('avg_giveback', 0.0)):.4f} "
        f"hold_h(avg/med/p90/max)={avg_trade_hours:.2f}/{med_trade_hours:.2f}/{p90_trade_hours:.2f}/{max_trade_hours:.2f} "
        f"time={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    if bool(s.save_plot):
        print(f"[rl-single] plot: {plot_path}", flush=True)
    if bool(s.save_timeline):
        print(f"[rl-single] timeline: {timeline_path}", flush=True)
    return out


def _parse_args(argv: Iterable[str] | None = None) -> SingleSymbolRLDemoSettings:
    root = _default_hybrid_root()
    ap = argparse.ArgumentParser(description="Backtest/plot single-symbol para policy RL")
    ap.add_argument("--symbol", default="STXUSDT")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--signals", default=str(root / "supervised_signals.parquet"))
    ap.add_argument("--run-dir", default=str(root / "rl_runs" / "default"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fold-id", type=int, default=-1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--plot-out", default=str(root / "reports" / "single_symbol_stxusdt_rl.html"))
    ap.add_argument("--timeline-out", default=str(root / "reports" / "single_symbol_stxusdt_rl_timeline.parquet"))
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-save-plot", action="store_true")
    ap.add_argument("--no-save-timeline", action="store_true")
    ap.add_argument("--plot-line", action="store_true")
    ns = ap.parse_args(list(argv) if argv is not None else None)
    return SingleSymbolRLDemoSettings(
        symbol=str(ns.symbol),
        days=int(ns.days),
        signals_path=str(ns.signals),
        run_dir=str(ns.run_dir),
        checkpoint=(None if ns.checkpoint in (None, "", "none") else str(ns.checkpoint)),
        fold_id=int(ns.fold_id),
        device=str(ns.device),
        plot_out=str(ns.plot_out),
        timeline_out=str(ns.timeline_out),
        show_plot=bool(not ns.no_show),
        save_plot=bool(not ns.no_save_plot),
        save_timeline=bool(not ns.no_save_timeline),
        plot_candles=bool(not ns.plot_line),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = run(cfg)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
