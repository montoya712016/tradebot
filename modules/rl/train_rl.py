# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Treino RL (DQN) sobre sinais supervisionados exportados.

Split temporal estrito (walk-forward global):
- mesma faixa de timestamps para todos os ativos
- sem mistura temporal entre treino e validação
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
import sys
import time
from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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

from env_rl import TradingEnvConfig, HybridTradingEnv  # type: ignore
from env_rl.action_space import (  # type: ignore
    ALL_ACTIONS,
    ACTION_HOLD,
    ACTION_OPEN_LONG_SMALL,
    ACTION_OPEN_LONG_BIG,
    ACTION_OPEN_SHORT_SMALL,
    ACTION_OPEN_SHORT_BIG,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)
from env_rl.reward import RewardConfig  # type: ignore
from rl.walkforward import build_time_folds, split_by_fold, FoldSpec  # type: ignore


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


STATE_COLS = [
    "mu_long_norm",
    "mu_short_norm",
    "edge_norm",
    "strength_norm",
    "uncertainty_norm",
    "vol_short_norm",
    "vol_long_norm",
    "trend_strength_norm",
    "shock_flag",
]


@dataclass
class TrainRLConfig:
    signals_path: str = str(_default_storage_root() / "supervised_signals.parquet")
    out_dir: str = str(_default_storage_root() / "rl_runs" / "default")
    seed: int = 42
    train_days: int = 720
    valid_days: int = 365
    step_days: int = 365
    embargo_minutes: int = 0
    max_folds: int = 1
    min_rows_symbol: int = 800
    single_split_mode: bool = True
    holdout_days: int = 365

    epochs: int = 20
    episodes_per_epoch: int = 48
    episode_bars: int = 10_080
    random_starts: bool = True
    batch_size: int = 1024
    replay_size: int = 200_000
    gamma: float = 0.995
    lr: float = 3e-4
    hidden_dim: int = 128
    target_update_steps: int = 250
    warmup_steps: int = 3_000
    train_steps_per_epoch: int = 4_000
    eval_every_epochs: int = 5
    grad_clip: float = 1.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 8
    expert_mix_start: float = 0.6
    expert_mix_end: float = 0.0
    expert_mix_decay_epochs: int = 10

    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    turnover_penalty: float = 0.0007
    dd_penalty: float = 0.20
    regret_penalty: float = 0.02
    idle_penalty: float = 0.0
    cooldown_bars: int = 8
    regret_window: int = 10
    small_size: float = 0.2
    big_size: float = 0.5
    min_reentry_gap_bars: int = 10_080
    min_hold_bars: int = 240
    edge_entry_threshold: float = 0.40
    strength_entry_threshold: float = 0.70
    target_trades_per_week: float = 1.0
    trade_rate_penalty: float = 0.05

    device: str = "cuda"


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.adv = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(max(1_000, capacity))
        self.buf = deque(maxlen=self.capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool) -> None:
        self.buf.append((s.astype(np.float32, copy=False), int(a), float(r), s2.astype(np.float32, copy=False), float(d)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self.buf), size=int(batch_size))
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _pick_device(pref: str) -> torch.device:
    if str(pref).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _epsilon(epoch: int, cfg: TrainRLConfig) -> float:
    k = int(max(1, cfg.epsilon_decay_epochs))
    t = min(1.0, max(0.0, float(epoch) / float(k)))
    return float(cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * t)


def _expert_prob(epoch: int, cfg: TrainRLConfig) -> float:
    k = int(max(1, cfg.expert_mix_decay_epochs))
    t = min(1.0, max(0.0, float(epoch) / float(k)))
    return float(cfg.expert_mix_start + (cfg.expert_mix_end - cfg.expert_mix_start) * t)


def _expert_action(env: HybridTradingEnv, t: int) -> int:
    edge = float(env.edge_norm[t])
    strength = float(env.strength_norm[t])
    mu_l = float(env.mu_long_norm[t])
    mu_s = float(env.mu_short_norm[t])
    side = int(env.position_side)
    edge_thr = float(env.cfg.edge_entry_threshold)
    strength_thr = float(env.cfg.strength_entry_threshold)
    if side > 0:
        if strength < strength_thr or edge < 0.0:
            return ACTION_CLOSE_LONG
        if edge > edge_thr:
            return ACTION_OPEN_LONG_BIG
        if mu_l >= mu_s:
            return ACTION_OPEN_LONG_SMALL
        return ACTION_HOLD
    if side < 0:
        if strength < strength_thr or edge > 0.0:
            return ACTION_CLOSE_SHORT
        if edge < -edge_thr:
            return ACTION_OPEN_SHORT_BIG
        if mu_s > mu_l:
            return ACTION_OPEN_SHORT_SMALL
        return ACTION_HOLD
    if strength < strength_thr:
        return ACTION_HOLD
    if edge > edge_thr:
        return ACTION_OPEN_LONG_BIG
    if edge < -edge_thr:
        return ACTION_OPEN_SHORT_BIG
    return ACTION_OPEN_LONG_SMALL if mu_l >= mu_s else ACTION_OPEN_SHORT_SMALL


def _load_signals(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"signals_path nao encontrado: {p}")
    df = pd.read_parquet(p)
    if "symbol" not in df.columns:
        raise RuntimeError("Arquivo de sinais precisa da coluna 'symbol'")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def _build_env_cfg(cfg: TrainRLConfig) -> TradingEnvConfig:
    return TradingEnvConfig(
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        small_size=float(cfg.small_size),
        big_size=float(cfg.big_size),
        cooldown_bars=int(cfg.cooldown_bars),
        regret_window=int(cfg.regret_window),
        min_reentry_gap_bars=int(cfg.min_reentry_gap_bars),
        min_hold_bars=int(cfg.min_hold_bars),
        use_signal_gate=True,
        edge_entry_threshold=float(cfg.edge_entry_threshold),
        strength_entry_threshold=float(cfg.strength_entry_threshold),
        reward=RewardConfig(
            dd_penalty=float(cfg.dd_penalty),
            turnover_penalty=float(cfg.turnover_penalty),
            regret_penalty=float(cfg.regret_penalty),
            idle_penalty=float(cfg.idle_penalty),
        ),
    )


def _split_symbols(df: pd.DataFrame, min_rows: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, sdf in df.groupby("symbol", sort=True):
        sdf2 = sdf.sort_index().copy()
        if len(sdf2) < int(min_rows):
            continue
        if "close" not in sdf2.columns:
            continue
        if float(np.nanmax(sdf2["close"].to_numpy(dtype=np.float64, copy=False))) <= 0.0:
            continue
        out[str(sym)] = sdf2
    return out


def _build_single_recent_fold(
    timestamps: pd.DatetimeIndex,
    *,
    train_days: int,
    holdout_days: int,
    embargo_minutes: int = 0,
) -> FoldSpec:
    ts = pd.to_datetime(pd.Index(timestamps)).sort_values().unique()
    if len(ts) == 0:
        raise RuntimeError("timestamps vazio para split unico")
    t_min = pd.Timestamp(ts.min())
    t_max = pd.Timestamp(ts.max())
    d_train = pd.Timedelta(days=int(max(1, train_days)))
    d_hold = pd.Timedelta(days=int(max(1, holdout_days)))
    d_emb = pd.Timedelta(minutes=int(max(0, embargo_minutes)))

    valid_end = t_max + pd.Timedelta(minutes=1)
    valid_start = max(t_min, t_max - d_hold)
    train_end = valid_start - d_emb
    train_start = max(t_min, train_end - d_train)
    if train_start >= train_end:
        raise RuntimeError("split unico invalido: ajuste train_days/holdout_days")
    if valid_start >= valid_end:
        raise RuntimeError("split unico invalido: janela de validacao vazia")
    return FoldSpec(
        fold_id=0,
        train_start=pd.Timestamp(train_start),
        train_end=pd.Timestamp(train_end),
        valid_start=pd.Timestamp(valid_start),
        valid_end=pd.Timestamp(valid_end),
    )


def _choose_action_masked(q: QNet, state: np.ndarray, valid_actions: list[int], device: torch.device) -> int:
    if not valid_actions:
        valid_actions = list(ALL_ACTIONS)
    with torch.no_grad():
        qs = q(torch.from_numpy(state).to(device).unsqueeze(0))
        idx = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        qv = qs.index_select(1, idx)
        best_i = int(torch.argmax(qv, dim=1).item())
        return int(valid_actions[best_i])


def _run_episode_collect(
    env: HybridTradingEnv,
    q: QNet,
    *,
    eps: float,
    buffer: ReplayBuffer,
    device: torch.device,
    expert_prob: float = 0.0,
    start_idx: int = 0,
    max_steps: int = 0,
) -> dict[str, float]:
    s = env.reset(start_idx=int(max(0, start_idx)))
    done = False
    steps = 0
    while not done:
        valid_actions = env.valid_actions()
        if not valid_actions:
            valid_actions = list(ALL_ACTIONS)
        if np.random.rand() < float(max(0.0, min(1.0, expert_prob))):
            a = int(_expert_action(env, env.t))
            if a not in valid_actions:
                a = int(valid_actions[0])
        elif np.random.rand() < float(eps):
            a = int(np.random.choice(valid_actions))
        else:
            a = _choose_action_masked(q, s, valid_actions, device)
        s2, r, done, _info = env.step(a)
        steps += 1
        if int(max_steps) > 0 and steps >= int(max_steps):
            done = True
        buffer.push(s, a, r, s2, done)
        s = s2
    return env.summary()


def _optimize(
    q: QNet,
    target: QNet,
    opt: torch.optim.Optimizer,
    batch,
    *,
    gamma: float,
    device: torch.device,
    grad_clip: float,
) -> float:
    s, a, r, s2, d = batch
    s_t = torch.from_numpy(s).to(device)
    a_t = torch.from_numpy(a).to(device).long().view(-1, 1)
    r_t = torch.from_numpy(r).to(device).float().view(-1, 1)
    s2_t = torch.from_numpy(s2).to(device)
    d_t = torch.from_numpy(d).to(device).float().view(-1, 1)

    q_sa = q(s_t).gather(1, a_t)
    with torch.no_grad():
        next_a = q(s2_t).argmax(dim=1, keepdim=True)
        q_next = target(s2_t).gather(1, next_a)
        q_target = r_t + (1.0 - d_t) * float(gamma) * q_next
    loss = F.smooth_l1_loss(q_sa, q_target)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(q.parameters(), float(grad_clip))
    opt.step()
    return float(loss.item())


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
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
            "avg_turnover_mean": 0.0,
        }
    ret = np.asarray([r.get("ret_total", 0.0) for r in rows], dtype=np.float64)
    dd = np.asarray([r.get("max_dd", 0.0) for r in rows], dtype=np.float64)
    pf = np.asarray([r.get("profit_factor", 0.0) for r in rows], dtype=np.float64)
    wr = np.asarray([r.get("win_rate", 0.0) for r in rows], dtype=np.float64)
    tr = np.asarray([r.get("trades", 0.0) for r in rows], dtype=np.float64)
    trd = np.asarray([r.get("trades_per_day", 0.0) for r in rows], dtype=np.float64)
    trw = np.asarray([r.get("trades_per_week", 0.0) for r in rows], dtype=np.float64)
    ahb = np.asarray([r.get("avg_hold_bars", 0.0) for r in rows], dtype=np.float64)
    to = np.asarray([r.get("avg_turnover", 0.0) for r in rows], dtype=np.float64)
    return {
        "ret_total_mean": float(np.nanmean(ret)),
        "max_dd_mean": float(np.nanmean(dd)),
        "profit_factor_median": float(np.nanmedian(pf)),
        "win_rate_mean": float(np.nanmean(wr)),
        "trades_total": float(np.nansum(tr)),
        "trades_per_day_mean": float(np.nanmean(trd)),
        "trades_per_week_mean": float(np.nanmean(trw)),
        "avg_hold_bars_mean": float(np.nanmean(ahb)),
        "avg_turnover_mean": float(np.nanmean(to)),
    }


def _evaluate_policy(
    q: QNet,
    symbol_frames: dict[str, pd.DataFrame],
    env_cfg: TradingEnvConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    q.eval()
    rows: list[dict[str, float]] = []
    for _sym, sdf in symbol_frames.items():
        env = HybridTradingEnv(sdf, env_cfg)
        s = env.reset()
        done = False
        while not done:
            a = _choose_action_masked(q, s, env.valid_actions(), device)
            s, _r, done, _info = env.step(a)
        rows.append(env.summary())
    return _aggregate_metrics(rows)


def _train_single_fold(
    fold: FoldSpec,
    train_frames: dict[str, pd.DataFrame],
    valid_frames: dict[str, pd.DataFrame],
    cfg: TrainRLConfig,
    *,
    out_dir: Path,
) -> dict[str, float]:
    if not train_frames or not valid_frames:
        return {"fold_id": float(fold.fold_id), "skipped": 1.0}

    env_cfg = _build_env_cfg(cfg)
    sample_env = HybridTradingEnv(next(iter(train_frames.values())), env_cfg)
    state_dim = sample_env.state_dim
    n_actions = len(ALL_ACTIONS)

    device = _pick_device(cfg.device)
    q = QNet(state_dim, n_actions, int(cfg.hidden_dim)).to(device)
    target = QNet(state_dim, n_actions, int(cfg.hidden_dim)).to(device)
    target.load_state_dict(q.state_dict())
    target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=float(cfg.lr))
    buffer = ReplayBuffer(int(cfg.replay_size))
    train_syms = list(train_frames.keys())
    step_count = 0
    fold_dir = out_dir / f"fold_{int(fold.fold_id):03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    ckpt_meta = {
        "state_dim": int(state_dim),
        "n_actions": int(n_actions),
        "hidden_dim": int(cfg.hidden_dim),
        "env_cfg": {
            "fee_rate": float(cfg.fee_rate),
            "slippage_rate": float(cfg.slippage_rate),
            "small_size": float(cfg.small_size),
            "big_size": float(cfg.big_size),
            "cooldown_bars": int(cfg.cooldown_bars),
            "regret_window": int(cfg.regret_window),
            "min_reentry_gap_bars": int(cfg.min_reentry_gap_bars),
            "min_hold_bars": int(cfg.min_hold_bars),
            "use_signal_gate": True,
            "edge_entry_threshold": float(cfg.edge_entry_threshold),
            "strength_entry_threshold": float(cfg.strength_entry_threshold),
            "reward": {
                "dd_penalty": float(cfg.dd_penalty),
                "turnover_penalty": float(cfg.turnover_penalty),
                "regret_penalty": float(cfg.regret_penalty),
                "idle_penalty": float(cfg.idle_penalty),
            },
        },
    }

    train_log: list[dict[str, float]] = []
    best_score = -np.inf
    best_epoch = -1
    best_state: dict | None = None
    last_valid_metrics = {
        "ret_total_mean": 0.0,
        "max_dd_mean": 0.0,
        "profit_factor_median": 0.0,
        "trades_total": 0.0,
        "trades_per_day_mean": 0.0,
        "trades_per_week_mean": 0.0,
        "avg_hold_bars_mean": 0.0,
    }
    t_fold_start = time.perf_counter()
    for epoch in range(int(cfg.epochs)):
        t_ep_start = time.perf_counter()
        eps = _epsilon(epoch, cfg)
        exp_p = _expert_prob(epoch, cfg)
        rows_epoch: list[dict[str, float]] = []
        q.train()
        t_collect_start = time.perf_counter()
        for _ep in range(int(cfg.episodes_per_epoch)):
            sym = str(np.random.choice(train_syms))
            sdf = train_frames[sym]
            env = HybridTradingEnv(sdf, env_cfg)
            max_steps = int(max(0, cfg.episode_bars))
            if bool(cfg.random_starts) and max_steps > 0 and len(sdf) > (max_steps + 2):
                hi = int(len(sdf) - max_steps - 1)
                start_idx = int(np.random.randint(0, max(1, hi)))
            else:
                start_idx = 0
            rows_epoch.append(
                _run_episode_collect(
                    env,
                    q,
                    eps=eps,
                    buffer=buffer,
                    device=device,
                    expert_prob=float(exp_p),
                    start_idx=int(start_idx),
                    max_steps=int(max_steps),
                )
            )
        dt_collect = time.perf_counter() - t_collect_start

        losses: list[float] = []
        t_opt_start = time.perf_counter()
        if len(buffer) >= int(cfg.batch_size):
            n_steps = int(cfg.train_steps_per_epoch)
            for _ in range(n_steps):
                batch = buffer.sample(int(cfg.batch_size))
                loss = _optimize(
                    q,
                    target,
                    opt,
                    batch,
                    gamma=float(cfg.gamma),
                    device=device,
                    grad_clip=float(cfg.grad_clip),
                )
                losses.append(loss)
                step_count += 1
                if step_count % int(max(1, cfg.target_update_steps)) == 0:
                    target.load_state_dict(q.state_dict())
        dt_opt = time.perf_counter() - t_opt_start

        train_metrics = _aggregate_metrics(rows_epoch)
        eval_every = int(max(1, cfg.eval_every_epochs))
        need_eval = bool(((epoch + 1) % eval_every == 0) or ((epoch + 1) == int(cfg.epochs)))
        if need_eval:
            t_eval_start = time.perf_counter()
            valid_metrics = _evaluate_policy(q, valid_frames, env_cfg, device=device)
            dt_eval = time.perf_counter() - t_eval_start
            last_valid_metrics = dict(valid_metrics)
        else:
            valid_metrics = dict(last_valid_metrics)
            dt_eval = 0.0
        rec = {
            "epoch": float(epoch),
            "epsilon": float(eps),
            "expert_prob": float(exp_p),
            "buffer_size": float(len(buffer)),
            "loss_mean": float(np.mean(losses)) if losses else 0.0,
            "train_ret_total_mean": float(train_metrics["ret_total_mean"]),
            "train_max_dd_mean": float(train_metrics["max_dd_mean"]),
            "train_pf_median": float(train_metrics["profit_factor_median"]),
            "valid_ret_total_mean": float(valid_metrics["ret_total_mean"]),
            "valid_max_dd_mean": float(valid_metrics["max_dd_mean"]),
            "valid_pf_median": float(valid_metrics["profit_factor_median"]),
            "valid_trades_total": float(valid_metrics["trades_total"]),
            "valid_trades_per_day_mean": float(valid_metrics["trades_per_day_mean"]),
            "valid_trades_per_week_mean": float(valid_metrics["trades_per_week_mean"]),
            "valid_avg_hold_bars_mean": float(valid_metrics["avg_hold_bars_mean"]),
            "valid_eval_fresh": float(1.0 if need_eval else 0.0),
            "time_collect_s": float(dt_collect),
            "time_optimize_s": float(dt_opt),
            "time_eval_s": float(dt_eval),
        }
        train_log.append(rec)
        trw_excess = float(max(0.0, rec["valid_trades_per_week_mean"] - float(cfg.target_trades_per_week)))
        trade_rate_pen = float(cfg.trade_rate_penalty) * float(trw_excess * trw_excess)
        score = float(rec["valid_ret_total_mean"] - 0.5 * rec["valid_max_dd_mean"] - trade_rate_pen)
        if need_eval and (score > best_score):
            best_score = score
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
            torch.save({"state_dict": best_state, **ckpt_meta}, fold_dir / "policy_dqn.pt")

        # persistencia incremental por epoca (permite interromper sem perder tudo)
        t_save_start = time.perf_counter()
        torch.save({"state_dict": q.state_dict(), **ckpt_meta}, fold_dir / "policy_last.pt")
        _atomic_write_json(fold_dir / "train_log.json", train_log)
        _atomic_write_json(
            fold_dir / "summary.partial.json",
            {
                "fold_id": float(fold.fold_id),
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "valid_start": str(fold.valid_start),
                "valid_end": str(fold.valid_end),
                "train_symbols": float(len(train_frames)),
                "valid_symbols": float(len(valid_frames)),
                "completed_epochs": float(epoch + 1),
                "best_epoch_so_far": float(best_epoch),
                "best_score_so_far": float(best_score),
                "target_trades_per_week": float(cfg.target_trades_per_week),
                "trade_rate_penalty": float(cfg.trade_rate_penalty),
                "last_expert_prob": float(rec["expert_prob"]),
                "best_valid_ret_total_mean_so_far": float(max(x["valid_ret_total_mean"] for x in train_log)),
                "best_valid_max_dd_mean_so_far": float(min(x["valid_max_dd_mean"] for x in train_log)),
                "last_valid_ret_total_mean": float(rec["valid_ret_total_mean"]),
                "last_valid_max_dd_mean": float(rec["valid_max_dd_mean"]),
                "last_valid_pf_median": float(rec["valid_pf_median"]),
                "last_valid_trades_total": float(rec["valid_trades_total"]),
                "last_valid_trades_per_week_mean": float(rec["valid_trades_per_week_mean"]),
                "last_valid_avg_hold_bars_mean": float(rec["valid_avg_hold_bars_mean"]),
                "last_valid_eval_fresh": float(rec["valid_eval_fresh"]),
                "last_time_collect_s": float(rec["time_collect_s"]),
                "last_time_optimize_s": float(rec["time_optimize_s"]),
                "last_time_eval_s": float(rec["time_eval_s"]),
            },
        )
        dt_save = time.perf_counter() - t_save_start
        dt_ep = time.perf_counter() - t_ep_start
        print(
            f"[rl][fold {fold.fold_id}] epoch={epoch+1}/{cfg.epochs} "
            f"eps={eps:.3f} exp={exp_p:.3f} loss={rec['loss_mean']:.6f} "
            f"valid_ret={rec['valid_ret_total_mean']:+.3%} valid_dd={rec['valid_max_dd_mean']:.2%} "
            f"valid_pf={rec['valid_pf_median']:.3f} trades/wk={rec['valid_trades_per_week_mean']:.3f} "
            f"score={score:+.4f} eval_fresh={int(need_eval)} "
            f"time(ep/collect/opt/eval/save)={dt_ep:.1f}/{dt_collect:.1f}/{dt_opt:.1f}/{dt_eval:.1f}/{dt_save:.1f}s",
            flush=True,
        )

    torch.save({"state_dict": q.state_dict(), **ckpt_meta}, fold_dir / "policy_last.pt")
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
    torch.save({"state_dict": best_state, **ckpt_meta}, fold_dir / "policy_dqn.pt")
    _atomic_write_json(fold_dir / "train_log.json", train_log)

    def _score_row(x: dict[str, float]) -> float:
        trw_excess = float(max(0.0, x.get("valid_trades_per_week_mean", 0.0) - float(cfg.target_trades_per_week)))
        trade_rate_pen = float(cfg.trade_rate_penalty) * float(trw_excess * trw_excess)
        return float(x["valid_ret_total_mean"] - 0.5 * x["valid_max_dd_mean"] - trade_rate_pen)

    eval_rows = [x for x in train_log if float(x.get("valid_eval_fresh", 1.0)) > 0.5]
    best_pool = eval_rows if eval_rows else train_log
    best = max(best_pool, key=_score_row)
    summary = {
        "fold_id": float(fold.fold_id),
        "train_start": str(fold.train_start),
        "train_end": str(fold.train_end),
        "valid_start": str(fold.valid_start),
        "valid_end": str(fold.valid_end),
        "train_symbols": float(len(train_frames)),
        "valid_symbols": float(len(valid_frames)),
        "best_valid_ret_total_mean": float(best["valid_ret_total_mean"]),
        "best_valid_max_dd_mean": float(best["valid_max_dd_mean"]),
        "best_valid_pf_median": float(best["valid_pf_median"]),
        "best_valid_trades_total": float(best["valid_trades_total"]),
        "best_valid_trades_per_week_mean": float(best.get("valid_trades_per_week_mean", 0.0)),
        "best_valid_avg_hold_bars_mean": float(best.get("valid_avg_hold_bars_mean", 0.0)),
        "target_trades_per_week": float(cfg.target_trades_per_week),
        "trade_rate_penalty": float(cfg.trade_rate_penalty),
        "best_epoch": float(best_epoch),
        "best_score": float(_score_row(best)),
    }
    _atomic_write_json(fold_dir / "summary.json", summary)
    try:
        (fold_dir / "summary.partial.json").unlink(missing_ok=True)
    except Exception:
        pass
    dt_fold = time.perf_counter() - t_fold_start
    print(f"[rl][time] fold={fold.fold_id} total={dt_fold:.2f}s", flush=True)
    return summary


def train_rl_walkforward(cfg: TrainRLConfig) -> Path:
    t_total = time.perf_counter()
    _set_seed(int(cfg.seed))
    t0 = time.perf_counter()
    df = _load_signals(cfg.signals_path)
    print(f"[rl][time] load_signals={time.perf_counter() - t0:.2f}s rows={len(df)}", flush=True)
    t0 = time.perf_counter()
    if bool(cfg.single_split_mode):
        folds = [
            _build_single_recent_fold(
                pd.to_datetime(df.index),
                train_days=int(cfg.train_days),
                holdout_days=int(cfg.holdout_days),
                embargo_minutes=int(cfg.embargo_minutes),
            )
        ]
        print(
            f"[rl][time] build_single_split={time.perf_counter() - t0:.2f}s "
            f"holdout_days={int(cfg.holdout_days)}",
            flush=True,
        )
    else:
        folds = build_time_folds(
            df.index,
            train_days=int(cfg.train_days),
            valid_days=int(cfg.valid_days),
            step_days=int(cfg.step_days),
            embargo_minutes=int(cfg.embargo_minutes),
            max_folds=int(cfg.max_folds),
        )
        print(f"[rl][time] build_folds={time.perf_counter() - t0:.2f}s n_folds={len(folds)}", flush=True)
    if not folds:
        raise RuntimeError("Nenhum fold gerado; ajuste train_days/valid_days/step_days")

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    fold_rows: list[dict[str, float]] = []
    for fold in folds:
        t_fold = time.perf_counter()
        df_train, df_valid = split_by_fold(df, fold)
        train_frames = _split_symbols(df_train, int(cfg.min_rows_symbol))
        valid_frames = _split_symbols(df_valid, int(cfg.min_rows_symbol))
        print(
            f"[rl] fold={fold.fold_id} train={fold.train_start}..{fold.train_end} "
            f"valid={fold.valid_start}..{fold.valid_end} train_syms={len(train_frames)} valid_syms={len(valid_frames)}",
            flush=True,
        )
        row = _train_single_fold(fold, train_frames, valid_frames, cfg, out_dir=out_dir)
        fold_rows.append(row)
        _atomic_write_json(out_dir / "fold_summaries.json", fold_rows)
        print(f"[rl][time] fold={fold.fold_id} end_to_end={time.perf_counter() - t_fold:.2f}s", flush=True)

    _atomic_write_json(out_dir / "fold_summaries.json", fold_rows)
    print(f"[rl][time] train_total={time.perf_counter() - t_total:.2f}s out_dir={out_dir}", flush=True)
    return out_dir


def _parse_args(argv: Iterable[str] | None = None) -> TrainRLConfig:
    storage_root = _default_storage_root()
    ap = argparse.ArgumentParser(description="Treino RL walk-forward (DQN)")
    ap.add_argument("--signals", default=str(storage_root / "supervised_signals.parquet"))
    ap.add_argument("--out-dir", default=str(storage_root / "rl_runs" / "default"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-days", type=int, default=720)
    ap.add_argument("--valid-days", type=int, default=365)
    ap.add_argument("--step-days", type=int, default=365)
    ap.add_argument("--embargo-minutes", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--single-split-mode", action="store_true", default=True)
    ap.add_argument("--wf-mode", dest="single_split_mode", action="store_false")
    ap.add_argument("--holdout-days", type=int, default=365)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--episodes-per-epoch", type=int, default=48)
    ap.add_argument("--episode-bars", type=int, default=10080)
    ap.add_argument("--no-random-starts", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--replay-size", type=int, default=200000)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--target-update-steps", type=int, default=250)
    ap.add_argument("--train-steps-per-epoch", type=int, default=4000)
    ap.add_argument("--eval-every-epochs", type=int, default=5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-epochs", type=int, default=8)
    ap.add_argument("--expert-mix-start", type=float, default=0.6)
    ap.add_argument("--expert-mix-end", type=float, default=0.0)
    ap.add_argument("--expert-mix-decay-epochs", type=int, default=10)

    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--turnover-penalty", type=float, default=0.0007)
    ap.add_argument("--dd-penalty", type=float, default=0.20)
    ap.add_argument("--regret-penalty", type=float, default=0.02)
    ap.add_argument("--idle-penalty", type=float, default=0.0)
    ap.add_argument("--cooldown-bars", type=int, default=8)
    ap.add_argument("--min-reentry-gap-bars", type=int, default=10080)
    ap.add_argument("--min-hold-bars", type=int, default=240)
    ap.add_argument("--edge-entry-threshold", type=float, default=0.40)
    ap.add_argument("--strength-entry-threshold", type=float, default=0.70)
    ap.add_argument("--target-trades-per-week", type=float, default=1.0)
    ap.add_argument("--trade-rate-penalty", type=float, default=0.05)
    ap.add_argument("--regret-window", type=int, default=10)
    ap.add_argument("--small-size", type=float, default=0.2)
    ap.add_argument("--big-size", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")

    ns = ap.parse_args(list(argv) if argv is not None else None)
    return TrainRLConfig(
        signals_path=str(ns.signals),
        out_dir=str(ns.out_dir),
        seed=int(ns.seed),
        train_days=int(ns.train_days),
        valid_days=int(ns.valid_days),
        step_days=int(ns.step_days),
        embargo_minutes=int(ns.embargo_minutes),
        max_folds=int(ns.max_folds),
        min_rows_symbol=int(ns.min_rows_symbol),
        single_split_mode=bool(ns.single_split_mode),
        holdout_days=int(ns.holdout_days),
        epochs=int(ns.epochs),
        episodes_per_epoch=int(ns.episodes_per_epoch),
        episode_bars=int(ns.episode_bars),
        random_starts=bool(not ns.no_random_starts),
        batch_size=int(ns.batch_size),
        replay_size=int(ns.replay_size),
        gamma=float(ns.gamma),
        lr=float(ns.lr),
        hidden_dim=int(ns.hidden_dim),
        target_update_steps=int(ns.target_update_steps),
        train_steps_per_epoch=int(ns.train_steps_per_epoch),
        eval_every_epochs=int(ns.eval_every_epochs),
        grad_clip=float(ns.grad_clip),
        epsilon_start=float(ns.eps_start),
        epsilon_end=float(ns.eps_end),
        epsilon_decay_epochs=int(ns.eps_decay_epochs),
        expert_mix_start=float(ns.expert_mix_start),
        expert_mix_end=float(ns.expert_mix_end),
        expert_mix_decay_epochs=int(ns.expert_mix_decay_epochs),
        fee_rate=float(ns.fee_rate),
        slippage_rate=float(ns.slippage_rate),
        turnover_penalty=float(ns.turnover_penalty),
        dd_penalty=float(ns.dd_penalty),
        regret_penalty=float(ns.regret_penalty),
        idle_penalty=float(ns.idle_penalty),
        cooldown_bars=int(ns.cooldown_bars),
        min_reentry_gap_bars=int(ns.min_reentry_gap_bars),
        min_hold_bars=int(ns.min_hold_bars),
        regret_window=int(ns.regret_window),
        small_size=float(ns.small_size),
        big_size=float(ns.big_size),
        edge_entry_threshold=float(ns.edge_entry_threshold),
        strength_entry_threshold=float(ns.strength_entry_threshold),
        target_trades_per_week=float(ns.target_trades_per_week),
        trade_rate_penalty=float(ns.trade_rate_penalty),
        device=str(ns.device),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = train_rl_walkforward(cfg)
    print(f"[rl] treino finalizado: {out}")


if __name__ == "__main__":
    main()
