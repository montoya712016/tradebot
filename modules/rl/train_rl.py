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
    ACTION_OPEN_SHORT_SMALL,
    ACTION_CLOSE_LONG,
    ACTION_CLOSE_SHORT,
)
from env_rl.reward import RewardConfig  # type: ignore
from rl.walkforward import build_time_folds, split_by_fold, FoldSpec  # type: ignore
from utils.progress import ProgressPrinter  # type: ignore


def _default_storage_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


STATE_SIGNAL_COLS = [
    "mu_long_norm",
    "mu_short_norm",
]

STATE_INTERNAL_COLS = [
    "pos_flat",
    "pos_long",
    "pos_short",
    "time_in_trade_norm",
    "unrealized_norm",
    "trade_drawdown_norm",
    "cooldown_norm",
]


def _active_state_signal_cols(cfg: "TrainRLConfig") -> list[str]:
    cols = list(STATE_SIGNAL_COLS)
    if not bool(cfg.state_use_uncertainty):
        cols = [c for c in cols if c != "uncertainty_norm"]
    if not bool(cfg.state_use_vol_long):
        cols = [c for c in cols if c != "vol_long_norm"]
    if not bool(cfg.state_use_shock):
        cols = [c for c in cols if c != "shock_flag"]
    return cols


@dataclass
class TrainRLConfig:
    signals_path: str = str(_default_storage_root() / "supervised_signals.parquet")
    signals_contract_path: str = ""
    out_dir: str = str(_default_storage_root() / "rl_runs" / "default")
    seed: int = 42
    train_days: int = 720
    valid_days: int = 180
    step_days: int = 365
    embargo_minutes: int = 0
    max_folds: int = 1
    min_rows_symbol: int = 800
    single_split_mode: bool = True
    holdout_days: int = 180

    epochs: int = 28
    episodes_per_epoch: int = 20
    episode_bars: int = 5_760
    random_starts: bool = True
    batch_size: int = 512
    replay_size: int = 200_000
    gamma: float = 0.995
    lr: float = 1e-4
    hidden_dim: int = 96
    target_update_steps: int = 1_000
    warmup_steps: int = 3_000
    train_steps_per_epoch: int = 1_200
    eval_every_epochs: int = 1
    grad_clip: float = 1.0

    epsilon_start: float = 0.50
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 8
    expert_mix_start: float = 0.45
    expert_mix_end: float = 0.0
    expert_mix_decay_epochs: int = 14
    strict_best_update: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0005

    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    turnover_penalty: float = 0.0020
    dd_penalty: float = 0.35
    dd_level_penalty: float = 0.08
    dd_soft_limit: float = 0.15
    dd_excess_penalty: float = 2.5
    dd_hard_limit: float = 0.30
    dd_hard_penalty: float = 0.05
    hold_bar_penalty: float = 0.0
    hold_soft_bars: int = 180
    hold_excess_penalty: float = 0.0
    hold_regret_penalty: float = 0.09
    stagnation_bars: int = 180
    stagnation_ret_epsilon: float = 0.00003
    stagnation_penalty: float = 0.008
    reverse_penalty: float = 0.0010
    entry_penalty: float = 0.0000
    weak_entry_penalty: float = 0.0000
    regret_penalty: float = 0.02
    idle_penalty: float = 0.0
    cooldown_bars: int = 12
    regret_window: int = 10
    small_size: float = 0.10
    big_size: float = 0.25
    min_reentry_gap_bars: int = 180
    min_hold_bars: int = 90
    max_hold_bars: int = 720
    edge_entry_threshold: float = 0.45
    strength_entry_threshold: float = 0.82
    force_close_on_adverse: bool = True
    edge_exit_threshold: float = 0.22
    strength_exit_threshold: float = 0.18
    allow_direct_flip: bool = True
    allow_scale_in: bool = False
    state_use_uncertainty: bool = False
    state_use_vol_long: bool = False
    state_use_shock: bool = False
    state_use_window_context: bool = True
    state_window_short_bars: int = 30
    state_window_long_bars: int = 30
    target_trades_per_week: float = 4.0
    min_trades_per_week_floor: float = 0.00
    trade_rate_penalty: float = 0.00
    trade_rate_under_penalty: float = 0.00
    score_min_trades_floor_penalty: float = 0.00
    score_ret_weight: float = 0.45
    score_calmar_weight: float = 0.30
    score_pf_weight: float = 0.35
    score_efficiency_weight: float = 0.08
    score_giveback_penalty: float = 0.20
    score_hold_hours_target: float = 8.0
    score_hold_hours_penalty: float = 0.002
    score_hold_hours_min: float = 0.0
    score_hold_hours_under_penalty: float = 0.00
    score_hold_hours_excess_cap: float = 24.0
    score_dd_soft_limit: float = 0.15
    score_dd_excess_penalty: float = 4.0

    eval_symbols_per_epoch: int = 4
    eval_tail_bars: int = 86_400
    eval_full_every_epochs: int = 4

    device: str = "cuda"
    eval_device: str = "cpu"


def _load_signals_contract(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_signals_contract(df: pd.DataFrame, contract: dict) -> None:
    if not contract:
        return
    req = list(contract.get("columns_required", []) or [])
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"signals sem colunas do contrato: {missing}")
    if "oof_ok" in df.columns:
        bad = int((df["oof_ok"].astype(np.int8) <= 0).sum())
        if bad > 0:
            raise RuntimeError(f"signals com OOF invalido: rows_bad={bad}")
    raw_mu_expected = bool(contract.get("raw_mu_for_rl", True))
    if raw_mu_expected and {"mu_long", "mu_long_norm", "mu_short", "mu_short_norm"}.issubset(df.columns):
        # se contrato exige mu bruto no RL, os norm devem bater com o valor bruto.
        d1 = float(np.nanmean(np.abs(df["mu_long"].to_numpy(dtype=np.float64) - df["mu_long_norm"].to_numpy(dtype=np.float64))))
        d2 = float(np.nanmean(np.abs(df["mu_short"].to_numpy(dtype=np.float64) - df["mu_short_norm"].to_numpy(dtype=np.float64))))
        if not (d1 <= 1e-6 and d2 <= 1e-6):
            raise RuntimeError(
                f"contrato raw_mu_for_rl violado: mae_long={d1:.6g} mae_short={d2:.6g}"
            )


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
    unc = float(env.uncertainty_norm[t]) if hasattr(env, "uncertainty_norm") else 0.0
    trend = float(env.trend_strength_norm[t]) if hasattr(env, "trend_strength_norm") else 0.0
    tit = int(getattr(env, "time_in_trade", 0))
    side = int(env.position_side)
    edge_thr = float(env.cfg.edge_entry_threshold)
    strength_thr = float(env.cfg.strength_entry_threshold)
    hold_soft = int(max(1, int(getattr(env.cfg.reward, "hold_soft_bars", 180))))
    stale_trade = bool(tit >= hold_soft)
    edge_soft = float(max(0.05, 0.65 * edge_thr))
    edge_hard = float(max(0.08, edge_thr))
    strength_soft = float(max(0.20, 0.80 * strength_thr))
    if side > 0:
        adverse = bool(
            (edge <= -edge_soft)
            or (mu_l < mu_s and edge < edge_soft)
            or (strength < strength_soft)
            or (trend < -0.35)
            or (unc > 2.5)
        )
        stale = bool(stale_trade and (edge < edge_hard or strength < strength_thr))
        if adverse or stale:
            return ACTION_CLOSE_LONG
        if edge > edge_soft and mu_l >= mu_s:
            return ACTION_OPEN_LONG_SMALL
        return ACTION_HOLD
    if side < 0:
        adverse = bool(
            (edge >= edge_soft)
            or (mu_s < mu_l and edge > -edge_soft)
            or (strength < strength_soft)
            or (trend > 0.35)
            or (unc > 2.5)
        )
        stale = bool(stale_trade and (-edge < edge_hard or strength < strength_thr))
        if adverse or stale:
            return ACTION_CLOSE_SHORT
        if edge < -edge_soft and mu_s >= mu_l:
            return ACTION_OPEN_SHORT_SMALL
        return ACTION_HOLD
    if strength < strength_thr:
        return ACTION_HOLD
    if edge > edge_hard and mu_l >= mu_s:
        return ACTION_OPEN_LONG_SMALL
    if edge < -edge_hard and mu_s >= mu_l:
        return ACTION_OPEN_SHORT_SMALL
    return ACTION_HOLD


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
        max_hold_bars=int(cfg.max_hold_bars),
        use_signal_gate=True,
        edge_entry_threshold=float(cfg.edge_entry_threshold),
        strength_entry_threshold=float(cfg.strength_entry_threshold),
        force_close_on_adverse=bool(cfg.force_close_on_adverse),
        edge_exit_threshold=float(cfg.edge_exit_threshold),
        strength_exit_threshold=float(cfg.strength_exit_threshold),
        allow_direct_flip=bool(cfg.allow_direct_flip),
        allow_scale_in=bool(cfg.allow_scale_in),
        state_use_uncertainty=bool(cfg.state_use_uncertainty),
        state_use_vol_long=bool(cfg.state_use_vol_long),
        state_use_shock=bool(cfg.state_use_shock),
        state_use_window_context=bool(cfg.state_use_window_context),
        state_window_short_bars=int(cfg.state_window_short_bars),
        state_window_long_bars=int(cfg.state_window_long_bars),
        reward=RewardConfig(
            dd_penalty=float(cfg.dd_penalty),
            dd_level_penalty=float(cfg.dd_level_penalty),
            dd_soft_limit=float(cfg.dd_soft_limit),
            dd_excess_penalty=float(cfg.dd_excess_penalty),
            dd_hard_limit=float(cfg.dd_hard_limit),
            dd_hard_penalty=float(cfg.dd_hard_penalty),
            hold_bar_penalty=float(cfg.hold_bar_penalty),
            hold_soft_bars=int(cfg.hold_soft_bars),
            hold_excess_penalty=float(cfg.hold_excess_penalty),
            hold_regret_penalty=float(cfg.hold_regret_penalty),
            stagnation_bars=int(cfg.stagnation_bars),
            stagnation_ret_epsilon=float(cfg.stagnation_ret_epsilon),
            stagnation_penalty=float(cfg.stagnation_penalty),
            reverse_penalty=float(cfg.reverse_penalty),
            entry_penalty=float(cfg.entry_penalty),
            weak_entry_penalty=float(cfg.weak_entry_penalty),
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


def _choose_action_masked(
    q: QNet,
    state: np.ndarray,
    valid_actions: list[int],
    device: torch.device,
    *,
    state_buf: torch.Tensor | None = None,
    idx_cache: dict[tuple[int, ...], torch.Tensor] | None = None,
) -> int:
    if not valid_actions:
        valid_actions = list(ALL_ACTIONS)
    s_np = state.astype(np.float32, copy=False)
    if state_buf is None or int(state_buf.shape[1]) != int(s_np.shape[0]):
        state_buf = torch.empty((1, int(s_np.shape[0])), dtype=torch.float32, device=device)
    with torch.inference_mode():
        state_buf[0].copy_(torch.from_numpy(s_np), non_blocking=True)
        qs = q(state_buf)
        key = tuple(int(a) for a in valid_actions)
        if idx_cache is not None:
            idx = idx_cache.get(key)
            if idx is None:
                idx = torch.as_tensor(key, dtype=torch.long, device=device)
                idx_cache[key] = idx
        else:
            idx = torch.as_tensor(key, dtype=torch.long, device=device)
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
    state_buf: torch.Tensor | None = None,
    idx_cache: dict[tuple[int, ...], torch.Tensor] | None = None,
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
            a = _choose_action_masked(q, s, valid_actions, device, state_buf=state_buf, idx_cache=idx_cache)
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
    ret = np.asarray([r.get("ret_total", 0.0) for r in rows], dtype=np.float64)
    dd = np.asarray([r.get("max_dd", 0.0) for r in rows], dtype=np.float64)
    pf = np.asarray([r.get("profit_factor", 0.0) for r in rows], dtype=np.float64)
    wr = np.asarray([r.get("win_rate", 0.0) for r in rows], dtype=np.float64)
    tr = np.asarray([r.get("trades", 0.0) for r in rows], dtype=np.float64)
    trd = np.asarray([r.get("trades_per_day", 0.0) for r in rows], dtype=np.float64)
    trw = np.asarray([r.get("trades_per_week", 0.0) for r in rows], dtype=np.float64)
    ahb = np.asarray([r.get("avg_hold_bars", 0.0) for r in rows], dtype=np.float64)
    ahh = np.asarray([r.get("avg_hold_hours", 0.0) for r in rows], dtype=np.float64)
    ahlh = np.asarray([r.get("avg_hold_long_hours", 0.0) for r in rows], dtype=np.float64)
    ahsh = np.asarray([r.get("avg_hold_short_hours", 0.0) for r in rows], dtype=np.float64)
    trl = np.asarray([r.get("trades_long", 0.0) for r in rows], dtype=np.float64)
    trs = np.asarray([r.get("trades_short", 0.0) for r in rows], dtype=np.float64)
    apr = np.asarray([r.get("avg_peak_ret", 0.0) for r in rows], dtype=np.float64)
    agb = np.asarray([r.get("avg_giveback", 0.0) for r in rows], dtype=np.float64)
    tef = np.asarray([r.get("trade_efficiency_mean", 0.0) for r in rows], dtype=np.float64)
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
        "avg_hold_hours_mean": float(np.nanmean(ahh)),
        "avg_hold_long_hours_mean": float(np.nanmean(ahlh)),
        "avg_hold_short_hours_mean": float(np.nanmean(ahsh)),
        "trades_long_total": float(np.nansum(trl)),
        "trades_short_total": float(np.nansum(trs)),
        "avg_peak_ret_mean": float(np.nanmean(apr)),
        "avg_giveback_mean": float(np.nanmean(agb)),
        "trade_efficiency_mean": float(np.nanmean(tef)),
        "avg_turnover_mean": float(np.nanmean(to)),
    }


def _selection_score(
    valid_ret: float,
    valid_dd: float,
    valid_pf: float,
    valid_trades_per_week: float,
    valid_hold_hours: float,
    valid_trade_efficiency: float,
    valid_trade_giveback: float,
    cfg: TrainRLConfig,
) -> float:
    dd = float(max(1e-6, valid_dd))
    calmar = float(valid_ret) / dd
    trw_excess = float(max(0.0, float(valid_trades_per_week) - float(cfg.target_trades_per_week)))
    trw_under = float(max(0.0, float(cfg.target_trades_per_week) - float(valid_trades_per_week)))
    trade_rate_pen = float(cfg.trade_rate_penalty) * float(trw_excess * trw_excess)
    trade_rate_under_pen = float(cfg.trade_rate_under_penalty) * float(trw_under * trw_under)
    trw_floor_def = float(max(0.0, float(cfg.min_trades_per_week_floor) - float(valid_trades_per_week)))
    trw_floor_pen = float(cfg.score_min_trades_floor_penalty) * float(trw_floor_def * trw_floor_def)
    hold_h_under = float(max(0.0, float(cfg.score_hold_hours_min) - float(valid_hold_hours)))
    hold_h_under_pen = float(cfg.score_hold_hours_under_penalty) * float(hold_h_under * hold_h_under)
    hold_h_excess = float(max(0.0, float(valid_hold_hours) - float(cfg.score_hold_hours_target)))
    hold_h_excess = float(min(hold_h_excess, float(max(1.0, cfg.score_hold_hours_excess_cap))))
    # penalidade suave: evita score dominado por cenarios extremos de hold.
    hold_h_pen = float(cfg.score_hold_hours_penalty) * float(np.log1p(hold_h_excess) ** 2.0)
    quality_bonus = float(cfg.score_efficiency_weight) * float(valid_trade_efficiency)
    giveback_pen = float(cfg.score_giveback_penalty) * float(max(0.0, valid_trade_giveback))
    dd_excess = float(max(0.0, float(valid_dd) - float(cfg.score_dd_soft_limit)))
    dd_excess_pen = float(cfg.score_dd_excess_penalty) * float(dd_excess * dd_excess)
    pf_centered = float(np.clip(float(valid_pf) - 1.0, -1.0, 3.0))
    pf_bonus = float(cfg.score_pf_weight) * pf_centered
    return float(
        float(cfg.score_ret_weight) * float(valid_ret)
        + float(cfg.score_calmar_weight) * calmar
        + pf_bonus
        + quality_bonus
        - trade_rate_pen
        - trade_rate_under_pen
        - trw_floor_pen
        - hold_h_under_pen
        - hold_h_pen
        - giveback_pen
        - dd_excess_pen
    )


def _evaluate_policy(
    q: QNet,
    symbol_frames: dict[str, pd.DataFrame],
    env_cfg: TradingEnvConfig,
    *,
    device: torch.device,
    progress_prefix: str = "",
) -> dict[str, float]:
    q.eval()
    state_buf: torch.Tensor | None = None
    idx_cache: dict[tuple[int, ...], torch.Tensor] = {}
    rows: list[dict[str, float]] = []
    prog: ProgressPrinter | None = None
    total_syms = int(max(1, len(symbol_frames)))
    if progress_prefix:
        prog = ProgressPrinter(prefix=progress_prefix, total=total_syms, print_every_s=2.0)
        prog.update(0, suffix=f"sym=0/{total_syms}")
    done_syms = 0
    for _sym, sdf in symbol_frames.items():
        env = HybridTradingEnv(sdf, env_cfg)
        if state_buf is None or int(state_buf.shape[1]) != int(env.state_dim):
            state_buf = torch.empty((1, int(env.state_dim)), dtype=torch.float32, device=device)
        s = env.reset()
        done = False
        while not done:
            a = _choose_action_masked(q, s, env.valid_actions(), device, state_buf=state_buf, idx_cache=idx_cache)
            s, _r, done, _info = env.step(a)
        rows.append(env.summary())
        done_syms += 1
        if prog is not None:
            prog.update(done_syms, suffix=f"sym={done_syms}/{total_syms}")
    if prog is not None:
        prog.update(total_syms, suffix=f"sym={total_syms}/{total_syms}")
        prog.close()
    return _aggregate_metrics(rows)


def _select_eval_frames(
    valid_frames: dict[str, pd.DataFrame],
    cfg: TrainRLConfig,
    *,
    epoch: int,
) -> tuple[dict[str, pd.DataFrame], bool]:
    if not valid_frames:
        return {}, True
    full_every = int(max(1, cfg.eval_full_every_epochs))
    use_full = bool(((epoch + 1) % full_every == 0) or ((epoch + 1) == int(cfg.epochs)))
    if use_full:
        return valid_frames, True

    names = sorted(valid_frames.keys())
    n_take = int(max(0, cfg.eval_symbols_per_epoch))
    if n_take <= 0 or n_take >= len(names):
        picked = list(names)
    else:
        rng = np.random.default_rng(int(cfg.seed) + (int(epoch) + 1) * 7919)
        picked = sorted(list(rng.choice(np.asarray(names, dtype=object), size=int(n_take), replace=False)))

    tail_bars = int(max(0, cfg.eval_tail_bars))
    out: dict[str, pd.DataFrame] = {}
    for sym in picked:
        sdf = valid_frames[str(sym)]
        if tail_bars > 0 and len(sdf) > tail_bars:
            out[str(sym)] = sdf.iloc[-tail_bars:].copy()
        else:
            out[str(sym)] = sdf
    return out, False


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
    eval_device = _pick_device(cfg.eval_device)
    q = QNet(state_dim, n_actions, int(cfg.hidden_dim)).to(device)
    target = QNet(state_dim, n_actions, int(cfg.hidden_dim)).to(device)
    target.load_state_dict(q.state_dict())
    target.eval()
    q_eval: QNet | None = None
    if str(eval_device) != str(device):
        q_eval = QNet(state_dim, n_actions, int(cfg.hidden_dim)).to(eval_device)
        q_eval.eval()
    opt = torch.optim.Adam(q.parameters(), lr=float(cfg.lr))
    buffer = ReplayBuffer(int(cfg.replay_size))
    train_syms = list(train_frames.keys())
    step_count = 0
    fold_dir = out_dir / f"fold_{int(fold.fold_id):03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    est_collect_steps = int(cfg.epochs) * int(cfg.episodes_per_epoch) * int(max(1, cfg.episode_bars))
    est_opt_steps = int(cfg.epochs) * int(cfg.train_steps_per_epoch)
    print(
        f"[rl][fold {fold.fold_id}] budget collect_steps<=~{est_collect_steps:,} "
        f"opt_steps={est_opt_steps:,} batch={int(cfg.batch_size)} hidden={int(cfg.hidden_dim)} device={device}",
        flush=True,
    )
    state_signal_cols = _active_state_signal_cols(cfg)
    print(
        f"[rl][fold {fold.fold_id}] state_signals={state_signal_cols} "
        f"state_internal={STATE_INTERNAL_COLS} "
        f"state_dim={int(state_dim)} "
        f"ctx={int(cfg.state_window_short_bars)}/{int(cfg.state_window_long_bars)} "
        f"use_ctx={int(bool(cfg.state_use_window_context))}",
        flush=True,
    )

    ckpt_meta = {
        "state_dim": int(state_dim),
        "n_actions": int(n_actions),
        "hidden_dim": int(cfg.hidden_dim),
        "state_signal_cols": list(state_signal_cols),
        "state_internal_cols": list(STATE_INTERNAL_COLS),
        "env_cfg": {
            "fee_rate": float(cfg.fee_rate),
            "slippage_rate": float(cfg.slippage_rate),
            "small_size": float(cfg.small_size),
            "big_size": float(cfg.big_size),
            "cooldown_bars": int(cfg.cooldown_bars),
            "regret_window": int(cfg.regret_window),
            "min_reentry_gap_bars": int(cfg.min_reentry_gap_bars),
            "min_hold_bars": int(cfg.min_hold_bars),
            "max_hold_bars": int(cfg.max_hold_bars),
            "use_signal_gate": True,
            "edge_entry_threshold": float(cfg.edge_entry_threshold),
            "strength_entry_threshold": float(cfg.strength_entry_threshold),
            "force_close_on_adverse": bool(cfg.force_close_on_adverse),
            "edge_exit_threshold": float(cfg.edge_exit_threshold),
            "strength_exit_threshold": float(cfg.strength_exit_threshold),
            "allow_direct_flip": bool(cfg.allow_direct_flip),
            "allow_scale_in": bool(cfg.allow_scale_in),
            "state_use_uncertainty": bool(cfg.state_use_uncertainty),
            "state_use_vol_long": bool(cfg.state_use_vol_long),
            "state_use_shock": bool(cfg.state_use_shock),
            "state_use_window_context": bool(cfg.state_use_window_context),
            "state_window_short_bars": int(cfg.state_window_short_bars),
            "state_window_long_bars": int(cfg.state_window_long_bars),
            "strict_best_update": bool(cfg.strict_best_update),
            "early_stop_patience": int(cfg.early_stop_patience),
            "early_stop_min_delta": float(cfg.early_stop_min_delta),
            "reward": {
                "dd_penalty": float(cfg.dd_penalty),
                "dd_level_penalty": float(cfg.dd_level_penalty),
                "dd_soft_limit": float(cfg.dd_soft_limit),
                "dd_excess_penalty": float(cfg.dd_excess_penalty),
                "dd_hard_limit": float(cfg.dd_hard_limit),
                "dd_hard_penalty": float(cfg.dd_hard_penalty),
                "hold_bar_penalty": float(cfg.hold_bar_penalty),
                "hold_soft_bars": int(cfg.hold_soft_bars),
                "hold_excess_penalty": float(cfg.hold_excess_penalty),
                "hold_regret_penalty": float(cfg.hold_regret_penalty),
                "stagnation_bars": int(cfg.stagnation_bars),
                "stagnation_ret_epsilon": float(cfg.stagnation_ret_epsilon),
                "stagnation_penalty": float(cfg.stagnation_penalty),
                "reverse_penalty": float(cfg.reverse_penalty),
                "entry_penalty": float(cfg.entry_penalty),
                "weak_entry_penalty": float(cfg.weak_entry_penalty),
                "turnover_penalty": float(cfg.turnover_penalty),
                "regret_penalty": float(cfg.regret_penalty),
                "idle_penalty": float(cfg.idle_penalty),
            },
        },
    }

    train_log: list[dict[str, float]] = []
    state_buf_train = torch.empty((1, int(state_dim)), dtype=torch.float32, device=device)
    idx_cache_train: dict[tuple[int, ...], torch.Tensor] = {}
    best_score = -np.inf
    best_epoch = -1
    best_state: dict | None = None
    no_improve_epochs = 0
    last_valid_metrics = {
        "ret_total_mean": 0.0,
        "max_dd_mean": 0.0,
        "profit_factor_median": 0.0,
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
    }
    t_fold_start = time.perf_counter()
    for epoch in range(int(cfg.epochs)):
        t_ep_start = time.perf_counter()
        eps = _epsilon(epoch, cfg)
        exp_p = _expert_prob(epoch, cfg)
        rows_epoch: list[dict[str, float]] = []
        q.train()
        t_collect_start = time.perf_counter()
        collect_total = int(max(1, cfg.episodes_per_epoch))
        collect_prog = ProgressPrinter(
            prefix=f"[rl][fold {fold.fold_id}][ep {epoch+1}/{cfg.epochs}] collect",
            total=collect_total,
            print_every_s=2.0,
        )
        collect_prog.update(0, suffix=f"eps={eps:.3f} exp={exp_p:.3f}")
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
                    state_buf=state_buf_train,
                    idx_cache=idx_cache_train,
                )
            )
            collect_prog.update(int(_ep + 1), suffix=f"buf={len(buffer)}")
        collect_prog.update(collect_total, suffix=f"buf={len(buffer)}")
        collect_prog.close()
        dt_collect = time.perf_counter() - t_collect_start

        losses: list[float] = []
        t_opt_start = time.perf_counter()
        if len(buffer) >= int(cfg.batch_size):
            n_steps = int(cfg.train_steps_per_epoch)
            opt_prog = ProgressPrinter(
                prefix=f"[rl][fold {fold.fold_id}][ep {epoch+1}/{cfg.epochs}] optimize",
                total=max(1, n_steps),
                print_every_s=2.0,
            )
            opt_prog.update(0, suffix=f"batch={int(cfg.batch_size)}")
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
                if (_ + 1) % 16 == 0 or (_ + 1) == n_steps:
                    lm = float(np.mean(losses[-32:])) if losses else 0.0
                    opt_prog.update(int(_ + 1), suffix=f"loss~{lm:.6f}")
            opt_prog.update(max(1, n_steps), suffix=f"loss={float(np.mean(losses)) if losses else 0.0:.6f}")
            opt_prog.close()
        dt_opt = time.perf_counter() - t_opt_start

        train_metrics = _aggregate_metrics(rows_epoch)
        eval_every = int(max(1, cfg.eval_every_epochs))
        need_eval = bool((epoch == 0) or ((epoch + 1) % eval_every == 0) or ((epoch + 1) == int(cfg.epochs)))
        eval_is_full = False
        eval_scope_symbols = float(len(valid_frames))
        if need_eval:
            eval_frames, eval_is_full = _select_eval_frames(valid_frames, cfg, epoch=epoch)
            eval_scope_symbols = float(len(eval_frames))
            t_eval_start = time.perf_counter()
            model_eval = q
            dev_eval = device
            if q_eval is not None:
                q_eval.load_state_dict(q.state_dict())
                q_eval.eval()
                model_eval = q_eval
                dev_eval = eval_device
            valid_metrics = _evaluate_policy(
                model_eval,
                eval_frames,
                env_cfg,
                device=dev_eval,
                progress_prefix=(
                    f"[rl][fold {fold.fold_id}][ep {epoch+1}/{cfg.epochs}] eval"
                    f"{'-full' if eval_is_full else '-slice'}"
                ),
            )
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
            "valid_trades_long_total": float(valid_metrics["trades_long_total"]),
            "valid_trades_short_total": float(valid_metrics["trades_short_total"]),
            "valid_trades_per_day_mean": float(valid_metrics["trades_per_day_mean"]),
            "valid_trades_per_week_mean": float(valid_metrics["trades_per_week_mean"]),
            "valid_avg_hold_bars_mean": float(valid_metrics["avg_hold_bars_mean"]),
            "valid_avg_hold_hours_mean": float(valid_metrics["avg_hold_hours_mean"]),
            "valid_avg_hold_long_hours_mean": float(valid_metrics["avg_hold_long_hours_mean"]),
            "valid_avg_hold_short_hours_mean": float(valid_metrics["avg_hold_short_hours_mean"]),
            "valid_avg_peak_ret_mean": float(valid_metrics["avg_peak_ret_mean"]),
            "valid_avg_giveback_mean": float(valid_metrics["avg_giveback_mean"]),
            "valid_trade_efficiency_mean": float(valid_metrics["trade_efficiency_mean"]),
            "valid_eval_fresh": float(1.0 if need_eval else 0.0),
            "valid_eval_full": float(1.0 if eval_is_full else 0.0),
            "valid_eval_symbols": float(eval_scope_symbols),
            "time_collect_s": float(dt_collect),
            "time_optimize_s": float(dt_opt),
            "time_eval_s": float(dt_eval),
        }
        train_log.append(rec)
        trw_excess = float(max(0.0, float(rec["valid_trades_per_week_mean"]) - float(cfg.target_trades_per_week)))
        trw_under = float(max(0.0, float(cfg.target_trades_per_week) - float(rec["valid_trades_per_week_mean"])))
        trw_floor_def = float(max(0.0, float(cfg.min_trades_per_week_floor) - float(rec["valid_trades_per_week_mean"])))
        score_trade_excess_pen = float(cfg.trade_rate_penalty) * float(trw_excess * trw_excess)
        score_trade_under_pen = float(cfg.trade_rate_under_penalty) * float(trw_under * trw_under)
        score_trade_floor_pen = float(cfg.score_min_trades_floor_penalty) * float(trw_floor_def * trw_floor_def)
        dd_excess = float(max(0.0, float(rec["valid_max_dd_mean"]) - float(cfg.score_dd_soft_limit)))
        score_dd_excess_pen = float(cfg.score_dd_excess_penalty) * float(dd_excess * dd_excess)
        hold_h_under = float(max(0.0, float(cfg.score_hold_hours_min) - float(rec["valid_avg_hold_hours_mean"])))
        score_hold_h_under_pen = float(cfg.score_hold_hours_under_penalty) * float(hold_h_under * hold_h_under)
        hold_h_excess = float(max(0.0, float(rec["valid_avg_hold_hours_mean"]) - float(cfg.score_hold_hours_target)))
        hold_h_excess = float(min(hold_h_excess, float(max(1.0, cfg.score_hold_hours_excess_cap))))
        score_hold_h_pen = float(cfg.score_hold_hours_penalty) * float(np.log1p(hold_h_excess) ** 2.0)
        score_eff_bonus = float(cfg.score_efficiency_weight) * float(rec["valid_trade_efficiency_mean"])
        score_giveback_pen = float(cfg.score_giveback_penalty) * float(max(0.0, rec["valid_avg_giveback_mean"]))
        score = _selection_score(
            rec["valid_ret_total_mean"],
            rec["valid_max_dd_mean"],
            rec["valid_pf_median"],
            rec["valid_trades_per_week_mean"],
            rec["valid_avg_hold_hours_mean"],
            rec["valid_trade_efficiency_mean"],
            rec["valid_avg_giveback_mean"],
            cfg,
        )
        calmar = float(rec["valid_ret_total_mean"]) / float(max(1e-6, rec["valid_max_dd_mean"]))
        dd_excess = float(max(0.0, rec["valid_max_dd_mean"] - float(cfg.score_dd_soft_limit)))
        improved = bool(need_eval and (score > (best_score + float(cfg.early_stop_min_delta))))
        if improved:
            best_score = score
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
            torch.save({"state_dict": best_state, **ckpt_meta}, fold_dir / "policy_dqn.pt")
            no_improve_epochs = 0
        elif need_eval:
            no_improve_epochs += 1
            if bool(cfg.strict_best_update) and best_state is not None:
                q.load_state_dict(best_state)
                target.load_state_dict(best_state)

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
                "strict_best_update": float(1.0 if cfg.strict_best_update else 0.0),
                "early_stop_patience": float(cfg.early_stop_patience),
                "early_stop_min_delta": float(cfg.early_stop_min_delta),
                "no_improve_epochs": float(no_improve_epochs),
                "target_trades_per_week": float(cfg.target_trades_per_week),
                "min_trades_per_week_floor": float(cfg.min_trades_per_week_floor),
                "trade_rate_penalty": float(cfg.trade_rate_penalty),
                "trade_rate_under_penalty": float(cfg.trade_rate_under_penalty),
                "score_min_trades_floor_penalty": float(cfg.score_min_trades_floor_penalty),
                "score_efficiency_weight": float(cfg.score_efficiency_weight),
                "score_pf_weight": float(cfg.score_pf_weight),
                "score_giveback_penalty": float(cfg.score_giveback_penalty),
                "score_hold_hours_target": float(cfg.score_hold_hours_target),
                "score_hold_hours_penalty": float(cfg.score_hold_hours_penalty),
                "score_hold_hours_min": float(cfg.score_hold_hours_min),
                "score_hold_hours_under_penalty": float(cfg.score_hold_hours_under_penalty),
                "last_expert_prob": float(rec["expert_prob"]),
                "best_valid_ret_total_mean_so_far": float(
                    max((x["valid_ret_total_mean"] for x in train_log if float(x.get("valid_eval_fresh", 0.0)) > 0.5), default=0.0)
                ),
                "best_valid_max_dd_mean_so_far": float(
                    min((x["valid_max_dd_mean"] for x in train_log if float(x.get("valid_eval_fresh", 0.0)) > 0.5), default=0.0)
                ),
                "last_valid_ret_total_mean": float(rec["valid_ret_total_mean"]),
                "last_valid_max_dd_mean": float(rec["valid_max_dd_mean"]),
                "last_valid_calmar": float(calmar),
                "last_valid_dd_excess": float(dd_excess),
                "last_valid_pf_median": float(rec["valid_pf_median"]),
                "last_valid_trades_total": float(rec["valid_trades_total"]),
                "last_valid_trades_long_total": float(rec["valid_trades_long_total"]),
                "last_valid_trades_short_total": float(rec["valid_trades_short_total"]),
                "last_valid_trades_per_week_mean": float(rec["valid_trades_per_week_mean"]),
                "last_valid_avg_hold_bars_mean": float(rec["valid_avg_hold_bars_mean"]),
                "last_valid_avg_hold_hours_mean": float(rec["valid_avg_hold_hours_mean"]),
                "last_valid_avg_hold_long_hours_mean": float(rec["valid_avg_hold_long_hours_mean"]),
                "last_valid_avg_hold_short_hours_mean": float(rec["valid_avg_hold_short_hours_mean"]),
                "last_valid_avg_peak_ret_mean": float(rec["valid_avg_peak_ret_mean"]),
                "last_valid_avg_giveback_mean": float(rec["valid_avg_giveback_mean"]),
                "last_valid_trade_efficiency_mean": float(rec["valid_trade_efficiency_mean"]),
                "score_trade_excess_pen": float(score_trade_excess_pen),
                "score_trade_under_pen": float(score_trade_under_pen),
                "score_trade_floor_pen": float(score_trade_floor_pen),
                "score_dd_excess_pen": float(score_dd_excess_pen),
                "score_hold_hours_under_pen": float(score_hold_h_under_pen),
                "score_hold_hours_pen": float(score_hold_h_pen),
                "score_eff_bonus": float(score_eff_bonus),
                "score_giveback_pen": float(score_giveback_pen),
                "last_valid_eval_fresh": float(rec["valid_eval_fresh"]),
                "last_valid_eval_full": float(rec["valid_eval_full"]),
                "last_valid_eval_symbols": float(rec["valid_eval_symbols"]),
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
            f"calmar={calmar:+.3f} dd_excess={dd_excess:.2%} "
            f"valid_pf={rec['valid_pf_median']:.3f} trades/wk={rec['valid_trades_per_week_mean']:.3f} "
            f"hold_h={rec['valid_avg_hold_hours_mean']:.2f} (L={rec['valid_avg_hold_long_hours_mean']:.2f} S={rec['valid_avg_hold_short_hours_mean']:.2f}) "
            f"eff={rec['valid_trade_efficiency_mean']:+.3f} giveback={rec['valid_avg_giveback_mean']:.4f} "
            f"score={score:+.4f} (tr_ex={score_trade_excess_pen:.3f} tr_un={score_trade_under_pen:.3f} tr_fl={score_trade_floor_pen:.3f} "
            f"dd={score_dd_excess_pen:.3f} hold_u={score_hold_h_under_pen:.3f} hold_o={score_hold_h_pen:.3f} "
            f"eff={score_eff_bonus:+.3f} gb={score_giveback_pen:.3f}) "
            f"eval_fresh={int(need_eval)} full={int(eval_is_full)} eval_syms={int(eval_scope_symbols)} "
            f"time(ep/collect/opt/eval/save)={dt_ep:.1f}/{dt_collect:.1f}/{dt_opt:.1f}/{dt_eval:.1f}/{dt_save:.1f}s",
            flush=True,
        )
        if need_eval and int(cfg.early_stop_patience) > 0 and no_improve_epochs >= int(cfg.early_stop_patience):
            print(
                f"[rl][fold {fold.fold_id}] early_stop epoch={epoch+1} "
                f"no_improve={no_improve_epochs} patience={int(cfg.early_stop_patience)}",
                flush=True,
            )
            break

    torch.save({"state_dict": q.state_dict(), **ckpt_meta}, fold_dir / "policy_last.pt")
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in q.state_dict().items()}
    torch.save({"state_dict": best_state, **ckpt_meta}, fold_dir / "policy_dqn.pt")
    _atomic_write_json(fold_dir / "train_log.json", train_log)

    def _score_row(x: dict[str, float]) -> float:
        return _selection_score(
            float(x.get("valid_ret_total_mean", 0.0)),
            float(x.get("valid_max_dd_mean", 0.0)),
            float(x.get("valid_pf_median", 0.0)),
            float(x.get("valid_trades_per_week_mean", 0.0)),
            float(x.get("valid_avg_hold_hours_mean", 0.0)),
            float(x.get("valid_trade_efficiency_mean", 0.0)),
            float(x.get("valid_avg_giveback_mean", 0.0)),
            cfg,
        )

    eval_rows_full = [x for x in train_log if float(x.get("valid_eval_full", 0.0)) > 0.5]
    eval_rows = [x for x in train_log if float(x.get("valid_eval_fresh", 1.0)) > 0.5]
    best_pool = eval_rows_full if eval_rows_full else (eval_rows if eval_rows else train_log)
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
        "best_valid_calmar": float(best["valid_ret_total_mean"]) / float(max(1e-6, best["valid_max_dd_mean"])),
        "best_valid_pf_median": float(best["valid_pf_median"]),
        "best_valid_trades_total": float(best["valid_trades_total"]),
        "best_valid_trades_long_total": float(best.get("valid_trades_long_total", 0.0)),
        "best_valid_trades_short_total": float(best.get("valid_trades_short_total", 0.0)),
        "best_valid_trades_per_week_mean": float(best.get("valid_trades_per_week_mean", 0.0)),
        "best_valid_avg_hold_bars_mean": float(best.get("valid_avg_hold_bars_mean", 0.0)),
        "best_valid_avg_hold_hours_mean": float(best.get("valid_avg_hold_hours_mean", 0.0)),
        "best_valid_avg_hold_long_hours_mean": float(best.get("valid_avg_hold_long_hours_mean", 0.0)),
        "best_valid_avg_hold_short_hours_mean": float(best.get("valid_avg_hold_short_hours_mean", 0.0)),
        "best_valid_avg_peak_ret_mean": float(best.get("valid_avg_peak_ret_mean", 0.0)),
        "best_valid_avg_giveback_mean": float(best.get("valid_avg_giveback_mean", 0.0)),
        "best_valid_trade_efficiency_mean": float(best.get("valid_trade_efficiency_mean", 0.0)),
        "target_trades_per_week": float(cfg.target_trades_per_week),
        "min_trades_per_week_floor": float(cfg.min_trades_per_week_floor),
        "trade_rate_penalty": float(cfg.trade_rate_penalty),
        "trade_rate_under_penalty": float(cfg.trade_rate_under_penalty),
        "score_min_trades_floor_penalty": float(cfg.score_min_trades_floor_penalty),
        "score_efficiency_weight": float(cfg.score_efficiency_weight),
        "score_pf_weight": float(cfg.score_pf_weight),
        "score_giveback_penalty": float(cfg.score_giveback_penalty),
        "score_hold_hours_target": float(cfg.score_hold_hours_target),
        "score_hold_hours_penalty": float(cfg.score_hold_hours_penalty),
        "score_hold_hours_min": float(cfg.score_hold_hours_min),
        "score_hold_hours_under_penalty": float(cfg.score_hold_hours_under_penalty),
        "score_hold_hours_excess_cap": float(cfg.score_hold_hours_excess_cap),
        "strict_best_update": float(1.0 if cfg.strict_best_update else 0.0),
        "early_stop_patience": float(cfg.early_stop_patience),
        "early_stop_min_delta": float(cfg.early_stop_min_delta),
        "eval_symbols_per_epoch": float(cfg.eval_symbols_per_epoch),
        "eval_tail_bars": float(cfg.eval_tail_bars),
        "eval_full_every_epochs": float(cfg.eval_full_every_epochs),
        "state_signal_cols": list(state_signal_cols),
        "state_internal_cols": list(STATE_INTERNAL_COLS),
        "state_use_uncertainty": float(1.0 if cfg.state_use_uncertainty else 0.0),
        "state_use_vol_long": float(1.0 if cfg.state_use_vol_long else 0.0),
        "state_use_shock": float(1.0 if cfg.state_use_shock else 0.0),
        "state_use_window_context": float(1.0 if cfg.state_use_window_context else 0.0),
        "state_window_short_bars": float(cfg.state_window_short_bars),
        "state_window_long_bars": float(cfg.state_window_long_bars),
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
    cpath = str(cfg.signals_contract_path or "").strip()
    if not cpath:
        cpath = str(Path(cfg.signals_path).with_suffix(Path(cfg.signals_path).suffix + ".contract.json"))
    contract = _load_signals_contract(cpath)
    if contract:
        _validate_signals_contract(df, contract)
        print(f"[rl] signals_contract=ok path={Path(cpath).expanduser().resolve()}", flush=True)
    else:
        print(f"[rl] signals_contract=missing path={Path(cpath).expanduser().resolve()} (seguindo sem contrato)", flush=True)
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
    if contract:
        _atomic_write_json(out_dir / "signals_contract.json", contract)

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
    ap.add_argument("--signals-contract", default="")
    ap.add_argument("--out-dir", default=str(storage_root / "rl_runs" / "default"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-days", type=int, default=720)
    ap.add_argument("--valid-days", type=int, default=180)
    ap.add_argument("--step-days", type=int, default=365)
    ap.add_argument("--embargo-minutes", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--min-rows-symbol", type=int, default=800)
    ap.add_argument("--single-split-mode", action="store_true", default=True)
    ap.add_argument("--wf-mode", dest="single_split_mode", action="store_false")
    ap.add_argument("--holdout-days", type=int, default=180)

    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--episodes-per-epoch", type=int, default=20)
    ap.add_argument("--episode-bars", type=int, default=5760)
    ap.add_argument("--no-random-starts", action="store_true")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--replay-size", type=int, default=200000)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--target-update-steps", type=int, default=1000)
    ap.add_argument("--train-steps-per-epoch", type=int, default=1200)
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eps-start", type=float, default=0.50)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-epochs", type=int, default=8)
    ap.add_argument("--expert-mix-start", type=float, default=0.45)
    ap.add_argument("--expert-mix-end", type=float, default=0.0)
    ap.add_argument("--expert-mix-decay-epochs", type=int, default=14)
    ap.add_argument("--strict-best-update", dest="strict_best_update", action="store_true", default=True)
    ap.add_argument("--no-strict-best-update", dest="strict_best_update", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0005)

    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--turnover-penalty", type=float, default=0.0020)
    ap.add_argument("--dd-penalty", type=float, default=0.35)
    ap.add_argument("--dd-level-penalty", type=float, default=0.08)
    ap.add_argument("--dd-soft-limit", type=float, default=0.15)
    ap.add_argument("--dd-excess-penalty", type=float, default=2.5)
    ap.add_argument("--dd-hard-limit", type=float, default=0.30)
    ap.add_argument("--dd-hard-penalty", type=float, default=0.05)
    ap.add_argument("--hold-bar-penalty", type=float, default=0.0)
    ap.add_argument("--hold-soft-bars", type=int, default=180)
    ap.add_argument("--hold-excess-penalty", type=float, default=0.0)
    ap.add_argument("--hold-regret-penalty", type=float, default=0.09)
    ap.add_argument("--stagnation-bars", type=int, default=180)
    ap.add_argument("--stagnation-ret-epsilon", type=float, default=0.00003)
    ap.add_argument("--stagnation-penalty", type=float, default=0.008)
    ap.add_argument("--reverse-penalty", type=float, default=0.0010)
    ap.add_argument("--entry-penalty", type=float, default=0.0000)
    ap.add_argument("--weak-entry-penalty", type=float, default=0.0000)
    ap.add_argument("--regret-penalty", type=float, default=0.02)
    ap.add_argument("--idle-penalty", type=float, default=0.0)
    ap.add_argument("--cooldown-bars", type=int, default=12)
    ap.add_argument("--min-reentry-gap-bars", type=int, default=180)
    ap.add_argument("--min-hold-bars", type=int, default=90)
    ap.add_argument("--max-hold-bars", type=int, default=720)
    ap.add_argument("--edge-entry-threshold", type=float, default=0.45)
    ap.add_argument("--strength-entry-threshold", type=float, default=0.82)
    ap.add_argument("--force-close-on-adverse", dest="force_close_on_adverse", action="store_true", default=True)
    ap.add_argument("--no-force-close-on-adverse", dest="force_close_on_adverse", action="store_false")
    ap.add_argument("--edge-exit-threshold", type=float, default=0.22)
    ap.add_argument("--strength-exit-threshold", type=float, default=0.18)
    ap.add_argument("--allow-direct-flip", dest="allow_direct_flip", action="store_true", default=True)
    ap.add_argument("--no-allow-direct-flip", dest="allow_direct_flip", action="store_false")
    ap.add_argument("--allow-scale-in", dest="allow_scale_in", action="store_true", default=False)
    ap.add_argument("--no-allow-scale-in", dest="allow_scale_in", action="store_false")
    ap.add_argument("--state-use-uncertainty", dest="state_use_uncertainty", action="store_true", default=False)
    ap.add_argument("--no-state-use-uncertainty", dest="state_use_uncertainty", action="store_false")
    ap.add_argument("--state-use-vol-long", dest="state_use_vol_long", action="store_true", default=False)
    ap.add_argument("--no-state-use-vol-long", dest="state_use_vol_long", action="store_false")
    ap.add_argument("--state-use-shock", dest="state_use_shock", action="store_true", default=False)
    ap.add_argument("--no-state-use-shock", dest="state_use_shock", action="store_false")
    ap.add_argument("--state-use-window-context", dest="state_use_window_context", action="store_true", default=True)
    ap.add_argument("--no-state-use-window-context", dest="state_use_window_context", action="store_false")
    ap.add_argument("--state-window-short-bars", type=int, default=30)
    ap.add_argument("--state-window-long-bars", type=int, default=30)
    ap.add_argument("--target-trades-per-week", type=float, default=4.0)
    ap.add_argument("--min-trades-per-week-floor", type=float, default=0.00)
    ap.add_argument("--trade-rate-penalty", type=float, default=0.00)
    ap.add_argument("--trade-rate-under-penalty", type=float, default=0.00)
    ap.add_argument("--score-min-trades-floor-penalty", type=float, default=0.00)
    ap.add_argument("--score-ret-weight", type=float, default=0.45)
    ap.add_argument("--score-calmar-weight", type=float, default=0.30)
    ap.add_argument("--score-pf-weight", type=float, default=0.35)
    ap.add_argument("--score-efficiency-weight", type=float, default=0.08)
    ap.add_argument("--score-giveback-penalty", type=float, default=0.20)
    ap.add_argument("--score-hold-hours-target", type=float, default=8.0)
    ap.add_argument("--score-hold-hours-penalty", type=float, default=0.002)
    ap.add_argument("--score-hold-hours-min", type=float, default=0.0)
    ap.add_argument("--score-hold-hours-under-penalty", type=float, default=0.00)
    ap.add_argument("--score-hold-hours-excess-cap", type=float, default=24.0)
    ap.add_argument("--score-dd-soft-limit", type=float, default=0.15)
    ap.add_argument("--score-dd-excess-penalty", type=float, default=4.0)
    ap.add_argument("--eval-symbols-per-epoch", type=int, default=4)
    ap.add_argument("--eval-tail-bars", type=int, default=86400)
    ap.add_argument("--eval-full-every-epochs", type=int, default=4)
    ap.add_argument("--regret-window", type=int, default=10)
    ap.add_argument("--small-size", type=float, default=0.10)
    ap.add_argument("--big-size", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-device", default="cpu")

    ns = ap.parse_args(list(argv) if argv is not None else None)
    return TrainRLConfig(
        signals_path=str(ns.signals),
        signals_contract_path=str(ns.signals_contract or ""),
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
        strict_best_update=bool(ns.strict_best_update),
        early_stop_patience=int(ns.early_stop_patience),
        early_stop_min_delta=float(ns.early_stop_min_delta),
        fee_rate=float(ns.fee_rate),
        slippage_rate=float(ns.slippage_rate),
        turnover_penalty=float(ns.turnover_penalty),
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
        stagnation_bars=int(ns.stagnation_bars),
        stagnation_ret_epsilon=float(ns.stagnation_ret_epsilon),
        stagnation_penalty=float(ns.stagnation_penalty),
        reverse_penalty=float(ns.reverse_penalty),
        entry_penalty=float(ns.entry_penalty),
        weak_entry_penalty=float(ns.weak_entry_penalty),
        regret_penalty=float(ns.regret_penalty),
        idle_penalty=float(ns.idle_penalty),
        cooldown_bars=int(ns.cooldown_bars),
        min_reentry_gap_bars=int(ns.min_reentry_gap_bars),
        min_hold_bars=int(ns.min_hold_bars),
        max_hold_bars=int(ns.max_hold_bars),
        regret_window=int(ns.regret_window),
        small_size=float(ns.small_size),
        big_size=float(ns.big_size),
        edge_entry_threshold=float(ns.edge_entry_threshold),
        strength_entry_threshold=float(ns.strength_entry_threshold),
        force_close_on_adverse=bool(ns.force_close_on_adverse),
        edge_exit_threshold=float(ns.edge_exit_threshold),
        strength_exit_threshold=float(ns.strength_exit_threshold),
        allow_direct_flip=bool(ns.allow_direct_flip),
        allow_scale_in=bool(ns.allow_scale_in),
        state_use_uncertainty=bool(ns.state_use_uncertainty),
        state_use_vol_long=bool(ns.state_use_vol_long),
        state_use_shock=bool(ns.state_use_shock),
        state_use_window_context=bool(ns.state_use_window_context),
        state_window_short_bars=int(ns.state_window_short_bars),
        state_window_long_bars=int(ns.state_window_long_bars),
        target_trades_per_week=float(ns.target_trades_per_week),
        min_trades_per_week_floor=float(ns.min_trades_per_week_floor),
        trade_rate_penalty=float(ns.trade_rate_penalty),
        trade_rate_under_penalty=float(ns.trade_rate_under_penalty),
        score_min_trades_floor_penalty=float(ns.score_min_trades_floor_penalty),
        score_ret_weight=float(ns.score_ret_weight),
        score_calmar_weight=float(ns.score_calmar_weight),
        score_pf_weight=float(ns.score_pf_weight),
        score_efficiency_weight=float(ns.score_efficiency_weight),
        score_giveback_penalty=float(ns.score_giveback_penalty),
        score_hold_hours_target=float(ns.score_hold_hours_target),
        score_hold_hours_penalty=float(ns.score_hold_hours_penalty),
        score_hold_hours_min=float(ns.score_hold_hours_min),
        score_hold_hours_under_penalty=float(ns.score_hold_hours_under_penalty),
        score_hold_hours_excess_cap=float(ns.score_hold_hours_excess_cap),
        score_dd_soft_limit=float(ns.score_dd_soft_limit),
        score_dd_excess_penalty=float(ns.score_dd_excess_penalty),
        eval_symbols_per_epoch=int(ns.eval_symbols_per_epoch),
        eval_tail_bars=int(ns.eval_tail_bars),
        eval_full_every_epochs=int(ns.eval_full_every_epochs),
        device=str(ns.device),
        eval_device=str(ns.eval_device),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = train_rl_walkforward(cfg)
    print(f"[rl] treino finalizado: {out}")


if __name__ == "__main__":
    main()
