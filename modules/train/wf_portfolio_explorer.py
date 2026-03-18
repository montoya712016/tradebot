# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Hierarchical portfolio explorer:
1) refresh labels for a label/contract configuration
2) run one or more retrains for the same refreshed labels
3) run one or more portfolio backtests per retrain

Designed to be dashboard-friendly and cheap to iterate before a full GA.
"""

from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            for cand in (p, p.parent):
                sp = str(cand)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from utils.paths import resolve_generated_path, models_root_for_asset  # noqa: E402
from backtest.portfolio import (  # noqa: E402
    PortfolioDemoSettings,
    _default_portfolio_cfg,
    prepare_portfolio_data,
    run_prepared_portfolio,
)
from config.trade_contract import TradeContract, bars_from_minutes, apply_crypto_pipeline_env  # noqa: E402
from prepare_features.refresh_sniper_labels_in_cache import RefreshLabelsSettings, run as refresh_labels  # noqa: E402


RESULTS_HEADER = [
    "label_id",
    "model_id",
    "backtest_id",
    "train_id",
    "stage",
    "status",
    "start_utc",
    "end_utc",
    "duration_sec",
    "seed",
    "train_run_dir",
    "bt_out_dir",
    "equity_html",
    "equity_png",
    "score",
    "eq_end",
    "ret_pct",
    "max_dd",
    "win_rate",
    "profit_factor",
    "trades",
    "top_symbols",
    "error",
    "label_profit_thr",
    "exit_ema_span_min",
    "exit_ema_init_offset_pct",
    "entry_ratio_neg_per_pos",
    "calib_tail_blend",
    "calib_tail_boost",
    "top_metric_qs",
    "top_metric_min_count",
    "tau_entry",
    "corr_enabled",
    "corr_max_with_market",
    "corr_max_pair",
    "corr_open_reduce_start",
    "corr_open_hard_reject",
    "corr_open_min_weight_mult",
    "max_positions",
    "total_exposure",
    "max_trade_exposure",
    "min_trade_exposure",
    "month_mean",
    "month_p50",
    "month_pos_frac",
    "month_worst",
    "month_best",
    "semester_mean",
    "semester_p50",
    "semester_pos_frac",
    "semester_worst",
    "semester_best",
    "clusters",
    "avg_trades_per_cluster",
    "cluster_pos_frac",
    "top1_share_pos",
    "top3_share_pos",
    "ret_wo_best_weighted",
    "ret_wo_top3_weighted",
]


@dataclass
class ExploreSettings:
    out_root: str = "wf_portfolio_explore"
    results_csv: str = "explore_runs.csv"
    seed: int = 42
    max_label_trials: int = 3
    retrains_per_label: int = 3
    backtests_per_retrain: int = 15
    days: int = 4 * 360
    max_symbols: int = 0
    candle_sec: int = 300
    safe_threads: int = 1
    label_profit_choices: tuple[float, ...] = (0.01, 0.025, 0.04, 0.06, 0.10)
    exit_span_choices: tuple[int, ...] = (30, 60, 90, 150, 300, 600)
    exit_offset_choices: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.04)
    neg_pos_choices: tuple[float, ...] = (1.5, 3.0, 5.0, 8.0, 12.0)
    tail_blend_choices: tuple[float, ...] = (0.40, 0.60, 0.80, 0.95)
    tail_boost_choices: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0)
    top_q_presets: tuple[str, ...] = ("0.001,0.0025,0.005", "0.0005,0.001,0.0025", "0.005,0.01,0.02", "0.01,0.02,0.05")
    top_min_count_choices: tuple[int, ...] = (12, 32, 48, 96, 128)
    tau_entry_choices: tuple[float, ...] = (0.45, 0.60, 0.75, 0.85)
    corr_enabled_choices: tuple[bool, ...] = (False, True)
    corr_max_with_market_choices: tuple[float, ...] = (0.60, 0.80, 0.95)
    corr_max_pair_choices: tuple[float, ...] = (0.70, 0.85, 0.98)
    corr_open_reduce_start_choices: tuple[float, ...] = (0.30, 0.60, 0.80)
    corr_open_hard_reject_choices: tuple[float, ...] = (0.80, 0.92, 0.99)
    corr_open_min_weight_mult_choices: tuple[float, ...] = (0.05, 0.25, 0.50)
    max_positions_choices: tuple[int, ...] = (5, 15, 30, 60, 100)
    total_exposure_choices: tuple[float, ...] = (0.50, 1.0, 2.0, 4.0)
    max_trade_exposure_choices: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50)
    min_trade_exposure_choices: tuple[float, ...] = (0.005, 0.02, 0.05, 0.10)
    exposure_multiplier_choices: tuple[float, ...] = (1.0, 3.0, 6.0, 12.0)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v or default


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().tz_localize(None).isoformat()


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULTS_HEADER})


class _Tee:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, msg: str) -> None:
        print(msg, flush=True)
        with self.path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")


@contextmanager
def _temp_env(overrides: dict[str, str]):
    old: dict[str, str | None] = {}
    try:
        for k, v in overrides.items():
            old[k] = os.environ.get(k)
            os.environ[k] = str(v)
        yield
    finally:
        for k, old_v in old.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest_wf_mtime(asset: str = "crypto") -> tuple[Path | None, float]:
    root = models_root_for_asset(asset).resolve()
    runs = [p for p in root.glob("wf_*") if p.is_dir()]
    if not runs:
        return None, 0.0
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = runs[0]
    return p, float(p.stat().st_mtime)


def _run_train_subprocess(env_overrides: dict[str, str], log_path: Path) -> str:
    before_run, before_mtime = _latest_wf_mtime("crypto")
    cmd = [sys.executable, "-u", str((_repo_root() / "scripts" / "train.py").resolve())]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace"
    )
    
    from utils.progress import LineProgressPrinter
    
    log_lines = []
    run_dir_found = None
    
    prog_pool = None
    prog_wf = None
    
    for line in proc.stdout:
        log_lines.append(line)
        line_lower = line.strip().lower()
        
        # Catch run_dir early
        m_run = re.search(r"run_dir:\s*(.+)", line)
        if m_run:
            run_dir_found = m_run.group(1).strip()
            
        # Parse dataset progress
        if "sniper-ds" in line_lower:
            m_ds = re.search(r"(\d+)/(\d+)", line_lower)
            if m_ds:
                done = int(m_ds.group(1))
                total = int(m_ds.group(2))
                if prog_pool is None:
                    prog_pool = LineProgressPrinter(prefix="train-data", total=total, width=26)
                prog_pool.update(done, current="Reading Parquets...")
                
        # Transition to walk-forward
        m_per = re.search(r"periodos=(\d+)", line_lower)
        if m_per and prog_wf is None:
            if prog_pool is not None:
                prog_pool.update(prog_pool.total, current="Pools Assembled")
                prog_pool.close()
            total_steps = int(m_per.group(1))
            prog_wf = LineProgressPrinter(prefix="train-wf", total=total_steps, width=26)
            prog_wf.update(0, current="Starting walk-forward...")
            
        # Track walk-forward steps
        if "d salvo em" in line_lower or "salvo em wf_" in line_lower:
            if prog_wf is not None:
                if not hasattr(prog_wf, "steps_done"):
                    prog_wf.steps_done = 0
                prog_wf.steps_done += 1
                prog_wf.update(prog_wf.steps_done, current="Training models...")
                
        if "modelos salvos em:" in line_lower and prog_wf is not None:
            if not hasattr(prog_wf, "steps_done"):
                prog_wf.steps_done = prog_wf.total
            prog_wf.update(prog_wf.total, current="Done")
            prog_wf.close()

    proc.wait()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(log_lines), encoding="utf-8", errors="replace")
    
    if proc.returncode != 0:
        raise RuntimeError(f"train subprocess failed code={proc.returncode}")
        
    if run_dir_found:
        return run_dir_found
        
    after_run, after_mtime = _latest_wf_mtime("crypto")
    if after_run is not None and after_mtime >= before_mtime and (before_run is None or after_run != before_run or after_mtime > before_mtime):
        return str(after_run)
    raise RuntimeError("could not resolve train run_dir")


def _build_contract(candle_sec: int, label_profit_thr: float, exit_span_min: int, exit_offset: float) -> TradeContract:
    return TradeContract(
        timeframe_sec=int(candle_sec),
        entry_label_windows_minutes=(240,),
        entry_label_min_profit_pcts=(float(label_profit_thr),),
        entry_label_weight_alpha=0.01,
        exit_ema_span=bars_from_minutes(float(exit_span_min), int(candle_sec), min_bars=2),
        exit_ema_init_offset_pct=float(exit_offset),
        fee_pct_per_side=0.0005,
        slippage_pct=0.0,
        max_adds=0,
        add_spacing_pct=0.0,
        add_sizing=(1.0,),
        risk_max_cycle_pct=0.0,
        dd_intermediate_limit_pct=0.0,
        danger_drop_pct=0.0,
        danger_recovery_pct=0.0,
        danger_timeout_hours=0.0,
        danger_fast_minutes=0.0,
        danger_drop_pct_critical=0.0,
        danger_stabilize_recovery_pct=0.0,
        danger_stabilize_bars=0,
        forbid_exit_on_gap=False,
        gap_hours_forbidden=0.0,
    )


def _refresh_labels(contract: TradeContract, candle_sec: int) -> dict[str, object]:
    env = {
        "SNIPER_ASSET_CLASS": "crypto",
        "SNIPER_LABELS_REFRESH_WORKERS": "1",
        "CRYPTO_PIPELINE_CANDLE_SEC": str(int(candle_sec)),
        "SNIPER_CANDLE_SEC": str(int(candle_sec)),
        "PF_CRYPTO_CANDLE_SEC": str(int(candle_sec)),
        "PF_ENTRY_LABEL_NET_PROFIT_THR": str(contract.entry_label_min_profit_pcts[0]),
        "PF_ENTRY_LABEL_PROFIT_ONLY": "1",
        "PF_ENTRY_LABEL_REQUIRE_NO_DIP": "0",
        "PF_ENTRY_LABEL_ENABLE_NEUTRAL": "0",
        "PF_ENTRY_LABEL_ANY_CANONICAL": "0",
    }
    with _temp_env(env):
        apply_crypto_pipeline_env(int(candle_sec))
        s = RefreshLabelsSettings(contract=contract, candle_sec=int(candle_sec))
        s.workers = int(os.environ.get("SNIPER_DATASET_WORKERS", 8))
        s.max_ram_pct = 72.0
        s.min_free_mb = 4096.0
        s.per_worker_mem_mb = 1024.0
        return refresh_labels(s)


def _metrics_from_portfolio_output(res_obj: object) -> dict[str, float]:
    res = res_obj["result"] if isinstance(res_obj, dict) else getattr(res_obj, "result")
    trades = list(getattr(res, "trades", []) or [])
    weighted = np.asarray([float(getattr(t, "r_net", 0.0)) * float(getattr(t, "weight", 0.0)) for t in trades], dtype=np.float64)
    raw = np.asarray([float(getattr(t, "r_net", 0.0)) for t in trades], dtype=np.float64)
    wins = raw[raw > 0.0]
    losses = raw[raw < 0.0]
    pos_weighted = weighted[weighted > 0.0]
    out: dict[str, float] = {
        "eq_end": float(getattr(res_obj, "eq_end", 1.0) if not isinstance(res_obj, dict) else res_obj.get("eq_end", 1.0)),
        "ret_pct": float(((getattr(res_obj, "ret_total", 0.0) if not isinstance(res_obj, dict) else res_obj.get("ret_total", 0.0)) or 0.0)),
        "max_dd": float(getattr(res, "max_dd", 0.0)),
        "trades": float(len(trades)),
        "win_rate": float(np.mean(raw > 0.0)) if raw.size else 0.0,
        "profit_factor": float(np.sum(wins) / max(1e-12, -float(np.sum(losses)))) if losses.size else (float("inf") if wins.size else 0.0),
        "trade_p50_raw": float(np.nanmedian(raw)) if raw.size else 0.0,
    }
    if pos_weighted.size:
        pos_sorted = np.sort(pos_weighted)[::-1]
        pos_sum = float(np.sum(pos_sorted))
        out["top1_share_pos"] = float(pos_sorted[0] / pos_sum) if pos_sum > 0.0 else 0.0
        out["top3_share_pos"] = float(np.sum(pos_sorted[:3]) / pos_sum) if pos_sum > 0.0 else 0.0
    else:
        out["top1_share_pos"] = 0.0
        out["top3_share_pos"] = 0.0
    out["ret_wo_best_weighted"] = float(np.sum(weighted) - (float(np.max(weighted)) if weighted.size else 0.0))
    if weighted.size >= 3:
        w_sorted = np.sort(weighted)[::-1]
        out["ret_wo_top3_weighted"] = float(np.sum(weighted) - float(np.sum(w_sorted[:3])))
    else:
        out["ret_wo_top3_weighted"] = float(np.sum(weighted))

    eq_curve = getattr(res, "equity_curve", None)
    if eq_curve is not None and len(eq_curve):
        eq_s = pd.Series(eq_curve).astype(float)
        month_ret = eq_s.resample("ME").last().pct_change().dropna()
        sem_ret = eq_s.resample("2QE-DEC").last().pct_change().dropna()
        out["month_mean"] = float(month_ret.mean()) if len(month_ret) else 0.0
        out["month_p50"] = float(month_ret.median()) if len(month_ret) else 0.0
        out["month_pos_frac"] = float((month_ret > 0.0).mean()) if len(month_ret) else 0.0
        out["month_worst"] = float(month_ret.min()) if len(month_ret) else 0.0
        out["month_best"] = float(month_ret.max()) if len(month_ret) else 0.0
        out["semester_mean"] = float(sem_ret.mean()) if len(sem_ret) else 0.0
        out["semester_p50"] = float(sem_ret.median()) if len(sem_ret) else 0.0
        out["semester_pos_frac"] = float((sem_ret > 0.0).mean()) if len(sem_ret) else 0.0
        out["semester_worst"] = float(sem_ret.min()) if len(sem_ret) else 0.0
        out["semester_best"] = float(sem_ret.max()) if len(sem_ret) else 0.0
    else:
        for k in ("month_mean", "month_p50", "month_pos_frac", "month_worst", "month_best", "semester_mean", "semester_p50", "semester_pos_frac", "semester_worst", "semester_best"):
            out[k] = 0.0

    entry_ts = [pd.to_datetime(getattr(t, "entry_ts")) for t in trades]
    if entry_ts:
        counts = Counter(entry_ts)
        cl_sizes = np.asarray(list(counts.values()), dtype=np.float64)
        out["clusters"] = float(len(counts))
        out["avg_trades_per_cluster"] = float(np.mean(cl_sizes))
    else:
        out["clusters"] = 0.0
        out["avg_trades_per_cluster"] = 0.0
    out["cluster_pos_frac"] = out["win_rate"]
    return out


def _score_from_metrics(m: dict[str, float]) -> float:
    ret_wo_best = max(0.0, float(m.get("ret_wo_best_weighted", 0.0)))
    month_pos = float(m.get("month_pos_frac", 0.0))
    sem_pos = float(m.get("semester_pos_frac", 0.0))
    max_dd = max(0.0, float(m.get("max_dd", 0.0)))
    top1 = max(0.0, float(m.get("top1_share_pos", 0.0)))
    avg_cluster = max(0.0, float(m.get("avg_trades_per_cluster", 0.0)))
    if ret_wo_best <= 0.0:
        return 0.0
    return float(
        ret_wo_best
        * (0.5 + month_pos)
        * (0.5 + sem_pos)
        * math.exp(-3.0 * max_dd)
        * math.exp(-2.0 * max(0.0, top1 - 0.20))
        * math.exp(-2.5 * max(0.0, avg_cluster - 1.15))
    )


def _sample_label_cfg(rng: random.Random, s: ExploreSettings) -> dict[str, object]:
    return {
        "label_profit_thr": float(rng.choice(s.label_profit_choices)),
        "exit_span_min": int(rng.choice(s.exit_span_choices)),
        "exit_offset": float(rng.choice(s.exit_offset_choices)),
    }


def _sample_train_cfg(rng: random.Random, s: ExploreSettings) -> dict[str, object]:
    return {
        "neg_per_pos": float(rng.choice(s.neg_pos_choices)),
        "tail_blend": float(rng.choice(s.tail_blend_choices)),
        "tail_boost": float(rng.choice(s.tail_boost_choices)),
        "top_qs": str(rng.choice(s.top_q_presets)),
        "top_min_count": int(rng.choice(s.top_min_count_choices)),
    }


def _sample_bt_cfg(rng: random.Random, s: ExploreSettings) -> dict[str, object]:
    corr_enabled = bool(rng.random() < 0.4) 
    tau_entry = round(rng.uniform(0.30, 0.95), 2)
    
    return {
        "tau_entry": float(tau_entry),
        "corr_enabled": corr_enabled,
        "corr_max_with_market": round(rng.uniform(0.50, 0.98), 2),
        "corr_max_pair": round(rng.uniform(0.60, 0.99), 2),
        "corr_open_reduce_start": round(rng.uniform(0.20, 0.85), 2),
        "corr_open_hard_reject": round(rng.uniform(0.70, 0.99), 2),
        "corr_open_min_weight_mult": round(rng.uniform(0.01, 0.60), 2),
        "max_positions": rng.randint(2, 120),
        "total_exposure": round(rng.uniform(0.20, 5.0), 2),
        "max_trade_exposure": round(rng.uniform(0.01, 0.60), 2),
        "min_trade_exposure": round(rng.uniform(0.001, 0.15), 2),
        "exposure_multiplier": round(rng.uniform(1.0, 15.0), 1),
    }


def run(settings: ExploreSettings | None = None) -> None:
    s = settings or ExploreSettings()
    random_seed = int(s.seed)
    # If seed is 0 or negative, we use current time to make it truly random every run
    if random_seed <= 0:
        random_seed = int(time.time() * 1000) % 2**32
        
    rng = random.Random(random_seed)
    out_root = resolve_generated_path(s.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / s.results_csv
    loop_log = out_root / "loop.log"
    log = _Tee(loop_log)
    log.write(f"[explore] out_root={out_root}")
    log.write(f"[explore] seed={s.seed} label_trials={s.max_label_trials} retrains_per_label={s.retrains_per_label} backtests_per_retrain={s.backtests_per_retrain}")

    start_label_idx = 1
    while (out_root / f"label_{start_label_idx:03d}").exists():
        start_label_idx += 1
        
    if start_label_idx > s.max_label_trials:
        log.write(f"[explore] Already reached max_label_trials={s.max_label_trials}. Increase WF_EXPLORE_LABEL_TRIALS to run more.")
        return

    for label_idx in range(start_label_idx, int(s.max_label_trials) + 1):
        label_id = f"label_{label_idx:03d}"
        label_dir = out_root / label_id
        label_dir.mkdir(parents=True, exist_ok=True)
        label_cfg = _sample_label_cfg(rng, s)
        contract = _build_contract(
            candle_sec=int(s.candle_sec),
            label_profit_thr=float(label_cfg["label_profit_thr"]),
            exit_span_min=int(label_cfg["exit_span_min"]),
            exit_offset=float(label_cfg["exit_offset"]),
        )
        start_utc = _utc_now_iso()
        status = "ok"
        err = ""
        refresh_info: dict[str, object] = {}
        try:
            log.write(f"[explore] {label_id} refresh start profit_thr={label_cfg['label_profit_thr']} exit_span={label_cfg['exit_span_min']} offset={label_cfg['exit_offset']}")
            refresh_info = _refresh_labels(contract, int(s.candle_sec))
            log.write(f"[explore] {label_id} refresh done ok={refresh_info.get('ok', 0)} fail={refresh_info.get('fail', 0)}")
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"
            (label_dir / "refresh_error.txt").write_text(err, encoding="utf-8")
            log.write(f"[explore] {label_id} refresh error {err}")
        end_utc = _utc_now_iso()
        _append_csv(
            results_csv,
            {
                "label_id": label_id,
                "model_id": "",
                "backtest_id": "",
                "train_id": label_id,
                "stage": "refresh",
                "status": status,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "duration_sec": max(0.0, (pd.to_datetime(end_utc) - pd.to_datetime(start_utc)).total_seconds()),
                "seed": s.seed,
                "error": err,
                "label_profit_thr": label_cfg["label_profit_thr"],
                "exit_ema_span_min": label_cfg["exit_span_min"],
                "exit_ema_init_offset_pct": label_cfg["exit_offset"],
            },
        )
        if status != "ok":
            continue

        for model_idx in range(1, int(s.retrains_per_label) + 1):
            model_id = f"model_{model_idx:03d}"
            model_dir = label_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            train_cfg = _sample_train_cfg(rng, s)
            train_start = _utc_now_iso()
            train_status = "ok"
            train_err = ""
            train_run_dir = ""
            try:
                log.write(
                    f"[explore] {label_id}/{model_id} train start neg_per_pos={train_cfg['neg_per_pos']} "
                    f"tail_blend={train_cfg['tail_blend']} tail_boost={train_cfg['tail_boost']} qs={train_cfg['top_qs']}"
                )
                env = {
                    "CRYPTO_PIPELINE_CANDLE_SEC": str(int(s.candle_sec)),
                    "SNIPER_CANDLE_SEC": str(int(s.candle_sec)),
                    "PF_CRYPTO_CANDLE_SEC": str(int(s.candle_sec)),
                    "CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT": str(label_cfg["label_profit_thr"]),
                    "CRYPTO_EXIT_EMA_SPAN_MINUTES": str(label_cfg["exit_span_min"]),
                    "CRYPTO_EXIT_EMA_INIT_OFFSET_PCT": str(label_cfg["exit_offset"]),
                    "TRAIN_ENTRY_RATIO_NEG_PER_POS": str(train_cfg["neg_per_pos"]),
                    "TRAIN_REFRESH_LABELS": "0",
                    "TRAIN_MAX_SYMBOLS": "0" if int(s.max_symbols) <= 0 else str(int(s.max_symbols)),
                    "SNIPER_CACHE_WORKERS": str(int(s.safe_threads)),
                    "SNIPER_DATASET_WORKERS": str(int(s.safe_threads)),
                    "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BLEND": str(train_cfg["tail_blend"]),
                    "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BOOST": str(train_cfg["tail_boost"]),
                    "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_QS": str(train_cfg["top_qs"]),
                    "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_MIN_COUNT": str(train_cfg["top_min_count"]),
                }
                
                # OPTIMIZATION: Only train offsets actually used in the backtest segment
                # Each segment in walk-forward is 180 days. We need models at the START of each segment.
                # PLUS the offset 0 (end of window) which is the model we are actually optimizing.
                num_segments = max(1, math.ceil(int(s.days) / 180))
                active_offsets = [(i + 1) * 180 for i in range(num_segments)]
                env["TRAIN_OFFSETS_DAYS"] = ",".join(map(str, active_offsets))
                
                train_run_dir = _run_train_subprocess(env, model_dir / "train.log")
                log.write(f"[explore] {label_id}/{model_id} train done run_dir={train_run_dir}")
            except Exception as e:
                train_status = "error"
                train_err = f"{type(e).__name__}: {e}"
                log.write(f"[explore] {label_id}/{model_id} train error {train_err}")
            train_end = _utc_now_iso()
            _append_csv(
                results_csv,
                {
                    "label_id": label_id,
                    "model_id": model_id,
                    "backtest_id": "",
                    "train_id": f"{label_id}/{model_id}",
                    "stage": "train",
                    "status": train_status,
                    "start_utc": train_start,
                    "end_utc": train_end,
                    "duration_sec": max(0.0, (pd.to_datetime(train_end) - pd.to_datetime(train_start)).total_seconds()),
                    "seed": s.seed,
                    "train_run_dir": train_run_dir,
                    "error": train_err,
                    "label_profit_thr": label_cfg["label_profit_thr"],
                    "exit_ema_span_min": label_cfg["exit_span_min"],
                    "exit_ema_init_offset_pct": label_cfg["exit_offset"],
                    "entry_ratio_neg_per_pos": train_cfg["neg_per_pos"],
                    "calib_tail_blend": train_cfg["tail_blend"],
                    "calib_tail_boost": train_cfg["tail_boost"],
                    "top_metric_qs": train_cfg["top_qs"],
                    "top_metric_min_count": train_cfg["top_min_count"],
                },
            )
            if train_status != "ok":
                continue

            prepared_portfolio = None
            try:
                prepare_cfg = _default_portfolio_cfg()
                prepare_cfg.corr_filter_enabled = True
                prepare_cfg.corr_open_filter_enabled = True
                contract_prepare = _build_contract(
                    candle_sec=int(s.candle_sec),
                    label_profit_thr=float(label_cfg["label_profit_thr"]),
                    exit_span_min=int(label_cfg["exit_span_min"]),
                    exit_offset=float(label_cfg["exit_offset"]),
                )
                prepare_settings = PortfolioDemoSettings(
                    asset_class="crypto",
                    run_dir=train_run_dir,
                    days=int(s.days),
                    max_symbols=int(s.max_symbols),
                    total_days_cache=0,
                    symbols=[],
                    cfg=prepare_cfg,
                    save_plot=False,
                    plot_out=None,
                    override_tau_entry=None,
                    candle_sec=int(s.candle_sec),
                    contract=contract_prepare,
                    long_only=True,
                    require_feature_cache=True,
                    rebuild_on_score_error=False,
                    align_global_window=True,
                )
                log.write(f"[explore] {label_id}/{model_id} prepare start")
                prepared_portfolio = prepare_portfolio_data(prepare_settings)
                log.write(
                    f"[explore] {label_id}/{model_id} prepare done symbols={prepared_portfolio.symbols_total} "
                    f"tau_default={prepared_portfolio.tau_entry_default:.2f}"
                )
            except Exception as e:
                train_err = f"prepare_portfolio failed: {type(e).__name__}: {e}"
                log.write(f"[explore] {label_id}/{model_id} prepare error {train_err}")
                _append_csv(
                    results_csv,
                    {
                        "label_id": label_id,
                        "model_id": model_id,
                        "backtest_id": "",
                        "train_id": f"{label_id}/{model_id}",
                        "stage": "train",
                        "status": "error",
                        "start_utc": train_end,
                        "end_utc": _utc_now_iso(),
                        "duration_sec": 0.0,
                        "seed": s.seed,
                        "train_run_dir": train_run_dir,
                        "error": train_err,
                        "label_profit_thr": label_cfg["label_profit_thr"],
                        "exit_ema_span_min": label_cfg["exit_span_min"],
                        "exit_ema_init_offset_pct": label_cfg["exit_offset"],
                        "entry_ratio_neg_per_pos": train_cfg["neg_per_pos"],
                        "calib_tail_blend": train_cfg["tail_blend"],
                        "calib_tail_boost": train_cfg["tail_boost"],
                        "top_metric_qs": train_cfg["top_qs"],
                        "top_metric_min_count": train_cfg["top_min_count"],
                    },
                )
                continue

            for bt_idx in range(1, int(s.backtests_per_retrain) + 1):
                bt_id = f"bt_{bt_idx:03d}"
                bt_dir = model_dir / bt_id
                bt_dir.mkdir(parents=True, exist_ok=True)
                bt_cfg = _sample_bt_cfg(rng, s)
                bt_start = _utc_now_iso()
                bt_status = "ok"
                bt_err = ""
                metrics: dict[str, float] = {}
                score = 0.0
                top_symbols = ""
                equity_html = ""
                try:
                    cfg = _default_portfolio_cfg()
                    cfg.max_positions = int(bt_cfg["max_positions"])
                    cfg.total_exposure = float(bt_cfg["total_exposure"])
                    cfg.max_trade_exposure = float(bt_cfg["max_trade_exposure"])
                    cfg.min_trade_exposure = float(bt_cfg["min_trade_exposure"])
                    cfg.corr_filter_enabled = bool(bt_cfg["corr_enabled"])
                    cfg.corr_open_filter_enabled = bool(bt_cfg["corr_enabled"])
                    cfg.corr_max_with_market = float(bt_cfg["corr_max_with_market"])
                    cfg.corr_max_pair = float(bt_cfg["corr_max_pair"])
                    cfg.corr_open_reduce_start = float(bt_cfg["corr_open_reduce_start"])
                    cfg.corr_open_hard_reject = float(bt_cfg["corr_open_hard_reject"])
                    cfg.corr_open_min_weight_mult = float(bt_cfg["corr_open_min_weight_mult"])
                    cfg.exposure_multiplier = float(bt_cfg.get("exposure_multiplier", 0.0) or 0.0)
                    contract_bt = _build_contract(
                        candle_sec=int(s.candle_sec),
                        label_profit_thr=float(label_cfg["label_profit_thr"]),
                        exit_span_min=int(label_cfg["exit_span_min"]),
                        exit_offset=float(label_cfg["exit_offset"]),
                    )
                    log.write(
                        f"[explore] {label_id}/{model_id}/{bt_id} backtest start tau={bt_cfg['tau_entry']} "
                        f"corr={int(bool(bt_cfg['corr_enabled']))} max_pos={bt_cfg['max_positions']}"
                    )
                    bt_out = run_prepared_portfolio(
                        prepared_portfolio,
                        cfg=cfg,
                        days=int(s.days),
                        override_tau_entry=float(bt_cfg["tau_entry"]),
                        save_plot=True,
                        plot_out=str((bt_dir / "portfolio_equity.html").resolve()),
                    )
                    metrics = _metrics_from_portfolio_output(bt_out)
                    score = _score_from_metrics(metrics)
                    equity_html = str(bt_out.get("plot_path") or "")
                    trades_obj = list(getattr(bt_out["result"], "trades", []) or [])
                    sym_weight = Counter()
                    for tr in trades_obj:
                        sym_weight[str(getattr(tr, "symbol", ""))] += float(getattr(tr, "r_net", 0.0)) * float(getattr(tr, "weight", 0.0))
                    top_symbols = ",".join([s0 for s0, _ in sym_weight.most_common(10)])
                    log.write(
                        f"[explore] {label_id}/{model_id}/{bt_id} score={score:.6f} ret={metrics.get('ret_pct', 0.0):+.2%} "
                        f"dd={metrics.get('max_dd', 0.0):.2%} trades={int(metrics.get('trades', 0.0))}"
                    )
                except Exception as e:
                    bt_status = "error"
                    bt_err = f"{type(e).__name__}: {e}"
                    (bt_dir / "error.txt").write_text(bt_err, encoding="utf-8")
                    log.write(f"[explore] {label_id}/{model_id}/{bt_id} backtest error {bt_err}")
                bt_end = _utc_now_iso()
                _append_csv(
                    results_csv,
                    {
                        "label_id": label_id,
                        "model_id": model_id,
                        "backtest_id": bt_id,
                        "train_id": f"{label_id}/{model_id}",
                        "stage": "backtest",
                        "status": bt_status,
                        "start_utc": bt_start,
                        "end_utc": bt_end,
                        "duration_sec": max(0.0, (pd.to_datetime(bt_end) - pd.to_datetime(bt_start)).total_seconds()),
                        "seed": s.seed,
                        "train_run_dir": train_run_dir,
                        "bt_out_dir": str(bt_dir),
                        "equity_html": equity_html,
                        "equity_png": "",
                        "score": score,
                        "eq_end": metrics.get("eq_end", ""),
                        "ret_pct": metrics.get("ret_pct", ""),
                        "max_dd": metrics.get("max_dd", ""),
                        "win_rate": metrics.get("win_rate", ""),
                        "profit_factor": metrics.get("profit_factor", ""),
                        "trades": int(metrics.get("trades", 0.0)) if metrics else "",
                        "top_symbols": top_symbols,
                        "error": bt_err,
                        "label_profit_thr": label_cfg["label_profit_thr"],
                        "exit_ema_span_min": label_cfg["exit_span_min"],
                        "exit_ema_init_offset_pct": label_cfg["exit_offset"],
                        "entry_ratio_neg_per_pos": train_cfg["neg_per_pos"],
                        "calib_tail_blend": train_cfg["tail_blend"],
                        "calib_tail_boost": train_cfg["tail_boost"],
                        "top_metric_qs": train_cfg["top_qs"],
                        "top_metric_min_count": train_cfg["top_min_count"],
                        "tau_entry": bt_cfg["tau_entry"],
                        "corr_enabled": int(bool(bt_cfg["corr_enabled"])),
                        "corr_max_with_market": bt_cfg["corr_max_with_market"],
                        "corr_max_pair": bt_cfg["corr_max_pair"],
                        "corr_open_reduce_start": bt_cfg["corr_open_reduce_start"],
                        "corr_open_hard_reject": bt_cfg["corr_open_hard_reject"],
                        "corr_open_min_weight_mult": bt_cfg["corr_open_min_weight_mult"],
                        "max_positions": bt_cfg["max_positions"],
                        "total_exposure": bt_cfg["total_exposure"],
                        "max_trade_exposure": bt_cfg["max_trade_exposure"],
                        "min_trade_exposure": bt_cfg["min_trade_exposure"],
                        "month_mean": metrics.get("month_mean", ""),
                        "month_p50": metrics.get("month_p50", ""),
                        "month_pos_frac": metrics.get("month_pos_frac", ""),
                        "month_worst": metrics.get("month_worst", ""),
                        "month_best": metrics.get("month_best", ""),
                        "semester_mean": metrics.get("semester_mean", ""),
                        "semester_p50": metrics.get("semester_p50", ""),
                        "semester_pos_frac": metrics.get("semester_pos_frac", ""),
                        "semester_worst": metrics.get("semester_worst", ""),
                        "semester_best": metrics.get("semester_best", ""),
                        "clusters": int(metrics.get("clusters", 0.0)) if metrics else "",
                        "avg_trades_per_cluster": metrics.get("avg_trades_per_cluster", ""),
                        "cluster_pos_frac": metrics.get("cluster_pos_frac", ""),
                        "top1_share_pos": metrics.get("top1_share_pos", ""),
                        "top3_share_pos": metrics.get("top3_share_pos", ""),
                        "ret_wo_best_weighted": metrics.get("ret_wo_best_weighted", ""),
                        "ret_wo_top3_weighted": metrics.get("ret_wo_top3_weighted", ""),
                    },
                )


def main() -> None:
    s = ExploreSettings(
        out_root=_env_str("WF_EXPLORE_OUT_ROOT", "wf_portfolio_explore"),
        results_csv=_env_str("WF_EXPLORE_RESULTS_CSV", "explore_runs.csv"),
        seed=_env_int("WF_EXPLORE_SEED", 42),
        max_label_trials=_env_int("WF_EXPLORE_LABEL_TRIALS", 3),
        retrains_per_label=_env_int("WF_EXPLORE_RETRAINS_PER_LABEL", 3),
        backtests_per_retrain=_env_int("WF_EXPLORE_BACKTESTS_PER_RETRAIN", 15),
        days=_env_int("WF_EXPLORE_DAYS", 4 * 360),
        max_symbols=_env_int("WF_EXPLORE_MAX_SYMBOLS", 0),
        candle_sec=_env_int("WF_EXPLORE_CANDLE_SEC", 300),
        safe_threads=_env_int("WF_EXPLORE_SAFE_THREADS", 1),
    )
    run(s)


if __name__ == "__main__":
    main()
