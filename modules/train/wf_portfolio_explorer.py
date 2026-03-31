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
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import duckdb  # type: ignore


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
from train.sniper_dataflow import _cache_dir, _cache_format  # noqa: E402


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
    "phase",
    "generation",
    "source_cluster_id",
    "source_label_id",
    "source_model_id",
    "source_backtest_id",
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
    "window_info",
    "forced_period_days",
    "period_train_end_utc",
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
    # Kept in the CSV schema only so older analysis code can still read mixed roots.
    # New explores no longer vary or populate max_positions.
    "max_positions",
    "total_exposure",
    "max_trade_exposure",
    "min_trade_exposure",
    "exposure_multiplier",
    "month_mean",
    "month_p50",
    "month_pos_frac",
    "month_worst",
    "month_best",
    "max_neg_month_streak",
    "underwater_frac",
    "worst_rolling_90d",
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

VARIED_PARAM_COLS = [
    "label_profit_thr",
    "exit_ema_span_min",
    "exit_ema_init_offset_pct",
    "entry_ratio_neg_per_pos",
    "calib_tail_blend",
    "calib_tail_boost",
    "tau_entry",
]


@dataclass
class ExploreSettings:
    out_root: str = "wf_portfolio_explore"
    results_csv: str = "explore_runs.csv"
    seed: int = 42
    max_label_trials: int = 56
    retrains_per_label: int = 2
    backtests_per_retrain: int = 21
    days: int = 4 * 360
    max_symbols: int = 0
    candle_sec: int = 300
    safe_threads: int = 1
    # V6: keep structural policy fixed, expand edge/training modestly.
    label_profit_choices: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    exit_span_choices: tuple[int, ...] = (60, 90, 120, 180, 300)
    exit_offset_choices: tuple[float, ...] = (0.00, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
    # V6: small train/calibration preset space, not the free-for-all from v4.
    neg_pos_choices: tuple[float, ...] = (4.0, 6.0, 8.0)
    tail_blend_choices: tuple[float, ...] = (0.60, 0.70, 0.80)
    tail_boost_choices: tuple[float, ...] = (1.75, 2.25, 2.75)
    top_q_presets: tuple[str, ...] = (
        "0.0005,0.001,0.0025",
        "0.0025,0.005,0.01",
        "0.001,0.0025,0.005",
    )
    top_min_count_choices: tuple[int, ...] = (48, 64)
    tau_entry_choices: tuple[float, ...] = tuple(round(0.70 + (0.01 * i), 2) for i in range(21))
    # V6: portfolio risk is controlled only by exposure budget. The practical
    # simultaneous-position cap now comes from total_exposure/max_trade_exposure.
    total_exposure_choices: tuple[float, ...] = (1.00,)
    max_trade_exposure_choices: tuple[float, ...] = (0.10,)
    min_trade_exposure_choices: tuple[float, ...] = (0.02,)
    exposure_multiplier_choices: tuple[float, ...] = (0.0,)
    feature_preset: str = "core80"


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v or default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().tz_localize(None).isoformat()


def _fmt_duration(seconds: float) -> str:
    try:
        total = max(0.0, float(seconds))
    except Exception:
        total = 0.0
    if total < 60.0:
        return f"{total:.1f}s"
    minutes = total / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}m"
    return f"{(minutes / 60.0):.2f}h"


def _read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULTS_HEADER)
    try:
        rel = duckdb.sql(f"SELECT * FROM read_csv_auto('{path.as_posix()}', all_varchar=true)")
        rows = rel.fetchall()
        cols = [str(c) for c in rel.columns]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(12):
        try:
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
                if write_header:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in RESULTS_HEADER})
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.25 * (attempt + 1))
        except OSError as exc:
            last_err = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"failed to append results row to {path} after repeated retries") from last_err


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


def _resolve_period_window(run_dir: str, period_days: int, test_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    meta_path = Path(run_dir) / f"period_{int(period_days)}d" / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"missing meta for forced period: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_end_raw = str(meta.get("train_end_utc") or "").strip()
    if not train_end_raw:
        raise RuntimeError(f"missing train_end_utc in {meta_path}")
    train_end = pd.to_datetime(train_end_raw)
    test_end = train_end + pd.Timedelta(days=int(test_days))
    return train_end, test_end


def _infer_max_period_days(run_dir: str) -> int:
    pd_dirs = [
        p for p in Path(run_dir).iterdir()
        if p.is_dir() and p.name.startswith("period_") and p.name.endswith("d")
    ]
    if not pd_dirs:
        raise RuntimeError(f"no period_* dirs found in {run_dir}")
    return max(int(p.name.replace("period_", "").replace("d", "")) for p in pd_dirs)


def _validate_required_period_meta(run_dir: str, period_days: int) -> Path:
    meta_path = Path(run_dir) / f"period_{int(period_days)}d" / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"missing meta for required period: {meta_path}")
    return meta_path


def _effective_forced_period_days(forced_step_days: int) -> int:
    forced_step_days = int(forced_step_days)
    if forced_step_days <= 0:
        return 0
    remove_tail_days = _env_int("SNIPER_REMOVE_TAIL_DAYS", 0)
    if remove_tail_days <= 0:
        return int(forced_step_days)
    effective = int(forced_step_days) - int(remove_tail_days)
    return int(max(1, effective))


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
    pool_read_complete = False
    
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
                current = "Assembling pools..." if done >= total else "Reading Parquets..."
                prog_pool.update(done, current=current)
                if done >= total:
                    pool_read_complete = True
                
        # Transition to walk-forward
        m_per = re.search(r"periodos=(\d+)", line_lower)
        if m_per and prog_wf is None:
            if prog_pool is not None:
                current = "Pools Assembled" if pool_read_complete else "Preparing dataset..."
                prog_pool.update(prog_pool.total, current=current)
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


def _refresh_labels(contract: TradeContract, candle_sec: int, feature_preset: str) -> dict[str, object]:
    refresh_workers = max(1, int(os.environ.get("SNIPER_LABELS_REFRESH_WORKERS", os.environ.get("SNIPER_DATASET_WORKERS", "4"))))
    env = {
        "SNIPER_ASSET_CLASS": "crypto",
        "SNIPER_FEATURE_PRESET": str(feature_preset or "full"),
        "SNIPER_LABELS_REFRESH_WORKERS": str(refresh_workers),
        "SNIPER_TRAIN_EXIT_MODEL": "0",
        "CRYPTO_PIPELINE_CANDLE_SEC": str(int(candle_sec)),
        "SNIPER_CANDLE_SEC": str(int(candle_sec)),
        "PF_CRYPTO_CANDLE_SEC": str(int(candle_sec)),
        "PF_ENTRY_LABEL_NET_PROFIT_THR": str(contract.entry_label_min_profit_pcts[0]),
        "PF_ENTRY_LABEL_PROFIT_ONLY": "1",
        "PF_ENTRY_LABEL_REQUIRE_NO_DIP": "0",
        "PF_ENTRY_LABEL_ENABLE_NEUTRAL": "0",
        "PF_ENTRY_LABEL_ANY_CANONICAL": "0",
        "PF_REFRESH_CONTRACT_LABELS_ONLY": "1",
    }
    with _temp_env(env):
        apply_crypto_pipeline_env(int(candle_sec))
        s = RefreshLabelsSettings(contract=contract, candle_sec=int(candle_sec))
        s.workers = int(os.environ.get("SNIPER_LABELS_REFRESH_WORKERS", refresh_workers))
        s.max_ram_pct = 72.0
        s.min_free_mb = 4096.0
        s.per_worker_mem_mb = 1024.0
        return refresh_labels(s)


def _list_cached_symbols(candle_sec: int, feature_preset: str) -> list[str]:
    with _temp_env({"SNIPER_FEATURE_PRESET": str(feature_preset or "full")}):
        cache_dir = _cache_dir("crypto", int(candle_sec))
        fmt = _cache_format()
        suffix = ".parquet" if fmt == "parquet" else ".pkl"
        symbols: list[str] = []
        for p in sorted(cache_dir.glob(f"*{suffix}")):
            sym = str(p.stem or "").strip().upper()
            if sym:
                symbols.append(sym)
        return symbols


def _write_symbol_manifest(label_dir: Path, symbols: list[str]) -> str:
    manifest = label_dir / "symbols_ok.txt"
    manifest.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    return str(manifest)


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
        peaks = eq_s.cummax()
        underwater = (eq_s < peaks).astype(float)
        max_streak = 0
        cur_streak = 0
        for value in month_ret.to_numpy(dtype=float):
            if value < 0.0:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
        rolling_90d = (eq_s / eq_s.shift(90)) - 1.0
        out["month_mean"] = float(month_ret.mean()) if len(month_ret) else 0.0
        out["month_p50"] = float(month_ret.median()) if len(month_ret) else 0.0
        out["month_pos_frac"] = float((month_ret > 0.0).mean()) if len(month_ret) else 0.0
        out["month_worst"] = float(month_ret.min()) if len(month_ret) else 0.0
        out["month_best"] = float(month_ret.max()) if len(month_ret) else 0.0
        out["max_neg_month_streak"] = float(max_streak)
        out["underwater_frac"] = float(underwater.mean()) if len(underwater) else 0.0
        out["worst_rolling_90d"] = float(rolling_90d.min()) if rolling_90d.notna().any() else 0.0
        out["semester_mean"] = float(sem_ret.mean()) if len(sem_ret) else 0.0
        out["semester_p50"] = float(sem_ret.median()) if len(sem_ret) else 0.0
        out["semester_pos_frac"] = float((sem_ret > 0.0).mean()) if len(sem_ret) else 0.0
        out["semester_worst"] = float(sem_ret.min()) if len(sem_ret) else 0.0
        out["semester_best"] = float(sem_ret.max()) if len(sem_ret) else 0.0
    else:
        for k in ("month_mean", "month_p50", "month_pos_frac", "month_worst", "month_best", "max_neg_month_streak", "underwater_frac", "worst_rolling_90d", "semester_mean", "semester_p50", "semester_pos_frac", "semester_worst", "semester_best"):
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
    ret_pct = max(0.0, float(m.get("ret_pct", 0.0)) * 100.0)
    max_dd = max(0.0, float(m.get("max_dd", 0.0)))
    month_pos = float(m.get("month_pos_frac", 0.0))
    streak = max(0.0, float(m.get("max_neg_month_streak", 0.0)))
    underwater = min(1.0, max(0.0, float(m.get("underwater_frac", 1.0))))
    worst_90d = float(m.get("worst_rolling_90d", 0.0))
    pf = max(0.0, float(m.get("profit_factor", 1.0)))
    trades = max(0.0, float(m.get("trades", 0.0)))
    if ret_pct <= 0.0:
        return 0.0
    if max_dd > 0.30:
        return 0.0
    return float(
        (math.sqrt(ret_pct) * 9.0)
        * math.exp(-8.0 * max_dd)
        * math.exp(-18.0 * max(0.0, max_dd - 0.12))
        * (0.80 + 0.20 * min(1.0, max(0.0, month_pos)))
        * math.exp(-0.55 * streak)
        * math.exp(-2.25 * underwater)
        * math.exp(-7.5 * max(0.0, -worst_90d))
        * min(1.25, max(0.85, pf))
        * min(1.0, trades / 120.0)
    )


def _vdc(index: int, base: int) -> float:
    idx = max(1, int(index))
    b = max(2, int(base))
    out = 0.0
    denom = 1.0
    while idx > 0:
        idx, rem = divmod(idx, b)
        denom *= float(b)
        out += float(rem) / denom
    return float(out)


def _sample_choice(choices: tuple, seq_idx: int, *, base: int, salt: int = 0):
    if not choices:
        raise ValueError("choices must not be empty")
    idx = max(1, int(seq_idx) + int(salt))
    pos = min(len(choices) - 1, int(_vdc(idx, int(base)) * len(choices)))
    return choices[pos]


def _grid_choice(grid_idx: int, choices: tuple):
    if not choices:
        raise ValueError("choices must not be empty")
    return choices[int(grid_idx) % len(choices)]


def _label_cfg_from_grid(label_ordinal: int, s: ExploreSettings) -> dict[str, object]:
    profit_choices = tuple(float(v) for v in s.label_profit_choices)
    span_choices = tuple(int(v) for v in s.exit_span_choices)
    offset_choices = tuple(float(v) for v in s.exit_offset_choices)
    n_profit = len(profit_choices)
    n_span = len(span_choices)
    n_offset = len(offset_choices)
    total = n_profit * n_span * n_offset
    if total <= 0:
        raise ValueError("label grid must not be empty")
    # Deterministic permutation: covers the full 245-combo grid if requested,
    # and gives a well-spread subset for smaller runs such as the default 49.
    permuted = (int(label_ordinal) * 73) % total
    profit_idx = permuted // (n_span * n_offset)
    rem = permuted % (n_span * n_offset)
    span_idx = rem // n_offset
    offset_idx = rem % n_offset
    return {
        "label_profit_thr": _grid_choice(profit_idx, profit_choices),
        "exit_span_min": _grid_choice(span_idx, span_choices),
        "exit_offset": _grid_choice(offset_idx, offset_choices),
    }


def _bt_cfg_from_grid(bt_ordinal: int, s: ExploreSettings) -> dict[str, object]:
    return {
        "tau_entry": float(_grid_choice(int(bt_ordinal), tuple(float(v) for v in s.tau_entry_choices))),
        "total_exposure": float(_grid_choice(0, s.total_exposure_choices)),
        "max_trade_exposure": float(_grid_choice(0, s.max_trade_exposure_choices)),
        "min_trade_exposure": float(_grid_choice(0, s.min_trade_exposure_choices)),
        "exposure_multiplier": float(_grid_choice(0, s.exposure_multiplier_choices)),
    }


def _train_cfg_from_grid(train_ordinal: int, s: ExploreSettings) -> dict[str, object]:
    neg_choices = tuple(float(v) for v in s.neg_pos_choices)
    blend_choices = tuple(float(v) for v in s.tail_blend_choices)
    boost_choices = tuple(float(v) for v in s.tail_boost_choices)
    q_choices = tuple(str(v) for v in s.top_q_presets)
    min_count_choices = tuple(int(v) for v in s.top_min_count_choices)
    total = len(neg_choices) * len(blend_choices) * len(boost_choices) * len(q_choices) * len(min_count_choices)
    if total <= 0:
        raise ValueError("train grid must not be empty")
    permuted = (int(train_ordinal) * 97) % total
    n0 = len(blend_choices) * len(boost_choices) * len(q_choices) * len(min_count_choices)
    n1 = len(boost_choices) * len(q_choices) * len(min_count_choices)
    n2 = len(q_choices) * len(min_count_choices)
    n3 = len(min_count_choices)
    neg_idx = permuted // n0
    rem = permuted % n0
    blend_idx = rem // n1
    rem = rem % n1
    boost_idx = rem // n2
    rem = rem % n2
    q_idx = rem // n3
    min_idx = rem % n3
    return {
        "neg_per_pos": _grid_choice(neg_idx, neg_choices),
        "tail_blend": _grid_choice(blend_idx, blend_choices),
        "tail_boost": _grid_choice(boost_idx, boost_choices),
        "top_qs": _grid_choice(q_idx, q_choices),
        "top_min_count": _grid_choice(min_idx, min_count_choices),
    }


def _choice_index(choices: tuple, value: object) -> int:
    if not choices:
        raise ValueError("choices must not be empty")
    vals = list(choices)
    try:
        target = float(value)
        dists = [abs(float(v) - target) for v in vals]
        return int(min(range(len(vals)), key=lambda i: dists[i]))
    except Exception:
        text = str(value)
        for idx, item in enumerate(vals):
            if str(item) == text:
                return int(idx)
        return 0


def _neighbor_choice(choices: tuple, center_value: object, seq_idx: int, *, base: int, salt: int = 0, radius: int = 1):
    center_idx = _choice_index(choices, center_value)
    lo = max(0, center_idx - int(radius))
    hi = min(len(choices) - 1, center_idx + int(radius))
    return _sample_choice(tuple(choices[lo : hi + 1]), seq_idx, base=base, salt=salt)


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return series.rank(pct=True, ascending=ascending, method="average").fillna(0.0)


def _assign_clusters(arr: np.ndarray) -> tuple[np.ndarray, float]:
    n = int(len(arr))
    if n <= 0:
        return np.empty((0,), dtype=int), 0.0
    if n == 1:
        return np.zeros((1,), dtype=int), 0.0
    dmat = np.sqrt(np.sum((arr[:, None, :] - arr[None, :, :]) ** 2, axis=2))
    k = max(2, min(8, n - 1))
    kth = np.partition(dmat, kth=k, axis=1)[:, k]
    radius = float(np.median(kth) * 1.15)
    radius = max(radius, 0.40)
    labels = np.full(n, -1, dtype=int)
    cid = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = cid
        while stack:
            cur = stack.pop()
            neigh = np.where(dmat[cur] <= radius)[0]
            for nb in neigh:
                if labels[nb] < 0:
                    labels[nb] = cid
                    stack.append(int(nb))
        cid += 1
    return labels, radius


def _refine_radius(generation: int) -> int:
    generation = int(generation)
    if generation <= 3:
        return 2
    return 1


def _pick_seed(seeds: list[dict], seq_idx: int) -> dict | None:
    if not seeds:
        return None
    idx = min(len(seeds) - 1, int(_vdc(int(seq_idx) + 97, 89) * len(seeds)))
    return dict(seeds[idx])


def _load_refine_seeds(results_csv: Path, generation: int) -> list[dict]:
    if not results_csv.exists():
        return []
    try:
        df = _read_results_csv(results_csv)
    except Exception:
        return []
    if df.empty:
        return []
    df = df[(df.get("stage").astype(str) == "backtest") & (df.get("status").astype(str) == "ok")].copy()
    if df.empty:
        return []
    if "generation" in df.columns:
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(0).astype(int)
        df = df[df["generation"] < int(generation)].copy()
    if df.empty:
        return []
    for col in VARIED_PARAM_COLS + ["score", "max_dd"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["score"].notna()].copy()
    if df.empty:
        return []

    feats = df[VARIED_PARAM_COLS].copy()
    med = feats.median(numeric_only=True)
    feats = feats.fillna(med).fillna(0.0)
    mins = feats.min()
    span = (feats.max() - mins).replace(0.0, 1.0)
    arr = ((feats - mins) / span).to_numpy(dtype=float)
    cluster_ids, _ = _assign_clusters(arr)
    df["cluster_id"] = cluster_ids
    df["cluster_dist"] = 0.0

    centers: dict[int, np.ndarray] = {}
    for cid in sorted(df["cluster_id"].unique()):
        mask = df["cluster_id"] == cid
        center = np.nanmedian(arr[mask.to_numpy(), :], axis=0)
        centers[int(cid)] = center
        dists = np.sqrt(np.sum((arr[mask.to_numpy(), :] - center) ** 2, axis=1))
        df.loc[mask, "cluster_dist"] = dists

    stats = (
        df.groupby("cluster_id", dropna=False)
        .agg(
            cluster_size=("cluster_id", "size"),
            score_q25=("score", lambda x: float(pd.Series(x).quantile(0.25))),
            score_median=("score", "median"),
            dd_median=("max_dd", "median"),
        )
        .reset_index()
    )
    stats["score_rank"] = _safe_rank(stats["score_q25"], ascending=True)
    stats["size_rank"] = _safe_rank(stats["cluster_size"], ascending=True)
    stats["dd_rank"] = _safe_rank(-pd.to_numeric(stats["dd_median"], errors="coerce").fillna(1.0), ascending=True)
    stats["cluster_score"] = stats["score_rank"] * 0.55 + stats["size_rank"] * 0.25 + stats["dd_rank"] * 0.20
    stats = stats.sort_values(["cluster_score", "score_q25", "cluster_size"], ascending=False, na_position="last")

    seeds: list[dict] = []
    top_clusters = stats.head(min(6, len(stats)))
    for _, cluster_row in top_clusters.iterrows():
        cid = int(cluster_row["cluster_id"])
        cluster_df = df[df["cluster_id"] == cid].copy()
        if cluster_df.empty:
            continue
        cluster_df = cluster_df.sort_values(["score", "cluster_dist"], ascending=[False, True], na_position="last")
        rep = cluster_df.iloc[0]
        seeds.append(
            {
                "cluster_id": cid,
                "cluster_score": float(cluster_row["cluster_score"]),
                "label_profit_thr": float(rep.get("label_profit_thr")),
                "exit_ema_span_min": int(float(rep.get("exit_ema_span_min"))),
                "exit_ema_init_offset_pct": float(rep.get("exit_ema_init_offset_pct")),
                "entry_ratio_neg_per_pos": float(rep.get("entry_ratio_neg_per_pos")),
                "calib_tail_blend": float(rep.get("calib_tail_blend")),
                "calib_tail_boost": float(rep.get("calib_tail_boost")),
                "tau_entry": float(rep.get("tau_entry")),
                "total_exposure": float(rep.get("total_exposure")),
                "max_trade_exposure": float(rep.get("max_trade_exposure")),
                "source_label_id": str(rep.get("label_id") or ""),
                "source_model_id": str(rep.get("model_id") or ""),
                "source_backtest_id": str(rep.get("backtest_id") or ""),
            }
        )
    return seeds


def _sample_label_cfg(seq_idx: int, s: ExploreSettings) -> dict[str, object]:
    return {
        "label_profit_thr": float(_sample_choice(s.label_profit_choices, seq_idx, base=2, salt=3)),
        "exit_span_min": int(_sample_choice(s.exit_span_choices, seq_idx, base=3, salt=11)),
        "exit_offset": float(_sample_choice(s.exit_offset_choices, seq_idx, base=5, salt=23)),
    }


def _sample_train_cfg(seq_idx: int, s: ExploreSettings) -> dict[str, object]:
    return {
        "neg_per_pos": float(_sample_choice(s.neg_pos_choices, seq_idx, base=7, salt=5)),
        "tail_blend": float(_sample_choice(s.tail_blend_choices, seq_idx, base=11, salt=17)),
        "tail_boost": float(_sample_choice(s.tail_boost_choices, seq_idx, base=13, salt=29)),
        "top_qs": str(_sample_choice(s.top_q_presets, seq_idx, base=17, salt=37)),
        "top_min_count": int(_sample_choice(s.top_min_count_choices, seq_idx, base=19, salt=43)),
    }


def _sample_bt_cfg(seq_idx: int, s: ExploreSettings) -> dict[str, object]:
    tau_entry = float(_sample_choice(s.tau_entry_choices, seq_idx, base=23, salt=7))

    return {
        "tau_entry": float(tau_entry),
        "total_exposure": float(_sample_choice(s.total_exposure_choices, seq_idx, base=59, salt=19)),
        "max_trade_exposure": float(_sample_choice(s.max_trade_exposure_choices, seq_idx, base=61, salt=31)),
        "min_trade_exposure": float(_sample_choice(s.min_trade_exposure_choices, seq_idx, base=67, salt=41)),
        "exposure_multiplier": float(_sample_choice(s.exposure_multiplier_choices, seq_idx, base=71, salt=47)),
    }


def _sample_label_cfg_refine(seq_idx: int, s: ExploreSettings, seed: dict, generation: int) -> dict[str, object]:
    radius = _refine_radius(generation)
    return {
        "label_profit_thr": float(_neighbor_choice(s.label_profit_choices, seed.get("label_profit_thr"), seq_idx, base=2, salt=3, radius=radius)),
        "exit_span_min": int(_neighbor_choice(s.exit_span_choices, seed.get("exit_ema_span_min"), seq_idx, base=3, salt=11, radius=radius)),
        "exit_offset": float(_neighbor_choice(s.exit_offset_choices, seed.get("exit_ema_init_offset_pct"), seq_idx, base=5, salt=23, radius=radius)),
    }


def _sample_train_cfg_refine(seq_idx: int, s: ExploreSettings, seed: dict, generation: int) -> dict[str, object]:
    radius = _refine_radius(generation)
    return {
        "neg_per_pos": float(_neighbor_choice(s.neg_pos_choices, seed.get("entry_ratio_neg_per_pos"), seq_idx, base=7, salt=5, radius=radius)),
        "tail_blend": float(_neighbor_choice(s.tail_blend_choices, seed.get("calib_tail_blend"), seq_idx, base=11, salt=17, radius=radius)),
        "tail_boost": float(_neighbor_choice(s.tail_boost_choices, seed.get("calib_tail_boost"), seq_idx, base=13, salt=29, radius=radius)),
        "top_qs": str(_sample_choice(s.top_q_presets, seq_idx, base=17, salt=37)),
        "top_min_count": int(_sample_choice(s.top_min_count_choices, seq_idx, base=19, salt=43)),
    }


def _sample_bt_cfg_refine(seq_idx: int, s: ExploreSettings, seed: dict, generation: int) -> dict[str, object]:
    radius = _refine_radius(generation)
    return {
        "tau_entry": float(_neighbor_choice(s.tau_entry_choices, seed.get("tau_entry"), seq_idx, base=23, salt=7, radius=radius)),
        "total_exposure": float(_neighbor_choice(s.total_exposure_choices, seed.get("total_exposure"), seq_idx, base=59, salt=19, radius=radius)),
        "max_trade_exposure": float(_neighbor_choice(s.max_trade_exposure_choices, seed.get("max_trade_exposure"), seq_idx, base=61, salt=31, radius=radius)),
        "min_trade_exposure": float(_sample_choice(s.min_trade_exposure_choices, seq_idx, base=67, salt=41)),
        "exposure_multiplier": float(_sample_choice(s.exposure_multiplier_choices, seq_idx, base=71, salt=47)),
    }


def run(settings: ExploreSettings | None = None) -> None:
    s = settings or ExploreSettings()
    forced_step_days = _env_int("WF_EXPLORE_STEP_DAYS", 0)
    forced_period_days = _effective_forced_period_days(forced_step_days)
    base_sequence = int(_env_int("WF_EXPLORE_SEQUENCE_ID", 0))
    phase = _env_str("WF_EXPLORE_PHASE", "broad").strip().lower() or "broad"
    generation = int(_env_int("WF_EXPLORE_GENERATION", 1))
    label_start = max(1, int(_env_int("WF_EXPLORE_LABEL_START", 1)))
    label_count = max(1, int(_env_int("WF_EXPLORE_LABEL_COUNT", s.max_label_trials)))
    label_end = label_start + label_count - 1
    if base_sequence <= 0:
        base_sequence = int(forced_step_days or s.seed or 1)
    out_root = resolve_generated_path(s.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / s.results_csv
    loop_log = out_root / "loop.log"
    log = _Tee(loop_log)
    log.write(f"[explore] out_root={out_root}")
    log.write(
        f"[explore] seed={s.seed} seq_base={base_sequence} "
        f"phase={phase} generation={generation} labels={label_start}..{label_end} "
        f"retrains_per_label={s.retrains_per_label} backtests_per_retrain={s.backtests_per_retrain} "
        f"feature_preset={s.feature_preset}"
    )
    if forced_step_days > 0:
        log.write(
            f"[explore] fair-step mapping: step_days={int(forced_step_days)} "
            f"tail_days={_env_int('SNIPER_REMOVE_TAIL_DAYS', 0)} "
            f"effective_train_period_days={int(forced_period_days)}"
        )

    finished_set = set() # (stage, label_id, model_id, backtest_id)
    train_run_dirs = {} # (label_id, model_id) -> run_dir
    if results_csv.exists():
        try:
            df_old = _read_results_csv(results_csv)
            # Ensure columns are treated as strings to avoid issues
            for _, row in df_old.iterrows():
                st = str(row.get("stage", ""))
                lid = str(row.get("label_id", ""))
                mid = str(row.get("model_id", "")).replace("nan", "")
                bid = str(row.get("backtest_id", "")).replace("nan", "")
                if str(row.get("status", "")) == "ok":
                    finished_set.add((st, lid, mid, bid))
                    if st == "train" and lid and mid:
                        train_run_dirs[(lid, mid)] = str(row.get("train_run_dir", ""))
            log.write(f"[explore] Loaded {len(finished_set)} existing successful stages from CSV.")
        except Exception as e:
            log.write(f"[explore] Warning: Could not load existing CSV for resume: {e}")

    refine_seeds = _load_refine_seeds(results_csv, generation) if phase == "refine" else []
    if phase == "refine":
        log.write(f"[explore] refine seeds loaded: {len(refine_seeds)}")
    if phase == "refine" and not refine_seeds:
        log.write("[explore] refine requested but no prior viable clusters were found; falling back to broad sampling")
        phase = "broad"
    run_started = time.perf_counter()
    refresh_count = 0
    train_count = 0
    prepare_count = 0
    backtest_count = 0
    refresh_total_sec = 0.0
    train_total_sec = 0.0
    prepare_total_sec = 0.0
    backtest_total_sec = 0.0

    for label_idx in range(int(label_start), int(label_end) + 1):
        label_id = f"label_{label_idx:03d}"
        label_dir = out_root / label_id
        label_seq = int(base_sequence) * 10_000 + int(label_idx)
        label_seed = _pick_seed(refine_seeds, label_seq) if phase == "refine" else None
        label_cfg = (
            _sample_label_cfg_refine(label_seq, s, label_seed, generation)
            if label_seed is not None
            else _label_cfg_from_grid(int(label_idx) - 1, s)
        )
        
        if ("refresh", label_id, "", "") in finished_set:
            log.write(f"[explore] {label_id} refresh skip (already done)")
            status = "ok"
            label_symbols_manifest = ""
            try:
                syms_cached = _list_cached_symbols(int(s.candle_sec), str(s.feature_preset))
                if syms_cached:
                    label_symbols_manifest = _write_symbol_manifest(label_dir, syms_cached)
            except Exception:
                label_symbols_manifest = ""
        else:
            label_dir.mkdir(parents=True, exist_ok=True)
            contract = _build_contract(
                candle_sec=int(s.candle_sec),
                label_profit_thr=float(label_cfg["label_profit_thr"]),
                exit_span_min=int(label_cfg["exit_span_min"]),
                exit_offset=float(label_cfg["exit_offset"]),
            )
            start_utc = _utc_now_iso()
            refresh_started = time.perf_counter()
            status = "ok"
            err = ""
            refresh_info: dict[str, object] = {}
            label_symbols_manifest = ""
            try:
                log.write(f"[explore] {label_id} refresh start profit_thr={label_cfg['label_profit_thr']} exit_span={label_cfg['exit_span_min']} offset={label_cfg['exit_offset']}")
                refresh_info = _refresh_labels(contract, int(s.candle_sec), str(s.feature_preset))
                syms_cached = _list_cached_symbols(int(s.candle_sec), str(s.feature_preset))
                if syms_cached:
                    label_symbols_manifest = _write_symbol_manifest(label_dir, syms_cached)
            except Exception as e:
                status = "error"
                err = f"{type(e).__name__}: {e}"
                (label_dir / "refresh_error.txt").write_text(err, encoding="utf-8")
                log.write(f"[explore] {label_id} refresh error {err}")
            end_utc = _utc_now_iso()
            refresh_elapsed = time.perf_counter() - refresh_started
            refresh_count += 1
            refresh_total_sec += refresh_elapsed
            if status == "ok":
                log.write(
                    f"[explore] {label_id} refresh done ok={refresh_info.get('ok', 0)} fail={refresh_info.get('fail', 0)} "
                    f"dur={_fmt_duration(refresh_elapsed)} avg={_fmt_duration(refresh_total_sec / max(1, refresh_count))} "
                    f"symbols_cached={len(syms_cached) if 'syms_cached' in locals() else 0}"
                )
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
                    "phase": phase,
                    "generation": generation,
                    "source_cluster_id": (label_seed.get("cluster_id", "") if label_seed else ""),
                    "source_label_id": (label_seed.get("source_label_id", "") if label_seed else ""),
                    "source_model_id": (label_seed.get("source_model_id", "") if label_seed else ""),
                    "source_backtest_id": (label_seed.get("source_backtest_id", "") if label_seed else ""),
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
            train_seq = int(base_sequence) * 100_000 + ((int(label_idx) - 1) * int(s.retrains_per_label)) + int(model_idx)
            train_seed = _pick_seed(refine_seeds, train_seq) if phase == "refine" else label_seed
            train_cfg = (
                _sample_train_cfg_refine(train_seq, s, train_seed, generation)
                if train_seed is not None
                else _train_cfg_from_grid(train_seq - 1, s)
            )
            forced_period = int(forced_period_days) if int(forced_period_days) > 0 else None

            train_status = "ok"
            train_run_dir = ""
            should_skip_train = ("train", label_id, model_id, "") in finished_set
            if should_skip_train:
                candidate_run_dir = str(train_run_dirs.get((label_id, model_id), "") or "").strip()
                skip_reason = ""
                if not candidate_run_dir:
                    skip_reason = "missing cached train_run_dir"
                elif forced_period is not None:
                    try:
                        _validate_required_period_meta(candidate_run_dir, int(forced_period))
                    except Exception as e:
                        skip_reason = str(e)
                else:
                    try:
                        _infer_max_period_days(candidate_run_dir)
                    except Exception as e:
                        skip_reason = str(e)
                if skip_reason:
                    should_skip_train = False
                    log.write(f"[explore] {label_id}/{model_id} train re-run required ({skip_reason})")
                else:
                    log.write(f"[explore] {label_id}/{model_id} train skip (already done)")
                    train_run_dir = candidate_run_dir

            if not should_skip_train:
                model_dir.mkdir(parents=True, exist_ok=True)
                train_start = _utc_now_iso()
                train_started = time.perf_counter()
                train_err = ""
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
                        "TRAIN_MIN_SYMBOLS_USED_PER_PERIOD": ("10" if forced_period is not None else "30"),
                        "TRAIN_FEATURE_PRESET": str(s.feature_preset),
                        "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BLEND": str(train_cfg["tail_blend"]),
                        "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BOOST": str(train_cfg["tail_boost"]),
                        "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_QS": str(train_cfg["top_qs"]),
                        "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_MIN_COUNT": str(train_cfg["top_min_count"]),
                    }
                    if label_symbols_manifest:
                        env["TRAIN_SYMBOLS_FILE"] = str(label_symbols_manifest)
                    if forced_period is not None:
                        active_offsets = [int(forced_period)]
                    else:
                        num_segments = max(1, math.ceil(int(s.days) / 180))
                        active_offsets = [(i + 1) * 180 for i in range(num_segments)]
                    env["TRAIN_OFFSETS_DAYS"] = ",".join(map(str, active_offsets))
                    
                    train_run_dir = _run_train_subprocess(env, model_dir / "train.log")
                    if forced_period is not None:
                        _validate_required_period_meta(train_run_dir, int(forced_period))
                    else:
                        _infer_max_period_days(train_run_dir)
                except Exception as e:
                    train_status = "error"
                    train_err = f"{type(e).__name__}: {e}"
                    log.write(f"[explore] {label_id}/{model_id} train error {train_err}")
                train_end = _utc_now_iso()
                train_elapsed = time.perf_counter() - train_started
                train_count += 1
                train_total_sec += train_elapsed
                if train_status == "ok":
                    log.write(
                        f"[explore] {label_id}/{model_id} train done run_dir={train_run_dir} "
                        f"dur={_fmt_duration(train_elapsed)} avg={_fmt_duration(train_total_sec / max(1, train_count))}"
                    )
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
                        "phase": phase,
                        "generation": generation,
                        "source_cluster_id": (train_seed.get("cluster_id", "") if train_seed else ""),
                        "source_label_id": (train_seed.get("source_label_id", "") if train_seed else ""),
                        "source_model_id": (train_seed.get("source_model_id", "") if train_seed else ""),
                        "source_backtest_id": (train_seed.get("source_backtest_id", "") if train_seed else ""),
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
            forced_period = int(forced_period) if forced_period is not None else _infer_max_period_days(train_run_dir)
            period_train_end = None
            for bt_idx in range(1, int(s.backtests_per_retrain) + 1):
                bt_id = f"bt_{bt_idx:03d}"
                bt_dir = model_dir / bt_id
                bt_seq = (
                    int(base_sequence) * 1_000_000
                    + (((int(label_idx) - 1) * int(s.retrains_per_label) + (int(model_idx) - 1)) * int(s.backtests_per_retrain))
                    + int(bt_idx)
                )
                bt_seed = _pick_seed(refine_seeds, bt_seq) if phase == "refine" else train_seed
                bt_cfg = (
                    _sample_bt_cfg_refine(bt_seq, s, bt_seed, generation)
                    if bt_seed is not None
                    else _bt_cfg_from_grid(int(bt_idx) - 1, s)
                )
                if ("backtest", label_id, model_id, bt_id) in finished_set:
                    continue
                
                if prepared_portfolio is None:
                    try:
                        prepare_started = time.perf_counter()
                        period_train_end, period_test_end = _resolve_period_window(
                            train_run_dir,
                            forced_period,
                            int(s.days),
                        )
                        prepare_cfg = _default_portfolio_cfg()
                        prepare_cfg.corr_filter_enabled = False
                        prepare_cfg.corr_open_filter_enabled = False
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
                            rebuild_on_score_error=True,
                            align_global_window=True,
                            force_period_days=(int(forced_period),),
                            explicit_window_start=str(period_train_end),
                            explicit_window_end=str(period_test_end),
                        )
                        log.write(f"[explore] {label_id}/{model_id} prepare start")
                        prepared_portfolio = prepare_portfolio_data(prepare_settings)
                        prepare_elapsed = time.perf_counter() - prepare_started
                        prepare_count += 1
                        prepare_total_sec += prepare_elapsed
                        log.write(
                            f"[explore] {label_id}/{model_id} prepare done symbols={prepared_portfolio.symbols_total} "
                            f"tau_default={prepared_portfolio.tau_entry_default:.2f} "
                            f"window={prepared_portfolio.window_info} "
                            f"dur={_fmt_duration(prepare_elapsed)} avg={_fmt_duration(prepare_total_sec / max(1, prepare_count))}"
                        )
                    except Exception as e:
                        prepare_err = f"{type(e).__name__}: {e}"
                        (model_dir / "prepare_error.txt").write_text(prepare_err, encoding="utf-8")
                        _append_csv(
                            results_csv,
                            {
                                "label_id": label_id,
                                "model_id": model_id,
                                "backtest_id": "",
                                "train_id": f"{label_id}/{model_id}",
                                "stage": "prepare",
                                "status": "error",
                                "start_utc": _utc_now_iso(),
                                "end_utc": _utc_now_iso(),
                                "duration_sec": 0.0,
                                "seed": s.seed,
                                "phase": phase,
                                "generation": generation,
                                "source_cluster_id": (bt_seed.get("cluster_id", "") if bt_seed else ""),
                                "source_label_id": (bt_seed.get("source_label_id", "") if bt_seed else ""),
                                "source_model_id": (bt_seed.get("source_model_id", "") if bt_seed else ""),
                                "source_backtest_id": (bt_seed.get("source_backtest_id", "") if bt_seed else ""),
                                "train_run_dir": train_run_dir,
                                "window_info": f"{period_train_end}..{period_test_end}",
                                "forced_period_days": forced_period,
                                "period_train_end_utc": str(period_train_end),
                                "error": prepare_err,
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
                        log.write(f"[explore] {label_id}/{model_id} prepare error {prepare_err}")
                        break # Skip this model's backtests if prepare fails
                
                bt_dir.mkdir(parents=True, exist_ok=True)
                bt_start = _utc_now_iso()
                bt_started = time.perf_counter()
                bt_status = "ok"
                bt_err = ""
                metrics: dict[str, float] = {}
                score = 0.0
                top_symbols = ""
                equity_html = ""
                try:
                    cfg = _default_portfolio_cfg()
                    # max_positions is intentionally disabled for new explores.
                    # Exposure budget provides the practical cap on simultaneous entries.
                    cfg.max_positions = 0
                    cfg.total_exposure = float(bt_cfg["total_exposure"])
                    cfg.max_trade_exposure = float(bt_cfg["max_trade_exposure"])
                    cfg.min_trade_exposure = float(bt_cfg["min_trade_exposure"])
                    cfg.corr_filter_enabled = False
                    cfg.corr_open_filter_enabled = False
                    cfg.exposure_multiplier = float(bt_cfg.get("exposure_multiplier", 0.0) or 0.0)
                    contract_bt = _build_contract(
                        candle_sec=int(s.candle_sec),
                        label_profit_thr=float(label_cfg["label_profit_thr"]),
                        exit_span_min=int(label_cfg["exit_span_min"]),
                        exit_offset=float(label_cfg["exit_offset"]),
                    )
                    log.write(
                        f"[explore] {label_id}/{model_id}/{bt_id} backtest start tau={bt_cfg['tau_entry']} "
                        f"total_exp={bt_cfg['total_exposure']:.2f} max_trade={bt_cfg['max_trade_exposure']:.2f} "
                        f"phase={phase} gen={generation} "
                        f"seed_cluster={bt_seed.get('cluster_id', '') if bt_seed else ''}"
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
                bt_elapsed = time.perf_counter() - bt_started
                backtest_count += 1
                backtest_total_sec += bt_elapsed
                if bt_status == "ok":
                    log.write(
                        f"[explore] {label_id}/{model_id}/{bt_id} timing dur={_fmt_duration(bt_elapsed)} "
                        f"avg={_fmt_duration(backtest_total_sec / max(1, backtest_count))}"
                    )
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
                        "phase": phase,
                        "generation": generation,
                        "source_cluster_id": (bt_seed.get("cluster_id", "") if bt_seed else ""),
                        "source_label_id": (bt_seed.get("source_label_id", "") if bt_seed else ""),
                        "source_model_id": (bt_seed.get("source_model_id", "") if bt_seed else ""),
                        "source_backtest_id": (bt_seed.get("source_backtest_id", "") if bt_seed else ""),
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
                        "window_info": (prepared_portfolio.window_info if prepared_portfolio is not None else ""),
                        "forced_period_days": int(forced_period),
                        "period_train_end_utc": (str(period_train_end) if period_train_end is not None else ""),
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
                        "max_positions": "",
                        "total_exposure": bt_cfg["total_exposure"],
                        "max_trade_exposure": bt_cfg["max_trade_exposure"],
                        "min_trade_exposure": bt_cfg["min_trade_exposure"],
                        "exposure_multiplier": bt_cfg["exposure_multiplier"],
                        "month_mean": metrics.get("month_mean", ""),
                        "month_p50": metrics.get("month_p50", ""),
                        "month_pos_frac": metrics.get("month_pos_frac", ""),
                        "month_worst": metrics.get("month_worst", ""),
                        "month_best": metrics.get("month_best", ""),
                        "max_neg_month_streak": metrics.get("max_neg_month_streak", ""),
                        "underwater_frac": metrics.get("underwater_frac", ""),
                        "worst_rolling_90d": metrics.get("worst_rolling_90d", ""),
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
    total_elapsed = time.perf_counter() - run_started
    log.write(
        "[explore] generation summary "
        f"phase={phase} gen={generation} labels={label_start}..{label_end} "
        f"total={_fmt_duration(total_elapsed)} "
        f"refresh={refresh_count}x/{_fmt_duration(refresh_total_sec)} "
        f"train={train_count}x/{_fmt_duration(train_total_sec)} "
        f"prepare={prepare_count}x/{_fmt_duration(prepare_total_sec)} "
        f"backtest={backtest_count}x/{_fmt_duration(backtest_total_sec)}"
    )


def main() -> None:
    s = ExploreSettings(
        out_root=_env_str("WF_EXPLORE_OUT_ROOT", "wf_portfolio_explore"),
        results_csv=_env_str("WF_EXPLORE_RESULTS_CSV", "explore_runs.csv"),
        seed=_env_int("WF_EXPLORE_SEED", 42),
        max_label_trials=_env_int("WF_EXPLORE_LABEL_TRIALS", 56),
        retrains_per_label=_env_int("WF_EXPLORE_RETRAINS_PER_LABEL", 2),
        backtests_per_retrain=_env_int("WF_EXPLORE_BACKTESTS_PER_RETRAIN", 21),
        days=_env_int("WF_EXPLORE_DAYS", 4 * 360),
        max_symbols=_env_int("WF_EXPLORE_MAX_SYMBOLS", 0),
        candle_sec=_env_int("WF_EXPLORE_CANDLE_SEC", 300),
        safe_threads=_env_int("WF_EXPLORE_SAFE_THREADS", 8),
        feature_preset=_env_str("WF_EXPLORE_FEATURE_PRESET", "core80"),
    )
    run(s)


if __name__ == "__main__":
    main()
