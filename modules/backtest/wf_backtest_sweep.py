# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Run multiple WF portfolio backtests with random params, using a fixed train run_dir.
No retraining. Optionally refresh feature cache once before the sweep.
"""

from dataclasses import dataclass, field
from pathlib import Path
import ast
import argparse
import csv
import os
import random
import re
import subprocess
import sys
import time
import json

import numpy as np
import pandas as pd


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from backtest.wf_backtest_params import BacktestOptimizedRanges  # noqa: E402
from utils.paths import resolve_generated_path  # noqa: E402
from config.symbols import load_top_market_cap_symbols  # noqa: E402


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().tz_localize(None).isoformat()


def _rand_float(rng: random.Random, lo: float, hi: float, step: float | None = None, decimals: int = 6) -> float:
    if step and step > 0:
        n = int(round((hi - lo) / step))
        val = lo + step * rng.randint(0, max(0, n))
        return float(round(val, decimals))
    return float(round(rng.uniform(lo, hi), decimals))


def _rand_int(rng: random.Random, lo: int, hi: int, step: int | None = None) -> int:
    if step and step > 0:
        n = int((hi - lo) // step)
        return int(lo + step * rng.randint(0, max(0, n)))
    return int(rng.randint(lo, hi))


def _append_csv(path: Path, row: dict, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        out = {k: row.get(k, "") for k in header}
        w.writerow(out)


def _max_drawdown(eq: np.ndarray) -> float:
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    return float(abs(np.min(dd)))


def _compute_metrics(out_dir: Path) -> dict:
    eq_end = 1.0
    ret_pct = 0.0
    max_dd = 0.0
    top_symbols = ""

    eq_path = out_dir / "wf_equity_curve.csv"
    if eq_path.exists():
        eq_df = pd.read_csv(eq_path)
        if "equity" in eq_df.columns and len(eq_df) > 0:
            eq = eq_df["equity"].to_numpy(np.float64, copy=False)
            eq_end = float(eq[-1])
            ret_pct = float((eq_end - 1.0) * 100.0)
            max_dd = _max_drawdown(eq)

    trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    trades_path = out_dir / "wf_trades.csv"
    if trades_path.exists():
        df_tr = pd.read_csv(trades_path, usecols=["pnl_w"])
        pnl = df_tr["pnl_w"].to_numpy(np.float64, copy=False)
        trades = int(len(pnl))
        if trades > 0:
            wins = float(np.sum(pnl[pnl > 0.0]))
            losses = float(np.sum(-pnl[pnl < 0.0]))
            win_rate = float(np.mean(pnl > 0.0))
            profit_factor = float(wins / max(1e-12, losses)) if losses > 0.0 else float("inf")

    sym_path = out_dir / "wf_symbol_stats.csv"
    if sym_path.exists():
        try:
            df_sym = pd.read_csv(sym_path)
            if "symbol" in df_sym.columns and "pnl_w_sum" in df_sym.columns:
                g = df_sym.groupby("symbol")["pnl_w_sum"].sum().sort_values(ascending=False)
                top_symbols = ",".join([str(s) for s in g.head(10).index.to_list()])
        except Exception:
            pass

    return {
        "eq_end": float(eq_end),
        "ret_pct": float(ret_pct),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "trades": int(trades),
        "top_symbols": top_symbols,
    }


def _parse_sweep_start_params(loop_log: Path) -> dict[str, dict]:
    params_by_run: dict[str, dict] = {}
    if not loop_log.exists():
        return params_by_run
    pat = re.compile(r"^\[sweep\]\s+start\s+(\S+)\s+params=(\{.*\})\s*$")
    try:
        for line in loop_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.match(line.strip())
            if not m:
                continue
            run_id = m.group(1)
            raw = m.group(2)
            try:
                params = ast.literal_eval(raw)
                if isinstance(params, dict):
                    params_by_run[run_id] = params
            except Exception:
                continue
    except Exception:
        return params_by_run
    return params_by_run


def _parse_run_id_start_utc(run_id: str) -> str:
    m = re.match(r"^sweep_\d+_(\d{8})_(\d{6})$", str(run_id))
    if not m:
        return ""
    try:
        ts = pd.to_datetime(f"{m.group(1)}_{m.group(2)}", format="%Y%m%d_%H%M%S")
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return ""


def _guess_end_utc(out_dir: Path) -> str:
    cand = [
        out_dir / "wf_equity_curve.csv",
        out_dir / "wf_trades.csv",
        out_dir / "wf_steps.csv",
        out_dir / "wf_symbol_stats.csv",
    ]
    times: list[float] = []
    for p in cand:
        if p.exists():
            try:
                times.append(p.stat().st_mtime)
            except Exception:
                continue
    if not times:
        try:
            times.append(out_dir.stat().st_mtime)
        except Exception:
            return ""
    try:
        ts = pd.to_datetime(max(times), unit="s")
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return ""


def rebuild_results_csv(settings: SweepSettings | None = None) -> Path | None:
    s = settings or SweepSettings()
    out_root = resolve_generated_path(s.out_root)
    if not out_root.exists():
        return None
    results_csv = out_root / s.results_csv
    loop_log = out_root / "loop.log"
    params_by_run = _parse_sweep_start_params(loop_log)
    header = list(DASH_RESULTS_HEADER)

    run_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("sweep_")])
    if not run_dirs:
        return None

    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    if results_csv.exists():
        backup = results_csv.with_name(f"{results_csv.stem}_backup_{stamp}{results_csv.suffix}")
        try:
            results_csv.rename(backup)
        except Exception:
            pass

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for out_dir in run_dirs:
            run_id = out_dir.name
            bt_params = params_by_run.get(run_id, {})
            metrics = _compute_metrics(out_dir)
            equity_png_path = out_dir / "wf_equity.png"
            equity_png = str(equity_png_path) if equity_png_path.exists() else ""
            start_utc = _parse_run_id_start_utc(run_id)
            end_utc = _guess_end_utc(out_dir)
            duration_sec = 0.0
            if start_utc and end_utc:
                try:
                    duration_sec = float((pd.to_datetime(end_utc) - pd.to_datetime(start_utc)).total_seconds())
                except Exception:
                    duration_sec = 0.0
            row = {
                "train_id": run_id,
                "backtest_id": "bt_01",
                "stage": "backtest",
                "status": "ok",
                "start_utc": start_utc,
                "end_utc": end_utc,
                "duration_sec": float(max(0.0, duration_sec)),
                "seed": "",
                "train_run_dir": str(s.run_dir),
                "bt_out_dir": str(out_dir),
                "equity_png": equity_png,
                "eq_end": metrics.get("eq_end", ""),
                "ret_pct": metrics.get("ret_pct", ""),
                "max_dd": metrics.get("max_dd", ""),
                "win_rate": metrics.get("win_rate", ""),
                "profit_factor": metrics.get("profit_factor", ""),
                "trades": metrics.get("trades", ""),
                "top_symbols": metrics.get("top_symbols", ""),
                "error": "",
                "entry_window_min": "",
                "exit_ema_span": "",
                "exit_ema_init_offset_pct": "",
                "train_total_days": "",
                "train_offsets_years": "",
                "train_offsets_step_days": "",
                "train_max_symbols": "",
                "train_min_symbols_used_per_period": "",
                "train_max_rows_entry": "",
                "train_xgb_device": "",
                "tau_entry": bt_params.get("tau_entry", ""),
                "bt_years": int(s.years),
                "bt_step_days": int(s.step_days),
                "bt_bar_stride": int(s.bar_stride),
                "bt_max_symbols": int(s.max_symbols),
                "bt_max_positions": bt_params.get("max_positions", ""),
                "bt_total_exposure": bt_params.get("total_exposure", ""),
                "bt_max_trade_exposure": bt_params.get("max_trade_exposure", ""),
                "bt_min_trade_exposure": bt_params.get("min_trade_exposure", ""),
                "bt_exit_confirm_bars": bt_params.get("exit_confirm_bars", ""),
                "bt_universe_history_mode": "rolling",
                "bt_universe_history_days": bt_params.get("universe_history_days", ""),
                "bt_universe_min_pf": bt_params.get("universe_min_pf", ""),
                "bt_universe_min_win": bt_params.get("universe_min_win", ""),
                "bt_universe_max_dd": bt_params.get("universe_max_dd", ""),
                "bt_step_cache_mode": str(s.step_cache_mode),
                "bt_plot_save_only": bool(s.plot_save_only),
            }
            out = {k: row.get(k, "") for k in header}
            w.writerow(out)
    return results_csv


# Mesmo schema esperado pelo dashboard (`modules/train/wf_dashboard_server.py`).
try:
    from train.wf_random_loop import RESULTS_HEADER as DASH_RESULTS_HEADER  # type: ignore
except Exception:
    DASH_RESULTS_HEADER = [
        "train_id",
        "backtest_id",
        "stage",
        "status",
        "start_utc",
        "end_utc",
        "duration_sec",
        "seed",
        "train_run_dir",
        "bt_out_dir",
        "equity_png",
        "eq_end",
        "ret_pct",
        "max_dd",
        "win_rate",
        "profit_factor",
        "trades",
        "top_symbols",
        "error",
        # contract params
        "entry_window_min",
        "exit_ema_span",
        "exit_ema_init_offset_pct",
        # train params
        "train_total_days",
        "train_offsets_years",
        "train_offsets_step_days",
        "train_max_symbols",
        "train_min_symbols_used_per_period",
        "train_max_rows_entry",
        "train_xgb_device",
        # backtest thresholds
        "tau_entry",
        # backtest params
        "bt_years",
        "bt_step_days",
        "bt_bar_stride",
        "bt_max_symbols",
        "bt_max_positions",
        "bt_total_exposure",
        "bt_max_trade_exposure",
        "bt_min_trade_exposure",
        "bt_exit_confirm_bars",
        "bt_universe_history_mode",
        "bt_universe_history_days",
        "bt_universe_min_pf",
        "bt_universe_min_win",
        "bt_universe_max_dd",
        "bt_step_cache_mode",
        "bt_plot_save_only",
    ]


def _ensure_dashboard_csv_compatible(path: Path) -> None:
    """
    O dashboard usa o header (1ª linha) do CSV para ler as colunas.
    Se existir um CSV legado do sweep (sem train_id/stage), renomeia para não "sumir" no dashboard.
    """
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip("\r\n")
    except Exception:
        return
    if not first:
        return
    cols = [c.strip() for c in first.split(",") if c.strip()]
    if "train_id" in cols and "stage" in cols:
        return
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    legacy = path.with_name(f"{path.stem}_legacy_{stamp}{path.suffix}")
    try:
        path.rename(legacy)
    except Exception:
        pass


@dataclass
class SweepSettings:
    out_root: str = "wf_backtest_sweep"
    results_csv: str = "sweep_runs.csv"
    max_runs: int = 10
    run_dir: str = "D:/astra/models_sniper/crypto/wf_002"
    contract_json: str = ""
    years: int = 6
    step_days: int = 180
    bar_stride: int = 1
    max_symbols: int = 0
    min_market_cap: float = 50_000_000
    step_cache_mode: str = "memory"
    plot_save_only: bool = True
    refresh_cache_once: bool = False
    start_dashboard: bool = True
    dashboard_port: int = 5055
    use_run_dir_symbols: bool = True
    fixed_tau_entry: float = 0.775
    backtest_ranges: BacktestOptimizedRanges = field(default_factory=BacktestOptimizedRanges)


def _load_symbols_from_run_dir(run_dir: str) -> list[str]:
    base = Path(run_dir).expanduser().resolve()
    if not base.is_dir():
        return []
    meta_path = base / "period_0d" / "meta.json"
    if not meta_path.exists():
        for cand in sorted(base.glob("period_*")):
            mp = cand / "meta.json"
            if mp.exists():
                meta_path = mp
                break
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    for key in ("symbols_used", "symbols"):
        vals = meta.get(key)
        if isinstance(vals, list) and vals:
            out = [str(s).strip().upper() for s in vals if str(s).strip()]
            return out
    return []


def _sample_backtest_params(rng: random.Random, s: SweepSettings) -> dict:
    r = s.backtest_ranges
    max_positions = _rand_int(rng, *r.max_positions_range)
    total_exposure = _rand_float(rng, *r.total_exposure_range, decimals=3)
    max_trade = _rand_float(rng, *r.max_trade_exposure_range, decimals=3)
    min_trade_lo = float(r.min_trade_exposure_range[0])
    min_trade_hi = float(min(max_trade, r.min_trade_exposure_range[1]))
    if min_trade_hi < min_trade_lo:
        min_trade = min_trade_hi
    else:
        min_trade = _rand_float(rng, min_trade_lo, min_trade_hi, r.min_trade_exposure_range[2], decimals=3)
    exit_confirm_bars = _rand_int(rng, *r.exit_confirm_bars_range)
    universe_history_days = _rand_int(rng, *r.universe_history_days_range)
    universe_min_pf = _rand_float(rng, *r.universe_min_pf_range, decimals=3)
    universe_min_win = _rand_float(rng, *r.universe_min_win_range, decimals=3)
    universe_max_dd = _rand_float(rng, *r.universe_max_dd_range, decimals=3)
    tau_entry = float(getattr(s, "fixed_tau_entry", 0.0) or 0.0)
    if tau_entry <= 0.0:
        tau_entry = _rand_float(rng, *r.tau_entry_range, decimals=3)
    return {
        "max_positions": int(max_positions),
        "total_exposure": float(total_exposure),
        "max_trade_exposure": float(max_trade),
        "min_trade_exposure": float(min_trade),
        "exit_confirm_bars": int(exit_confirm_bars),
        "universe_history_days": int(universe_history_days),
        "universe_min_pf": float(universe_min_pf),
        "universe_min_win": float(universe_min_win),
        "universe_max_dd": float(universe_max_dd),
        "tau_entry": float(tau_entry),
    }


def _run_backtest(out_dir: Path, s: SweepSettings, bt_params: dict, *, refresh_cache: bool, log_fh, symbols_csv: str | None) -> None:
    wf_script = (Path(__file__).resolve().parents[1] / "backtest" / "wf_portfolio.py").resolve()
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(wf_script),
        "--run-dir",
        str(s.run_dir),
        "--out-dir",
        str(out_dir),
        "--years",
        str(int(s.years)),
        "--step-days",
        str(int(s.step_days)),
        "--bar-stride",
        str(int(s.bar_stride)),
        "--max-symbols",
        str(int(s.max_symbols)),
        "--min-market-cap",
        str(float(s.min_market_cap)),
        "--max-positions",
        str(int(bt_params["max_positions"])),
        "--total-exposure",
        str(float(bt_params["total_exposure"])),
        "--max-trade-exposure",
        str(float(bt_params["max_trade_exposure"])),
        "--min-trade-exposure",
        str(float(bt_params["min_trade_exposure"])),
        "--exit-confirm-bars",
        str(int(bt_params["exit_confirm_bars"])),
        "--universe-history-mode",
        "rolling",
        "--universe-history-days",
        str(int(bt_params["universe_history_days"])),
        "--universe-min-pf",
        str(float(bt_params["universe_min_pf"])),
        "--universe-min-win",
        str(float(bt_params["universe_min_win"])),
        "--universe-max-dd",
        str(float(bt_params["universe_max_dd"])),
        "--step-cache-mode",
        str(s.step_cache_mode),
        "--tau-entry",
        str(float(bt_params["tau_entry"])),
    ]
    if symbols_csv:
        cmd.extend(["--symbols", symbols_csv])
    if s.contract_json:
        cmd.extend(["--contract-json", str(s.contract_json)])
    if bool(s.plot_save_only):
        cmd.append("--plot-save-only")
    if refresh_cache:
        cmd.append("--refresh-cache")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def run(settings: SweepSettings | None = None) -> None:
    s = settings or SweepSettings()
    out_root = resolve_generated_path(s.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / s.results_csv
    loop_log = out_root / "loop.log"
    dash_log = out_root / "dash.log"

    _ensure_dashboard_csv_compatible(results_csv)
    header = list(DASH_RESULTS_HEADER)

    dash_proc = None
    if bool(s.start_dashboard):
        dash_script = (Path(__file__).resolve().parents[1] / "train" / "wf_dashboard_ngrok_monolith.py").resolve()
        env = os.environ.copy()
        env["WF_DASH_OUT_ROOT"] = str(out_root)
        env["WF_DASH_RESULTS_CSV"] = str(results_csv)
        env["WF_DASH_LOOP_LOG"] = str(loop_log)
        env["WF_DASH_LOG"] = str(dash_log)
        env["WF_NGROK_PORT"] = str(int(s.dashboard_port))
        env["WF_DASH_HOST"] = "127.0.0.1"
        dash_log.parent.mkdir(parents=True, exist_ok=True)
        dash_fh = dash_log.open("a", encoding="utf-8", newline="")
        dash_proc = subprocess.Popen(
            [sys.executable, str(dash_script)],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=dash_fh,
            stderr=dash_fh,
        )
        print(f"[sweep] dashboard (ngrok): https://<seu-dominio-ngrok>", flush=True)

    symbols_csv = ""
    run_syms: list[str] = []
    if bool(s.use_run_dir_symbols):
        run_syms = _load_symbols_from_run_dir(str(s.run_dir))
        if run_syms:
            print(f"[sweep] symbols from run_dir: {len(run_syms)}", flush=True)
    cap_syms = load_top_market_cap_symbols(
        min_cap=float(getattr(s, "min_market_cap", 0.0) or 0.0) or None,
    )
    if cap_syms:
        print(f"[sweep] symbols from top_market_cap: {len(cap_syms)} (min_cap={s.min_market_cap})", flush=True)
    if run_syms or cap_syms:
        merged: list[str] = []
        seen = set()
        for sym in cap_syms + run_syms:
            s2 = str(sym).strip().upper()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            merged.append(s2)
        symbols_csv = ",".join(merged)
    rng = random.Random(int(time.time_ns() % (2**31 - 1)))
    with loop_log.open("a", encoding="utf-8", newline="") as log_fh:
        for i in range(int(max(1, s.max_runs))):
            run_id = f"sweep_{i+1:03d}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
            seed = int(time.time_ns() % (2**31 - 1))
            out_dir = out_root / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            bt_params = _sample_backtest_params(rng, s)

            start_utc = _utc_now_iso()
            status = "ok"
            err_msg = ""
            metrics: dict = {}
            equity_png = ""
            log_fh.write(f"\n[sweep] start {run_id} params={bt_params}\n")
            log_fh.flush()
            try:
                _run_backtest(
                    out_dir,
                    s,
                    bt_params,
                    refresh_cache=bool(s.refresh_cache_once and i == 0),
                    log_fh=log_fh,
                    symbols_csv=symbols_csv or None,
                )
                metrics = _compute_metrics(out_dir)
                equity_png_path = out_dir / "wf_equity.png"
                if equity_png_path.exists():
                    equity_png = str(equity_png_path)
            except Exception as e:
                status = "error"
                err_msg = f"{type(e).__name__}: {e}"
            end_utc = _utc_now_iso()
            duration_sec = max(0.0, (pd.to_datetime(end_utc) - pd.to_datetime(start_utc)).total_seconds())

            row = {
                "train_id": run_id,
                "backtest_id": "bt_01",
                "stage": "backtest",
                "status": status,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "duration_sec": float(duration_sec),
                "seed": int(seed),
                "train_run_dir": str(s.run_dir),
                "bt_out_dir": str(out_dir),
                "equity_png": equity_png,
                "eq_end": metrics.get("eq_end", ""),
                "ret_pct": metrics.get("ret_pct", ""),
                "max_dd": metrics.get("max_dd", ""),
                "win_rate": metrics.get("win_rate", ""),
                "profit_factor": metrics.get("profit_factor", ""),
                "trades": metrics.get("trades", ""),
                "top_symbols": metrics.get("top_symbols", ""),
                "error": err_msg,
                # contract params (não aplicável aqui)
                "entry_window_min": "",
                "exit_ema_span": "",
                "exit_ema_init_offset_pct": "",
                # train params (não aplicável aqui)
                "train_total_days": "",
                "train_offsets_years": "",
                "train_offsets_step_days": "",
                "train_max_symbols": "",
                "train_min_symbols_used_per_period": "",
                "train_max_rows_entry": "",
                "train_xgb_device": "",
                # backtest thresholds
                "tau_entry": bt_params.get("tau_entry", ""),
                # backtest params
                "bt_years": int(s.years),
                "bt_step_days": int(s.step_days),
                "bt_bar_stride": int(s.bar_stride),
                "bt_max_symbols": int(s.max_symbols),
                "bt_max_positions": bt_params.get("max_positions", ""),
                "bt_total_exposure": bt_params.get("total_exposure", ""),
                "bt_max_trade_exposure": bt_params.get("max_trade_exposure", ""),
                "bt_min_trade_exposure": bt_params.get("min_trade_exposure", ""),
                "bt_exit_confirm_bars": bt_params.get("exit_confirm_bars", ""),
                "bt_universe_history_mode": "rolling",
                "bt_universe_history_days": bt_params.get("universe_history_days", ""),
                "bt_universe_min_pf": bt_params.get("universe_min_pf", ""),
                "bt_universe_min_win": bt_params.get("universe_min_win", ""),
                "bt_universe_max_dd": bt_params.get("universe_max_dd", ""),
                "bt_step_cache_mode": str(s.step_cache_mode),
                "bt_plot_save_only": bool(s.plot_save_only),
            }
            _append_csv(results_csv, row, header)
            log_fh.write(f"[sweep] done {run_id} status={status} ret={row.get('ret_pct')} dd={row.get('max_dd')}\n")
            log_fh.flush()

    if dash_proc is not None:
        try:
            dash_proc.terminate()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-results", action="store_true", help="Rebuild sweep_runs.csv from existing sweep_* dirs.")
    args = ap.parse_args()
    if bool(getattr(args, "rebuild_results", False)):
        rebuild_results_csv()
        return
    run()


if __name__ == "__main__":
    main()
