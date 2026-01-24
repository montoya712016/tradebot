# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Random loop:
- sample contract params (labels)
- refresh labels
- train WF (entry)
- run multiple backtests with same models (vary thresholds)
- append metrics to CSV
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import json
import os
import random
import subprocess
import sys
import time
import traceback

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

from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract
from prepare_features.refresh_sniper_labels_in_cache import RefreshLabelsSettings, run as refresh_labels
from train.wf_train_params import TrainOptimizedContractRanges
from backtest.wf_backtest_params import BacktestOptimizedRanges
from train.sniper_trainer import (
    TrainConfig,
    train_sniper_models,
    DEFAULT_ENTRY_PARAMS,
)
from utils.paths import resolve_generated_path


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


def _find_latest_wf_dir() -> Path | None:
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        asset = os.getenv("SNIPER_ASSET_CLASS", "crypto")
        models_root = _models_root_for_asset(asset).resolve()
    except Exception:
        asset = os.getenv("SNIPER_ASSET_CLASS", "crypto").strip().lower()
        models_root = (Path(__file__).resolve().parents[2].parent / "models_sniper" / asset).resolve()
    if not models_root.is_dir():
        return None
    wf_list = sorted([p for p in models_root.glob("wf_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return wf_list[-1] if wf_list else None


def _load_contract_from_json(path: Path) -> TradeContract:
    base = DEFAULT_TRADE_CONTRACT
    if not path.exists():
        return base
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return replace(base, **data)
    except Exception:
        pass
    return base


def _compute_metrics(out_dir: Path) -> dict:
    eq_end = 1.0
    ret_pct = 0.0
    max_dd = 0.0
    win_rate = 0.0
    profit_factor = 0.0
    trades = 0
    top_symbols = ""

    eq_path = out_dir / "wf_equity_curve.csv"
    if eq_path.exists():
        eq_df = pd.read_csv(eq_path)
        if "equity" in eq_df.columns and len(eq_df) > 0:
            eq = eq_df["equity"].to_numpy(np.float64, copy=False)
            eq_end = float(eq[-1])
            ret_pct = float((eq_end - 1.0) * 100.0)
            max_dd = _max_drawdown(eq)

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
        df_sym = pd.read_csv(sym_path)
        if "symbol" in df_sym.columns and "pnl_w_sum" in df_sym.columns:
            g = df_sym.groupby("symbol")["pnl_w_sum"].sum().sort_values(ascending=False)
            top_symbols = ",".join([str(s) for s in g.head(10).index.to_list()])

    return {
        "eq_end": float(eq_end),
        "ret_pct": float(ret_pct),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "trades": int(trades),
        "top_symbols": top_symbols,
    }


RESULTS_HEADER = [
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
    "entry_min_profit",
    "entry_weight_alpha",
    "exit_ema_span",
    "exit_ema_init_offset_pct",
    # train params
    "entry_ratio_neg_per_pos",
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


@dataclass
class RandomLoopSettings:
    # output
    out_root: str = "wf_random_loop"
    results_csv: str = "random_runs.csv"
    max_runs: int = 0  # 0 => infinite
    sleep_seconds_on_error: int = 30

    # fixed params (nao otimizados pelo wf_random_loop)
    candle_sec: int = 60
    refresh_verbose: bool = True
    skip_refresh: bool = True
    rebuild_cache: bool = True
    rebuild_cache_mode: str = "once"
    total_days: int = 0
    offsets_step_days: int = 180
    offsets_years: int = 6
    max_symbols_train: int = 100
    min_symbols_used_per_period: int = 30
    max_rows_entry: int = 2_000_000
    xgb_device: str = "cuda:0"
    entry_ratio_neg_per_pos: float = 6.0
    backtests_per_train: int = 5
    years: int = 6
    step_days: int = 180
    bar_stride: int = 1
    max_symbols_backtest: int = 100
    step_cache_mode: str = "memory"
    plot_save_only: bool = True
    universe_history_mode: str = "rolling"

    # otimizados pelo wf_random_loop
    train_contract_ranges: TrainOptimizedContractRanges = field(default_factory=TrainOptimizedContractRanges)
    backtest_ranges: BacktestOptimizedRanges = field(default_factory=BacktestOptimizedRanges)


def _sample_contract(rng: random.Random, s: RandomLoopSettings) -> tuple[TradeContract, dict]:
    r = s.train_contract_ranges
    win_lo, win_hi, win_step = r.entry_window
    w = _rand_int(rng, int(win_lo), int(win_hi), int(win_step))
    p = _rand_float(rng, *r.entry_min_profit, decimals=4)
    windows = (int(w),)
    profits = [float(p)]
    alpha = _rand_float(rng, *r.entry_weight_alpha, decimals=3)
    ema_span = int(max(1, round((float(w) * 60.0) / float(max(1, int(s.candle_sec))))))
    ema_offset = _rand_float(rng, *r.exit_ema_init_offset_pct, decimals=4)
    overrides = {
        "entry_label_windows_minutes": tuple(int(w) for w in windows),
        "entry_label_min_profit_pcts": tuple(float(p) for p in profits),
        "entry_label_weight_alpha": float(alpha),
        "exit_ema_span": int(ema_span),
        "exit_ema_init_offset_pct": float(ema_offset),
    }
    contract = replace(DEFAULT_TRADE_CONTRACT, **overrides)
    return contract, overrides


def _sample_backtest_params(rng: random.Random, s: RandomLoopSettings) -> dict:
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
        "tau_entry": _rand_float(rng, *r.tau_entry_range, decimals=3),
    }


def _run_backtest(out_dir: Path, run_dir: Path, contract_json: Path, bt_params: dict, s: RandomLoopSettings) -> None:
    wf_script = (Path(__file__).resolve().parents[1] / "backtest" / "wf_portfolio.py").resolve()
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(wf_script),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--years",
        str(int(s.years)),
        "--step-days",
        str(int(s.step_days)),
        "--bar-stride",
        str(int(s.bar_stride)),
        "--max-symbols",
        str(int(s.max_symbols_backtest)),
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
        str(s.universe_history_mode),
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
        "--contract-json",
        str(contract_json),
        "--tau-entry",
        str(float(bt_params["tau_entry"])),
        "--no-pushover",
    ]
    if bool(s.plot_save_only):
        cmd.append("--plot-save-only")
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def run(settings: RandomLoopSettings | None = None) -> None:
    s = settings or RandomLoopSettings()
    env_out_root = os.getenv("WF_OUT_ROOT", "").strip()
    if env_out_root:
        s.out_root = str(env_out_root)
    env_results_csv = os.getenv("WF_RESULTS_CSV", "").strip()
    if env_results_csv:
        s.results_csv = str(env_results_csv)
    env_skip = os.getenv("WF_SKIP_REFRESH", "").strip().lower()
    if env_skip in {"1", "true", "yes", "y", "on"}:
        s.skip_refresh = True
    env_force_refresh = os.getenv("WF_FORCE_REFRESH", "").strip().lower()
    if env_force_refresh in {"1", "true", "yes", "y", "on"}:
        s.skip_refresh = False
    env_rebuild = os.getenv("WF_REBUILD_CACHE", "").strip().lower()
    if env_rebuild in {"1", "true", "yes", "y", "on"}:
        s.rebuild_cache = True
    env_rebuild_mode = os.getenv("WF_REBUILD_CACHE_MODE", "").strip().lower()
    if env_rebuild_mode in {"once", "always", "off"}:
        s.rebuild_cache_mode = env_rebuild_mode
    env_max_train = os.getenv("WF_MAX_SYMBOLS_TRAIN", "").strip()
    if env_max_train:
        try:
            s.max_symbols_train = int(env_max_train)
        except Exception:
            pass
    env_max_bt = os.getenv("WF_MAX_SYMBOLS_BACKTEST", "").strip()
    if env_max_bt:
        try:
            s.max_symbols_backtest = int(env_max_bt)
        except Exception:
            pass
    env_min_used = os.getenv("WF_MIN_SYMBOLS_USED_PER_PERIOD", "").strip()
    if env_min_used:
        try:
            s.min_symbols_used_per_period = int(env_min_used)
        except Exception:
            pass
    env_skip_train = os.getenv("WF_SKIP_TRAIN", "").strip().lower()
    skip_train = env_skip_train in {"1", "true", "yes", "y", "on"}
    env_skip_train_mode = os.getenv("WF_SKIP_TRAIN_MODE", "").strip().lower()
    skip_train_mode = env_skip_train_mode if env_skip_train_mode in {"once", "always"} else "always"
    if skip_train:
        # no skip-train: reaproveita modelos existentes, entao nao precisa refresh de labels
        s.skip_refresh = True
    out_root = resolve_generated_path(s.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / s.results_csv

    header = list(RESULTS_HEADER)

    run_idx = 0
    skip_train_once_done = False
    cache_rebuilt_once = False
    while True:
        if s.max_runs > 0 and run_idx >= s.max_runs:
            break
        run_idx += 1
        seed = int(time.time_ns() % (2**31 - 1))
        rng = random.Random(seed)
        train_id = f"train_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{seed}"
        train_start_utc = _utc_now_iso()
        train_dir = out_root / train_id
        train_dir.mkdir(parents=True, exist_ok=True)
        contract, contract_overrides = _sample_contract(rng, s)
        if skip_train and (skip_train_mode == "always" or not skip_train_once_done):
            latest = _find_latest_wf_dir()
            if latest is None:
                raise RuntimeError("WF_SKIP_TRAIN ativo mas nenhum wf_* encontrado em models_sniper")
            contract_json = latest / "contract.json"
            contract = _load_contract_from_json(contract_json)
            contract_overrides = json.loads(contract_json.read_text(encoding="utf-8")) if contract_json.exists() else {}

        if bool(s.rebuild_cache):
            mode = str(getattr(s, "rebuild_cache_mode", "once") or "once").lower()
            if mode == "always":
                os.environ["SNIPER_CACHE_REFRESH"] = "1"
            elif mode == "once":
                if not cache_rebuilt_once:
                    os.environ["SNIPER_CACHE_REFRESH"] = "1"
                else:
                    os.environ["SNIPER_CACHE_REFRESH"] = "0"
            else:
                os.environ["SNIPER_CACHE_REFRESH"] = "0"
            print(
                f"[loop] cache_refresh={os.environ.get('SNIPER_CACHE_REFRESH','0')} mode={mode} rebuilt_once={cache_rebuilt_once}",
                flush=True,
            )

        contract_json = train_dir / "contract.json"
        contract_json.write_text(json.dumps(contract_overrides, indent=2, ensure_ascii=True), encoding="utf-8")

        train_meta = {
            "train_id": train_id,
            "seed": seed,
            "start_utc": train_start_utc,
            "contract_overrides": contract_overrides,
            "refresh": {
                "candle_sec": int(s.candle_sec),
                "verbose": bool(s.refresh_verbose),
                "skip": bool(s.skip_refresh),
            },
            "train": {
                "total_days": int(s.total_days),
                "offsets_years": int(s.offsets_years),
                "offsets_step_days": int(s.offsets_step_days),
                "max_symbols": int(s.max_symbols_train),
                "min_symbols_used_per_period": int(s.min_symbols_used_per_period),
                "max_rows_entry": int(s.max_rows_entry),
                "entry_ratio_neg_per_pos": float(s.entry_ratio_neg_per_pos),
                "xgb_device": str(s.xgb_device),
            },
            "backtest": {
                "backtests_per_train": int(s.backtests_per_train),
                "years": int(s.years),
                "step_days": int(s.step_days),
                "bar_stride": int(s.bar_stride),
                "max_symbols": int(s.max_symbols_backtest),
                "universe_history_mode": str(s.universe_history_mode),
                "step_cache_mode": str(s.step_cache_mode),
                "plot_save_only": bool(s.plot_save_only),
                "tau_entry_range": tuple(s.backtest_ranges.tau_entry_range),
                "max_positions_range": tuple(s.backtest_ranges.max_positions_range),
                "total_exposure_range": tuple(s.backtest_ranges.total_exposure_range),
                "max_trade_exposure_range": tuple(s.backtest_ranges.max_trade_exposure_range),
                "min_trade_exposure_range": tuple(s.backtest_ranges.min_trade_exposure_range),
                "exit_confirm_bars_range": tuple(s.backtest_ranges.exit_confirm_bars_range),
                "universe_history_days_range": tuple(s.backtest_ranges.universe_history_days_range),
                "universe_min_pf_range": tuple(s.backtest_ranges.universe_min_pf_range),
                "universe_min_win_range": tuple(s.backtest_ranges.universe_min_win_range),
                "universe_max_dd_range": tuple(s.backtest_ranges.universe_max_dd_range),
            },
        }
        (train_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=2, ensure_ascii=True), encoding="utf-8")

        train_run_dir = ""
        try:
            print(f"[loop] train {train_id} seed={seed}", flush=True)
            if bool(s.skip_refresh):
                print(
                    f"[loop] refresh labels skipped (WF_SKIP_REFRESH={os.getenv('WF_SKIP_REFRESH','')})",
                    flush=True,
                )
            else:
                print(f"[loop] refresh labels start contract={contract_overrides}", flush=True)
                refresh_info = refresh_labels(
                    RefreshLabelsSettings(
                        limit=0,
                        symbols=None,
                        candle_sec=int(s.candle_sec),
                        contract=contract,
                        max_symbols=int(s.max_symbols_train),
                        verbose=bool(s.refresh_verbose),
                    )
                )
                print(
                    f"[loop] refresh labels done ok={refresh_info.get('ok', 0)} "
                    f"fail={refresh_info.get('fail', 0)} sec={refresh_info.get('seconds', 0):.2f}",
                    flush=True,
                )

            if skip_train and (skip_train_mode == "always" or not skip_train_once_done):
                latest = _find_latest_wf_dir()
                if latest is None:
                    raise RuntimeError("WF_SKIP_TRAIN ativo mas nenhum wf_* encontrado em models_sniper")
                train_run_dir = str(latest)
                print(f"[loop] training skipped, using run_dir={train_run_dir}", flush=True)
                skip_train_once_done = True
            else:
                entry_params = dict(DEFAULT_ENTRY_PARAMS)
                entry_params["device"] = str(s.xgb_device)
                cfg = TrainConfig(
                    total_days=int(s.total_days),
                    offsets_days=tuple(range(int(s.offsets_step_days), int(365 * s.offsets_years) + 1, int(s.offsets_step_days))),
                    max_symbols=int(s.max_symbols_train),
                    min_symbols_used_per_period=int(s.min_symbols_used_per_period),
                    entry_ratio_neg_per_pos=float(s.entry_ratio_neg_per_pos),
                    max_rows_entry=int(s.max_rows_entry),
                    entry_params=entry_params,
                    use_feature_cache=True,
                    contract=contract,
                )
                print("[loop] training start", flush=True)
                train_run_dir = str(train_sniper_models(cfg))
                print(f"[loop] training done run_dir={train_run_dir}", flush=True)
                if bool(s.rebuild_cache) and str(getattr(s, "rebuild_cache_mode", "once") or "once").lower() == "once":
                    cache_rebuilt_once = True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            end_utc = _utc_now_iso()
            duration_sec = max(0.0, (pd.to_datetime(end_utc) - pd.to_datetime(train_start_utc)).total_seconds())
            err_msg = f"{type(e).__name__}: {e}"
            try:
                (out_root / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                (train_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            row = {
                "train_id": train_id,
                "backtest_id": "",
                "stage": "train",
                "status": "error",
                "start_utc": train_start_utc,
                "end_utc": end_utc,
                "duration_sec": float(duration_sec),
                "seed": seed,
                "train_run_dir": train_run_dir,
                "bt_out_dir": "",
                "equity_png": "",
                "eq_end": "",
                "ret_pct": "",
                "max_dd": "",
                "win_rate": "",
                "profit_factor": "",
                "trades": "",
                "top_symbols": "",
                "error": err_msg,
                # contract params
                "entry_window_min": (contract.entry_label_windows_minutes[0] if len(contract.entry_label_windows_minutes) > 0 else ""),
                "entry_min_profit": (contract.entry_label_min_profit_pcts[0] if len(contract.entry_label_min_profit_pcts) > 0 else ""),
                "entry_weight_alpha": float(getattr(contract, "entry_label_weight_alpha", 1.0)),
                "exit_ema_span": int(getattr(contract, "exit_ema_span", 0) or 0),
                "exit_ema_init_offset_pct": float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0),
                # train params
                "entry_ratio_neg_per_pos": float(s.entry_ratio_neg_per_pos),
                "train_total_days": int(s.total_days),
                "train_offsets_years": int(s.offsets_years),
                "train_offsets_step_days": int(s.offsets_step_days),
                "train_max_symbols": int(s.max_symbols_train),
                "train_min_symbols_used_per_period": int(s.min_symbols_used_per_period),
                "train_max_rows_entry": int(s.max_rows_entry),
                "train_xgb_device": str(s.xgb_device),
                # backtest thresholds
                "tau_entry": "",
                # backtest params
                "bt_years": "",
                "bt_step_days": "",
                "bt_bar_stride": "",
                "bt_max_symbols": "",
                "bt_max_positions": "",
                "bt_total_exposure": "",
                "bt_max_trade_exposure": "",
                "bt_min_trade_exposure": "",
                "bt_exit_confirm_bars": "",
                "bt_universe_history_mode": "",
                "bt_universe_history_days": "",
                "bt_universe_min_pf": "",
                "bt_universe_min_win": "",
                "bt_universe_max_dd": "",
                "bt_step_cache_mode": "",
                "bt_plot_save_only": "",
            }
            _append_csv(results_csv, row, header)
            time.sleep(max(1, int(s.sleep_seconds_on_error)))
            continue

        for bt_idx in range(max(0, int(s.backtests_per_train))):
            bt_id = f"bt_{bt_idx + 1:02d}"
            bt_out_dir = train_dir / bt_id
            bt_out_dir.mkdir(parents=True, exist_ok=True)
            bt_params = _sample_backtest_params(rng, s)
            bt_start_utc = _utc_now_iso()
            bt_meta = {
                "train_id": train_id,
                "backtest_id": bt_id,
                "start_utc": bt_start_utc,
                "params": bt_params,
            }
            (bt_out_dir / "backtest_meta.json").write_text(json.dumps(bt_meta, indent=2, ensure_ascii=True), encoding="utf-8")

            status = "ok"
            err_msg = ""
            metrics: dict = {}
            equity_png = ""
            try:
                print(
                    f"[loop] backtest start {train_id}/{bt_id} "
                    f"tau_entry={bt_params.get('tau_entry')} "
                    f"max_positions={bt_params.get('max_positions')} total_exposure={bt_params.get('total_exposure')}",
                    flush=True,
                )
                _run_backtest(bt_out_dir, Path(train_run_dir), contract_json, bt_params, s)
                metrics = _compute_metrics(bt_out_dir)
                equity_png_path = bt_out_dir / "wf_equity.png"
                if equity_png_path.exists():
                    equity_png = str(equity_png_path)
                print(
                    f"[loop] {train_id}/{bt_id} eq={metrics.get('eq_end', 0):.4f} "
                    f"ret={metrics.get('ret_pct', 0):+.2f}% dd={metrics.get('max_dd', 0):.2%} "
                    f"pf={metrics.get('profit_factor', 0):.2f} win={metrics.get('win_rate', 0):.2%} "
                    f"trades={metrics.get('trades', 0)}",
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                status = "error"
                err_msg = f"{type(e).__name__}: {e}"
                try:
                    (out_root / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    (bt_out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass
            finally:
                end_utc = _utc_now_iso()
                duration_sec = max(0.0, (pd.to_datetime(end_utc) - pd.to_datetime(bt_start_utc)).total_seconds())
                row = {
                    "train_id": train_id,
                    "backtest_id": bt_id,
                    "stage": "backtest",
                    "status": status,
                    "start_utc": bt_start_utc,
                    "end_utc": end_utc,
                    "duration_sec": float(duration_sec),
                    "seed": seed,
                    "train_run_dir": train_run_dir,
                    "bt_out_dir": str(bt_out_dir),
                    "equity_png": equity_png,
                    "eq_end": metrics.get("eq_end", ""),
                    "ret_pct": metrics.get("ret_pct", ""),
                    "max_dd": metrics.get("max_dd", ""),
                    "win_rate": metrics.get("win_rate", ""),
                    "profit_factor": metrics.get("profit_factor", ""),
                    "trades": metrics.get("trades", ""),
                    "top_symbols": metrics.get("top_symbols", ""),
                    "error": err_msg,
                    # contract params
                    "entry_window_min": (contract.entry_label_windows_minutes[0] if len(contract.entry_label_windows_minutes) > 0 else ""),
                    "entry_min_profit": (contract.entry_label_min_profit_pcts[0] if len(contract.entry_label_min_profit_pcts) > 0 else ""),
                    "entry_weight_alpha": float(getattr(contract, "entry_label_weight_alpha", 1.0)),
                    "exit_ema_span": int(getattr(contract, "exit_ema_span", 0) or 0),
                    "exit_ema_init_offset_pct": float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0),
                    # train params
                    "entry_ratio_neg_per_pos": float(s.entry_ratio_neg_per_pos),
                    "train_total_days": int(s.total_days),
                    "train_offsets_years": int(s.offsets_years),
                    "train_offsets_step_days": int(s.offsets_step_days),
                    "train_max_symbols": int(s.max_symbols_train),
                    "train_min_symbols_used_per_period": int(s.min_symbols_used_per_period),
                    "train_max_rows_entry": int(s.max_rows_entry),
                    "train_xgb_device": str(s.xgb_device),
                    # backtest thresholds
                    "tau_entry": bt_params.get("tau_entry", ""),
                    # backtest params
                    "bt_years": int(s.years),
                    "bt_step_days": int(s.step_days),
                    "bt_bar_stride": int(s.bar_stride),
                    "bt_max_symbols": int(s.max_symbols_backtest),
                    "bt_max_positions": bt_params.get("max_positions", ""),
                    "bt_total_exposure": bt_params.get("total_exposure", ""),
                    "bt_max_trade_exposure": bt_params.get("max_trade_exposure", ""),
                    "bt_min_trade_exposure": bt_params.get("min_trade_exposure", ""),
                    "bt_exit_confirm_bars": bt_params.get("exit_confirm_bars", ""),
                    "bt_universe_history_mode": str(s.universe_history_mode),
                    "bt_universe_history_days": bt_params.get("universe_history_days", ""),
                    "bt_universe_min_pf": bt_params.get("universe_min_pf", ""),
                    "bt_universe_min_win": bt_params.get("universe_min_win", ""),
                    "bt_universe_max_dd": bt_params.get("universe_max_dd", ""),
                    "bt_step_cache_mode": str(s.step_cache_mode),
                    "bt_plot_save_only": bool(s.plot_save_only),
                }
                _append_csv(results_csv, row, header)

            if status != "ok":
                time.sleep(max(1, int(s.sleep_seconds_on_error)))
                break


def main() -> None:
    run()


if __name__ == "__main__":
    main()
