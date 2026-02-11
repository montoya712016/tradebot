# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Pipeline completo híbrido (WF):
1) Exporta sinais supervisionados long/short
2) Treina RL (DQN) em walk-forward temporal global
3) Avalia RL vs baseline supervisionado
4) Roda backtests agregados (supervised-only e supervised+RL)
"""

from pathlib import Path
import argparse
import copy
import json
import sys
import time
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

from supervised.inference_long_short import SignalExportConfig, export_long_short_signals  # type: ignore
from rl.train_rl import TrainRLConfig, train_rl_walkforward  # type: ignore
from rl.evaluate_rl import EvaluateConfig, evaluate_rl_run  # type: ignore
from backtest.backtest_supervised_only import run as run_sup_only  # type: ignore
from backtest.backtest_supervised_plus_rl import run as run_sup_plus_rl  # type: ignore


def _default_storage_root() -> Path:
    # .../tradebot/modules/rl/run_hybrid_wf_pipeline.py -> .../astra/models_sniper/hybrid_rl
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root.parent / "models_sniper" / "hybrid_rl"


def _contract_path_for_signals(signals_path: str | Path) -> Path:
    p = Path(signals_path).expanduser().resolve()
    return p.with_suffix(p.suffix + ".contract.json")


def _compute_signal_drift(oof_path: Path, prod_path: Path, out_json: Path) -> Path:
    cols = [
        "mu_long_norm",
        "mu_short_norm",
        "edge_norm",
        "strength_norm",
        "uncertainty_norm",
        "vol_short_norm",
        "trend_strength_norm",
    ]
    a = pd.read_parquet(oof_path)
    b = pd.read_parquet(prod_path)
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)
    ka = a.reset_index().rename(columns={"index": "ts"})
    kb = b.reset_index().rename(columns={"index": "ts"})
    keep = ["ts", "symbol"] + [c for c in cols if c in ka.columns and c in kb.columns]
    ka = ka[keep].copy()
    kb = kb[keep].copy()
    merged = ka.merge(kb, on=["ts", "symbol"], suffixes=("_oof", "_prod"), how="inner")
    report: dict[str, object] = {
        "rows_oof": int(len(ka)),
        "rows_prod": int(len(kb)),
        "rows_overlap": int(len(merged)),
        "features": {},
    }
    for c in cols:
        co = f"{c}_oof"
        cp = f"{c}_prod"
        if co not in merged.columns or cp not in merged.columns:
            continue
        x = merged[co].to_numpy(dtype=np.float64, copy=False)
        y = merged[cp].to_numpy(dtype=np.float64, copy=False)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            continue
        x = x[m]
        y = y[m]
        dx = y - x
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 3 else 0.0
        report["features"][c] = {
            "n": int(x.size),
            "mean_oof": float(np.mean(x)),
            "mean_prod": float(np.mean(y)),
            "std_oof": float(np.std(x)),
            "std_prod": float(np.std(y)),
            "mae_prod_vs_oof": float(np.mean(np.abs(dx))),
            "mean_diff_prod_minus_oof": float(np.mean(dx)),
            "corr": float(corr if np.isfinite(corr) else 0.0),
            "p05_oof": float(np.quantile(x, 0.05)),
            "p95_oof": float(np.quantile(x, 0.95)),
            "p05_prod": float(np.quantile(y, 0.05)),
            "p95_prod": float(np.quantile(y, 0.95)),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def _check_signal_contract_compat(oof_contract_path: Path, prod_contract_path: Path) -> None:
    if not oof_contract_path.exists() or not prod_contract_path.exists():
        return
    try:
        a = json.loads(oof_contract_path.read_text(encoding="utf-8"))
        b = json.loads(prod_contract_path.read_text(encoding="utf-8"))
    except Exception:
        return
    ca = list(a.get("columns_required", []) or [])
    cb = list(b.get("columns_required", []) or [])
    if ca != cb:
        raise RuntimeError("Contrato de sinais incompativel entre OOF e PROD (columns_required divergiu)")
    if bool(a.get("raw_mu_for_rl", True)) != bool(b.get("raw_mu_for_rl", True)):
        raise RuntimeError("Contrato de sinais incompativel entre OOF e PROD (raw_mu_for_rl divergiu)")


def run_pipeline(
    *,
    signals_cfg: SignalExportConfig,
    rl_cfg: TrainRLConfig,
    eval_cfg: EvaluateConfig,
    base_out_dir: Path,
    prod_signals_cfg: SignalExportConfig | None = None,
    dev_wf_folds: int = 3,
    enable_prod_refit: bool = True,
    prod_holdout_days: int = 30,
) -> dict:
    def _fmt_secs(dt: float) -> str:
        return f"{dt:.2f}s"

    t0_all = time.perf_counter()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    signals_path = export_long_short_signals(signals_cfg)
    dt_signals = time.perf_counter() - t0
    print(f"[pipeline][time] export_signals_oof={_fmt_secs(dt_signals)} path={signals_path}", flush=True)

    # fase 1: validacao temporal multi-fold (dev_wf)
    dev_cfg = copy.deepcopy(rl_cfg)
    dev_cfg.signals_path = str(signals_path)
    dev_cfg.signals_contract_path = str(_contract_path_for_signals(signals_path))
    dev_cfg.single_split_mode = False
    dev_cfg.max_folds = int(max(2, int(dev_wf_folds)))
    if int(dev_cfg.step_days) >= int(dev_cfg.valid_days):
        dev_cfg.step_days = int(max(7, min(60, int(dev_cfg.valid_days) // 2)))
    dev_cfg.out_dir = str((Path(rl_cfg.out_dir).expanduser().resolve().parent / (Path(rl_cfg.out_dir).name + "_dev_wf")).resolve())
    t0 = time.perf_counter()
    run_dir_dev = train_rl_walkforward(dev_cfg)
    dt_train = time.perf_counter() - t0
    print(f"[pipeline][time] train_rl_dev_wf={_fmt_secs(dt_train)} run_dir={run_dir_dev}", flush=True)

    eval_cfg_dev = copy.deepcopy(eval_cfg)
    eval_cfg_dev.signals_path = str(signals_path)
    eval_cfg_dev.run_dir = str(run_dir_dev)
    t0 = time.perf_counter()
    eval_csv_dev = evaluate_rl_run(eval_cfg_dev)
    dt_eval = time.perf_counter() - t0
    print(f"[pipeline][time] evaluate_rl_dev_wf={_fmt_secs(dt_eval)} csv={eval_csv_dev}", flush=True)

    run_dir = run_dir_dev
    eval_csv = eval_csv_dev
    signals_path_prod = Path(signals_path)
    eval_csv_prod = None
    run_dir_prod = None
    drift_json = None
    dt_sig_prod = 0.0
    dt_train_prod = 0.0
    dt_eval_prod = 0.0
    if bool(enable_prod_refit):
        # sinais do regressor de producao (tipicamente wf mais recente / T-0)
        cfg_prod = copy.deepcopy(prod_signals_cfg or signals_cfg)
        if not str(getattr(cfg_prod, "out_path", "")).strip():
            cfg_prod.out_path = str((Path(signals_path).with_name("supervised_signals_prod.parquet")).resolve())
        t0 = time.perf_counter()
        signals_path_prod = export_long_short_signals(cfg_prod)
        dt_sig_prod = time.perf_counter() - t0
        print(f"[pipeline][time] export_signals_prod={_fmt_secs(dt_sig_prod)} path={signals_path_prod}", flush=True)
        _check_signal_contract_compat(
            _contract_path_for_signals(signals_path),
            _contract_path_for_signals(signals_path_prod),
        )
        t0 = time.perf_counter()
        drift_json = _compute_signal_drift(
            Path(signals_path).expanduser().resolve(),
            Path(signals_path_prod).expanduser().resolve(),
            (base_out_dir / "signals_drift_oof_vs_prod.json").resolve(),
        )
        print(f"[pipeline][time] signals_drift={_fmt_secs(time.perf_counter()-t0)} out={drift_json}", flush=True)

        # fase 2: refit final (split unico recente) usando contrato de producao
        prod_cfg = copy.deepcopy(rl_cfg)
        prod_cfg.signals_path = str(signals_path_prod)
        prod_cfg.signals_contract_path = str(_contract_path_for_signals(signals_path_prod))
        prod_cfg.single_split_mode = True
        prod_cfg.max_folds = 1
        prod_cfg.holdout_days = int(max(7, int(prod_holdout_days)))
        prod_cfg.out_dir = str((Path(rl_cfg.out_dir).expanduser().resolve().parent / (Path(rl_cfg.out_dir).name + "_prod_refit")).resolve())
        t0 = time.perf_counter()
        run_dir_prod = train_rl_walkforward(prod_cfg)
        dt_train_prod = time.perf_counter() - t0
        print(f"[pipeline][time] train_rl_prod_refit={_fmt_secs(dt_train_prod)} run_dir={run_dir_prod}", flush=True)

        eval_cfg_prod = copy.deepcopy(eval_cfg)
        eval_cfg_prod.signals_path = str(signals_path_prod)
        eval_cfg_prod.run_dir = str(run_dir_prod)
        t0 = time.perf_counter()
        eval_csv_prod = evaluate_rl_run(eval_cfg_prod)
        dt_eval_prod = time.perf_counter() - t0
        print(f"[pipeline][time] evaluate_rl_prod_refit={_fmt_secs(dt_eval_prod)} csv={eval_csv_prod}", flush=True)
        run_dir = run_dir_prod
        eval_csv = eval_csv_prod

    t0 = time.perf_counter()
    sup_only_json = run_sup_only(
        signals_path=str(signals_path_prod),
        out_json=str(base_out_dir / "backtest_supervised_only.json"),
        min_rows_symbol=int(rl_cfg.min_rows_symbol),
        edge_threshold=float(eval_cfg.edge_threshold),
        strength_threshold=float(eval_cfg.strength_threshold),
        fee_rate=float(rl_cfg.fee_rate),
        slippage_rate=float(rl_cfg.slippage_rate),
        small_size=float(rl_cfg.small_size),
        big_size=float(rl_cfg.big_size),
        cooldown_bars=int(rl_cfg.cooldown_bars),
        min_reentry_gap_bars=int(rl_cfg.min_reentry_gap_bars),
        min_hold_bars=int(rl_cfg.min_hold_bars),
        max_hold_bars=int(rl_cfg.max_hold_bars),
        edge_entry_threshold=float(rl_cfg.edge_entry_threshold),
        strength_entry_threshold=float(rl_cfg.strength_entry_threshold),
        dd_penalty=float(rl_cfg.dd_penalty),
        dd_level_penalty=float(rl_cfg.dd_level_penalty),
        dd_soft_limit=float(rl_cfg.dd_soft_limit),
        dd_excess_penalty=float(rl_cfg.dd_excess_penalty),
        dd_hard_limit=float(rl_cfg.dd_hard_limit),
        dd_hard_penalty=float(rl_cfg.dd_hard_penalty),
        hold_bar_penalty=float(rl_cfg.hold_bar_penalty),
        hold_soft_bars=int(rl_cfg.hold_soft_bars),
        hold_excess_penalty=float(rl_cfg.hold_excess_penalty),
        hold_regret_penalty=float(rl_cfg.hold_regret_penalty),
        turnover_penalty=float(rl_cfg.turnover_penalty),
        regret_penalty=float(rl_cfg.regret_penalty),
        idle_penalty=float(rl_cfg.idle_penalty),
    )
    dt_sup_only = time.perf_counter() - t0
    print(f"[pipeline][time] backtest_supervised_only={_fmt_secs(dt_sup_only)} out={sup_only_json}", flush=True)

    t0 = time.perf_counter()
    sup_plus_rl_json = run_sup_plus_rl(
        signals_path=str(signals_path_prod),
        checkpoint=None,
        run_dir=str(run_dir),
        fold_id=-1,
        out_json=str(base_out_dir / "backtest_supervised_plus_rl.json"),
        min_rows_symbol=int(rl_cfg.min_rows_symbol),
        fee_rate=float(rl_cfg.fee_rate),
        slippage_rate=float(rl_cfg.slippage_rate),
        small_size=float(rl_cfg.small_size),
        big_size=float(rl_cfg.big_size),
        cooldown_bars=int(rl_cfg.cooldown_bars),
        min_reentry_gap_bars=int(rl_cfg.min_reentry_gap_bars),
        min_hold_bars=int(rl_cfg.min_hold_bars),
        max_hold_bars=int(rl_cfg.max_hold_bars),
        edge_entry_threshold=float(rl_cfg.edge_entry_threshold),
        strength_entry_threshold=float(rl_cfg.strength_entry_threshold),
        dd_penalty=float(rl_cfg.dd_penalty),
        dd_level_penalty=float(rl_cfg.dd_level_penalty),
        dd_soft_limit=float(rl_cfg.dd_soft_limit),
        dd_excess_penalty=float(rl_cfg.dd_excess_penalty),
        dd_hard_limit=float(rl_cfg.dd_hard_limit),
        dd_hard_penalty=float(rl_cfg.dd_hard_penalty),
        hold_bar_penalty=float(rl_cfg.hold_bar_penalty),
        hold_soft_bars=int(rl_cfg.hold_soft_bars),
        hold_excess_penalty=float(rl_cfg.hold_excess_penalty),
        hold_regret_penalty=float(rl_cfg.hold_regret_penalty),
        turnover_penalty=float(rl_cfg.turnover_penalty),
        regret_penalty=float(rl_cfg.regret_penalty),
        idle_penalty=float(rl_cfg.idle_penalty),
        device=str(eval_cfg.device),
    )
    dt_sup_rl = time.perf_counter() - t0
    print(f"[pipeline][time] backtest_supervised_plus_rl={_fmt_secs(dt_sup_rl)} out={sup_plus_rl_json}", flush=True)

    out = {
        "signals_path_oof": str(signals_path),
        "signals_path_prod": str(signals_path_prod),
        "rl_run_dir_dev_wf": str(run_dir_dev),
        "rl_run_dir": str(run_dir),
        "evaluation_csv_dev_wf": str(eval_csv_dev),
        "evaluation_csv": str(eval_csv),
        "backtest_supervised_only": str(sup_only_json),
        "backtest_supervised_plus_rl": str(sup_plus_rl_json),
    }
    if run_dir_prod is not None:
        out["rl_run_dir_prod_refit"] = str(run_dir_prod)
    if eval_csv_prod is not None:
        out["evaluation_csv_prod_refit"] = str(eval_csv_prod)
    if drift_json is not None:
        out["signals_drift_json"] = str(drift_json)
    summary_path = base_out_dir / "pipeline_summary.json"
    total_s = time.perf_counter() - t0_all
    out["timings_seconds"] = {
        "export_signals": float(dt_signals),
        "train_rl_dev_wf": float(dt_train),
        "evaluate_rl_dev_wf": float(dt_eval),
        "export_signals_prod": float(dt_sig_prod),
        "train_rl_prod_refit": float(dt_train_prod),
        "evaluate_rl_prod_refit": float(dt_eval_prod),
        "backtest_supervised_only": float(dt_sup_only),
        "backtest_supervised_plus_rl": float(dt_sup_rl),
        "total": float(total_s),
    }
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pipeline][time] total={_fmt_secs(total_s)}", flush=True)
    out["summary_path"] = str(summary_path)
    return out


def _parse_args(argv: Iterable[str] | None = None):
    storage_root = _default_storage_root()
    default_signals_out = storage_root / "supervised_signals.parquet"
    default_rl_out_dir = storage_root / "rl_runs" / "default"
    default_pipeline_out = storage_root / "reports"
    default_symbols = [
        "SOLUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "LTCUSDT",
        "NEARUSDT",
        "INJUSDT",
        "OPUSDT",
        "ARBUSDT",
        "ADAUSDT",
        "XLMUSDT",
        "STXUSDT",
        "ATOMUSDT",
        "DOGEUSDT",
        "SHIBUSDT",
        "HBARUSDT",
        "SUIUSDT",
        "APEUSDT",
        "XRPUSDT",
    ]
    ap = argparse.ArgumentParser(description="Pipeline híbrido WF: regressão long/short + RL")
    ap.add_argument("--asset-class", default="crypto", choices=["crypto", "stocks"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", nargs="*", default=default_symbols)
    ap.add_argument("--symbols-limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--total-days-cache", type=int, default=0)
    ap.add_argument("--signals-out", default=str(default_signals_out))
    ap.add_argument("--prod-run-dir", default=None)
    ap.add_argument("--signals-prod-out", default=str(storage_root / "supervised_signals_prod.parquet"))

    ap.add_argument("--rl-out-dir", default=str(default_rl_out_dir))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-days", type=int, default=1080)
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
    ap.add_argument("--train-steps-per-epoch", type=int, default=1200)
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--expert-mix-start", type=float, default=0.45)
    ap.add_argument("--expert-mix-end", type=float, default=0.0)
    ap.add_argument("--expert-mix-decay-epochs", type=int, default=14)
    ap.add_argument("--strict-best-update", dest="strict_best_update", action="store_true", default=True)
    ap.add_argument("--no-strict-best-update", dest="strict_best_update", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-rate", type=float, default=0.0001)
    ap.add_argument("--small-size", type=float, default=0.10)
    ap.add_argument("--big-size", type=float, default=0.25)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-device", default="cpu")

    ap.add_argument("--edge-threshold", type=float, default=0.2)
    ap.add_argument("--strength-threshold", type=float, default=0.2)
    ap.add_argument("--pipeline-out", default=str(default_pipeline_out))
    ap.add_argument("--dev-wf-folds", type=int, default=3)
    ap.add_argument("--enable-prod-refit", dest="enable_prod_refit", action="store_true", default=True)
    ap.add_argument("--no-enable-prod-refit", dest="enable_prod_refit", action="store_false")
    ap.add_argument("--prod-holdout-days", type=int, default=30)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    ns = _parse_args(argv)
    signals_cfg = SignalExportConfig(
        asset_class=str(ns.asset_class),
        run_dir=ns.run_dir,
        symbols=tuple(ns.symbols or ()),
        symbols_file=ns.symbols_file,
        symbols_limit=int(ns.symbols_limit),
        days=int(ns.days),
        total_days_cache=int(ns.total_days_cache),
        out_path=str(ns.signals_out),
        enforce_oof=True,
    )
    prod_signals_cfg = copy.deepcopy(signals_cfg)
    prod_signals_cfg.run_dir = (str(ns.prod_run_dir).strip() if ns.prod_run_dir else None)
    prod_signals_cfg.out_path = str(ns.signals_prod_out)
    rl_cfg = TrainRLConfig(
        signals_path=str(ns.signals_out),
        out_dir=str(ns.rl_out_dir),
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
        train_steps_per_epoch=int(ns.train_steps_per_epoch),
        eval_every_epochs=int(ns.eval_every_epochs),
        expert_mix_start=float(ns.expert_mix_start),
        expert_mix_end=float(ns.expert_mix_end),
        expert_mix_decay_epochs=int(ns.expert_mix_decay_epochs),
        strict_best_update=bool(ns.strict_best_update),
        early_stop_patience=int(ns.early_stop_patience),
        early_stop_min_delta=float(ns.early_stop_min_delta),
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
    eval_cfg = EvaluateConfig(
        signals_path=str(ns.signals_out),
        run_dir=str(ns.rl_out_dir),
        min_rows_symbol=int(ns.min_rows_symbol),
        edge_threshold=float(ns.edge_threshold),
        strength_threshold=float(ns.strength_threshold),
        device=str(ns.device),
    )
    out = run_pipeline(
        signals_cfg=signals_cfg,
        rl_cfg=rl_cfg,
        eval_cfg=eval_cfg,
        base_out_dir=Path(str(ns.pipeline_out)).expanduser().resolve(),
        prod_signals_cfg=prod_signals_cfg,
        dev_wf_folds=int(ns.dev_wf_folds),
        enable_prod_refit=bool(ns.enable_prod_refit),
        prod_holdout_days=int(ns.prod_holdout_days),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


