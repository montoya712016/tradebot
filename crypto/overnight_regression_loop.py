# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Overnight loop for supervised regressors only:
prepare -> refresh labels -> train long/short WF -> evaluate -> adjust.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import os
import random
import time
import traceback
import sys
import math
import numpy as np


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_add_repo_paths()

from config.symbols import load_top_market_cap_symbols, default_top_market_cap_path  # type: ignore
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore
from prepare_features.refresh_sniper_labels_in_cache import RefreshLabelsSettings, run as refresh_labels  # type: ignore
from train.train_sniper_wf import TrainSniperWFSettings, run as train_wf_run  # type: ignore
from crypto.trade_contract import TradeContract  # type: ignore
from crypto.trade_contract import DEFAULT_TRADE_CONTRACT as CRYPTO_CONTRACT  # type: ignore
from crypto.evaluate_regressors_wf import EvalConfig, run as eval_run  # type: ignore


@dataclass
class Candidate:
    side_mae_penalty: float = 1.25
    side_time_penalty: float = 0.35
    side_cross_penalty: float = 0.85
    entry_reg_weight_alpha: float = 1.00
    entry_reg_weight_power: float = 1.00
    entry_reg_balance_distance_power: float = 0.50
    entry_reg_balance_min_frac: float = 0.20
    max_rows_entry: int = 2_000_000
    max_symbols_train: int = 80


@dataclass
class LoopConfig:
    duration_hours: float = 8.0
    max_iters: int = 0
    eval_days: int = 120
    eval_symbols_limit: int = 80
    cache_symbols_limit: int = 120
    total_days_cache: int = 0
    train_total_days: int = 0
    output_root: str = "data/generated/overnight_regression"
    sleep_on_error_sec: int = 90
    xgb_device: str = "cuda:0"
    entry_model_type: str = "catboost"


CSV_HEADER = [
    "iter",
    "start_utc",
    "end_utc",
    "duration_sec",
    "status",
    "score",
    "run_dir",
    "eval_report",
    "side_mae_penalty",
    "side_time_penalty",
    "side_cross_penalty",
    "entry_reg_weight_alpha",
    "entry_reg_weight_power",
    "entry_reg_balance_distance_power",
    "entry_reg_balance_min_frac",
    "max_rows_entry",
    "max_symbols_train",
    "long_pearson",
    "short_pearson",
    "edge_pearson",
    "long_precision_top20",
    "short_precision_top20",
    "samples_n",
    "symbols_used",
    "error",
]


def _set_env_defaults() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SNIPER_TIMINGS", "1")
    os.environ.setdefault("SNIPER_FEATURE_TIMINGS", "1")
    os.environ.setdefault("SNIPER_CACHE_PROGRESS_EVERY_S", "5")
    # conservative defaults for 40GB RAM
    os.environ.setdefault("SNIPER_CACHE_WORKERS", "8")
    os.environ.setdefault("SNIPER_DATASET_WORKERS", "4")
    os.environ.setdefault("SNIPER_POOL_READERS", "4")
    os.environ.setdefault("SNIPER_POOL_CHUNK_ROWS", "150000")
    # keep weighting active for labels 0..100
    os.environ.setdefault("SNIPER_USE_TIMING_WEIGHT", "1")
    os.environ.setdefault("SNIPER_ENABLE_REG_SHAPE_WEIGHT", "1")
    os.environ.setdefault("SNIPER_TAIL_ABS", "4")
    os.environ.setdefault("SNIPER_TAIL_WEIGHT_MULT", "2")
    os.environ.setdefault("SNIPER_WEIGHT_MAX_MULT", "4")
    # split mode does not need sign balancing
    os.environ.setdefault("SNIPER_BALANCE_SIGN", "0")
    os.environ.setdefault("SNIPER_BALANCE_TAIL_SIGN", "0")
    os.environ.setdefault("SNIPER_BALANCE_BINS_SIGN", "0")
    os.environ.setdefault("SNIPER_TARGET_CENTER_WEIGHTED", "0")
    os.environ.setdefault("SNIPER_ENABLE_CALIBRATION", "1")
    os.environ.setdefault("SNIPER_FORCE_PIECEWISE_CALIB", "1")
    os.environ.setdefault("SNIPER_ASSET_CLASS", "crypto")


def _utc_now() -> str:
    return str(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _mutate_candidate(best: Candidate, rng: random.Random) -> Candidate:
    c = Candidate(**asdict(best))
    c.side_mae_penalty = _clip(c.side_mae_penalty + rng.gauss(0.0, 0.15), 0.8, 2.5)
    c.side_time_penalty = _clip(c.side_time_penalty + rng.gauss(0.0, 0.08), 0.05, 1.2)
    c.side_cross_penalty = _clip(c.side_cross_penalty + rng.gauss(0.0, 0.12), 0.2, 1.8)
    c.entry_reg_weight_alpha = _clip(c.entry_reg_weight_alpha + rng.gauss(0.0, 0.20), 0.4, 3.0)
    c.entry_reg_weight_power = _clip(c.entry_reg_weight_power + rng.gauss(0.0, 0.12), 0.5, 2.2)
    c.entry_reg_balance_distance_power = _clip(c.entry_reg_balance_distance_power + rng.gauss(0.0, 0.18), 0.0, 2.0)
    c.entry_reg_balance_min_frac = _clip(c.entry_reg_balance_min_frac + rng.gauss(0.0, 0.06), 0.05, 0.6)
    c.max_rows_entry = int(rng.choice([1_500_000, 2_000_000, 3_000_000]))
    c.max_symbols_train = int(rng.choice([60, 80, 100]))
    return c


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})


def _ensure_base_cache(cfg: LoopConfig) -> None:
    symbols = load_top_market_cap_symbols(
        path=str(default_top_market_cap_path()),
        limit=int(cfg.cache_symbols_limit),
        ensure_usdt=True,
    )
    if not symbols:
        raise RuntimeError("sem simbolos para montar cache base")
    flags = dict(GLOBAL_FLAGS_FULL)
    flags["_quiet"] = True
    print(
        f"[overnight] preparando cache base symbols={len(symbols)} total_days={int(cfg.total_days_cache)}",
        flush=True,
    )
    _ = ensure_feature_cache(
        symbols,
        total_days=int(cfg.total_days_cache),
        contract=CRYPTO_CONTRACT,
        flags=flags,
        asset_class="crypto",
        abort_ram_pct=85.0,
    )


def _refresh_labels(c: Candidate) -> dict:
    settings = RefreshLabelsSettings(
        limit=0,
        symbols=None,
        symbols_file=None,
        mcap_min_usd=50_000_000.0,
        mcap_max_usd=150_000_000_000.0,
        max_symbols=int(c.max_symbols_train),
        candle_sec=60,
        label_clip=0.20,
        label_center=True,
        use_dominant=True,
        dominant_mix=0.50,
        side_mae_penalty=float(c.side_mae_penalty),
        side_time_penalty=float(c.side_time_penalty),
        side_cross_penalty=float(c.side_cross_penalty),
        verbose=True,
    )
    return refresh_labels(settings)


def _train_once(c: Candidate, cfg: LoopConfig) -> str:
    contract = TradeContract(
        entry_label_windows_minutes=(60,),
        exit_ema_init_offset_pct=0.002,
    )
    min_syms = int(min(30, max(5, round(float(c.max_symbols_train) * 0.5))))
    if int(cfg.train_total_days) > 0:
        max_off = max(30, int(cfg.train_total_days) - 30)
        base = (90, 180, 270, 360, 540, 720, 900, 1080)
        offsets = tuple(int(x) for x in base if int(x) <= int(max_off))
        if not offsets:
            offsets = (max(30, int(cfg.train_total_days * 0.5)),)
    else:
        offsets = (180, 360, 540, 720, 1080)
    settings = TrainSniperWFSettings(
        asset_class="crypto",
        contract=contract,
        mcap_min_usd=50_000_000.0,
        offsets_step_days=180,
        offsets_days=offsets,
        entry_model_type=str(cfg.entry_model_type),
        entry_reg_window_min=60,
        entry_reg_weight_alpha=float(c.entry_reg_weight_alpha),
        entry_reg_weight_power=float(c.entry_reg_weight_power),
        entry_reg_balance_bins=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 80.0),
        entry_reg_balance_distance_power=float(c.entry_reg_balance_distance_power),
        entry_reg_balance_min_frac=float(c.entry_reg_balance_min_frac),
        entry_label_mode="split_0_100",
        entry_label_scale=100.0,
        entry_sides=("long", "short"),
        max_rows_entry=int(c.max_rows_entry),
        abort_ram_pct=85.0,
        entry_pool_full=True,
        entry_pool_prefiltered=True,
        use_feature_cache=True,
        total_days=int(cfg.train_total_days),
        max_symbols=int(c.max_symbols_train),
        min_symbols_used_per_period=int(min_syms),
        xgb_device=str(cfg.xgb_device),
        run_dir=None,
    )
    return str(train_wf_run(settings))


def _eval_once(run_dir: str, cfg: LoopConfig, iter_idx: int) -> tuple[Path, dict]:
    out_path = Path(cfg.output_root).expanduser().resolve() / f"eval_iter_{int(iter_idx):03d}.json"
    ecfg = EvalConfig(
        run_dir=str(run_dir),
        out_path=str(out_path),
        symbols_limit=int(cfg.eval_symbols_limit),
        days=int(cfg.eval_days),
        total_days_cache=int(cfg.total_days_cache),
        window_min=60,
        mcap_min_usd=50_000_000.0,
        mcap_max_usd=150_000_000_000.0,
    )
    p = eval_run(ecfg)
    rep = json.loads(Path(p).read_text(encoding="utf-8"))
    return Path(p), rep


def _candidate_score(report: dict) -> float:
    try:
        return float(report.get("score", float("nan")))
    except Exception:
        return float("nan")


def run(cfg: LoopConfig | None = None) -> None:
    cfg = cfg or LoopConfig()
    _set_env_defaults()

    out_root = Path(cfg.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "loop_records.csv"
    best_path = out_root / "best_candidate.json"
    status_path = out_root / "latest_status.json"

    def _write_status(state: str, extra: dict | None = None) -> None:
        payload = {
            "updated_utc": _utc_now(),
            "state": str(state),
            "config": asdict(cfg),
        }
        if extra:
            payload.update(dict(extra))
        status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_status("starting")
    _write_status("prepare_cache")
    _ensure_base_cache(cfg)
    _write_status("prepare_cache_done")

    rng = random.Random(int(time.time()))
    best_cand = Candidate()
    best_score = -math.inf
    if best_path.exists():
        try:
            prev = json.loads(best_path.read_text(encoding="utf-8"))
            best_cand = Candidate(**(prev.get("candidate") or {}))
            best_score = float(prev.get("score", -math.inf))
            print(f"[overnight] retomando best_score={best_score:.6f}", flush=True)
        except Exception:
            pass

    deadline = time.time() + float(cfg.duration_hours) * 3600.0
    iter_idx = 0
    while True:
        if cfg.max_iters > 0 and iter_idx >= int(cfg.max_iters):
            break
        if time.time() >= deadline:
            break
        iter_idx += 1
        t0 = time.time()
        start_utc = _utc_now()

        if iter_idx == 1 and best_score <= -1e20:
            cand = Candidate()
        else:
            # 70% exploit, 30% explore from baseline
            if rng.random() < 0.7:
                cand = _mutate_candidate(best_cand, rng)
            else:
                cand = _mutate_candidate(Candidate(), rng)
        if int(cfg.cache_symbols_limit) > 0:
            cand.max_symbols_train = int(max(1, min(int(cand.max_symbols_train), int(cfg.cache_symbols_limit))))

        status = "ok"
        err = ""
        run_dir = ""
        eval_report_path = ""
        score = float("nan")
        rep: dict = {}

        print(f"[overnight] iter={iter_idx} candidate={asdict(cand)}", flush=True)
        try:
            _write_status("refresh_labels", {"iter": int(iter_idx), "candidate": asdict(cand)})
            ref_info = _refresh_labels(cand)
            print(
                f"[overnight] iter={iter_idx} refresh ok={ref_info.get('ok')} fail={ref_info.get('fail')} sec={ref_info.get('seconds')}",
                flush=True,
            )
            _write_status("train_models", {"iter": int(iter_idx), "candidate": asdict(cand), "refresh": ref_info})
            run_dir = _train_once(cand, cfg)
            print(f"[overnight] iter={iter_idx} train run_dir={run_dir}", flush=True)
            _write_status("evaluate_models", {"iter": int(iter_idx), "candidate": asdict(cand), "run_dir": run_dir})
            ev_path, rep = _eval_once(run_dir, cfg, iter_idx)
            eval_report_path = str(ev_path)
            score = _candidate_score(rep)
            print(f"[overnight] iter={iter_idx} score={score:.6f}", flush=True)
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            print(f"[overnight] iter={iter_idx} ERROR {err}\n{tb}", flush=True)
            _write_status(
                "error",
                {
                    "iter": int(iter_idx),
                    "candidate": asdict(cand),
                    "error": err,
                    "traceback": tb,
                    "run_dir": run_dir,
                },
            )

        end_utc = _utc_now()
        dt = time.time() - t0
        long_m = ((rep.get("sides") or {}).get("long") or {}) if isinstance(rep, dict) else {}
        short_m = ((rep.get("sides") or {}).get("short") or {}) if isinstance(rep, dict) else {}
        edge_m = ((rep.get("sides") or {}).get("edge") or {}) if isinstance(rep, dict) else {}
        smp = (rep.get("samples") or {}) if isinstance(rep, dict) else {}

        row = {
            "iter": int(iter_idx),
            "start_utc": start_utc,
            "end_utc": end_utc,
            "duration_sec": float(dt),
            "status": status,
            "score": score,
            "run_dir": run_dir,
            "eval_report": eval_report_path,
            "side_mae_penalty": float(cand.side_mae_penalty),
            "side_time_penalty": float(cand.side_time_penalty),
            "side_cross_penalty": float(cand.side_cross_penalty),
            "entry_reg_weight_alpha": float(cand.entry_reg_weight_alpha),
            "entry_reg_weight_power": float(cand.entry_reg_weight_power),
            "entry_reg_balance_distance_power": float(cand.entry_reg_balance_distance_power),
            "entry_reg_balance_min_frac": float(cand.entry_reg_balance_min_frac),
            "max_rows_entry": int(cand.max_rows_entry),
            "max_symbols_train": int(cand.max_symbols_train),
            "long_pearson": float(long_m.get("pearson", float("nan"))),
            "short_pearson": float(short_m.get("pearson", float("nan"))),
            "edge_pearson": float(edge_m.get("pearson", float("nan"))),
            "long_precision_top20": float(long_m.get("precision_top20_above_target_p80", float("nan"))),
            "short_precision_top20": float(short_m.get("precision_top20_above_target_p80", float("nan"))),
            "samples_n": int(smp.get("n", 0) or 0),
            "symbols_used": int(smp.get("symbols_used", 0) or 0),
            "error": err,
        }
        _append_csv(csv_path, row)
        _write_status(
            "iteration_done",
            {
                "iter": int(iter_idx),
                "candidate": asdict(cand),
                "row": row,
            },
        )

        if status == "ok" and np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_cand = cand
            best_payload = {
                "updated_utc": end_utc,
                "score": float(best_score),
                "candidate": asdict(best_cand),
                "run_dir": run_dir,
                "eval_report": eval_report_path,
            }
            best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[overnight] novo melhor score={best_score:.6f}", flush=True)

        if status != "ok":
            time.sleep(max(5, int(cfg.sleep_on_error_sec)))

    _write_status(
        "finished",
        {
            "iters": int(iter_idx),
            "best_score": float(best_score) if np.isfinite(best_score) else None,
            "best_candidate": asdict(best_cand),
            "csv": str(csv_path),
            "best_json": str(best_path),
        },
    )
    print("[overnight] loop finalizado", flush=True)
    print(f"[overnight] csv={csv_path}", flush=True)
    print(f"[overnight] best={best_path}", flush=True)


def _parse_args(argv: list[str] | None = None) -> LoopConfig:
    ap = argparse.ArgumentParser(description="Loop noturno de treino/avaliacao dos regressors")
    ap.add_argument("--duration-hours", type=float, default=8.0)
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--eval-days", type=int, default=120)
    ap.add_argument("--eval-symbols-limit", type=int, default=80)
    ap.add_argument("--cache-symbols-limit", type=int, default=120)
    ap.add_argument("--total-days-cache", type=int, default=0)
    ap.add_argument("--train-total-days", type=int, default=0)
    ap.add_argument("--output-root", default="data/generated/overnight_regression")
    ap.add_argument("--sleep-on-error-sec", type=int, default=90)
    ap.add_argument("--xgb-device", default="cuda:0")
    ap.add_argument("--entry-model-type", default="catboost", choices=["catboost", "xgb"])
    ns = ap.parse_args(argv)
    return LoopConfig(
        duration_hours=float(ns.duration_hours),
        max_iters=int(ns.max_iters),
        eval_days=int(ns.eval_days),
        eval_symbols_limit=int(ns.eval_symbols_limit),
        cache_symbols_limit=int(ns.cache_symbols_limit),
        total_days_cache=int(ns.total_days_cache),
        train_total_days=int(ns.train_total_days),
        output_root=str(ns.output_root),
        sleep_on_error_sec=int(ns.sleep_on_error_sec),
        xgb_device=str(ns.xgb_device),
        entry_model_type=str(ns.entry_model_type),
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()
