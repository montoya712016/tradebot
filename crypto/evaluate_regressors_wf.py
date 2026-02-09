# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Evaluate walk-forward long/short regressors without RL.

Outputs aggregate metrics focused on:
- distribution and percentiles (pred/target)
- rank quality by percentiles/deciles
- long vs short asymmetry
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


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

from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward  # type: ignore
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore
from config.symbols import load_top_market_cap_symbols, default_top_market_cap_path  # type: ignore
from crypto.trade_contract import DEFAULT_TRADE_CONTRACT as CRYPTO_CONTRACT  # type: ignore


@dataclass
class EvalConfig:
    run_dir: str
    out_path: str = "data/generated/overnight_regression/eval_latest.json"
    symbols_limit: int = 80
    days: int = 120
    total_days_cache: int = 0
    window_min: int = 60
    mcap_symbols_file: str | None = None
    mcap_min_usd: float = 50_000_000.0
    mcap_max_usd: float = 150_000_000_000.0


def _resolve_symbols(cfg: EvalConfig) -> list[str]:
    path = str(cfg.mcap_symbols_file) if cfg.mcap_symbols_file else str(default_top_market_cap_path())
    syms = load_top_market_cap_symbols(path=path, limit=None, ensure_usdt=True)
    out: list[str] = []
    for s in syms:
        t = str(s).strip().upper()
        if not t:
            continue
        if not t.endswith("USDT"):
            t = f"{t}USDT"
        out.append(t)
        if cfg.symbols_limit > 0 and len(out) >= int(cfg.symbols_limit):
            break
    if not out:
        raise RuntimeError("lista de simbolos vazia para avaliacao")
    return out


def _dist_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    m = np.isfinite(a)
    if not np.any(m):
        return {"n": 0}
    a = a[m]
    q = np.quantile(a, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p50": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "nz": float(np.mean(np.abs(a) > 1e-12)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(rx, ry)


def _decile_table(pred: np.ndarray, target: np.ndarray, bins: int = 10) -> list[dict]:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    if p.size == 0:
        return []
    rp = pd.Series(p).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    idx = np.minimum(int(bins) - 1, np.maximum(0, np.floor(rp * float(bins)).astype(np.int32)))
    out: list[dict] = []
    for b in range(int(bins)):
        mb = idx == b
        if not np.any(mb):
            out.append({"bin": int(b), "n": 0})
            continue
        pb = p[mb]
        yb = y[mb]
        out.append(
            {
                "bin": int(b),
                "n": int(yb.size),
                "pred_mean": float(np.mean(pb)),
                "pred_p50": float(np.quantile(pb, 0.5)),
                "target_mean": float(np.mean(yb)),
                "target_p50": float(np.quantile(yb, 0.5)),
                "target_p90": float(np.quantile(yb, 0.9)),
            }
        )
    return out


def _pick_pred_from_map(pred_map: dict[str, np.ndarray], prefer_name: str) -> np.ndarray:
    if not pred_map:
        return np.array([], dtype=np.float32)
    if prefer_name in pred_map:
        return np.asarray(pred_map[prefer_name], dtype=np.float32)
    names = sorted(pred_map.keys())
    if len(names) == 1:
        return np.asarray(pred_map[names[0]], dtype=np.float32)
    mats = [np.asarray(pred_map[k], dtype=np.float32).reshape(-1, 1) for k in names]
    m = np.concatenate(mats, axis=1)
    m = np.where(np.isfinite(m), m, -np.inf)
    best = np.max(m, axis=1).astype(np.float32, copy=False)
    best[~np.isfinite(best)] = 0.0
    return best


def _side_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]
    if p.size == 0:
        return {"n": 0}
    err = p - y
    y_p80 = float(np.quantile(y, 0.8))
    p_p80 = float(np.quantile(p, 0.8))
    p_p20 = float(np.quantile(p, 0.2))
    top_mask = p >= p_p80
    bot_mask = p <= p_p20
    prec_top = float(np.mean(y[top_mask] >= y_p80)) if np.any(top_mask) else float("nan")
    return {
        "n": int(p.size),
        "pearson": _pearson(p, y),
        "spearman": _spearman(p, y),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "mean_pred": float(np.mean(p)),
        "mean_target": float(np.mean(y)),
        "pred_stats": _dist_stats(p),
        "target_stats": _dist_stats(y),
        "target_mean_when_pred_top20": float(np.mean(y[top_mask])) if np.any(top_mask) else float("nan"),
        "target_mean_when_pred_bot20": float(np.mean(y[bot_mask])) if np.any(bot_mask) else float("nan"),
        "precision_top20_above_target_p80": prec_top,
        "deciles": _decile_table(p, y, bins=10),
    }


def _extract_period_meta(run_dir: Path) -> dict:
    out: dict[str, list[dict]] = {"long": [], "short": []}
    for pd_dir in sorted([p for p in run_dir.glob("period_*d") if p.is_dir()]):
        meta_path = pd_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for side in ("long", "short"):
            for k, v in meta.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                m = re.match(rf"entry_{side}_(\d+)m$", k)
                if not m:
                    continue
                out[side].append(
                    {
                        "period": pd_dir.name,
                        "window_min": int(m.group(1)),
                        "best_iteration": int(v.get("best_iteration", 0) or 0),
                        "rmse": float((v.get("metrics") or {}).get("rmse", float("nan"))),
                        "mae": float((v.get("metrics") or {}).get("mae", float("nan"))),
                    }
                )
    return out


def evaluate(cfg: EvalConfig) -> dict:
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir nao encontrado: {run_dir}")
    periods = load_period_models(run_dir)
    if not periods:
        raise RuntimeError(f"nenhum period_* valido em {run_dir}")

    symbols = _resolve_symbols(cfg)
    flags = dict(GLOBAL_FLAGS_FULL)
    flags["_quiet"] = True
    cache_map = ensure_feature_cache(
        symbols,
        total_days=int(cfg.total_days_cache),
        contract=CRYPTO_CONTRACT,
        flags=flags,
        asset_class="crypto",
        abort_ram_pct=85.0,
    )

    prefer_name = f"ret_exp_{int(cfg.window_min)}m"
    all_pl: list[np.ndarray] = []
    all_ps: list[np.ndarray] = []
    all_yl: list[np.ndarray] = []
    all_ys: list[np.ndarray] = []
    used_symbols = 0
    skipped_symbols = 0

    for sym in symbols:
        p = cache_map.get(sym)
        if p is None or not Path(p).exists():
            skipped_symbols += 1
            continue
        df = pd.read_parquet(p) if str(p).lower().endswith(".parquet") else pd.read_pickle(p)
        if df.empty:
            skipped_symbols += 1
            continue
        if int(cfg.days) > 0:
            end_ts = pd.to_datetime(df.index.max())
            start_ts = end_ts - pd.Timedelta(days=int(cfg.days))
            df = df.loc[pd.to_datetime(df.index) >= start_ts].copy()
        if len(df) < 2_000:
            skipped_symbols += 1
            continue
        if "timing_label_long" not in df.columns or "timing_label_short" not in df.columns:
            skipped_symbols += 1
            continue

        p_long_map, p_short_map, _p_danger, _p_exit, _used, period_id = predict_scores_walkforward(
            df,
            periods=periods,
            return_period_id=True,
        )
        pl = _pick_pred_from_map(dict(p_long_map), prefer_name).astype(np.float64, copy=False)
        ps = _pick_pred_from_map(dict(p_short_map), prefer_name).astype(np.float64, copy=False)
        yl = df["timing_label_long"].to_numpy(dtype=np.float64, copy=False)
        ys = df["timing_label_short"].to_numpy(dtype=np.float64, copy=False)
        pid = np.asarray(period_id, dtype=np.int32)
        m = (pid >= 0) & np.isfinite(pl) & np.isfinite(ps) & np.isfinite(yl) & np.isfinite(ys)
        if not np.any(m):
            skipped_symbols += 1
            continue
        all_pl.append(pl[m])
        all_ps.append(ps[m])
        all_yl.append(yl[m])
        all_ys.append(ys[m])
        used_symbols += 1

    if not all_pl:
        raise RuntimeError("avaliacao sem amostras validas")

    pred_long = np.concatenate(all_pl).astype(np.float64, copy=False)
    pred_short = np.concatenate(all_ps).astype(np.float64, copy=False)
    y_long = np.concatenate(all_yl).astype(np.float64, copy=False)
    y_short = np.concatenate(all_ys).astype(np.float64, copy=False)
    edge_pred = pred_long - pred_short
    edge_true = y_long - y_short

    long_m = _side_metrics(pred_long, y_long)
    short_m = _side_metrics(pred_short, y_short)
    edge_m = _side_metrics(edge_pred, edge_true)

    asym = {
        "pred_long_gt_short_rate": float(np.mean(pred_long > pred_short)),
        "true_long_gt_short_rate": float(np.mean(y_long > y_short)),
        "mean_pred_gap": float(np.mean(pred_long) - np.mean(pred_short)),
        "mean_true_gap": float(np.mean(y_long) - np.mean(y_short)),
        "pearson_long_minus_short": float(long_m.get("pearson", np.nan)) - float(short_m.get("pearson", np.nan)),
    }

    train_meta = _extract_period_meta(run_dir)

    report = {
        "run_dir": str(run_dir),
        "window_min_eval": int(cfg.window_min),
        "samples": {
            "n": int(pred_long.size),
            "symbols_used": int(used_symbols),
            "symbols_skipped": int(skipped_symbols),
        },
        "sides": {
            "long": long_m,
            "short": short_m,
            "edge": edge_m,
        },
        "asymmetry": asym,
        "train_meta": train_meta,
    }
    return report


def _score(report: dict) -> float:
    try:
        long_m = report["sides"]["long"]
        short_m = report["sides"]["short"]
        edge_m = report["sides"]["edge"]
        c_long = float(long_m.get("pearson", 0.0) or 0.0)
        c_short = float(short_m.get("pearson", 0.0) or 0.0)
        s_long = float(long_m.get("spearman", 0.0) or 0.0)
        s_short = float(short_m.get("spearman", 0.0) or 0.0)
        c_edge = float(edge_m.get("pearson", 0.0) or 0.0)
        p_long = float(long_m.get("precision_top20_above_target_p80", 0.0) or 0.0)
        p_short = float(short_m.get("precision_top20_above_target_p80", 0.0) or 0.0)
        mae_long = float(long_m.get("mae", 0.0) or 0.0)
        mae_short = float(short_m.get("mae", 0.0) or 0.0)
        asym_gap = abs(float(report.get("asymmetry", {}).get("mean_pred_gap", 0.0) or 0.0))

        score = 0.0
        score += 1.4 * (c_long + c_short)
        score += 0.8 * (s_long + s_short)
        score += 0.7 * c_edge
        score += 1.0 * (p_long + p_short)
        score -= 0.02 * (mae_long + mae_short)
        score -= 0.01 * asym_gap
        return float(score)
    except Exception:
        return float("nan")


def run(cfg: EvalConfig) -> Path:
    report = evaluate(cfg)
    report["score"] = _score(report)
    out_path = Path(cfg.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _parse_args(argv: Iterable[str] | None = None) -> EvalConfig:
    ap = argparse.ArgumentParser(description="Avalia regressors walk-forward (long/short)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="data/generated/overnight_regression/eval_latest.json")
    ap.add_argument("--symbols-limit", type=int, default=80)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--total-days-cache", type=int, default=0)
    ap.add_argument("--window-min", type=int, default=60)
    ap.add_argument("--mcap-symbols-file", default=None)
    ap.add_argument("--mcap-min-usd", type=float, default=50_000_000.0)
    ap.add_argument("--mcap-max-usd", type=float, default=150_000_000_000.0)
    ns = ap.parse_args(list(argv) if argv is not None else None)
    return EvalConfig(
        run_dir=str(ns.run_dir),
        out_path=str(ns.out),
        symbols_limit=int(ns.symbols_limit),
        days=int(ns.days),
        total_days_cache=int(ns.total_days_cache),
        window_min=int(ns.window_min),
        mcap_symbols_file=ns.mcap_symbols_file,
        mcap_min_usd=float(ns.mcap_min_usd),
        mcap_max_usd=float(ns.mcap_max_usd),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = run(cfg)
    print(f"[eval-reg] report: {out}", flush=True)


if __name__ == "__main__":
    main()

