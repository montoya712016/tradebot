import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modules"))

from backtest.portfolio import (  # noqa: E402
    PortfolioDemoSettings,
    _default_portfolio_cfg,
    prepare_portfolio_data,
    run_prepared_portfolio,
)
from config.trade_contract import build_default_crypto_contract, apply_crypto_pipeline_env  # noqa: E402
from plotting.plotting import plot_equity_and_correlation  # noqa: E402


MILESTONES = [1440, 1260, 1080, 900, 720, 540, 360, 180]
PARAM_COLS = [
    "label_profit_thr",
    "exit_ema_span_min",
    "exit_ema_init_offset_pct",
    "entry_ratio_neg_per_pos",
    "calib_tail_blend",
    "calib_tail_boost",
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
]


def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _to_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(default)


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return series.rank(pct=True, ascending=ascending, method="average").fillna(0.0)


def _build_robust_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    numeric_cols = [
        "score",
        "ret_pct",
        "max_dd",
        "profit_factor",
        "trades",
        "month_pos_frac",
        "semester_pos_frac",
        "top1_share_pos",
        "avg_trades_per_cluster",
    ] + PARAM_COLS
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work["score"] = work.get("score", 0.0).fillna(0.0)
    work["ret_pct"] = work.get("ret_pct", 0.0).fillna(0.0)
    work["max_dd"] = work.get("max_dd", 1.0).fillna(1.0)
    work["profit_factor"] = work.get("profit_factor", 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["trades"] = work.get("trades", 0.0).fillna(0.0)
    work["month_pos_frac"] = work.get("month_pos_frac", 0.0).fillna(0.0)
    work["semester_pos_frac"] = work.get("semester_pos_frac", 0.0).fillna(0.0)
    work["top1_share_pos"] = work.get("top1_share_pos", 1.0).fillna(1.0)
    work["avg_trades_per_cluster"] = work.get("avg_trades_per_cluster", 10.0).fillna(10.0)

    viable = work[
        (work["score"] > 0.0)
        & (work["ret_pct"] > 0.0)
        & (work["max_dd"] > 0.0)
        & (work["max_dd"] <= 0.35)
        & (work["trades"] >= 30)
        & (work["profit_factor"] >= 1.15)
    ].copy()
    if viable.empty:
        viable = work[(work["score"] > 0.0) & (work["ret_pct"] > 0.0)].copy()
    if viable.empty:
        return viable

    viable["base_score_rank"] = _safe_rank(viable["score"], ascending=True)
    viable["dd_rank"] = _safe_rank(viable["max_dd"], ascending=False)
    viable["pf_rank"] = _safe_rank(viable["profit_factor"], ascending=True)
    viable["trade_rank"] = _safe_rank(viable["trades"], ascending=True)
    viable["month_rank"] = _safe_rank(viable["month_pos_frac"], ascending=True)
    viable["semester_rank"] = _safe_rank(viable["semester_pos_frac"], ascending=True)
    viable["top1_rank"] = _safe_rank(viable["top1_share_pos"], ascending=False)
    viable["cluster_rank"] = _safe_rank(viable["avg_trades_per_cluster"], ascending=False)

    q_cut = viable["score"].quantile(0.75)
    elite = viable[viable["score"] >= q_cut].copy()
    if len(elite) >= 40:
        viable = elite

    param_frame = viable.reindex(columns=PARAM_COLS).copy()
    param_frame = param_frame.fillna(param_frame.median(numeric_only=True)).fillna(0.0)
    scaled = pd.DataFrame(index=param_frame.index)
    for col in param_frame.columns:
        s = pd.to_numeric(param_frame[col], errors="coerce").fillna(0.0)
        lo = float(s.min())
        hi = float(s.max())
        if hi > lo:
            scaled[col] = (s - lo) / (hi - lo)
        else:
            scaled[col] = 0.5

    arr = scaled.to_numpy(dtype=float)
    n = len(scaled)
    k = max(12, min(36, int(np.sqrt(n) * 1.5)))
    robust_rows = []
    for row_idx, ix in enumerate(scaled.index):
        dists = np.sqrt(((arr - arr[row_idx]) ** 2).sum(axis=1))
        order = np.argsort(dists)
        neigh_idx = order[1 : min(len(order), k + 1)]
        if len(neigh_idx) == 0:
            neighbor_mean = float(viable.loc[ix, "score"])
            neighbor_q25 = float(viable.loc[ix, "score"])
            neighbor_std = 0.0
            neighbor_dd = float(viable.loc[ix, "max_dd"])
        else:
            neigh_scores = viable.iloc[neigh_idx]["score"].to_numpy(dtype=float)
            neigh_dd = viable.iloc[neigh_idx]["max_dd"].to_numpy(dtype=float)
            neighbor_mean = float(np.mean(neigh_scores))
            neighbor_q25 = float(np.quantile(neigh_scores, 0.25))
            neighbor_std = float(np.std(neigh_scores))
            neighbor_dd = float(np.median(neigh_dd))
        robust_rows.append(
            {
                "neighbor_mean_score": neighbor_mean,
                "neighbor_q25_score": neighbor_q25,
                "neighbor_score_std": neighbor_std,
                "neighbor_dd_median": neighbor_dd,
            }
        )
    robust_df = pd.DataFrame(robust_rows, index=viable.index)
    viable = viable.join(robust_df)

    viable["neighbor_mean_rank"] = _safe_rank(viable["neighbor_mean_score"], ascending=True)
    viable["neighbor_q25_rank"] = _safe_rank(viable["neighbor_q25_score"], ascending=True)
    viable["neighbor_std_rank"] = _safe_rank(viable["neighbor_score_std"], ascending=False)
    viable["neighbor_dd_rank"] = _safe_rank(viable["neighbor_dd_median"], ascending=False)

    viable["robust_score"] = (
        viable["base_score_rank"] * 0.18
        + viable["neighbor_mean_rank"] * 0.24
        + viable["neighbor_q25_rank"] * 0.26
        + viable["dd_rank"] * 0.08
        + viable["pf_rank"] * 0.06
        + viable["trade_rank"] * 0.04
        + viable["month_rank"] * 0.04
        + viable["semester_rank"] * 0.04
        + viable["top1_rank"] * 0.03
        + viable["cluster_rank"] * 0.03
    )
    viable["robust_score"] *= (0.85 + 0.15 * viable["neighbor_std_rank"])
    viable["robust_score"] *= (0.85 + 0.15 * viable["neighbor_dd_rank"])
    viable = viable.sort_values(
        ["robust_score", "neighbor_q25_score", "neighbor_mean_score", "score"],
        ascending=False,
        na_position="last",
    )
    return viable


def get_best_individual(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df = df[(df["stage"] == "backtest") & (df["status"] == "ok")].copy()
    if df.empty:
        return None

    robust = _build_robust_candidates(df)
    if not robust.empty:
        return robust.iloc[0].to_dict()

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.sort_values("score", ascending=False, na_position="last")
    else:
        df["ret_pct"] = pd.to_numeric(df.get("ret_pct"), errors="coerce")
        df = df.sort_values("ret_pct", ascending=False, na_position="last")
    return df.iloc[0].to_dict()


def build_contract_from_row(row, candle_sec: int):
    os.environ["CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT"] = str(_to_float(row.get("label_profit_thr"), 0.03))
    os.environ["CRYPTO_EXIT_EMA_SPAN_MINUTES"] = str(_to_int(row.get("exit_ema_span_min"), 120))
    os.environ["CRYPTO_EXIT_EMA_INIT_OFFSET_PCT"] = str(_to_float(row.get("exit_ema_init_offset_pct"), 0.005))
    return build_default_crypto_contract(candle_sec)


def build_cfg_from_row(row):
    cfg = _default_portfolio_cfg()
    cfg.max_positions = _to_int(row.get("max_positions"), cfg.max_positions)
    cfg.total_exposure = _to_float(row.get("total_exposure"), cfg.total_exposure)
    cfg.max_trade_exposure = _to_float(row.get("max_trade_exposure"), cfg.max_trade_exposure)
    cfg.min_trade_exposure = _to_float(row.get("min_trade_exposure"), cfg.min_trade_exposure)
    cfg.corr_filter_enabled = _to_bool(row.get("corr_enabled"), cfg.corr_filter_enabled)
    cfg.corr_open_filter_enabled = _to_bool(row.get("corr_enabled"), cfg.corr_open_filter_enabled)
    cfg.corr_max_with_market = _to_float(row.get("corr_max_with_market"), cfg.corr_max_with_market)
    cfg.corr_max_pair = _to_float(row.get("corr_max_pair"), cfg.corr_max_pair)
    cfg.corr_open_reduce_start = _to_float(row.get("corr_open_reduce_start"), cfg.corr_open_reduce_start)
    cfg.corr_open_hard_reject = _to_float(row.get("corr_open_hard_reject"), cfg.corr_open_hard_reject)
    cfg.corr_open_min_weight_mult = _to_float(row.get("corr_open_min_weight_mult"), cfg.corr_open_min_weight_mult)
    cfg.exposure_multiplier = _to_float(row.get("exposure_multiplier"), getattr(cfg, "exposure_multiplier", 0.0))
    return cfg


def _mode() -> str:
    raw = str(os.getenv("WF_OOS_MODE", "reuse") or "reuse").strip().lower()
    if raw in {"retrain", "train"}:
        return "retrain"
    return "reuse"


def build_train_env_from_row(row, oos_target_tail: int):
    return {
        "CRYPTO_PIPELINE_CANDLE_SEC": "300",
        "SNIPER_CANDLE_SEC": "300",
        "PF_CRYPTO_CANDLE_SEC": "300",
        "SNIPER_REMOVE_TAIL_DAYS": str(int(oos_target_tail)),
        "CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT": str(_to_float(row.get("label_profit_thr"), 0.03)),
        "CRYPTO_EXIT_EMA_SPAN_MINUTES": str(_to_int(row.get("exit_ema_span_min"), 120)),
        "CRYPTO_EXIT_EMA_INIT_OFFSET_PCT": str(_to_float(row.get("exit_ema_init_offset_pct"), 0.005)),
        "TRAIN_ENTRY_RATIO_NEG_PER_POS": str(_to_float(row.get("entry_ratio_neg_per_pos"), 4.0)),
        "TRAIN_REFRESH_LABELS": "1",
        "TRAIN_MAX_SYMBOLS": "0",
        "TRAIN_OFFSETS_DAYS": "180",
        "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BLEND": str(_to_float(row.get("calib_tail_blend"), 0.75)),
        "TRAIN_OVR_SNIPER_ENTRY_CALIB_TAIL_BOOST": str(_to_float(row.get("calib_tail_boost"), 1.25)),
        "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_QS": str(row.get("top_metric_qs") or "0.0005,0.001,0.0025"),
        "TRAIN_OVR_SNIPER_ENTRY_TOP_METRIC_MIN_COUNT": str(_to_int(row.get("top_metric_min_count"), 48)),
    }


def run_train_subprocess(env_overrides, log_path: Path):
    cmd = [sys.executable, "-u", str((repo_root / "scripts" / "train.py").resolve())]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    start_t = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_lines = []
    run_dir_found = None
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line)
        try:
            safe_line = line.rstrip().encode("ascii", errors="replace").decode("ascii", errors="replace")
            print(safe_line, flush=True)
        except Exception:
            pass
        match = re.search(r"run_dir:\s*(.+)", line)
        if match:
            run_dir_found = match.group(1).strip()
    proc.wait()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(log_lines), encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"train subprocess failed code={proc.returncode}")
    if not run_dir_found:
        raise RuntimeError("could not resolve retrain run_dir")
    duration_min = (time.time() - start_t) / 60.0
    return run_dir_found, duration_min


def main():
    fair_root = repo_root / "data" / "generated" / "fair_wf_explore"
    mode = _mode()
    completed_steps = {
        int(p.name.split("_")[1].replace("d", "")): p
        for p in fair_root.glob("step_*d")
        if (p / ".finished").exists()
    }

    if mode == "retrain":
        source_steps = [m for m in MILESTONES[:-2] if m in completed_steps]
    else:
        source_steps = [m for m in MILESTONES[:-1] if m in completed_steps]
    if not source_steps:
        print("[ERROR] No completed source steps found.")
        return

    combined_rets = []
    segment_rows = []
    report_dir = fair_root / "robustness_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] OOS mode: {mode}")

    for m_source in source_steps:
        idx = MILESTONES.index(m_source)
        if mode == "retrain":
            m_retrain = MILESTONES[idx + 1]
            m_target = MILESTONES[idx + 2]
        else:
            m_retrain = None
            m_target = MILESTONES[idx + 1]
        step_dir = fair_root / f"step_{m_source}d"

        best = get_best_individual(step_dir / "explore_runs.csv")
        if best is None:
            print(f"[SKIP] No viable best individual found in {m_source}d")
            continue

        days = 180
        os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(m_target)

        print(f"\n{'-' * 48}")
        if mode == "retrain":
            print(f"WF OOS Period: retrain T-{m_retrain} -> test T-{m_retrain}..T-{m_target}")
        else:
            print(f"WF OOS Period: reuse winner from T-{m_source} -> test T-{m_source}..T-{m_target}")
        print(f"Selected from step T-{m_source}: {best.get('label_id')}/{best.get('model_id')}/{best.get('backtest_id')}")
        print(
            "Selection: "
            f"score={_to_float(best.get('score'), 0.0):.4f} "
            f"robust={_to_float(best.get('robust_score'), 0.0):.4f} "
            f"nq25={_to_float(best.get('neighbor_q25_score'), 0.0):.4f} "
            f"nmean={_to_float(best.get('neighbor_mean_score'), 0.0):.4f}"
        )
        print(f"{'-' * 48}")

        try:
            candle_sec = apply_crypto_pipeline_env(300)
            contract = build_contract_from_row(best, candle_sec)
            cfg = build_cfg_from_row(best)
            tau_entry = _to_float(best.get("tau_entry"), 0.5)
            if mode == "retrain":
                retrain_env = build_train_env_from_row(best, m_target)
                retrain_log = report_dir / f"retrain_{m_source}d_to_{m_retrain}d_for_{m_target}d.log"
                retrain_run_dir, retrain_minutes = run_train_subprocess(retrain_env, retrain_log)
                run_dir = retrain_run_dir
                plot_out = str(report_dir / f"oos_segment_retrained_{m_retrain}d_to_{m_target}d.html")
            else:
                run_dir = str(best.get("train_run_dir") or "").strip()
                if not run_dir:
                    print(f"[SKIP] Missing train_run_dir in {m_source}d best individual")
                    continue
                retrain_run_dir = ""
                retrain_minutes = 0.0
                plot_out = str(report_dir / f"oos_segment_reuse_{m_source}d_to_{m_target}d.html")

            settings = PortfolioDemoSettings(
                asset_class="crypto",
                run_dir=run_dir,
                days=days,
                max_symbols=0,
                cfg=cfg,
                save_plot=True,
                plot_out=plot_out,
                override_tau_entry=tau_entry,
                candle_sec=candle_sec,
                contract=contract,
                long_only=True,
                align_global_window=True,
            )
            prepared = prepare_portfolio_data(settings)
            res = run_prepared_portfolio(
                prepared,
                cfg=cfg,
                days=days,
                override_tau_entry=tau_entry,
                save_plot=True,
                plot_out=plot_out,
            )
            curve = res["result"].equity_curve
            seg_rets = curve.pct_change().fillna(0)
            combined_rets.append(seg_rets)

            ret_total = float(res.get("ret_total", 0.0))
            max_dd = float(getattr(res["result"], "max_dd", 0.0))
            trades = len(getattr(res["result"], "trades", []))
            segment_rows.append(
                {
                    "mode": mode,
                    "source_step": m_source,
                    "retrain_step": m_retrain if m_retrain is not None else "",
                    "target_step": m_target,
                    "label_id": best.get("label_id", ""),
                    "model_id": best.get("model_id", ""),
                    "backtest_id": best.get("backtest_id", ""),
                    "selected_train_run_dir": str(best.get("train_run_dir") or ""),
                    "retrained_run_dir": retrain_run_dir,
                    "retrain_minutes": retrain_minutes,
                    "tau_entry": tau_entry,
                    "is_score": _to_float(best.get("score"), 0.0),
                    "robust_score": _to_float(best.get("robust_score"), 0.0),
                    "neighbor_q25_score": _to_float(best.get("neighbor_q25_score"), 0.0),
                    "neighbor_mean_score": _to_float(best.get("neighbor_mean_score"), 0.0),
                    "ret_total": ret_total,
                    "max_dd": max_dd,
                    "trades": trades,
                    "plot_out": plot_out,
                    "window_info": res.get("window_info", ""),
                }
            )
            print(f"  Result -> Return: {ret_total:+.2%}, MaxDD: {max_dd:.2%}, Trades: {trades}")
        except Exception as e:
            if mode == "retrain":
                print(f"  [ERROR] OOS retrain/backtest failed for selection {m_source}d -> retrain {m_retrain}d -> test {m_target}d: {e}")
            else:
                print(f"  [ERROR] OOS reuse backtest failed for selection {m_source}d -> test {m_target}d: {e}")

    if not combined_rets:
        print("\n[ERROR] No OOS segments generated.")
        return

    full_rets = pd.concat(combined_rets)
    full_rets = full_rets[~full_rets.index.duplicated(keep="first")]
    full_rets = full_rets.sort_index()
    full_equity = (1.0 + full_rets).cumprod()

    mode_suffix = "retrain" if mode == "retrain" else "reuse"
    full_equity.to_csv(report_dir / f"walkforward_oos_equity_{mode_suffix}.csv")
    full_equity.to_csv(report_dir / "walkforward_oos_equity.csv")
    pd.DataFrame(segment_rows).to_csv(report_dir / f"walkforward_oos_segments_{mode_suffix}.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(report_dir / "walkforward_oos_segments.csv", index=False)

    ret = (full_equity.iloc[-1] / full_equity.iloc[0]) - 1
    peaks = full_equity.cummax()
    dd = (full_equity - peaks) / peaks
    max_dd = abs(dd.min())
    full_plot_path = report_dir / f"walkforward_oos_equity_{mode_suffix}.html"
    full_plot_latest_path = report_dir / "walkforward_oos_equity.html"
    plot_title = (
        f"OOS Walk-Forward Equity ({mode_suffix}) | "
        f"{full_equity.index[0].date()}..{full_equity.index[-1].date()} | "
        f"ret={ret:+.2%} maxDD={max_dd:.2%} segments={len(segment_rows)}"
    )
    plot_equity_and_correlation(
        full_equity,
        title=plot_title,
        save_path=full_plot_path,
        show=False,
    )
    plot_equity_and_correlation(
        full_equity,
        title=plot_title,
        save_path=full_plot_latest_path,
        show=False,
    )

    print(f"\n[OK] Stitched OOS Equity saved to {report_dir / f'walkforward_oos_equity_{mode_suffix}.csv'}")
    print(f"[OK] Segment summary saved to {report_dir / f'walkforward_oos_segments_{mode_suffix}.csv'}")
    print(f"[OK] Full OOS HTML saved to {full_plot_path}")
    print("\n" + "=" * 60)
    print("FINAL CONSOLIDATED OOS ROBUSTNESS RESULTS")
    print("=" * 60)
    print(f"Total Period: {full_equity.index[0].date()} to {full_equity.index[-1].date()}")
    print(f"Cumulative Return: {ret:+.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Recovery Factor: {abs(ret / max_dd):.2f}" if max_dd > 0 else "Recovery Factor: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
