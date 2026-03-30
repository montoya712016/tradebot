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
    "max_positions",
    "total_exposure",
    "max_trade_exposure",
    "min_trade_exposure",
    "exposure_multiplier",
]
CLUSTER_PARAM_COLS = [
    "label_profit_thr",
    "exit_ema_span_min",
    "exit_ema_init_offset_pct",
    "entry_ratio_neg_per_pos",
    "calib_tail_blend",
    "calib_tail_boost",
    "top_metric_min_count",
    "tau_entry",
    "max_positions",
    "total_exposure",
    "max_trade_exposure",
    "min_trade_exposure",
    "exposure_multiplier",
]


def _fair_root() -> Path:
    raw = str(os.getenv("WF_FAIR_ROOT", "fair_wf_explore_v5") or "fair_wf_explore_v5").strip()
    return repo_root / "data" / "generated" / raw


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


def _build_global_param_scaler(step_paths: dict[int, Path]) -> dict[str, pd.Series]:
    frames = []
    cols = list(dict.fromkeys(PARAM_COLS + CLUSTER_PARAM_COLS))
    for step_dir in step_paths.values():
        csv_path = step_dir / "explore_runs.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        df = df[(df["stage"].astype(str) == "backtest") & (df["status"].astype(str) == "ok")].copy()
        if df.empty:
            continue
        frame = df.reindex(columns=cols).copy()
        for col in cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame)
    if not frames:
        idx = pd.Index(cols)
        return {
            "mins": pd.Series(0.0, index=idx),
            "span": pd.Series(1.0, index=idx),
        }
    pool = pd.concat(frames, axis=0, ignore_index=True)
    med = pool.median(numeric_only=True)
    pool = pool.fillna(med).fillna(0.0)
    mins = pool.min()
    maxs = pool.max()
    span = (maxs - mins).replace(0.0, 1.0)
    return {"mins": mins, "span": span}


def _normalize_with_scaler(frame: pd.DataFrame, cols: list[str], scaler: dict[str, pd.Series] | None) -> np.ndarray:
    work = frame.reindex(columns=cols).copy()
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if scaler is None:
        med = work.median(numeric_only=True)
        work = work.fillna(med).fillna(0.0)
        mins = work.min()
        span = (work.max() - mins).replace(0.0, 1.0)
    else:
        mins = scaler["mins"].reindex(cols).fillna(0.0)
        span = scaler["span"].reindex(cols).fillna(1.0).replace(0.0, 1.0)
        med = mins + (span * 0.5)
        work = work.fillna(med).fillna(0.0)
    return ((work - mins) / span).to_numpy(dtype=float)


def _calc_selection_score(row: pd.Series) -> float:
    ret = _to_float(row.get("ret_pct"), 0.0) * 100.0
    dd = _to_float(row.get("max_dd"), 0.0)
    pf = _to_float(row.get("profit_factor"), 1.0)
    trades = _to_float(row.get("trades"), 100.0)
    month_pos = _to_float(row.get("month_pos_frac"), 0.0)
    streak = _to_float(row.get("max_neg_month_streak"), 0.0)
    underwater = _to_float(row.get("underwater_frac"), 1.0)
    worst_90d = _to_float(row.get("worst_rolling_90d"), 0.0)
    if ret <= 0.0 or dd > 0.30:
        return 0.0
    dd_penalty = np.exp(-8.0 * dd) * np.exp(-18.0 * max(0.0, dd - 0.12))
    smoothed_ret = np.sqrt(ret) * 9.0
    trade_mult = min(1.0, trades / 120.0)
    pf_mult = min(1.25, max(0.85, pf))
    month_support = 0.80 + 0.20 * np.clip(month_pos, 0.0, 1.0)
    streak_penalty = np.exp(-0.55 * max(0.0, streak))
    underwater_penalty = np.exp(-2.25 * np.clip(underwater, 0.0, 1.0))
    regime_penalty = np.exp(-7.5 * max(0.0, -worst_90d))
    return float(smoothed_ret * dd_penalty * trade_mult * pf_mult * month_support * streak_penalty * underwater_penalty * regime_penalty)


def _equity_metrics_from_bt_dir(bt_out_dir: str) -> dict[str, float]:
    csv_path = Path(str(bt_out_dir or "").strip()) / "portfolio_equity.csv"
    metrics = {
        "month_pos_frac": 0.0,
        "month_worst": 0.0,
        "max_neg_month_streak": 0.0,
        "underwater_frac": 1.0,
        "worst_rolling_90d": 0.0,
    }
    try:
        eq = pd.read_csv(csv_path, index_col=0)
        eq.index = pd.to_datetime(eq.index)
        if "equity" in eq.columns and not eq.empty:
            eq_s = pd.to_numeric(eq["equity"], errors="coerce").dropna()
            month_ret = eq_s.resample("ME").last().pct_change().dropna()
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
            metrics = {
                "month_pos_frac": float((month_ret > 0.0).mean()) if len(month_ret) else 0.0,
                "month_worst": float(month_ret.min()) if len(month_ret) else 0.0,
                "max_neg_month_streak": float(max_streak),
                "underwater_frac": float(underwater.mean()) if len(underwater) else 1.0,
                "worst_rolling_90d": float(rolling_90d.min()) if rolling_90d.notna().any() else 0.0,
            }
    except Exception:
        pass
    return metrics


def _assign_param_clusters(arr: np.ndarray) -> tuple[np.ndarray, float]:
    n = int(len(arr))
    if n <= 0:
        return np.empty((0,), dtype=int), 0.0
    if n == 1:
        return np.zeros((1,), dtype=int), 0.0

    dmat = np.sqrt(np.sum((arr[:, None, :] - arr[None, :, :]) ** 2, axis=2))
    k = max(2, min(8, n - 1))
    kth = np.partition(dmat, kth=k, axis=1)[:, k]
    radius = float(np.median(kth) * 1.15)
    radius = max(radius, 0.55)

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


def _build_robust_candidates(
    df: pd.DataFrame,
    *,
    prior_cluster_memory: list[dict] | None = None,
    global_scaler: dict[str, pd.Series] | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    selector_mode = _selector_mode()
    work = df.copy()
    numeric_cols = [
        "score",
        "ret_pct",
        "max_dd",
        "profit_factor",
        "trades",
        "month_pos_frac",
        "max_neg_month_streak",
        "underwater_frac",
        "worst_rolling_90d",
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
    work["max_neg_month_streak"] = work.get("max_neg_month_streak", 0.0).fillna(0.0)
    work["underwater_frac"] = work.get("underwater_frac", 1.0).fillna(1.0)
    work["worst_rolling_90d"] = work.get("worst_rolling_90d", 0.0).fillna(0.0)
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
    viable["streak_rank"] = _safe_rank(viable["max_neg_month_streak"], ascending=False)
    viable["underwater_rank"] = _safe_rank(viable["underwater_frac"], ascending=False)
    viable["roll90_rank"] = _safe_rank(viable["worst_rolling_90d"], ascending=True)
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

    viable["local_robust_score"] = (
        viable["base_score_rank"] * 0.18
        + viable["neighbor_mean_rank"] * 0.24
        + viable["neighbor_q25_rank"] * 0.26
        + viable["dd_rank"] * 0.08
        + viable["pf_rank"] * 0.06
        + viable["trade_rank"] * 0.04
        + viable["month_rank"] * 0.03
        + viable["streak_rank"] * 0.05
        + viable["underwater_rank"] * 0.05
        + viable["roll90_rank"] * 0.04
        + viable["top1_rank"] * 0.03
        + viable["cluster_rank"] * 0.02
    )
    viable["local_robust_score"] *= (0.85 + 0.15 * viable["neighbor_std_rank"])
    viable["local_robust_score"] *= (0.85 + 0.15 * viable["neighbor_dd_rank"])
    viable["local_dd_score"] = (
        viable["base_score_rank"] * 0.12
        + viable["neighbor_mean_rank"] * 0.18
        + viable["neighbor_q25_rank"] * 0.22
        + viable["dd_rank"] * 0.16
        + viable["neighbor_dd_rank"] * 0.10
        + viable["pf_rank"] * 0.06
        + viable["trade_rank"] * 0.05
        + viable["month_rank"] * 0.03
        + viable["streak_rank"] * 0.07
        + viable["underwater_rank"] * 0.07
        + viable["roll90_rank"] * 0.05
        + viable["top1_rank"] * 0.02
        + viable["cluster_rank"] * 0.01
    )
    viable["local_dd_score"] *= (0.82 + 0.18 * viable["neighbor_std_rank"])

    cluster_frame = viable.reindex(columns=CLUSTER_PARAM_COLS).copy()
    cluster_frame = cluster_frame.fillna(cluster_frame.median(numeric_only=True)).fillna(0.0)
    cluster_scaled = pd.DataFrame(index=cluster_frame.index)
    for col in cluster_frame.columns:
        s = pd.to_numeric(cluster_frame[col], errors="coerce").fillna(0.0)
        lo = float(s.min())
        hi = float(s.max())
        if hi > lo:
            cluster_scaled[col] = (s - lo) / (hi - lo)
        else:
            cluster_scaled[col] = 0.5
    cluster_arr = cluster_scaled.to_numpy(dtype=float)
    global_cluster_arr = _normalize_with_scaler(viable, CLUSTER_PARAM_COLS, global_scaler)

    labels, cluster_radius = _assign_param_clusters(cluster_arr)
    viable["cluster_id"] = labels
    viable["cluster_radius"] = float(cluster_radius)

    centers = {}
    center_dist = np.zeros(len(viable), dtype=float)
    for cid in sorted(pd.unique(viable["cluster_id"])):
        mask = viable["cluster_id"].to_numpy(dtype=int) == int(cid)
        center = np.median(cluster_arr[mask], axis=0)
        centers[int(cid)] = center
        center_dist[mask] = np.sqrt(np.sum((cluster_arr[mask] - center) ** 2, axis=1))
    viable["cluster_center_dist"] = center_dist
    viable["cluster_center_rank"] = _safe_rank(viable["cluster_center_dist"], ascending=False)
    global_centers = {
        int(cid): np.median(global_cluster_arr[viable["cluster_id"].to_numpy(dtype=int) == int(cid)], axis=0)
        for cid in sorted(pd.unique(viable["cluster_id"]))
    }

    cluster_stats = (
        viable.groupby("cluster_id", as_index=False)
        .agg(
            cluster_size=("cluster_id", "size"),
            cluster_local_score_median=("local_robust_score", "median"),
            cluster_local_score_q25=("local_robust_score", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.25))),
            cluster_raw_score_median=("score", "median"),
            cluster_raw_score_q25=("score", lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.25))),
            cluster_dd_median=("max_dd", "median"),
            cluster_pf_median=("profit_factor", "median"),
            cluster_trade_median=("trades", "median"),
            cluster_month_median=("month_pos_frac", "median"),
            cluster_streak_median=("max_neg_month_streak", "median"),
            cluster_underwater_median=("underwater_frac", "median"),
            cluster_roll90_median=("worst_rolling_90d", "median"),
            cluster_neighbor_q25=("neighbor_q25_score", "median"),
            cluster_neighbor_mean=("neighbor_mean_score", "median"),
            cluster_dispersion=("cluster_center_dist", "median"),
        )
        .copy()
    )
    cluster_stats["cluster_global_center"] = cluster_stats["cluster_id"].map(global_centers)
    cluster_stats["cluster_size_rank"] = _safe_rank(cluster_stats["cluster_size"], ascending=True)
    cluster_stats["cluster_local_median_rank"] = _safe_rank(cluster_stats["cluster_local_score_median"], ascending=True)
    cluster_stats["cluster_local_q25_rank"] = _safe_rank(cluster_stats["cluster_local_score_q25"], ascending=True)
    cluster_stats["cluster_raw_median_rank"] = _safe_rank(cluster_stats["cluster_raw_score_median"], ascending=True)
    cluster_stats["cluster_raw_q25_rank"] = _safe_rank(cluster_stats["cluster_raw_score_q25"], ascending=True)
    cluster_stats["cluster_dd_rank"] = _safe_rank(cluster_stats["cluster_dd_median"], ascending=False)
    cluster_stats["cluster_pf_rank"] = _safe_rank(cluster_stats["cluster_pf_median"], ascending=True)
    cluster_stats["cluster_trade_rank"] = _safe_rank(cluster_stats["cluster_trade_median"], ascending=True)
    cluster_stats["cluster_month_rank"] = _safe_rank(cluster_stats["cluster_month_median"], ascending=True)
    cluster_stats["cluster_streak_rank"] = _safe_rank(cluster_stats["cluster_streak_median"], ascending=False)
    cluster_stats["cluster_underwater_rank"] = _safe_rank(cluster_stats["cluster_underwater_median"], ascending=False)
    cluster_stats["cluster_roll90_rank"] = _safe_rank(cluster_stats["cluster_roll90_median"], ascending=True)
    cluster_stats["cluster_neighbor_q25_rank"] = _safe_rank(cluster_stats["cluster_neighbor_q25"], ascending=True)
    cluster_stats["cluster_neighbor_mean_rank"] = _safe_rank(cluster_stats["cluster_neighbor_mean"], ascending=True)
    cluster_stats["cluster_dispersion_rank"] = _safe_rank(cluster_stats["cluster_dispersion"], ascending=False)
    cluster_stats["cluster_score"] = (
        cluster_stats["cluster_local_q25_rank"] * 0.24
        + cluster_stats["cluster_local_median_rank"] * 0.18
        + cluster_stats["cluster_raw_q25_rank"] * 0.12
        + cluster_stats["cluster_raw_median_rank"] * 0.08
        + cluster_stats["cluster_neighbor_q25_rank"] * 0.12
        + cluster_stats["cluster_neighbor_mean_rank"] * 0.08
        + cluster_stats["cluster_dd_rank"] * 0.08
        + cluster_stats["cluster_pf_rank"] * 0.04
        + cluster_stats["cluster_trade_rank"] * 0.03
        + cluster_stats["cluster_month_rank"] * 0.02
        + cluster_stats["cluster_streak_rank"] * 0.04
        + cluster_stats["cluster_underwater_rank"] * 0.04
        + cluster_stats["cluster_roll90_rank"] * 0.03
        + cluster_stats["cluster_size_rank"] * 0.05
        + cluster_stats["cluster_dispersion_rank"] * 0.02
    )
    prior_mem = list(prior_cluster_memory or [])
    if prior_mem:
        continuity_vals = []
        continuity_cov = []
        for _, row in cluster_stats.iterrows():
            center = row["cluster_global_center"]
            sims = []
            steps_seen = set()
            for mem in prior_mem:
                prev_center = mem.get("center_vec")
                if prev_center is None:
                    continue
                step_gap = 1.0
                if step is not None and mem.get("step") is not None:
                    step_gap = max(1.0, abs(float(mem["step"]) - float(step)) / 180.0)
                dist = float(np.sqrt(np.sum((center - prev_center) ** 2)))
                sim = float(np.exp(-3.2 * dist) * np.exp(-0.30 * (step_gap - 1.0)))
                quality = float(mem.get("quality", 0.5))
                sims.append(sim * (0.75 + 0.25 * quality))
                if sim >= 0.55:
                    steps_seen.add(int(mem.get("step", -1)))
            top = sorted(sims, reverse=True)[:4]
            continuity_vals.append(float(np.mean(top)) if top else 0.0)
            continuity_cov.append(float(len(steps_seen)))
        cluster_stats["causal_continuity"] = continuity_vals
        cluster_stats["causal_coverage_steps"] = continuity_cov
    else:
        cluster_stats["causal_continuity"] = 0.0
        cluster_stats["causal_coverage_steps"] = 0.0
    cluster_stats["causal_cont_rank"] = _safe_rank(cluster_stats["causal_continuity"], ascending=True)
    cluster_stats["causal_cov_rank"] = _safe_rank(cluster_stats["causal_coverage_steps"], ascending=True)
    cluster_stats["causal_cluster_score"] = (
        cluster_stats["cluster_score"] * 0.72
        + cluster_stats["causal_cont_rank"] * 0.20
        + cluster_stats["causal_cov_rank"] * 0.08
    )
    viable = viable.merge(cluster_stats, on="cluster_id", how="left")
    viable["cluster_score"] = viable["cluster_score"].fillna(0.0)
    viable["causal_cluster_score"] = viable["causal_cluster_score"].fillna(0.0)
    viable["causal_continuity"] = viable["causal_continuity"].fillna(0.0)
    viable["causal_coverage_steps"] = viable["causal_coverage_steps"].fillna(0.0)

    if selector_mode == "causal_cluster":
        viable["robust_score"] = (
            viable["causal_cluster_score"] * 0.62
            + viable["local_dd_score"] * 0.23
            + viable["local_robust_score"] * 0.10
            + viable["cluster_center_rank"] * 0.05
        )
        sort_cols = [
            "causal_cluster_score",
            "robust_score",
            "cluster_score",
            "cluster_center_rank",
            "neighbor_q25_score",
            "score",
        ]
    elif selector_mode == "cluster_blend":
        viable["robust_score"] = (
            viable["cluster_score"] * 0.55
            + viable["local_dd_score"] * 0.30
            + viable["local_robust_score"] * 0.10
            + viable["cluster_center_rank"] * 0.05
        )
        sort_cols = ["cluster_score", "robust_score", "cluster_local_score_q25", "cluster_size", "cluster_center_rank", "score"]
    elif selector_mode == "cluster_filter":
        viable["cluster_guard"] = (
            (viable["cluster_size"] >= 2).astype(float) * 0.5
            + (viable["cluster_local_score_q25"] >= viable["cluster_local_score_q25"].median()).astype(float) * 0.3
            + (viable["cluster_dd_median"] <= viable["cluster_dd_median"].median()).astype(float) * 0.2
        )
        viable["robust_score"] = viable["local_dd_score"] * (0.92 + 0.12 * viable["cluster_guard"])
        sort_cols = ["cluster_guard", "robust_score", "cluster_size", "neighbor_q25_score", "score"]
    elif selector_mode == "local":
        viable["robust_score"] = viable["local_robust_score"]
        sort_cols = ["robust_score", "neighbor_q25_score", "neighbor_mean_score", "score"]
    else:
        viable["robust_score"] = viable["local_dd_score"]
        sort_cols = ["robust_score", "dd_rank", "neighbor_q25_score", "neighbor_mean_score", "score"]

    viable["selector_mode"] = selector_mode
    viable = viable.sort_values(sort_cols, ascending=False, na_position="last")
    return viable


def get_best_individual(csv_path: Path):
    return _select_individual(csv_path)


def _extract_cluster_memory(robust: pd.DataFrame, step: int, limit: int = 3) -> list[dict]:
    if robust.empty:
        return []
    cluster_col = "causal_cluster_score" if "causal_cluster_score" in robust.columns else "cluster_score"
    cluster_rows = (
        robust.sort_values(
            [cluster_col, "cluster_score", "cluster_center_rank", "robust_score"],
            ascending=False,
            na_position="last",
        )
        .drop_duplicates(subset=["cluster_id"])
        .head(limit)
    )
    out = []
    for _, row in cluster_rows.iterrows():
        out.append(
            {
                "step": int(step),
                "cluster_id": int(_to_int(row.get("cluster_id"), -1)),
                "center_vec": row.get("cluster_global_center"),
                "quality": float(_to_float(row.get(cluster_col), 0.0)),
                "cluster_score": float(_to_float(row.get("cluster_score"), 0.0)),
            }
        )
    return out


def _build_causal_history_context(
    completed_steps: dict[int, Path],
    prior_steps: list[int],
    *,
    memory_limit: int = 24,
) -> tuple[dict[str, pd.Series] | None, list[dict]]:
    if not prior_steps:
        return None, []

    scaler_steps = {m: completed_steps[m] for m in prior_steps if m in completed_steps}
    global_scaler = _build_global_param_scaler(scaler_steps) if scaler_steps else None

    rolling_memory: list[dict] = []
    for prior_step in prior_steps:
        step_dir = completed_steps.get(int(prior_step))
        if step_dir is None:
            continue
        selected = _select_individual(
            step_dir / "explore_runs.csv",
            prior_cluster_memory=rolling_memory,
            global_scaler=global_scaler,
            step=int(prior_step),
        )
        if not selected:
            continue
        step_memory = list(selected.get("_cluster_memory", []))
        if not step_memory:
            continue
        rolling_memory.extend(step_memory)
        rolling_memory = rolling_memory[-int(memory_limit):]
    return global_scaler, rolling_memory


def _select_individual(
    csv_path: Path,
    *,
    prior_cluster_memory: list[dict] | None = None,
    global_scaler: dict[str, pd.Series] | None = None,
    step: int | None = None,
) -> dict | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df = df[(df["stage"] == "backtest") & (df["status"] == "ok")].copy()
    if df.empty:
        return None
    for col in ("max_neg_month_streak", "underwater_frac", "worst_rolling_90d"):
        if col not in df.columns:
            df[col] = np.nan
    missing_mask = df["max_neg_month_streak"].isna() | df["underwater_frac"].isna() | df["worst_rolling_90d"].isna()
    if missing_mask.any():
        enriched = df.loc[missing_mask, "bt_out_dir"].apply(_equity_metrics_from_bt_dir)
        for col in ("month_pos_frac", "month_worst", "max_neg_month_streak", "underwater_frac", "worst_rolling_90d"):
            if col not in df.columns:
                df[col] = np.nan
            df.loc[missing_mask, col] = enriched.apply(lambda x: x.get(col, np.nan))
    df["score"] = df.apply(_calc_selection_score, axis=1)

    robust = _build_robust_candidates(
        df,
        prior_cluster_memory=prior_cluster_memory,
        global_scaler=global_scaler,
        step=step,
    )
    if not robust.empty:
        cluster_score_col = "causal_cluster_score" if ("causal_cluster_score" in robust.columns and prior_cluster_memory) else "cluster_score"
        best_cluster_id = (
            robust.sort_values(
                [
                    cluster_score_col,
                    "cluster_score",
                    "cluster_size",
                    "cluster_local_score_q25",
                    "robust_score",
                ],
                ascending=False,
                na_position="last",
            )
            .iloc[0]
            .get("cluster_id")
        )
        cluster_rows = robust[robust["cluster_id"] == best_cluster_id].copy()
        if cluster_rows.empty:
            cluster_rows = robust.copy()
        cluster_rows["cluster_center_dist"] = pd.to_numeric(cluster_rows.get("cluster_center_dist"), errors="coerce").fillna(np.inf)
        cluster_rows["neighbor_q25_score"] = pd.to_numeric(cluster_rows.get("neighbor_q25_score"), errors="coerce").fillna(0.0)
        cluster_rows["score"] = pd.to_numeric(cluster_rows.get("score"), errors="coerce").fillna(0.0)
        cluster_rows["max_dd"] = pd.to_numeric(cluster_rows.get("max_dd"), errors="coerce").fillna(np.inf)
        cluster_rows["total_exposure"] = pd.to_numeric(cluster_rows.get("total_exposure"), errors="coerce").fillna(np.inf)
        cluster_rows["max_trade_exposure"] = pd.to_numeric(cluster_rows.get("max_trade_exposure"), errors="coerce").fillna(np.inf)
        cluster_rows = cluster_rows.sort_values(
            [
                cluster_score_col,
                "robust_score",
                "cluster_center_dist",
                "max_dd",
                "total_exposure",
                "max_trade_exposure",
                "neighbor_q25_score",
                "score",
            ],
            ascending=[False, False, True, True, True, True, False, False],
            na_position="last",
        )
        selected = cluster_rows.iloc[0].to_dict()
        selected["_cluster_memory"] = _extract_cluster_memory(robust, int(step or -1))
        return selected

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.sort_values("score", ascending=False, na_position="last")
    else:
        df["ret_pct"] = pd.to_numeric(df.get("ret_pct"), errors="coerce")
        df = df.sort_values("ret_pct", ascending=False, na_position="last")
    selected = df.iloc[0].to_dict()
    selected["_cluster_memory"] = []
    return selected


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
    max_pos_override = os.getenv("WF_OOS_MAX_POSITIONS")
    total_exp_override = os.getenv("WF_OOS_TOTAL_EXPOSURE")
    max_trade_override = os.getenv("WF_OOS_MAX_TRADE_EXPOSURE")
    min_trade_override = os.getenv("WF_OOS_MIN_TRADE_EXPOSURE")
    if max_pos_override not in {None, ""}:
        cfg.max_positions = _to_int(max_pos_override, cfg.max_positions)
    if total_exp_override not in {None, ""}:
        cfg.total_exposure = _to_float(total_exp_override, cfg.total_exposure)
    if max_trade_override not in {None, ""}:
        cfg.max_trade_exposure = _to_float(max_trade_override, cfg.max_trade_exposure)
    if min_trade_override not in {None, ""}:
        cfg.min_trade_exposure = _to_float(min_trade_override, cfg.min_trade_exposure)
    cfg.corr_filter_enabled = False
    cfg.corr_open_filter_enabled = False
    cfg.exposure_multiplier = _to_float(row.get("exposure_multiplier"), getattr(cfg, "exposure_multiplier", 0.0))
    return cfg


def _mode() -> str:
    raw = str(os.getenv("WF_OOS_MODE", "reuse") or "reuse").strip().lower()
    if raw in {"retrain", "train"}:
        return "retrain"
    return "reuse"


def _selector_mode() -> str:
    raw = str(os.getenv("WF_OOS_SELECTOR", "causal_cluster") or "causal_cluster").strip().lower()
    if raw in {"local", "local_dd", "cluster_filter", "cluster_blend", "causal_cluster"}:
        return raw
    return "causal_cluster"


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
    fair_root = _fair_root()
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
    print(f"[INFO] Selector mode: {_selector_mode()}")

    for source_idx, m_source in enumerate(source_steps):
        idx = MILESTONES.index(m_source)
        prior_steps = list(source_steps[:source_idx])
        global_scaler, prior_cluster_memory = _build_causal_history_context(
            completed_steps,
            prior_steps,
        )
        if mode == "retrain":
            m_retrain = MILESTONES[idx + 1]
            m_target = MILESTONES[idx + 2]
            m_test_start = m_retrain
        else:
            m_retrain = None
            m_test_start = MILESTONES[idx + 1]
            m_target = max(0, int(m_source) - 360)
        step_dir = fair_root / f"step_{m_source}d"

        best = _select_individual(
            step_dir / "explore_runs.csv",
            prior_cluster_memory=prior_cluster_memory,
            global_scaler=global_scaler,
            step=m_source,
        )
        if best is None:
            print(f"[SKIP] No viable best individual found in {m_source}d")
            continue
        best.pop("_cluster_memory", None)

        days = 180
        os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(m_target)

        print(f"\n{'-' * 48}")
        if mode == "retrain":
            print(f"WF OOS Period: retrain T-{m_retrain} -> test T-{m_retrain}..T-{m_target}")
        else:
            print(f"WF OOS Period: reuse stable-region representative from T-{m_source} -> test next block T-{m_test_start}..T-{m_target}")
        print(f"Selected representative from step T-{m_source}: {best.get('label_id')}/{best.get('model_id')}/{best.get('backtest_id')}")
        print(
            "Selection: "
            f"score={_to_float(best.get('score'), 0.0):.4f} "
            f"robust={_to_float(best.get('robust_score'), 0.0):.4f} "
            f"cluster={_to_int(best.get('cluster_id'), -1)} "
            f"cluster_score={_to_float(best.get('cluster_score'), 0.0):.4f} "
            f"causal_cluster={_to_float(best.get('causal_cluster_score'), 0.0):.4f} "
            f"continuity={_to_float(best.get('causal_continuity'), 0.0):.4f} "
            f"coverage={_to_float(best.get('causal_coverage_steps'), 0.0):.1f} "
            f"cluster_size={_to_int(best.get('cluster_size'), 0)} "
            f"prior_steps={len(prior_steps)} "
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
                plot_out = str(report_dir / f"oos_segment_reuse_{m_test_start}d_to_{m_target}d_from_{m_source}d.html")

            # Mirror the explore pipeline as closely as possible.
            # The previous OOS path used align_global_window_use_max_end=True,
            # which shifts the effective test window and skips symbols, causing
            # large divergences versus the metrics recorded in explore_runs.csv.
            prepare_cfg = _default_portfolio_cfg()
            prepare_cfg.corr_filter_enabled = False
            prepare_cfg.corr_open_filter_enabled = False
            settings = PortfolioDemoSettings(
                asset_class="crypto",
                run_dir=run_dir,
                days=days,
                max_symbols=0,
                total_days_cache=0,
                symbols=[],
                cfg=prepare_cfg,
                save_plot=False,
                plot_out=None,
                override_tau_entry=None,
                candle_sec=candle_sec,
                contract=contract,
                long_only=True,
                require_feature_cache=True,
                rebuild_on_score_error=False,
                align_global_window=True,
                align_global_window_use_max_end=False,
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
                    "selector_mode": _selector_mode(),
                    "source_step": m_source,
                    "retrain_step": m_retrain if m_retrain is not None else "",
                    "test_start_step": m_test_start,
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
                    "causal_cluster_score": _to_float(best.get("causal_cluster_score"), 0.0),
                    "causal_continuity": _to_float(best.get("causal_continuity"), 0.0),
                    "causal_coverage_steps": _to_float(best.get("causal_coverage_steps"), 0.0),
                    "prior_steps_used": len(prior_steps),
                    "cluster_id": _to_int(best.get("cluster_id"), -1),
                    "cluster_size": _to_int(best.get("cluster_size"), 0),
                    "cluster_score": _to_float(best.get("cluster_score"), 0.0),
                    "neighbor_q25_score": _to_float(best.get("neighbor_q25_score"), 0.0),
                    "neighbor_mean_score": _to_float(best.get("neighbor_mean_score"), 0.0),
                    "ret_total": ret_total,
                    "max_dd": max_dd,
                    "trades": trades,
                    "plot_out": plot_out,
                    "window_info": res.get("window_info", ""),
                    "symbols_total": int(getattr(prepared, "symbols_total", 0)),
                    "symbols_skipped_window": int(getattr(prepared, "symbols_skipped_window", 0)),
                }
            )
            print(f"  Result -> Return: {ret_total:+.2%}, MaxDD: {max_dd:.2%}, Trades: {trades}")
        except Exception as e:
            if mode == "retrain":
                print(f"  [ERROR] OOS retrain/backtest failed for selection {m_source}d -> retrain {m_retrain}d -> test {m_target}d: {e}")
            else:
                print(f"  [ERROR] OOS reuse backtest failed for selection {m_source}d -> test {m_test_start}d..{m_target}d: {e}")

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
