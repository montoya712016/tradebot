from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
FAIR_ROOT = REPO_ROOT / "data" / "generated" / "fair_wf_explore"
STEPS = [1440, 1260, 1080, 900, 720, 540, 360]
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
]
SUMMARY_COLS = [
    "score",
    "ret_pct",
    "max_dd",
    "profit_factor",
    "trades",
    "month_pos_frac",
    "max_neg_month_streak",
    "underwater_frac",
    "worst_rolling_90d",
]


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


def load_step(step: int) -> pd.DataFrame:
    csv_path = FAIR_ROOT / f"step_{step}d" / "explore_runs.csv"
    df = pd.read_csv(csv_path)
    df = df[(df["stage"].astype(str) == "backtest") & (df["status"].astype(str) == "ok")].copy()
    df["step"] = step
    for col in PARAM_COLS + SUMMARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["score"] = df.get("score", 0.0).fillna(0.0)
    return df


def build_good_set(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[
        (work["score"] > 0.0)
        & (work["ret_pct"] > 0.0)
        & (work["max_dd"] <= 0.30)
    ].copy()
    if work.empty:
        return work
    q = work["score"].quantile(0.88)
    good = work[work["score"] >= q].copy()
    if len(good) < 80:
        q = work["score"].quantile(0.80)
        good = work[work["score"] >= q].copy()
    if len(good) < 40:
        good = work.nlargest(min(80, len(work)), "score").copy()
    return good


def normalize_pair(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pool = pd.concat([a[PARAM_COLS], b[PARAM_COLS]], axis=0)
    med = pool.median(numeric_only=True)
    pool = pool.fillna(med).fillna(0.0)
    a_f = a[PARAM_COLS].fillna(med).fillna(0.0)
    b_f = b[PARAM_COLS].fillna(med).fillna(0.0)
    mins = pool.min()
    maxs = pool.max()
    span = (maxs - mins).replace(0.0, 1.0)
    a_n = ((a_f - mins) / span).to_numpy(dtype=float)
    b_n = ((b_f - mins) / span).to_numpy(dtype=float)
    return a_n, b_n


def pair_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    a_n, b_n = normalize_pair(a, b)
    cross = np.sqrt(((a_n[:, None, :] - b_n[None, :, :]) ** 2).sum(axis=2))
    a_to_b = cross.min(axis=1)
    b_to_a = cross.min(axis=0)
    aa = np.sqrt(((a_n[:, None, :] - a_n[None, :, :]) ** 2).sum(axis=2))
    bb = np.sqrt(((b_n[:, None, :] - b_n[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(aa, np.inf)
    np.fill_diagonal(bb, np.inf)
    a_local = np.median(np.min(aa, axis=1))
    b_local = np.median(np.min(bb, axis=1))
    thresh = max(0.18, float(np.median([a_local, b_local]) * 1.35))
    return {
        "threshold": thresh,
        "a_cover": float((a_to_b <= thresh).mean()),
        "b_cover": float((b_to_a <= thresh).mean()),
        "a_med_dist": float(np.median(a_to_b)),
        "b_med_dist": float(np.median(b_to_a)),
    }


def global_cluster_report(good_by_step: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    all_good = pd.concat([df.assign(step=int(step)) for step, df in good_by_step.items()], ignore_index=True)
    med = all_good[PARAM_COLS].median(numeric_only=True)
    feats = all_good[PARAM_COLS].fillna(med).fillna(0.0)
    mins = feats.min()
    maxs = feats.max()
    span = (maxs - mins).replace(0.0, 1.0)
    arr = ((feats - mins) / span).to_numpy(dtype=float)
    labels, radius = _assign_clusters(arr)
    all_good["cluster_id"] = labels
    rows = []
    for cid, grp in all_good.groupby("cluster_id"):
        step_list = sorted(int(x) for x in grp["step"].unique())
        rows.append(
            {
                "cluster_id": int(cid),
                "n": int(len(grp)),
                "steps": ",".join(map(str, step_list)),
                "step_count": int(len(step_list)),
                "score_med": float(grp["score"].median()),
                "ret_med": float(grp["ret_pct"].median() * 100.0),
                "dd_med": float(grp["max_dd"].median() * 100.0),
                "month_pos_med": float(grp["month_pos_frac"].median()),
                "streak_med": float(grp["max_neg_month_streak"].median()),
                "underwater_med": float(grp["underwater_frac"].median()),
                "roll90_worst_med": float(grp["worst_rolling_90d"].median() * 100.0),
                "tau_range": f"{grp['tau_entry'].min():.2f}..{grp['tau_entry'].max():.2f}",
                "maxpos_range": f"{int(grp['max_positions'].min())}..{int(grp['max_positions'].max())}",
                "totexp_range": f"{grp['total_exposure'].min():.2f}..{grp['total_exposure'].max():.2f}",
                "profit_thr_range": f"{grp['label_profit_thr'].min():.3f}..{grp['label_profit_thr'].max():.3f}",
            }
        )
    out = pd.DataFrame(rows).sort_values(["step_count", "n", "score_med"], ascending=[False, False, False])
    return out, radius


def main():
    step_data = {step: load_step(step) for step in STEPS if (FAIR_ROOT / f"step_{step}d" / "explore_runs.csv").exists()}
    good_by_step = {step: build_good_set(df) for step, df in step_data.items()}
    report_dir = FAIR_ROOT / "robustness_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    ordered = [s for s in STEPS if s in good_by_step]
    for left, right in zip(ordered[:-1], ordered[1:]):
        a = good_by_step[left]
        b = good_by_step[right]
        if a.empty or b.empty:
            continue
        overlap = pair_overlap(a, b)
        pair_rows.append(
            {
                "step_left": left,
                "step_right": right,
                "left_good_n": len(a),
                "right_good_n": len(b),
                **overlap,
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(report_dir / "fair_param_overlap_pairs.csv", index=False)

    cluster_df, radius = global_cluster_report(good_by_step)
    cluster_df.to_csv(report_dir / "fair_param_overlap_clusters.csv", index=False)

    print("=" * 72)
    print("PARAMETER OVERLAP VERIFICATION")
    print("=" * 72)
    if not pair_df.empty:
        print("\nAdjacent-step overlap among top-score regions:")
        for _, row in pair_df.iterrows():
            print(
                f"  {int(row['step_left'])}d -> {int(row['step_right'])}d | "
                f"cover L={row['a_cover']*100:.1f}% R={row['b_cover']*100:.1f}% | "
                f"med_dist L={row['a_med_dist']:.3f} R={row['b_med_dist']:.3f} | "
                f"th={row['threshold']:.3f}"
            )
    stable = cluster_df[cluster_df["step_count"] >= 3].copy()
    print(f"\nGlobal cluster radius: {radius:.3f}")
    print(f"Stable clusters across >=3 steps: {len(stable)}")
    if not stable.empty:
        for _, row in stable.head(12).iterrows():
            print(
                f"  cluster {int(row['cluster_id'])}: steps={row['steps']} n={int(row['n'])} "
                f"score_med={row['score_med']:.2f} ret_med={row['ret_med']:+.1f}% dd_med={row['dd_med']:.1f}% "
                f"month_pos={row['month_pos_med']:.2f} streak={row['streak_med']:.1f} "
                f"underwater={row['underwater_med']:.2f} roll90={row['roll90_worst_med']:+.1f}% "
                f"tau={row['tau_range']} max_pos={row['maxpos_range']} total_exp={row['totexp_range']}"
            )
    print("\nSaved:")
    print(f"  {report_dir / 'fair_param_overlap_pairs.csv'}")
    print(f"  {report_dir / 'fair_param_overlap_clusters.csv'}")


if __name__ == "__main__":
    main()
