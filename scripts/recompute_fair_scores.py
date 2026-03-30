import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
FAIR_ROOT = REPO_ROOT / "data" / "generated" / "fair_wf_explore"
TARGET_COLS = [
    "month_pos_frac",
    "month_worst",
    "max_neg_month_streak",
    "underwater_frac",
    "worst_rolling_90d",
    "score",
]


def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _equity_metrics(bt_out_dir: str) -> dict[str, float]:
    csv_path = Path(str(bt_out_dir or "").strip()) / "portfolio_equity.csv"
    metrics = {
        "month_pos_frac": 0.0,
        "month_worst": 0.0,
        "max_neg_month_streak": 0.0,
        "underwater_frac": 1.0,
        "worst_rolling_90d": 0.0,
    }
    if not csv_path.exists():
        return metrics
    eq = pd.read_csv(csv_path, index_col=0)
    eq.index = pd.to_datetime(eq.index)
    if "equity" not in eq.columns or eq.empty:
        return metrics
    eq_s = pd.to_numeric(eq["equity"], errors="coerce").dropna()
    if eq_s.empty:
        return metrics
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
    return {
        "month_pos_frac": float((month_ret > 0.0).mean()) if len(month_ret) else 0.0,
        "month_worst": float(month_ret.min()) if len(month_ret) else 0.0,
        "max_neg_month_streak": float(max_streak),
        "underwater_frac": float(underwater.mean()) if len(underwater) else 1.0,
        "worst_rolling_90d": float(rolling_90d.min()) if rolling_90d.notna().any() else 0.0,
    }


def _calc_score(row: pd.Series) -> float:
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
    return float(
        (math.sqrt(ret) * 9.0)
        * math.exp(-8.0 * dd)
        * math.exp(-18.0 * max(0.0, dd - 0.12))
        * (0.80 + 0.20 * min(1.0, max(0.0, month_pos)))
        * math.exp(-0.55 * max(0.0, streak))
        * math.exp(-2.25 * min(1.0, max(0.0, underwater)))
        * math.exp(-7.5 * max(0.0, -worst_90d))
        * min(1.25, max(0.85, pf))
        * min(1.0, trades / 120.0)
    )


def main():
    step_dirs = sorted([p for p in FAIR_ROOT.glob("step_*d") if p.is_dir()])
    for step_dir in step_dirs:
        csv_path = step_dir / "explore_runs.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        mask = (df.get("stage").astype(str) == "backtest") & (df.get("status").astype(str) == "ok")
        if not mask.any():
            continue
        for col in TARGET_COLS:
            if col not in df.columns:
                df[col] = np.nan
        enriched = df.loc[mask, "bt_out_dir"].apply(_equity_metrics)
        for col in TARGET_COLS:
            if col == "score":
                continue
            df.loc[mask, col] = enriched.apply(lambda x: x.get(col, np.nan))
        df.loc[mask, "score"] = df.loc[mask].apply(_calc_score, axis=1)
        df.to_csv(csv_path, index=False)
        print(f"[OK] updated {csv_path}")


if __name__ == "__main__":
    main()
