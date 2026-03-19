import pandas as pd
from pathlib import Path
import os

repo_root = Path(r"d:\astra\tradebot")
csv_path = repo_root / "data" / "generated" / "fair_wf_explore" / "step_1440d" / "explore_runs.csv"
if not csv_path.exists():
    print(f"File not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
df = df[(df["stage"] == "backtest") & (df["status"] == "ok")]
df["score"] = pd.to_numeric(df["score"], errors="coerce")
best = df.sort_values("score", ascending=False).iloc[0]

# Print key parameters for reconstruction
params = [
    "label_id", "model_id", "backtest_id", "train_id", "score", "ret_pct", "max_dd",
    "label_profit_thr", "exit_ema_span_min", "exit_ema_init_offset_pct",
    "entry_ratio_neg_per_pos", "calib_tail_blend", "calib_tail_boost",
    "top_metric_qs", "top_metric_min_count", "tau_entry", "corr_enabled",
    "max_positions", "total_exposure", "train_run_dir"
]

print("---BEST INDIVIDUAL PARAMETERS---")
for p in params:
    print(f"{p}: {best.get(p, 'N/A')}")
