import pandas as pd
from pathlib import Path

csv_path = Path(r"d:\astra\tradebot\data\generated\wf_portfolio_explore_runs\20260316_025259\explore_runs.csv")
df = pd.read_csv(csv_path)

bt = df[df["stage"] == "backtest"].copy()

# Look at runs that happened after 13:00 UTC (the "bad" ones from the resume)
bt["start_utc"] = pd.to_datetime(bt["start_utc"], format='mixed')
bt_recent = bt[bt["start_utc"] >= pd.to_datetime("2026-03-16 13:00:00")]

print(f"Recent runs: {len(bt_recent)}")
if len(bt_recent) > 0:
    for _, row in bt_recent.iterrows():
        print(f"{row['label_id']}/{row['model_id']} | Ret: {row['ret_pct']:>6.1f}% | DD: {row['max_dd']:>5.1%} | PF: {row['profit_factor']:.2f} | tau: {row['tau_entry']} | span: {row['exit_ema_span_min']} | thr: {row['label_profit_thr']} | max_pos: {row['max_positions']} | exp: {row['total_exposure']}")
