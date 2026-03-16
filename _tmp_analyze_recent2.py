import pandas as pd
from pathlib import Path

csv_path = Path(r"d:\astra\tradebot\data\generated\wf_portfolio_explore_runs\20260316_025259\explore_runs.csv")
df = pd.read_csv(csv_path)

bt = df[df["stage"] == "backtest"].copy()
bt["start_utc"] = pd.to_datetime(bt["start_utc"], format='mixed')
bt_recent = bt[bt["start_utc"] >= pd.to_datetime("2026-03-16 13:00:00")]

print(f"Recent runs: {len(bt_recent)}")
num_bad = 0
for _, row in bt_recent.iterrows():
    if row['max_dd'] > 0.40:
        num_bad += 1
        print("-" * 40)
        print(f"Run: {row['label_id']}/{row['model_id']}/{row['backtest_id']}")
        print(f"Ret: {row['ret_pct']:.1f}% | DD: {row['max_dd']:.1%}")
        print(f"tau: {row['tau_entry']}")
        print(f"span: {row['exit_ema_span_min']}")
        print(f"thr: {row['label_profit_thr']}")
        print(f"max_pos: {row['max_positions']}")
        print(f"exp: {row['total_exposure']}")
        if num_bad >= 5: break
