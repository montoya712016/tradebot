import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dna():
    csv_path = Path("d:/astra/tradebot/data/generated/wf_portfolio_explore_runs/20260316_025259/explore_runs.csv")
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Numeric conversion
    cols_to_fix = ['score', 'ret_pct', 'max_dd', 'win_rate', 'profit_factor', 
                   'trades', 'label_profit_thr', 'exit_ema_span_min', 'top_metric_min_count', 'tau_entry']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Define GOOD: Return > 100%, DD < 25%, High Win Rate, High PF
    # Using quartiles for Win Rate and PF to find "relatively high" values
    wr_q3 = df['win_rate'].quantile(0.75)
    pf_q3 = df['profit_factor'].quantile(0.75)
    
    good_mask = (df['ret_pct'] > 100) & (df['max_dd'] < 0.25) & (df['win_rate'] >= wr_q3) & (df['profit_factor'] >= pf_q3)
    good_runs = df[good_mask].copy()

    # Define BAD: Low Return (< 0) OR High DD (> 40%) OR Low Score
    bad_mask = (df['ret_pct'] < 20) | (df['max_dd'] > 0.40) | (df['score'] < 0.05)
    bad_runs = df[bad_mask].copy()

    print(f"Total Runs: {len(df)}")
    print(f"Good Runs found: {len(good_runs)}")
    print(f"Bad Runs found: {len(bad_runs)}")

    def get_dna_summary(subset, title):
        print(f"\n=== DNA Profile: {title} ===")
        # Aggregating by label_id
        dna = subset.groupby('label_id').agg({
            'score': 'mean',
            'ret_pct': 'mean',
            'max_dd': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
            'label_profit_thr': 'first',
            'exit_ema_span_min': 'first',
            'top_metric_min_count': 'first',
            'tau_entry': 'mean'
        }).sort_values('ret_pct', ascending=False)
        print(dna.head(15).to_string())
        return dna

    good_dna = get_dna_summary(good_runs, "GOOD (High Return, Low DD, High WR/PF)")
    bad_dna = get_dna_summary(bad_runs, "BAD (Low Return or High DD)")

    print("\n=== THE 'SUCCESS' DNA COMMONALITIES ===")
    params = ['label_profit_thr', 'exit_ema_span_min', 'top_metric_min_count']
    for p in params:
        print(f"\nParameter: {p}")
        print("GOOD (avg):", good_dna[p].mean())
        print("BAD  (avg):", bad_dna[p].mean())
        print("Winning values distribution in GOOD runs:")
        print(good_runs[p].value_counts().head(3).to_string())

    # Final Winner Selection
    # Highest score in GOOD runs
    if not good_runs.empty:
        absolute_winner = good_runs.sort_values('score', ascending=False).iloc[0]
        print("\n=== ABSOLUTE WINNER RUN ===")
        print(absolute_winner[['label_id', 'ret_pct', 'max_dd', 'win_rate', 'profit_factor', 'label_profit_thr', 'exit_ema_span_min', 'top_metric_min_count', 'tau_entry']])

if __name__ == "__main__":
    analyze_dna()
