import pandas as pd
from pathlib import Path

def main():
    repo_root = Path(r"d:\astra\tradebot")
    report_dir = repo_root / "data" / "generated" / "wf_portfolio_explore_runs" / "20260316_025259" / "robustness_report"
    
    df = pd.read_csv(report_dir / "methodology_equity.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    
    df['pct'] = df['equity'].pct_change()
    outliers = df[df['pct'] > 5.0] # More than 5x in a day
    if not outliers.empty:
        print("[OUTLIERS FOUND]")
        print(outliers)
    else:
        print("[NO 5X DAILY OUTLIERS FOUND]")

if __name__ == "__main__":
    main()
