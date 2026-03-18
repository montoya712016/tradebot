import pandas as pd
import numpy as np
from pathlib import Path

def get_stats(series):
    ret = (series.iloc[-1] / series.iloc[0]) - 1
    peaks = series.cummax()
    dd = (series - peaks) / peaks
    max_dd = abs(dd.min())
    daily_rets = series.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(365) if daily_rets.std() > 0 else 0
    return {"Return": f"{ret*100:.1f}%", "MaxDD": f"{max_dd*100:.1f}%", "Sharpe": f"{sharpe:.2f}"}

def main():
    repo_root = Path(r"d:\astra\tradebot")
    run_dir = repo_root / "data" / "generated" / "wf_portfolio_explore_runs" / "20260316_025259"
    report_dir = run_dir / "robustness_report"
    
    df_meth = pd.read_csv(report_dir / "methodology_equity.csv", index_col=0)
    df_meth.index = pd.to_datetime(df_meth.index)
    
    stats_m = get_stats(df_meth['equity'])
    print("METHODOLOGY (OOS):", stats_m)

if __name__ == "__main__":
    main()
