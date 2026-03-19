import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from datetime import timedelta

def calc_score(ret, dd, win, pf, trades, month_pos, sem_pos):
    """Refined score logic matching analyze_explore.py"""
    if ret <= 0: return ret
    
    dd_penalty = np.exp(-25.0 * (dd ** 2))
    smoothed_ret = np.sqrt(ret) * 10
    consistency_mult = 0.2 + month_pos + sem_pos
    trade_mult = min(1.0, trades / 100.0)
    pf_mult = min(3.0, pf)
    
    return smoothed_ret * dd_penalty * consistency_mult * trade_mult * pf_mult

def main():
    repo_root = Path(__file__).resolve().parent.parent
    
    # Path Configuration
    fair_root = repo_root / "data" / "generated" / "fair_wf_explore"
    legacy_runs_dir = repo_root / "data" / "generated" / "wf_portfolio_explore_runs"
    
    is_fair_mode = fair_root.exists() and any(fair_root.glob("step_*d"))
    
    all_curves = {}
    selection_pools = {} # Step -> List of (trial_id, metadata)
    
    if is_fair_mode:
        print(f"[INFO] Entering FAIR MODE (using independent step pools)")
        steps = sorted([int(p.name.split('_')[1].replace('d','')) for p in fair_root.glob("step_*d")])
        for s in steps:
            step_dir = fair_root / f"step_{s}d"
            csv_path = step_dir / "explore_runs.csv"
            if not csv_path.exists(): continue
            
            df_step = pd.read_csv(csv_path)
            df_step = df_step[(df_step["stage"] == "backtest") & (df_step["status"] == "ok")]
            
            selection_pools[s] = []
            for _, row in df_step.iterrows():
                bt_dir = Path(row["bt_out_dir"])
                eq_csv = bt_dir / "portfolio_equity.csv"
                if not eq_csv.exists(): continue
                
                # Load curve
                curve = pd.read_csv(eq_csv, index_col=0)
                curve.index = pd.to_datetime(curve.index)
                
                tid = str(row["train_id"])
                bid = str(row["backtest_id"])
                trial_key = f"s{s}_{bid}_{tid}" # Prefix with step to avoid collisions
                all_curves[trial_key] = curve
                selection_pools[s].append((trial_key, row))
        
        # Determine global min/max date from all curves
        min_date = min(c.index.min() for c in all_curves.values())
        max_date = max(c.index.max() for c in all_curves.values())
        
    else:
        # Legacy/Hindsight Mode
        print(f"[INFO] Entering HINDSIGHT MODE (using global survivor pool)")
        runs = sorted([p for p in legacy_runs_dir.glob("202*") if p.is_dir()], key=lambda p: p.name, reverse=True)
        target_run = None
        for run in runs:
            if (run / "explore_runs.csv").exists():
                target_run = run
                break
        if not target_run:
            print("[ERROR] No exploration data found.")
            return
            
        df_meta = pd.read_csv(target_run / "explore_runs.csv")
        df_meta = df_meta[(df_meta["stage"] == "backtest") & (df_meta["status"] == "ok")]
        
        for _, row in df_meta.iterrows():
            bt_dir = Path(row["bt_out_dir"])
            eq_csv = bt_dir / "portfolio_equity.csv"
            if not eq_csv.exists(): continue
            curve = pd.read_csv(eq_csv, index_col=0)
            curve.index = pd.to_datetime(curve.index)
            trial_key = f"{row['backtest_id']}_{row['train_id']}"
            all_curves[trial_key] = curve
            # In hindsight mode, all trials are available for all steps
            # (This is where the 'survivor bias' cheat happens)
            
        min_date = min(c.index.min() for c in all_curves.values())
        max_date = max(c.index.max() for c in all_curves.values())

    print(f"[INFO] Data period: {min_date.date()} to {max_date.date()}")
    
    # Parameters for Rolling WF
    step_days = 180
    current_date = min_date + timedelta(days=step_days) 
    
    oos_equity_segments = []
    selection_history = []
    
    while current_date + timedelta(days=step_days) <= max_date:
        is_end = current_date
        oos_end = current_date + timedelta(days=step_days)
        
        days_from_start = (is_end - min_date).days
        # Find the correct selection pool (snapshot of 'what was known')
        # Calculate T-minus days for the current date
        t_minus_now = (max_date - is_end).days
        
        if is_fair_mode:
            # We want the step that finished its OOS window at t_minus_now.
            # Example: step_1440d finishes its OOS at T-1260.
            # So if t_minus_now = 1260, we want target_step = 1440.
            target_step = t_minus_now + 180
            
            # Find the closest step available in pools within a reasonable margin
            valid_steps = [s for s in selection_pools.keys() if abs(s - target_step) <= 20]
            if not valid_steps:
                print(f"  [SKIP] No fair step data near {target_step}d (T-{t_minus_now}d)")
                current_date = oos_end
                continue
            active_step = min(valid_steps, key=lambda x: abs(x - target_step))
            candidates = selection_pools[active_step]
            print(f"[STEP] Date: {is_end.date()} | T-{t_minus_now:<4} | OOS -> {oos_end.date()} | POOL: {active_step}d")
        else:
            candidates = [(tid, None) for tid in all_curves.keys()]
            print(f"[STEP] Date: {is_end.date()} | OOS: {oos_end.date()} | USING HINDSIGHT POOL")
        
        trial_scores = {}
        for trial_id, _ in candidates:
            curve = all_curves[trial_id]
            # In-Sample Slicing
            is_curve = curve[:is_end]
            if len(is_curve) < 60: continue # Need at least 2 months of history
            
            # 1. Calc Return % (Scaled to 100-based as per explore script)
            ret_ratio = (is_curve["equity"].iloc[-1] / is_curve["equity"].iloc[0]) - 1
            ret_pct = ret_ratio * 100.0
            
            # 2. Risk Metrics (MaxDD)
            peaks = is_curve["equity"].cummax()
            drawdowns = (is_curve["equity"] - peaks) / peaks
            max_dd = abs(drawdowns.min())
            
            # Constraint: Skip individuals with extreme IS drawdown (not production-like)
            if max_dd > 0.25: continue 
            
            # 3. Robustness Proxies (from Daily Returns)
            daily_rets = is_curve["equity"].pct_change().dropna()
            # filter out non-active days if any (flat equity)
            active_rets = daily_rets[daily_rets != 0]
            if len(active_rets) < 30: continue # Minimum 30 days of activity
            
            win_rate = (active_rets > 0).mean()
            
            pos_sum = daily_rets[daily_rets > 0].sum()
            neg_sum = abs(daily_rets[daily_rets < 0].sum())
            pf = pos_sum / neg_sum if neg_sum > 0 else 2.0
            pf = min(5.0, pf) # Cap PF impact
            
            # 4. Consistency (Monthly positive pct)
            monthly = is_curve["equity"].resample("ME").last().pct_change().dropna()
            month_pos = (monthly > 0).mean() if not monthly.empty else 0.5
            
            # 5. Volatility (Critical for filtering glitches)
            daily_std = daily_rets.std()
            
            # Calculate final score with realistic numbers
            # (trades hardcoded to 100 as proxy for 'sufficient' sample size)
            score = calc_score(ret_pct, max_dd, win_rate, pf, 100, month_pos, 0.5)
            
            # PENALTY: Huge Daily StdDev = Data Glitch or Over-leverage
            # In crypto, Daily Std of 0.05 (5%) is high but normal. 0.20 is a glitch.
            vol_penalty = 1.0 / (1.0 + max(0, daily_std - 0.05) * 50.0)
            score *= vol_penalty
            
            trial_scores[trial_id] = score

        if not trial_scores:
            print(f"  [WARN] No viable trials at {is_end.date()} with DD < 25%")
            current_date = oos_end
            continue
            
        # Select best
        best_trial = max(trial_scores, key=trial_scores.get)
        best_score = trial_scores[best_trial]
        print(f"  [BEST] {best_trial} (IS Score: {best_score:.2f})")
        selection_history.append({"date": is_end, "trial": best_trial, "score": best_score})
        
        # OOS Performance
        full_curve = all_curves[best_trial]
        oos_curve = full_curve[is_end:oos_end]
        
        if len(oos_curve) > 1:
            # Each segment starts at 1.0 (relative).
            segments_rets = oos_curve["equity"].pct_change().fillna(0)
            # CAP OOS Daily Return: Anything > 30% in 1 day is treated as a glitch
            # This is key to PROVING robustness without glitch-compounding
            segments_rets = segments_rets.clip(upper=0.30, lower=-0.30)
            
            oos_ret = (1.0 + segments_rets).cumprod()
            oos_equity_segments.append(oos_ret)
            
        current_date = oos_end

    # Combine segments
    if not oos_equity_segments:
        print("[ERROR] No OOS segments generated.")
        return
        
    final_oos_curve = pd.concat(oos_equity_segments)
    # Correct for overlaps/multiplication
    methodology_equity = [1.0]
    for i in range(1, len(final_oos_curve)):
        # If it's the start of a new segment, we multiply by the new relative return
        # But wait, concat already gives us all points. 
        # We need to ensure continuity.
        pass
    
    # Robust continuity logic
    total_equity = 1.0
    combined_series = []
    for seg in oos_equity_segments:
        # Each segment starts at 1.0 (relative). We multiply the whole segment by total_equity.
        multiplied_seg = seg * total_equity
        combined_series.append(multiplied_seg)
        total_equity = multiplied_seg.iloc[-1]

    methodology_series = pd.concat(combined_series)
    # Remove duplicate index (segment boundaries)
    methodology_series = methodology_series[~methodology_series.index.duplicated(keep='first')]
    
    # Comparison with "Hindsight Absolute Winner"
    # Find the best trial across all available curves using the full period score
    print("[INFO] Calculating Hindsight Absolute Winner...")
    hindsight_scores = {}
    for tid, curve in all_curves.items():
        # Score on total available history
        ret = (curve["equity"].iloc[-1] / curve["equity"].iloc[0]) - 1
        peaks = curve["equity"].cummax()
        dd = (curve["equity"] - peaks) / peaks
        max_dd = abs(dd.min())
        if ret <= 0: continue
        score = np.sqrt(ret * 100) * np.exp(-25.0 * (max_dd**2))
        hindsight_scores[tid] = score
        
    if hindsight_scores:
        hindsight_id = max(hindsight_scores, key=hindsight_scores.get)
        hindsight_series = all_curves[hindsight_id]
        hindsight_series = hindsight_series["equity"] / hindsight_series["equity"].iloc[0]
        print(f"  [HINDSIGHT BEST] {hindsight_id}")
    else:
        print("  [WARN] No viable hindsight candidate found.")
        hindsight_series = methodology_series * 0.0 # Null series
    
    # Save Results
    if is_fair_mode:
        results_dir = fair_root / "robustness_report"
    else:
        results_dir = target_run / "robustness_report"
        
    results_dir.mkdir(parents=True, exist_ok=True)
    methodology_series.to_csv(results_dir / "methodology_equity.csv")
    
    # Calculate Final Stats
    def get_stats(series, name):
        ret = (series.iloc[-1] / series.iloc[0]) - 1
        peaks = series.cummax()
        dd = (series - peaks) / peaks
        max_dd = abs(dd.min())
        
        # Monthly Sharpe
        monthly_rets = series.resample("ME").last().pct_change().dropna()
        if not monthly_rets.empty and monthly_rets.std() > 0:
            sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12)
        else:
            sharpe = 0.0
            
        return {"Name": name, "Return": f"{ret*100:.1f}%", "MaxDD": f"{max_dd*100:.1f}%", "Sharpe": f"{sharpe:.2f}"}

    stats_methodology = get_stats(methodology_series, "Recursive Walk-Forward (OOS)")
    stats_hindsight = get_stats(hindsight_series, "Absolute Winner (Hindsight)")
    
    print("\n" + "="*40)
    print("ROBUSTNESS VERIFICATION RESULTS")
    print("="*40)
    print(f"{'Metric':<15} | {'Methodology (OOS)':<20} | {'Hindsight':<20}")
    print("-" * 60)
    for k in stats_methodology.keys():
        if k == "Name": continue
        print(f"{k:<15} | {stats_methodology[k]:<20} | {stats_hindsight[k]:<20}")
    print("="*40)
    
    print(f"\n[INFO] Methodology Equity Curve saved to {results_dir / 'methodology_equity.csv'}")

if __name__ == "__main__":
    main()
