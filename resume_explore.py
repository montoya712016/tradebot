import os
import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "data" / "generated" / "wf_portfolio_explore_runs"
    
    if not runs_dir.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}")
        sys.exit(1)
        
    runs = [p for p in runs_dir.glob("202*") if p.is_dir()]
    if not runs:
        print(f"[ERROR] No runs found in {runs_dir}")
        sys.exit(1)
        
    # Sort by name (which is a timestamp) to find the latest
    runs.sort(key=lambda p: p.name, reverse=True)
    latest_run = runs[0]
    
    # If the latest run is the one that was just created erroneously (e.g. empty or just started), 
    # we might want to allow the user to specify which one to resume or just pick the second latest 
    # if the latest has no explore_runs.csv? 
    # Actually, let's just find the latest one that has a substantial explore_runs.csv
    
    target_run = None
    for run in runs:
        csv_path = run / "explore_runs.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            target_run = run
            break
            
    if not target_run:
        print(f"[ERROR] Could not find any run with a valid explore_runs.csv")
        sys.exit(1)

    print(f"[INFO] Auto-resuming run: {target_run.name}")
    
    env = os.environ.copy()
    env["WF_EXPLORE_RESUME_DIR"] = str(target_run)
    env["WF_EXPLORE_LABEL_TRIALS"] = "100"
    
    monolith_script = repo_root / "crypto" / "wf_portfolio_explorer_monolith.py"
    
    try:
        subprocess.run([sys.executable, "-u", str(monolith_script)], env=env, cwd=str(repo_root))
    except KeyboardInterrupt:
        print("\n[INFO] Resume script stopped.")

if __name__ == "__main__":
    main()
