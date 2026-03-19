import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime

def run_step(repo_root: Path, days: int, trials_per_label: int, backtests_per_retrain: int, out_base: Path):
    step_dir = out_base / f"step_{days}d"
    step_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STARTING FAIR EXPLORATION STEP: {days} days")
    print(f"Target Directory: {step_dir}")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env["WF_EXPLORE_OUT_ROOT"] = str(step_dir)
    # Note: WF_EXPLORE_DAYS and SNIPER_REMOVE_TAIL_DAYS are already set in os.environ by the loop
    
    # Use explore.py directly instead of monolith to have a finite run
    explore_script = repo_root / "scripts" / "explore.py"
    
    # We run explore.py. It will finish after running the specified trials.
    try:
        start_t = time.time()
        result = subprocess.run(
            [sys.executable, "-u", str(explore_script)],
            env=env,
            cwd=str(repo_root),
            check=True
        )
        duration = time.time() - start_t
        with (step_dir / ".finished").open("w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"\n[OK] Step {days}d finished in {duration/60:.1f} minutes.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step {days}d failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted. Stopping orchestrator.")
        sys.exit(1)
        
    return True

def main():
    repo_root = Path(__file__).resolve().parent.parent
    fair_root = repo_root / "data" / "generated" / "fair_wf_explore"
    
    # Resume Capability: Don't move the folder, just append/skip
    fair_root.mkdir(parents=True, exist_ok=True)
    
    # Start Dashboard automatically
    
    # Start Dashboard automatically
    dash_script = repo_root / "scripts" / "fair_dashboard.py"
    dash_log = fair_root / "dashboard.log"
    print(f"[INFO] Launching Fair Dashboard: {dash_script} (Log: {dash_log})")
    
    with open(dash_log, "a") as log_f:
        dash_proc = subprocess.Popen([sys.executable, str(dash_script)], stdout=log_f, stderr=subprocess.STDOUT)
    
    # 180-day steps up to 4 years (1440 days)
    # The user wants to start from the oldest period (t-1440d).
    # Step 1440d: Train at T-1440, Test [T-1440, T-1260]. Anchor Tail = 1260.
    milestones = [1440, 1260, 1080, 900, 720, 540, 360, 180]
    
    # Each exploration window will be 180 days (the OOS segment).
    # This ensures that each step only backtests its targeted fair window.
    window_days = 180
    
    # Performance Tuning for the USER's PC (64GB RAM)
    os.environ["SNIPER_CACHE_WORKERS"] = "14"
    os.environ["SNIPER_DATASET_WORKERS"] = "14"
    
    try:
        success_count = 0
        for m in milestones:
            # Resume check: skip if .finished exists
            step_dir = fair_root / f"step_{m}d"
            if (step_dir / ".finished").exists():
                print(f"[INFO] Step {m}d already completed. Skipping.")
                success_count += 1
                continue
                
            # To test [T-m, T-(m-180)], we set simulation "Now" at T-(m-180)
            tail = max(0, m - 180)
            
            # Balanced Production parameters for Fair Run
            # Total: 10 labels * 3 models * 40 backtests = 1,200 per milestone
            # Entire pipeline: 1,200 * 8 = 9,600 backtests (~30 hours total)
            trials = 10
            retrains = 3
            backtests = 40
            
            # We override environment for run_step
            os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(tail)
            os.environ["WF_EXPLORE_DAYS"] = str(window_days)
            os.environ["WF_EXPLORE_LABEL_TRIALS"] = str(trials)
            os.environ["WF_EXPLORE_RETRAINS_PER_LABEL"] = str(retrains)
            os.environ["WF_EXPLORE_BACKTESTS_PER_RETRAIN"] = str(backtests)
            
            success = run_step(
                repo_root=repo_root,
                days=m,
                trials_per_label=trials,
                backtests_per_retrain=backtests,
                out_base=fair_root
            )
            if success:
                success_count += 1
            else:
                print(f"[FATAL] Orchestrator stopping due to failure in step {m}d")
                break
                
        print(f"\n{'='*60}")
        print(f"FAIR EXPLORATION STATUS: {success_count}/{len(milestones)} steps successful.")
        if success_count == len(milestones):
            print("You can now run 'verify_rolling_wf.py' to see the 100% fair robustness curve.")
        print(f"{'='*60}\n")
        
    finally:
        # Cleanup dashboard
        print("[INFO] Cleaning up dashboard...")
        dash_proc.terminate()
        try:
            dash_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dash_proc.kill()

if __name__ == "__main__":
    main()
