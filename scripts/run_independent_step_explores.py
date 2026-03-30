import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _fmt_duration(seconds: float) -> str:
    try:
        total = max(0.0, float(seconds))
    except Exception:
        total = 0.0
    if total < 60.0:
        return f"{total:.1f}s"
    minutes = total / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}m"
    return f"{(minutes / 60.0):.2f}h"


def _extract_max_numeric_suffix(series: pd.Series, prefix: str) -> int:
    max_val = 0
    for value in series.dropna().astype(str):
        value = value.strip()
        if not value.startswith(prefix):
            continue
        tail = value[len(prefix):]
        try:
            max_val = max(max_val, int(tail))
        except Exception:
            continue
    return int(max_val)


def _extract_numeric_suffix(value: object, prefix: str) -> int:
    text = str(value or "").strip()
    if not text.startswith(prefix):
        return 0
    try:
        return int(text[len(prefix):])
    except Exception:
        return 0


def _step_progress(step_dir: Path) -> dict:
    csv_path = step_dir / "explore_runs.csv"
    out = {
        "has_csv": False,
        "ok_refresh": 0,
        "max_label_idx": 0,
        "max_model_idx": 0,
        "max_bt_idx": 0,
        "ok_backtests": 0,
    }
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out

    df_ok = df[df.get("status").astype(str) == "ok"].copy() if "status" in df.columns else df.copy()
    out["has_csv"] = True
    if "stage" in df_ok.columns:
        out["ok_refresh"] = int((df_ok["stage"].astype(str) == "refresh").sum())
        bt_ok = df_ok[df_ok["stage"].astype(str) == "backtest"].copy()
    else:
        bt_ok = df_ok.copy()
    out["ok_backtests"] = int(len(bt_ok))
    if "label_id" in df_ok.columns:
        out["max_label_idx"] = _extract_max_numeric_suffix(df_ok["label_id"], "label_")
    if "model_id" in df_ok.columns:
        out["max_model_idx"] = _extract_max_numeric_suffix(df_ok["model_id"], "model_")
    if "backtest_id" in df_ok.columns:
        out["max_bt_idx"] = _extract_max_numeric_suffix(df_ok["backtest_id"], "bt_")
    return out


def _generation_progress(step_dir: Path, label_start: int, label_count: int, retrains: int, backtests: int) -> dict:
    csv_path = step_dir / "explore_runs.csv"
    out = {
        "ok_refresh": 0,
        "ok_backtests": 0,
        "labels_seen": 0,
        "models_seen": 0,
        "backtests_seen": 0,
    }
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return out
    if df.empty or "label_id" not in df.columns:
        return out
    label_end = int(label_start) + int(label_count) - 1
    label_nums = df["label_id"].apply(lambda x: _extract_numeric_suffix(x, "label_"))
    df = df[(label_nums >= int(label_start)) & (label_nums <= int(label_end))].copy()
    if df.empty:
        return out
    df_ok = df[df.get("status").astype(str) == "ok"].copy() if "status" in df.columns else df.copy()
    if "stage" in df_ok.columns:
        out["ok_refresh"] = int((df_ok["stage"].astype(str) == "refresh").sum())
        bt_ok = df_ok[df_ok["stage"].astype(str) == "backtest"].copy()
        train_ok = df_ok[df_ok["stage"].astype(str) == "train"].copy()
    else:
        bt_ok = df_ok.copy()
        train_ok = df_ok.copy()
    out["ok_backtests"] = int(len(bt_ok))
    out["labels_seen"] = int(df_ok["label_id"].astype(str).nunique()) if "label_id" in df_ok.columns else 0
    out["models_seen"] = int(train_ok["model_id"].astype(str).replace("nan", "").nunique()) if "model_id" in train_ok.columns else 0
    out["backtests_seen"] = int(bt_ok["backtest_id"].astype(str).replace("nan", "").nunique()) if "backtest_id" in bt_ok.columns else 0
    return out


def _generation_meets_targets(progress: dict, label_count: int, retrains: int, backtests: int) -> bool:
    return bool(
        int(progress.get("ok_refresh", 0)) >= int(label_count)
        and int(progress.get("labels_seen", 0)) >= int(label_count)
        and int(progress.get("models_seen", 0)) >= int(retrains)
        and int(progress.get("backtests_seen", 0)) >= int(backtests)
        and int(progress.get("ok_backtests", 0)) >= (int(label_count) * int(retrains) * int(backtests))
    )


def _target_counts(progress: dict, target_trials: int, target_retrains: int, target_backtests: int) -> tuple[int, int, int]:
    target_trials = max(int(target_trials), int(progress.get("max_label_idx", 0)))
    target_retrains = max(int(target_retrains), int(progress.get("max_model_idx", 0)))
    target_backtests = max(int(target_backtests), int(progress.get("max_bt_idx", 0)))
    return int(target_trials), int(target_retrains), int(target_backtests)

def _step_meets_targets(progress: dict, target_trials: int, target_retrains: int, target_backtests: int) -> bool:
    target_trials = int(target_trials)
    target_retrains = int(target_retrains)
    target_backtests = int(target_backtests)
    return bool(
        int(progress.get("ok_refresh", 0)) >= target_trials
        and int(progress.get("max_label_idx", 0)) >= target_trials
        and int(progress.get("max_model_idx", 0)) >= target_retrains
        and int(progress.get("max_bt_idx", 0)) >= target_backtests
        and int(progress.get("ok_backtests", 0)) >= (target_trials * target_retrains * target_backtests)
    )


def run_generation(
    repo_root: Path,
    days: int,
    phase: str,
    generation: int,
    label_start: int,
    label_count: int,
    retrains_per_label: int,
    backtests_per_retrain: int,
    out_base: Path,
):
    step_dir = out_base / f"step_{days}d"
    step_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STARTING FAIR EXPLORATION STEP: {days} days | {phase.upper()} GEN {generation}")
    print(f"Target Directory: {step_dir}")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env["WF_EXPLORE_OUT_ROOT"] = str(step_dir)
    env["WF_EXPLORE_LABEL_TRIALS"] = str(int(label_count))
    env["WF_EXPLORE_RETRAINS_PER_LABEL"] = str(int(retrains_per_label))
    env["WF_EXPLORE_BACKTESTS_PER_RETRAIN"] = str(int(backtests_per_retrain))
    env["WF_EXPLORE_SEQUENCE_ID"] = str(int(days) * 100 + int(generation))
    env["WF_EXPLORE_PHASE"] = str(phase)
    env["WF_EXPLORE_GENERATION"] = str(int(generation))
    env["WF_EXPLORE_LABEL_START"] = str(int(label_start))
    env["WF_EXPLORE_LABEL_COUNT"] = str(int(label_count))
    worker_count = int(_env_int("WF_EXPLORE_SAFE_THREADS", 8))
    env["WF_EXPLORE_SAFE_THREADS"] = str(worker_count)
    env["SNIPER_LABELS_REFRESH_WORKERS"] = str(worker_count)
    env["SNIPER_CACHE_WORKERS"] = str(worker_count)
    env["SNIPER_DATASET_WORKERS"] = str(worker_count)
    # Note: WF_EXPLORE_DAYS and SNIPER_REMOVE_TAIL_DAYS are already set in os.environ by the loop
    
    # Use explore.py directly instead of monolith to have a finite run
    explore_script = repo_root / "scripts" / "explore.py"
    
    # We run explore.py. It will finish after running the specified trials.
    try:
        finished_flag = step_dir / ".finished"
        if finished_flag.exists():
            finished_flag.unlink()
        start_t = time.time()
        result = subprocess.run(
            [sys.executable, "-u", str(explore_script)],
            env=env,
            cwd=str(repo_root),
            check=True
        )
        duration = time.time() - start_t
        progress = _generation_progress(step_dir, label_start, label_count, retrains_per_label, backtests_per_retrain)
        meets_targets = _generation_meets_targets(progress, label_count, retrains_per_label, backtests_per_retrain)
        if meets_targets:
            with (step_dir / f".generation_{int(generation)}_finished").open("w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(
                f"\n[OK] Step {days}d generation {generation} finished in {_fmt_duration(duration)} "
                f"(labels={progress.get('labels_seen', 0)}, models={progress.get('models_seen', 0)}, "
                f"ok_bt={progress.get('ok_backtests', 0)})."
            )
        else:
            print(
                f"\n[ERROR] Step {days}d generation {generation} exited without meeting targets after {_fmt_duration(duration)}. "
                f"Current(labels={progress.get('labels_seen', 0)}, models={progress.get('models_seen', 0)}, "
                f"bt={progress.get('backtests_seen', 0)}, ok_refresh={progress.get('ok_refresh', 0)}, "
                f"ok_bt={progress.get('ok_backtests', 0)}) "
                f"Target(labels={int(label_count)}, model={int(retrains_per_label)}, bt={int(backtests_per_retrain)}, "
                f"ok_bt={int(label_count) * int(retrains_per_label) * int(backtests_per_retrain)})"
            )
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step {days}d generation {generation} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted. Stopping orchestrator.")
        sys.exit(1)
        
    return True

def main():
    repo_root = Path(__file__).resolve().parent.parent
    fair_root_name = str(os.getenv("WF_FAIR_ROOT", "fair_wf_explore_v5") or "fair_wf_explore_v5").strip()
    fair_root = repo_root / "data" / "generated" / fair_root_name

    # Configuração local para rodar direto pelo VS Code, sem env vars.
    # Estas são as METAS FINAIS por step.
    # O orquestrador completa o que faltar até este alvo e depois para.
    TARGET_LABEL_TRIALS = 49
    TARGET_RETRAINS_PER_LABEL = 1
    TARGET_BACKTESTS_PER_RETRAIN = 26
    GENERATION_PLAN = [
        {"phase": "broad", "generation": 1, "label_start": 1, "label_count": 49},
    ]
    
    # Resume Capability: Don't move the folder, just append/skip
    fair_root.mkdir(parents=True, exist_ok=True)
    
    # Start Dashboard automatically
    
    # Start Dashboard automatically
    dash_script = repo_root / "scripts" / "fair_dashboard.py"
    dash_log = fair_root / "dashboard.log"
    print(f"[INFO] Fair root: {fair_root}")
    print(f"[INFO] Launching Fair Dashboard: {dash_script} (Log: {dash_log})")
    
    with open(dash_log, "a") as log_f:
        dash_env = os.environ.copy()
        dash_env["WF_FAIR_ROOT"] = fair_root_name
        dash_proc = subprocess.Popen([sys.executable, str(dash_script)], stdout=log_f, stderr=subprocess.STDOUT, env=dash_env)
    
    # 180-day steps up to 4 years (1440 days)
    # The user wants to start from the oldest period (t-1440d).
    # Step 1440d: Train at T-1440, Test [T-1440, T-1260]. Anchor Tail = 1260.
    # O step 180d foi removido do explore porque hoje ele não tem uma perna
    # OOS seguinte comparável dentro desta grade de 180 dias.
    milestones = [1440, 1260, 1080, 900, 720, 540, 360]
    
    # Each exploration window will be 180 days (the OOS segment).
    # This ensures that each step only backtests its targeted fair window.
    window_days = 180
    
    # Performance Tuning for the USER's PC (64GB RAM)
    os.environ.setdefault("WF_EXPLORE_SAFE_THREADS", "8")
    os.environ.setdefault("SNIPER_LABELS_REFRESH_WORKERS", "8")
    os.environ.setdefault("SNIPER_CACHE_WORKERS", "8")
    os.environ.setdefault("SNIPER_DATASET_WORKERS", "8")
    
    try:
        success_count = 0
        for m in milestones:
            step_started = time.time()
            step_dir = fair_root / f"step_{m}d"
            # To test [T-m, T-(m-180)], we set simulation "Now" at T-(m-180)
            tail = max(0, m - 180)
            
            progress = _step_progress(step_dir)

            step_finished = (step_dir / ".finished").exists()
            meets_targets = _step_meets_targets(progress, TARGET_LABEL_TRIALS, TARGET_RETRAINS_PER_LABEL, TARGET_BACKTESTS_PER_RETRAIN)
            if step_finished and meets_targets:
                print(
                    f"[INFO] Step {m}d already completed. "
                    f"Current=max(label={progress.get('max_label_idx', 0)}, model={progress.get('max_model_idx', 0)}, bt={progress.get('max_bt_idx', 0)}) "
                    f"Target=(label={TARGET_LABEL_TRIALS}, model={TARGET_RETRAINS_PER_LABEL}, bt={TARGET_BACKTESTS_PER_RETRAIN}). Skipping."
                )
                success_count += 1
                continue

            print(
                f"[INFO] Step {m}d progress: "
                f"current(label={progress.get('max_label_idx', 0)}, model={progress.get('max_model_idx', 0)}, bt={progress.get('max_bt_idx', 0)}, ok_bt={progress.get('ok_backtests', 0)}) "
                f"-> target(label={TARGET_LABEL_TRIALS}, model={TARGET_RETRAINS_PER_LABEL}, bt={TARGET_BACKTESTS_PER_RETRAIN})"
            )

            os.environ["SNIPER_REMOVE_TAIL_DAYS"] = str(tail)
            os.environ["WF_EXPLORE_DAYS"] = str(window_days)
            os.environ["WF_EXPLORE_STEP_DAYS"] = str(m)
            
            step_success = True
            for gen_cfg in GENERATION_PLAN:
                gen_progress = _generation_progress(
                    step_dir,
                    gen_cfg["label_start"],
                    gen_cfg["label_count"],
                    TARGET_RETRAINS_PER_LABEL,
                    TARGET_BACKTESTS_PER_RETRAIN,
                )
                gen_done = _generation_meets_targets(
                    gen_progress,
                    gen_cfg["label_count"],
                    TARGET_RETRAINS_PER_LABEL,
                    TARGET_BACKTESTS_PER_RETRAIN,
                )
                if gen_done:
                    print(
                        f"[INFO] Step {m}d generation {gen_cfg['generation']} already complete "
                        f"({gen_cfg['phase']}, labels {gen_cfg['label_start']}..{gen_cfg['label_start'] + gen_cfg['label_count'] - 1})."
                    )
                    continue

                success = run_generation(
                    repo_root=repo_root,
                    days=m,
                    phase=gen_cfg["phase"],
                    generation=gen_cfg["generation"],
                    label_start=gen_cfg["label_start"],
                    label_count=gen_cfg["label_count"],
                    retrains_per_label=TARGET_RETRAINS_PER_LABEL,
                    backtests_per_retrain=TARGET_BACKTESTS_PER_RETRAIN,
                    out_base=fair_root,
                )
                if not success:
                    print(f"[FATAL] Orchestrator stopping due to failure in step {m}d generation {gen_cfg['generation']}")
                    step_success = False
                    break

            if not step_success:
                break

            progress = _step_progress(step_dir)
            if _step_meets_targets(progress, TARGET_LABEL_TRIALS, TARGET_RETRAINS_PER_LABEL, TARGET_BACKTESTS_PER_RETRAIN):
                with (step_dir / ".finished").open("w") as f:
                    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(
                    f"[OK] Step {m}d complete in {_fmt_duration(time.time() - step_started)} "
                    f"(ok_refresh={progress.get('ok_refresh', 0)}, ok_bt={progress.get('ok_backtests', 0)})."
                )
                success_count += 1
            else:
                print(f"[FATAL] Step {m}d ended without reaching the final 2500-backtest target.")
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
