import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import duckdb  # type: ignore


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


def _extract_max_numeric_suffix(values: list[str], prefix: str) -> int:
    max_val = 0
    for raw in values:
        value = str(raw or "").strip()
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
        rel = duckdb.sql(f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}', all_varchar=true)")
        rows = rel.fetchall()
        cols = [str(c) for c in rel.columns]
    except Exception:
        return out
    if not rows:
        return out
    idx = {c: i for i, c in enumerate(cols)}
    status_i = idx.get("status")
    stage_i = idx.get("stage")
    label_i = idx.get("label_id")
    model_i = idx.get("model_id")
    bt_i = idx.get("backtest_id")
    ok_rows = [r for r in rows if status_i is None or str(r[status_i] or "") == "ok"]
    out["has_csv"] = True
    out["ok_refresh"] = int(sum(1 for r in ok_rows if stage_i is not None and str(r[stage_i] or "") == "refresh"))
    bt_ok = [r for r in ok_rows if stage_i is None or str(r[stage_i] or "") == "backtest"]
    out["ok_backtests"] = int(len(bt_ok))
    if label_i is not None:
        out["max_label_idx"] = _extract_max_numeric_suffix([r[label_i] for r in ok_rows], "label_")
    if model_i is not None:
        out["max_model_idx"] = _extract_max_numeric_suffix([r[model_i] for r in ok_rows], "model_")
    if bt_i is not None:
        out["max_bt_idx"] = _extract_max_numeric_suffix([r[bt_i] for r in ok_rows], "bt_")
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
        rel = duckdb.sql(f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}', all_varchar=true)")
        rows = rel.fetchall()
        cols = [str(c) for c in rel.columns]
    except Exception:
        return out
    if not rows:
        return out
    idx = {c: i for i, c in enumerate(cols)}
    label_i = idx.get("label_id")
    stage_i = idx.get("stage")
    status_i = idx.get("status")
    model_i = idx.get("model_id")
    bt_i = idx.get("backtest_id")
    if label_i is None:
        return out
    label_end = int(label_start) + int(label_count) - 1
    scoped_rows = []
    for row in rows:
        label_num = _extract_numeric_suffix(row[label_i], "label_")
        if int(label_start) <= label_num <= int(label_end):
            scoped_rows.append(row)
    if not scoped_rows:
        return out
    ok_rows = [r for r in scoped_rows if status_i is None or str(r[status_i] or "") == "ok"]
    out["ok_refresh"] = int(sum(1 for r in ok_rows if stage_i is not None and str(r[stage_i] or "") == "refresh"))
    bt_ok = [r for r in ok_rows if stage_i is None or str(r[stage_i] or "") == "backtest"]
    train_ok = [r for r in ok_rows if stage_i is None or str(r[stage_i] or "") == "train"]
    out["ok_backtests"] = int(len(bt_ok))
    out["labels_seen"] = int(len({str(r[label_i] or "") for r in ok_rows if str(r[label_i] or "").strip()}))
    if model_i is not None:
        out["models_seen"] = int(len({str(r[model_i] or "") for r in train_ok if str(r[model_i] or "").strip() and str(r[model_i] or "").strip().lower() != "nan"}))
    if bt_i is not None:
        out["backtests_seen"] = int(len({str(r[bt_i] or "") for r in bt_ok if str(r[bt_i] or "").strip() and str(r[bt_i] or "").strip().lower() != "nan"}))
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
    env.setdefault("WF_EXPLORE_FEATURE_PRESET", str(os.getenv("WF_EXPLORE_FEATURE_PRESET", "core80") or "core80"))
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
    fair_root_name = str(os.getenv("WF_FAIR_ROOT", "fair_wf_explore_v6") or "fair_wf_explore_v6").strip()
    os.environ["WF_FAIR_ROOT"] = fair_root_name
    os.environ.setdefault("WF_EXPLORE_FEATURE_PRESET", "core80")
    os.environ.setdefault("WF_EXPLORE_SAFE_THREADS", "8")
    os.environ.setdefault("WF_EXPLORE_CANDLE_SEC", "300")
    fair_root = repo_root / "data" / "generated" / fair_root_name

    # Configuração local para rodar direto pelo VS Code, sem env vars.
    # Estas são as METAS FINAIS por step.
    # O orquestrador completa o que faltar até este alvo e depois para.
    TARGET_LABEL_TRIALS = 56
    TARGET_RETRAINS_PER_LABEL = 2
    TARGET_BACKTESTS_PER_RETRAIN = 21
    GENERATION_PLAN = [
        {"phase": "broad", "generation": 1, "label_start": 1, "label_count": 56},
    ]
    
    # Resume Capability: Don't move the folder, just append/skip
    fair_root.mkdir(parents=True, exist_ok=True)
    
    # Start Dashboard automatically
    
    # Start Dashboard automatically
    dash_script = repo_root / "scripts" / "fair_dashboard.py"
    dash_log = fair_root / "dashboard.log"
    print(f"[INFO] Fair root: {fair_root}")
    print(
        f"[INFO] Explore defaults: feature_preset={os.environ.get('WF_EXPLORE_FEATURE_PRESET', 'core80')} "
        f"safe_threads={os.environ.get('WF_EXPLORE_SAFE_THREADS', '8')} "
        f"candle_sec={os.environ.get('WF_EXPLORE_CANDLE_SEC', '300')}"
    )
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
                target_bt = TARGET_LABEL_TRIALS * TARGET_RETRAINS_PER_LABEL * TARGET_BACKTESTS_PER_RETRAIN
                print(f"[FATAL] Step {m}d ended without reaching the final {target_bt}-backtest target.")
                break
                
        print(f"\n{'='*60}")
        print(f"FAIR EXPLORATION STATUS: {success_count}/{len(milestones)} steps successful.")
        if success_count == len(milestones):
            print("You can now run 'run_oos_walkforward.py' to see the fair OOS walk-forward curve.")
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
