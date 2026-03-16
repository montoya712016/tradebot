# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import shlex
import signal
import subprocess
import sys
import threading
import time


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_add_repo_paths()

from utils.paths import resolve_generated_path  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_args(arg_str: str) -> list[str]:
    if not arg_str:
        return []
    return shlex.split(arg_str, posix=(os.name != "nt"))


def _set_env_default(env: dict[str, str], key: str, value: str) -> None:
    if str(env.get(key, "")).strip():
        return
    env[key] = value


class ProcRunner:
    def __init__(self, name: str, cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str], restart_delay: int = 5) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.env = env
        self.restart_delay = int(max(1, restart_delay))
        self._proc: subprocess.Popen | None = None
        self._evt = threading.Event()
        self._th: threading.Thread | None = None

    def start(self) -> None:
        if self._th and self._th.is_alive():
            return
        self._evt.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _run_once(self) -> int:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", errors="replace") as log_fh:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self.env,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._evt.is_set():
                    break
                log_fh.write(line)
                log_fh.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return int(self._proc.wait())

    def _loop(self) -> None:
        while not self._evt.is_set():
            ret = self._run_once()
            if self._evt.is_set():
                break
            print(f"[{self.name}] exited code={ret}, restarting in {self.restart_delay}s...", flush=True)
            time.sleep(self.restart_delay)

    def stop(self) -> None:
        self._evt.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def main() -> None:
    restart_delay = int(os.getenv("WF_EXPLORE_RESTART_DELAY", "5") or "5")
    loop_args = os.getenv("WF_EXPLORE_ARGS", "")
    dash_args = os.getenv("WF_DASH_ARGS", "")

    root = _repo_root()
    resume_dir = os.getenv("WF_EXPLORE_RESUME_DIR", "").strip()
    if resume_dir:
        out_root = resolve_generated_path(Path(resume_dir))
        print(f"[INFO] Resuming explore run from: {out_root}", flush=True)
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_root = resolve_generated_path(Path("wf_portfolio_explore_runs") / run_id)
    out_root.mkdir(parents=True, exist_ok=True)

    loop_script = (root / "crypto" / "wf_portfolio_explorer.py").resolve()
    dash_script = (root / "modules" / "train" / "wf_dashboard_ngrok_monolith.py").resolve()

    loop_cmd = [sys.executable, "-u", str(loop_script)] + _split_args(loop_args)
    dash_cmd = [sys.executable, "-u", str(dash_script)] + _split_args(dash_args)

    loop_log = Path(os.getenv("WF_LOOP_LOG") or (out_root / "loop.log"))
    dash_log = Path(os.getenv("WF_DASH_LOG") or (out_root / "dash.log"))

    loop_env = os.environ.copy()
    loop_env["WF_EXPLORE_OUT_ROOT"] = str(out_root)
    loop_env["WF_LOOP_LOG"] = str(loop_log)
    loop_env["WF_DASH_LOG"] = str(dash_log)
    _set_env_default(loop_env, "WF_EXPLORE_SAFE_THREADS", "4")
    _set_env_default(loop_env, "SNIPER_CACHE_WORKERS", "14")
    _set_env_default(loop_env, "SNIPER_DATASET_WORKERS", "14")

    dash_env = os.environ.copy()
    dash_env["WF_DASH_OUT_ROOT"] = str(out_root)
    dash_env["WF_DASH_RESULTS_CSV"] = "explore_runs.csv"
    dash_env["WF_DASH_LOG"] = str(dash_log)
    dash_env["WF_DASH_LOOP_LOG"] = str(loop_log)

    loop_runner = ProcRunner("explore", loop_cmd, root, loop_log, loop_env, restart_delay=restart_delay)
    dash_runner = ProcRunner("dash", dash_cmd, root, dash_log, dash_env, restart_delay=restart_delay)

    stop_evt = threading.Event()

    def _cleanup() -> None:
        if stop_evt.is_set():
            return
        stop_evt.set()
        loop_runner.stop()
        dash_runner.stop()

    def _handler(signum, _frame) -> None:
        print(f"\n[STOP] signal={signum}", flush=True)
        _cleanup()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

    loop_runner.start()
    dash_runner.start()
    print("[INFO] Crypto explorer monolith running. Press Ctrl+C to stop.", flush=True)
    try:
        while not stop_evt.is_set():
            time.sleep(1)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
