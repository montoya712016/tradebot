#!/usr/bin/env python3
"""
Monolith:
- Run wf_random_loop (train + backtest loop)
- Run WF dashboard (ngrok + pushover)

Env overrides:
- WF_LOOP_ARGS: extra args for wf_random_loop.py
- WF_DASH_ARGS: extra args for wf_dashboard_ngrok_monolith.py
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from utils.paths import resolve_generated_path  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_args(arg_str: str) -> list[str]:
    if not arg_str:
        return []
    return shlex.split(arg_str, posix=(os.name != "nt"))


def _set_env_default(env: dict[str, str], key: str, value: str) -> None:
    if str(env.get(key, "")).strip():
        return
    env[key] = value


class ProcRunner:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        cwd: Path,
        restart_delay: int = 5,
        log_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.restart_delay = max(1, int(restart_delay))
        self.log_path = log_path
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._evt = threading.Event()
        self._th: threading.Thread | None = None
        self._log_fh = None

    def start(self) -> None:
        if self._th and self._th.is_alive():
            return
        self._evt.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _spawn(self) -> None:
        print(f"[{self.name}] starting: {' '.join(self.cmd)}", flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env,
        )

    def _run_once(self) -> int:
        self._spawn()
        assert self._proc is not None
        if self._proc.stdout is not None:
            for line in self._proc.stdout:
                if self._evt.is_set():
                    break
                if self._log_fh is not None:
                    self._log_fh.write(line)
                    self._log_fh.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        ret = self._proc.wait()
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        return int(ret)

    def _loop(self) -> None:
        while not self._evt.is_set():
            if self._proc is None:
                ret = self._run_once()
            else:
                ret = self._proc.poll()
                if ret is None:
                    time.sleep(1)
                    continue
                ret = int(ret)
            if self._evt.is_set():
                break
            print(f"[{self.name}] exited code={ret}, restarting in {self.restart_delay}s...", flush=True)
            self._proc = None
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
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-delay", type=int, default=5)
    ap.add_argument("--loop-args", type=str, default=os.getenv("WF_LOOP_ARGS", ""))
    ap.add_argument("--dash-args", type=str, default=os.getenv("WF_DASH_ARGS", ""))
    args = ap.parse_args()

    root = _repo_root()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = resolve_generated_path(Path("wf_runs") / run_id)
    out_root.mkdir(parents=True, exist_ok=True)
    loop_script = (root / "modules" / "train" / "wf_random_loop.py").resolve()
    dash_script = (root / "modules" / "train" / "wf_dashboard_ngrok_monolith.py").resolve()

    loop_cmd = [sys.executable, "-u", str(loop_script)] + _split_args(str(args.loop_args))
    dash_cmd = [sys.executable, "-u", str(dash_script)] + _split_args(str(args.dash_args))

    loop_log = Path(os.getenv("WF_LOOP_LOG") or (out_root / "loop.log"))
    dash_log = Path(os.getenv("WF_DASH_LOG") or (out_root / "dash.log"))

    loop_env = os.environ.copy()
    _set_env_default(loop_env, "PF_SYMBOL_WORKERS", "2")
    _set_env_default(loop_env, "SNIPER_CACHE_WORKERS", "2")
    _set_env_default(loop_env, "WF_REBUILD_CACHE_MODE", "off")
    _set_env_default(loop_env, "WF_REBUILD_CACHE", "0")
    _set_env_default(loop_env, "WF_SKIP_REFRESH", "0")
    _set_env_default(loop_env, "WF_FORCE_REFRESH", "1")
    _set_env_default(loop_env, "WF_MAX_SYMBOLS_TRAIN", "0")
    _set_env_default(loop_env, "WF_MAX_SYMBOLS_BACKTEST", "0")
    _set_env_default(loop_env, "WF_MIN_SYMBOLS_USED_PER_PERIOD", "30")
    loop_env["WF_OUT_ROOT"] = str(out_root)
    loop_env["WF_RESULTS_CSV"] = "random_runs.csv"
    _set_env_default(loop_env, "WF_SKIP_TRAIN", "0")
    _set_env_default(loop_env, "WF_SKIP_TRAIN_MODE", "always")
    loop_env["WF_LOOP_LOG"] = str(loop_log)
    loop_env["WF_DASH_LOG"] = str(dash_log)
    loop_runner = ProcRunner(
        "loop",
        loop_cmd,
        root,
        restart_delay=int(args.restart_delay),
        log_path=loop_log,
        env=loop_env,
    )
    dash_env = os.environ.copy()
    dash_env["WF_DASH_OUT_ROOT"] = str(out_root)
    dash_env["WF_DASH_RESULTS_CSV"] = "random_runs.csv"
    dash_env["WF_DASH_LOG"] = str(dash_log)
    dash_runner = ProcRunner("dash", dash_cmd, root, restart_delay=int(args.restart_delay), log_path=dash_log, env=dash_env)

    shutdown_evt = threading.Event()

    def _cleanup() -> None:
        if shutdown_evt.is_set():
            return
        shutdown_evt.set()
        loop_runner.stop()
        dash_runner.stop()

    def _handler(signum, _frame):
        print(f"\n[STOP] signal={signum} - shutting down...", flush=True)
        _cleanup()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

    loop_runner.start()
    dash_runner.start()
    print("[INFO] Monolith running. Press Ctrl+C to stop.", flush=True)

    try:
        while not shutdown_evt.is_set():
            time.sleep(1)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
