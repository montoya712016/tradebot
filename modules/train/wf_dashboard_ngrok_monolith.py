#!/usr/bin/env python3
"""
Monolith:
1) Start local WF dashboard (Flask)
2) Open ngrok tunnel with basic auth
3) Send Pushover message with link
4) Keep running (restart ngrok if it dies)

Env vars (WF_ overrides NGROK_ when present):
- WF_NGROK_DIR / NGROK_DIR
- WF_NGROK_DOMAIN / NGROK_DOMAIN
- WF_NGROK_PORT / NGROK_PORT
- WF_NGROK_BASIC_USER / NGROK_BASIC_USER
- WF_NGROK_BASIC_PASS / NGROK_BASIC_PASS
- WF_NGROK_AUTHTOKEN / NGROK_AUTHTOKEN

Dashboard env:
- WF_DASH_OUT_ROOT (default: wf_random_loop)
- WF_DASH_RESULTS_CSV (default: random_runs.csv)
- WF_DASH_MAX_ROWS (default: 200)
- WF_DASH_MAX_IMAGES (default: 24)
- WF_DASH_REFRESH_SEC (default: 6)
- WF_DASH_LOOP_LOG (default: generated/wf_random_loop/loop.log)
- WF_DASH_LOG (default: generated/wf_random_loop/dash.log)
- WF_DASH_LOG_MAX_LINES (default: 180)
- WF_DASH_DEMO_CSV (default: false)

Pushover (optional):
- PUSHOVER_USER_KEY
- PUSHOVER_TOKEN_DASH or PUSHOVER_TOKEN_TRADE

Run:
  python modules/train/wf_dashboard_ngrok_monolith.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import urllib.parse
import urllib.request

try:
    from .wf_dashboard_server import DashboardConfig, create_app
except Exception:
    import sys

    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break
    from train.wf_dashboard_server import DashboardConfig, create_app  # type: ignore

try:
    from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send
except Exception:
    try:
        from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore[import]
    except Exception:
        _pushover_load_default = None
        _pushover_send = None


def _env_first(names: list[str], default: str) -> str:
    for name in names:
        v = os.getenv(name)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _default_downloads() -> str:
    userprofile = os.getenv("USERPROFILE") or ""
    if userprofile:
        return str(Path(userprofile) / "Downloads")
    return r"C:\Users\Public\Downloads"


def _load_pushover() -> object | None:
    if _pushover_load_default is None:
        return None
    cfg = _pushover_load_default(
        user_env="PUSHOVER_USER_KEY",
        token_env="PUSHOVER_TOKEN_DASH",
        token_name_fallback="PUSHOVER_TOKEN_DASH",
        title="tradebot WF dashboard",
    )
    if cfg is None:
        cfg = _pushover_load_default(
            user_env="PUSHOVER_USER_KEY",
            token_env="PUSHOVER_TOKEN_TRADE",
            token_name_fallback="PUSHOVER_TOKEN_TRADE",
            title="tradebot WF dashboard",
        )
    return cfg


def _send_pushover(msg: str, url: str | None = None) -> None:
    if _pushover_send is None:
        return
    cfg = _load_pushover()
    if cfg is None:
        return
    try:
        _pushover_send(msg, cfg=cfg, url=url, url_title="WF dashboard")
    except Exception:
        pass


def _ngrok_has_tunnel(domain: str) -> bool:
    domain = (domain or "").strip().lower()
    if not domain:
        return False
    url = "http://127.0.0.1:4040/api/tunnels"
    try:
        raw = urllib.request.urlopen(url, timeout=1.0).read()
        obj = json.loads(raw.decode("utf-8", errors="replace"))
        tunnels = obj.get("tunnels") or []
        for t in tunnels:
            pu = str(t.get("public_url") or "").lower()
            if domain in pu:
                return True
        return False
    except Exception:
        return False


@dataclass
class NgrokConfig:
    downloads_dir: Path = field(default_factory=lambda: Path(_env_first(["WF_NGROK_DIR", "NGROK_DIR"], _default_downloads())))
    domain: str = field(default_factory=lambda: _env_first(["WF_NGROK_DOMAIN", "NGROK_DOMAIN"], "astra-assistent.ngrok.app"))
    port: int = field(default_factory=lambda: int(_env_first(["WF_NGROK_PORT", "NGROK_PORT"], "5055")))
    username: str = field(default_factory=lambda: _env_first(["WF_NGROK_BASIC_USER", "NGROK_BASIC_USER"], "astra"))
    password: str = field(default_factory=lambda: _env_first(["WF_NGROK_BASIC_PASS", "NGROK_BASIC_PASS"], "Peixe_2017."))
    authtoken: Optional[str] = field(default_factory=lambda: _env_first(["WF_NGROK_AUTHTOKEN", "NGROK_AUTHTOKEN"], "").strip() or None)

    def build_command(self) -> list[str]:
        exe = self.downloads_dir / "ngrok.exe"
        if not exe.exists():
            raise FileNotFoundError(f"ngrok.exe not found at {exe}")
        cmd = [
            str(exe),
            "http",
            "--domain",
            self.domain,
            "--basic-auth",
            f"{self.username}:{self.password}",
        ]
        if self.authtoken:
            cmd += ["--authtoken", self.authtoken]
        cmd.append(str(self.port))
        return cmd


class NgrokManager:
    def __init__(self, config: NgrokConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._evt = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._external_session = False

    def start(self) -> None:
        if self._proc is not None:
            return
        if _ngrok_has_tunnel(self.config.domain):
            self._external_session = True
            print(f"[OK] ngrok already running: https://{self.config.domain}")
            return
        self._spawn()
        self._evt.clear()
        self._th = threading.Thread(target=self._monitor_loop, daemon=True)
        self._th.start()
        print(f"[OK] ngrok started: https://{self.config.domain} (local port {self.config.port})")

    def _spawn(self) -> None:
        cmd = self.config.build_command()
        env = os.environ.copy()
        if self.config.authtoken and "NGROK_AUTHTOKEN" not in env:
            env["NGROK_AUTHTOKEN"] = self.config.authtoken
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.config.downloads_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def _monitor_loop(self) -> None:
        while not self._evt.is_set():
            time.sleep(2)
            if self._proc is None:
                continue
            ret = self._proc.poll()
            if ret is None:
                continue
            if _ngrok_has_tunnel(self.config.domain):
                self._external_session = True
                self._proc = None
                print(f"[OK] ngrok already running: https://{self.config.domain}")
                return
            print(f"[WARN] ngrok exited (code={ret}) - restarting in 2s...")
            time.sleep(2)
            self._spawn()

    def stop(self) -> None:
        self._evt.set()
        if self._external_session:
            print("[INFO] ngrok external session detected: leaving it running.")
            return
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        print("[OK] ngrok stopped.")


class DashboardServer:
    def __init__(self, host: str, port: int, cfg: DashboardConfig):
        from werkzeug.serving import make_server

        self.host = host
        self.port = int(port)
        self.app, _ = create_app(cfg)
        self._server = make_server(self.host, self.port, self.app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"[OK] Dashboard local at http://{self.host}:{self.port}")

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _install_signal_handlers(cleanup_cb) -> None:
    def _handler(signum, _frame):
        print(f"\n[STOP] Signal {signum} received - shutting down...")
        cleanup_cb()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def _auth_url(cfg: NgrokConfig) -> str:
    user = urllib.parse.quote(cfg.username or "", safe="")
    pwd = urllib.parse.quote(cfg.password or "", safe="")
    return f"https://{user}:{pwd}@{cfg.domain}"


def run_monolith() -> None:
    cfg = NgrokConfig()
    dash_cfg = DashboardConfig(
        out_root=_env_first(["WF_DASH_OUT_ROOT"], "wf_random_loop"),
        results_csv=_env_first(["WF_DASH_RESULTS_CSV"], "random_runs.csv"),
        max_rows=int(_env_first(["WF_DASH_MAX_ROWS"], "200")),
        max_images=int(_env_first(["WF_DASH_MAX_IMAGES"], "24")),
        refresh_sec=float(_env_first(["WF_DASH_REFRESH_SEC"], "6")),
        demo_csv=_env_bool("WF_DASH_DEMO_CSV", False),
        loop_log_path=_env_first(["WF_DASH_LOOP_LOG"], "") or None,
        dash_log_path=_env_first(["WF_DASH_LOG"], "") or None,
        log_max_lines=int(_env_first(["WF_DASH_LOG_MAX_LINES"], "180")),
    )
    host = _env_first(["WF_DASH_HOST"], "127.0.0.1")
    print(f"[INFO] Public domain: https://{cfg.domain} (user '{cfg.username}')")

    dashboard = DashboardServer(host=host, port=cfg.port, cfg=dash_cfg)
    dashboard.start()

    manager = NgrokManager(cfg)
    manager.start()

    msg = (
        "WF dashboard online\n"
        f"URL: https://{cfg.domain}"
    )
    _send_pushover(msg, url=f"https://{cfg.domain}")

    shutdown_evt = threading.Event()

    def _cleanup():
        if shutdown_evt.is_set():
            return
        shutdown_evt.set()
        manager.stop()
        dashboard.stop()

    _install_signal_handlers(_cleanup)
    print("[INFO] Monolith running. Press Ctrl+C to stop.")
    try:
        while not shutdown_evt.is_set():
            time.sleep(1)
    finally:
        _cleanup()


if __name__ == "__main__":
    run_monolith()
