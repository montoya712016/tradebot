#!/usr/bin/env python3
"""
Monolito para:
1) Subir o dashboard Flask local (porta 5000 por padrão)
2) Abrir túnel ngrok fixo (domínio custom + basic auth)
3) Manter tudo rodando (reinicia ngrok se cair)

Variáveis de ambiente:
- NGROK_DIR (ex.: C:\\Users\\<vc>\\Downloads)
- NGROK_DOMAIN (ex.: astra-assistent.ngrok.app)
- NGROK_PORT (porta local do dashboard)
- NGROK_BASIC_USER / NGROK_BASIC_PASS
- NGROK_AUTHTOKEN (opcional)

Execução:
  python realtime/realtime_dashboard_ngrok_monolith.py
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
import urllib.request

try:
    from .dashboard_server import create_app
except Exception:
    import sys

    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break
    from realtime.dashboard_server import create_app  # type: ignore


def _env_or_default(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v else default


def _default_downloads() -> str:
    userprofile = os.getenv("USERPROFILE") or ""
    if userprofile:
        return str(Path(userprofile) / "Downloads")
    return r"C:\Users\Public\Downloads"


@dataclass
class NgrokConfig:
    downloads_dir: Path = field(default_factory=lambda: Path(_env_or_default("NGROK_DIR", _default_downloads())))
    domain: str = field(default_factory=lambda: _env_or_default("NGROK_DOMAIN", "astra-assistent.ngrok.app"))
    port: int = field(default_factory=lambda: int(_env_or_default("NGROK_PORT", "5000")))
    username: str = field(default_factory=lambda: _env_or_default("NGROK_BASIC_USER", "astra"))
    password: str = field(default_factory=lambda: _env_or_default("NGROK_BASIC_PASS", "SenhaUltraSecreta!"))
    authtoken: Optional[str] = field(default_factory=lambda: os.getenv("NGROK_AUTHTOKEN"))

    def build_command(self) -> list[str]:
        exe = self.downloads_dir / "ngrok.exe"
        if not exe.exists():
            raise FileNotFoundError(f"ngrok.exe não encontrado em {exe}")
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
        # Se já existir um túnel local (4040), não tente abrir outro (ngrok free = 1 sessão).
        if _ngrok_has_tunnel(self.config.domain):
            self._external_session = True
            print(f"[OK] ngrok ja esta rodando: https://{self.config.domain} (reutilizando sessao existente)")
            return
        self._spawn()
        self._evt.clear()
        self._th = threading.Thread(target=self._monitor_loop, daemon=True)
        self._th.start()
        # pequena verificação: se cair imediatamente, a gente imprime o motivo no monitor.
        print(f"[OK] ngrok iniciado (tentando) em https://{self.config.domain} (porta local {self.config.port})")

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
            # Caso clássico: ERR_NGROK_108 (já existe sessão). Reusa em vez de loop infinito.
            if _ngrok_has_tunnel(self.config.domain):
                self._external_session = True
                self._proc = None
                print(f"[OK] ngrok ja esta rodando: https://{self.config.domain} (reutilizando sessao existente)")
                return

            print(f"[WARN] ngrok caiu (code={ret}) - tentando reiniciar em 2s...")
            time.sleep(2)
            self._spawn()

    def stop(self) -> None:
        self._evt.set()
        if self._external_session:
            print("[INFO] ngrok externo detectado: nao vou finalizar o processo existente.")
            return
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        print("[OK] ngrok desligado.")


class DashboardServer:
    def __init__(self, host: str, port: int):
        from werkzeug.serving import make_server

        self.host = host
        self.port = int(port)
        self.app, _ = create_app(demo=True, refresh_sec=2.0)
        self._server = make_server(self.host, self.port, self.app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()
        print(f"[OK] Dashboard local em http://{self.host}:{self.port}")

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _install_signal_handlers(cleanup_cb) -> None:
    def _handler(signum, _frame):
        print(f"\n[STOP] Sinal {signum} recebido - encerrando monolito...")
        cleanup_cb()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def executar_monolito_ngrok() -> None:
    cfg = NgrokConfig()
    print(f"[INFO] Dominio publico: https://{cfg.domain} (user '{cfg.username}')")

    # 1) Sobe dashboard local (thread)
    dashboard = DashboardServer(host="127.0.0.1", port=cfg.port)
    dashboard.start()

    # 2) Sobe túnel ngrok e monitora
    manager = NgrokManager(cfg)
    manager.start()

    # 3) Mantém vivo até sinal
    shutdown_evt = threading.Event()

    def _cleanup():
        if shutdown_evt.is_set():
            return
        shutdown_evt.set()
        manager.stop()
        dashboard.stop()

    _install_signal_handlers(_cleanup)
    print("[INFO] Monolito em execucao. Pressione Ctrl+C para sair.")
    try:
        while not shutdown_evt.is_set():
            time.sleep(1)
    finally:
        _cleanup()


if __name__ == "__main__":
    executar_monolito_ngrok()


def _ngrok_has_tunnel(domain: str) -> bool:
    """
    Heurística simples: checa a API local do ngrok (default 4040) e vê se existe o domínio.
    Se existir, significa que já há uma sessão/agent rodando (local) e não devemos abrir outra.
    """
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

