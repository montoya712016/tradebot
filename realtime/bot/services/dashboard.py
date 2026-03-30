from __future__ import annotations

import logging
import os
from typing import Any, Optional


log = logging.getLogger("realtime.services.dashboard")


def start_ngrok_tunnel(
    port: int,
    *,
    token_env: str = "NGROK_AUTHTOKEN",
    secrets_module: Any = None,
    domain: str = "",
) -> Optional[str]:
    try:
        from pyngrok import ngrok  # type: ignore
    except Exception:
        print("[ngrok][warn] pyngrok não disponível; túnel não iniciado (pip install pyngrok)", flush=True)
        return None

    token = (
        os.getenv("WF_NGROK_AUTHTOKEN")
        or os.getenv(token_env)
        or os.getenv("NGROK_AUTHTOKEN")
        or ""
    ).strip()
    if (not token) and secrets_module is not None:
        token = str(getattr(secrets_module, "NGROK_AUTHTOKEN", "") or "").strip()
    if token:
        try:
            ngrok.set_auth_token(token)
        except Exception:
            pass
    try:
        kwargs = {"bind_tls": True}
        if str(domain or "").strip():
            kwargs["domain"] = str(domain).strip()
        tunnel = ngrok.connect(int(port), **kwargs)
        return str(tunnel.public_url)
    except Exception as e:
        print(f"[ngrok][warn] falhou ao abrir túnel: {type(e).__name__}: {e}", flush=True)
        return None


def launch_dashboard_server(
    port: int,
    *,
    dashboard_state_cls: Any = None,
    account_summary_cls: Any = None,
) -> None:
    try:
        from modules.realtime.dashboard_server import create_app
    except ImportError:
        try:
            from realtime.dashboard_server import create_app
        except ImportError:
            print("[dash][warn] dashboard_server não encontrado", flush=True)
            return
    try:
        from werkzeug.serving import run_simple, WSGIRequestHandler
    except ImportError:
        print("[dash][warn] werkzeug não disponível; dashboard não iniciado", flush=True)
        return
    try:
        import logging as _logging

        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        WSGIRequestHandler.log = lambda *args, **kwargs: None
    except Exception:
        pass

    app, store = create_app(demo=False, refresh_sec=2.0)
    try:
        if dashboard_state_cls is not None and account_summary_cls is not None:
            empty = dashboard_state_cls(
                summary=account_summary_cls(
                    equity_usd=0.0,
                    cash_usd=0.0,
                    exposure_usd=0.0,
                    realized_pnl_usd=0.0,
                    unrealized_pnl_usd=0.0,
                ),
                positions=[],
                recent_trades=[],
                allocation={},
                meta={},
                equity_history=[],
            )
            store.set(empty)
    except Exception:
        log.warning("[dash] falha ao inicializar estado vazio", exc_info=True)

    try:
        run_simple("0.0.0.0", int(port), app, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[dash][warn] falhou ao iniciar: {type(e).__name__}: {e}", flush=True)
