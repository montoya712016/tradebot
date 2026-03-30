from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict
from typing import Any
from pathlib import Path

from flask import Flask, jsonify, render_template, render_template_string, request, redirect, session, url_for

from .auth import (
    delete_user,
    ensure_user_store,
    get_user,
    list_users,
    load_dashboard_auth_config,
    normalize_next_url,
    register_user,
    session_has_access,
    session_is_admin,
    session_is_owner,
    set_user_owner,
    set_user_admin,
    set_user_enabled,
    verify_user_login,
)

from .dashboard_state import DemoStateGenerator, StateStore, create_demo_state
# trade contract (para EMA de saída)
try:  # pragma: no cover
    from trade_contract import DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:  # pragma: no cover
    DEFAULT_TRADE_CONTRACT = None  # type: ignore
    exit_ema_span_from_window = None  # type: ignore

# OHLC helpers (opcionais)
try:  # pragma: no cover - dependência dinâmica
    from prepare_features.data import load_ohlc_1m_series
except Exception:  # pragma: no cover
    load_ohlc_1m_series = None  # type: ignore
try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _read_sysmon_tail(path: Path, limit: int = 300) -> list[dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and rec.get("ts_utc"):
                    rows.append(rec)
    except Exception:
        return []
    return list(rows)


def create_app(*, demo: bool = True, refresh_sec: float = 2.0) -> tuple[Flask, StateStore]:
    repo_root = Path(__file__).resolve().parents[2]
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["JSON_SORT_KEYS"] = False
    app.secret_key = os.getenv("ASTRA_DASHBOARD_SECRET_KEY", "astra-dashboard-dev")
    sysmon_path = Path(os.getenv("SYS_MON_LOG_PATH", "data/sysmon.jsonl"))
    auth_cfg = load_dashboard_auth_config(
        user_envs=("BOT_DASHBOARD_USER", "ASTRA_DASHBOARD_USER", "DASHBOARD_USER", "NGROK_BASIC_USER"),
        pass_envs=("BOT_DASHBOARD_PASS", "ASTRA_DASHBOARD_PASS", "DASHBOARD_PASS", "NGROK_BASIC_PASS"),
        session_key="astra_bot_dashboard_auth",
        default_db_relpath=str(repo_root / "local" / "dashboard_users.json"),
    )
    ensure_user_store(auth_cfg)

    store = StateStore(create_demo_state())
    demo_gen: DemoStateGenerator | None = None
    if demo:
        demo_gen = DemoStateGenerator(store, refresh_sec=refresh_sec)
        demo_gen.start()

    @app.context_processor
    def inject_auth_context() -> dict[str, Any]:
        current_user = session.get(auth_cfg.session_key)
        return {
            "auth_enabled": auth_cfg.enabled,
            "auth_is_admin": session_is_admin(auth_cfg, current_user),
            "auth_is_owner": session_is_owner(auth_cfg, current_user),
        }

    @app.before_request
    def require_dashboard_login() -> Any:
        if not auth_cfg.enabled:
            return None
        endpoint = (request.endpoint or "").strip()
        if endpoint in {"index", "login", "register", "logout", "static", "api_health", "api_update"}:
            return None
        if session_has_access(auth_cfg, session.get(auth_cfg.session_key)):
            return None
        session.pop(auth_cfg.session_key, None)
        next_url = normalize_next_url(request.full_path if request.query_string else request.path)
        return redirect(url_for("login", next=next_url))

    @app.route("/login", methods=["GET", "POST"])
    def login() -> Any:
        if not auth_cfg.enabled:
            return redirect(url_for("dashboard"))
        next_url = normalize_next_url(request.values.get("next"), fallback=url_for("dashboard"))
        error = ""
        username = ""
        if request.method == "POST":
            username = str(request.form.get("username", "") or "").strip()
            password = str(request.form.get("password", "") or "")
            ok, error = verify_user_login(auth_cfg, username, password)
            if ok:
                session.clear()
                session[auth_cfg.session_key] = str(get_user(auth_cfg, username).get("username"))
                return redirect(next_url)
        return render_template(
            "login.html",
            title="Astra — Login",
            badge="Realtime",
            eyebrow="Acesso protegido",
            heading="Entrar no dashboard do bot",
            subtitle="Use suas credenciais para acessar o monitoramento ao vivo e os controles do runtime.",
            error=error,
            next_url=next_url,
            username=username,
        )

    @app.route("/register", methods=["GET", "POST"])
    def register() -> Any:
        if not auth_cfg.enabled:
            return redirect(url_for("dashboard"))
        next_url = normalize_next_url(request.values.get("next"), fallback=url_for("dashboard"))
        error = ""
        success = ""
        full_name = ""
        cpf = ""
        phone = ""
        email = ""
        username = ""
        if request.method == "POST":
            full_name = str(request.form.get("full_name", "") or "").strip()
            cpf = str(request.form.get("cpf", "") or "").strip()
            phone = str(request.form.get("phone", "") or "").strip()
            email = str(request.form.get("email", "") or "").strip()
            username = str(request.form.get("username", "") or "").strip()
            password = str(request.form.get("password", "") or "")
            ok, message = register_user(
                auth_cfg,
                full_name=full_name,
                cpf=cpf,
                phone=phone,
                email=email,
                username=username,
                password=password,
            )
            if ok:
                success = message
                full_name = ""
                cpf = ""
                phone = ""
                email = ""
                username = ""
            else:
                error = message
        return render_template(
            "register.html",
            title="Astra — Criar conta",
            badge="Realtime",
            eyebrow="Cadastro",
            heading="Criar conta para o dashboard do bot",
            subtitle="Seu cadastro ficará pendente até você liberar manualmente o acesso na base local de usuários.",
            error=error,
            success=success,
            next_url=next_url,
            full_name=full_name,
            cpf=cpf,
            phone=phone,
            email=email,
            username=username,
        )

    @app.post("/logout")
    def logout() -> Any:
        session.pop(auth_cfg.session_key, None)
        return redirect(url_for("index"))

    @app.get("/admin/users")
    def admin_users() -> Any:
        if not session_is_admin(auth_cfg, session.get(auth_cfg.session_key)):
            return redirect(url_for("dashboard"))
        users = list_users(auth_cfg)
        flash_success = session.pop("admin_flash_success", "")
        flash_error = session.pop("admin_flash_error", "")
        return render_template(
            "admin_users.html",
            title="Astra — Usuários",
            badge="Realtime Admin",
            eyebrow="Administração",
            heading="Usuários do dashboard do bot",
            subtitle="Gerencie quem pode acessar o runtime e aprove cadastros pendentes.",
            total_users=len(users),
            pending_users=sum(1 for item in users if not item.get("enabled")),
            enabled_users=sum(1 for item in users if item.get("enabled")),
            admin_users=sum(1 for item in users if item.get("is_admin")),
            owner_users=sum(1 for item in users if item.get("is_owner")),
            users=users,
            action_url=url_for("admin_users_action"),
            home_url=url_for("dashboard"),
            flash_success=flash_success,
            flash_error=flash_error,
        )

    @app.post("/admin/users/action")
    def admin_users_action() -> Any:
        current_user = str(session.get(auth_cfg.session_key, "") or "")
        is_admin = session_is_admin(auth_cfg, current_user)
        is_owner = session_is_owner(auth_cfg, current_user)
        if not is_admin:
            return redirect(url_for("dashboard"))
        username = str(request.form.get("username", "") or "").strip()
        action = str(request.form.get("action", "") or "").strip().lower()
        target = get_user(auth_cfg, username)
        if target and target.get("is_owner") and not is_owner:
            ok, msg = False, "A conta owner só pode ser alterada pelo próprio owner."
            session["admin_flash_error"] = msg
            return redirect(url_for("admin_users"))
        if action == "enable":
            ok, msg = set_user_enabled(auth_cfg, username, True)
        elif action == "disable":
            if username == current_user and is_owner:
                ok, msg = False, "O owner não pode desabilitar a própria conta."
            else:
                ok, msg = set_user_enabled(auth_cfg, username, False)
        elif action == "make_admin":
            ok, msg = set_user_admin(auth_cfg, username, True)
        elif action == "remove_admin":
            admins = [u for u in list_users(auth_cfg) if u.get("is_admin")]
            if username == current_user:
                ok, msg = False, "Você não pode remover seu próprio papel de admin."
            elif len(admins) <= 1:
                ok, msg = False, "Não é possível remover o último admin."
            else:
                ok, msg = set_user_admin(auth_cfg, username, False)
        elif action == "make_owner":
            if not is_owner:
                ok, msg = False, "Só o owner pode promover outro owner."
            else:
                ok, msg = set_user_owner(auth_cfg, username, True)
        elif action == "remove_owner":
            owners = [u for u in list_users(auth_cfg) if u.get("is_owner")]
            if not is_owner:
                ok, msg = False, "Só o owner pode remover outro owner."
            elif username == current_user:
                ok, msg = False, "Você não pode remover o próprio papel de owner."
            elif len(owners) <= 1:
                ok, msg = False, "Não é possível remover o último owner."
            else:
                ok, msg = set_user_owner(auth_cfg, username, False)
        elif action == "delete":
            if username == current_user:
                ok, msg = False, "Você não pode remover a própria conta em uso."
            else:
                ok, msg = delete_user(auth_cfg, username)
        else:
            ok, msg = False, "Ação inválida."
        session["admin_flash_success" if ok else "admin_flash_error"] = msg
        return redirect(url_for("admin_users"))

    @app.get("/")
    def index() -> Any:
        contact_email = os.getenv("ASTRA_CONTACT_EMAIL", "astraquantlab@gmail.com").strip() or "astraquantlab@gmail.com"
        return render_template(
            "landing.html",
            title="Astra Tradebot",
            runtime_badge="Realtime",
            dashboard_url=url_for("dashboard"),
            login_url=url_for("login", next=url_for("dashboard")),
            register_url=url_for("register", next=url_for("dashboard")),
            contact_email=contact_email,
            contact_focus="Systematic operations, runtime supervision, and private partnerships",
            contact_href=f"mailto:{contact_email}?subject=Astra%20Tradebot%20Conversation",
            session_has_access=session_has_access(auth_cfg, session.get(auth_cfg.session_key)),
            hero_eyebrow="Astra • quantitative research • realtime operations",
            hero_prefix="Production-facing monitoring and",
            hero_accent="systematic execution infrastructure",
            hero_suffix=".",
            hero_copy=(
                "Astra develops Tradebot, a proprietary stack for research, deployment control, "
                "live monitoring, and systematic execution in digital asset markets."
            ),
            stats=[
                {"label": "Runtime modes", "value": "Live / Paper"},
                {"label": "Primary surface", "value": "Realtime"},
                {"label": "Auth model", "value": "Local"},
                {"label": "Ops focus", "value": "Monitoring"},
            ],
            summary_copy=(
                "This environment is designed for operational supervision, position visibility, "
                "trade inspection, and controlled runtime observation."
            ),
            chips=[
                "Live monitoring",
                "Execution controls",
                "Position visibility",
                "Protected access",
                "Quant + engineering",
            ],
            system_title="The realtime dashboard is the production-facing surface of Tradebot.",
            system_copy=(
                "It centralizes positions, equity, allocation, signals, trades, system health, "
                "and supporting diagnostics for controlled live or paper operation."
            ),
            info_cards=[
                {
                    "eyebrow": "Visibility",
                    "title": "Portfolio state",
                    "copy": "Open positions, allocation, equity history, PnL, and execution footprint in a single view.",
                },
                {
                    "eyebrow": "Monitoring",
                    "title": "Runtime supervision",
                    "copy": "Feed delay, process health, system telemetry, and realtime diagnostics for operational confidence.",
                },
                {
                    "eyebrow": "Discipline",
                    "title": "Controlled access",
                    "copy": "In-app login, local user base, admin controls, and owner-level authority for restricted collaboration.",
                },
            ],
            pipeline=[
                {
                    "step": "01",
                    "title": "Live state ingestion",
                    "copy": "The runtime consumes execution state and makes it available to the dashboard through a controlled internal API.",
                },
                {
                    "step": "02",
                    "title": "Positions and equity",
                    "copy": "Open positions, allocation, PnL, and equity evolution are rendered as the core operating surface.",
                },
                {
                    "step": "03",
                    "title": "Signals and trade inspection",
                    "copy": "Recent trades and model signals can be reviewed with supporting detail and context.",
                },
                {
                    "step": "04",
                    "title": "System telemetry",
                    "copy": "Feed delay, process status, and hardware observations support practical supervision.",
                },
                {
                    "step": "05",
                    "title": "Protected collaboration",
                    "copy": "Local access control and role separation keep the runtime visible without turning it into an open public surface.",
                },
                {
                    "step": "06",
                    "title": "Deployment review loop",
                    "copy": "The dashboard is designed as a real operating layer, not just a static report page.",
                },
            ],
        )
        shared_css_path = repo_root / "modules" / "realtime" / "static" / "astra_shared.css"
        try:
            shared_css = shared_css_path.read_text(encoding="utf-8")
        except Exception:
            shared_css = ""
        return render_template_string(
            """
            <!DOCTYPE html>
            <html lang="en" data-bs-theme="dark">
            <head>
                <title>Astra Tradebot</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link
                  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
                  rel="stylesheet"
                  integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
                  crossorigin="anonymous"
                />
                <link
                  rel="stylesheet"
                  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
                />
                <style>
                  {{ shared_css|safe }}
                  :root {
                    --text-main: #f8fafc;
                    --text-muted: rgba(255, 255, 255, 0.68);
                    --accent-2: #00ffb3;
                  }
                  body {
                    color: var(--text-main);
                    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                  }
                  .site-shell { max-width: 1340px; margin: 0 auto; }
                  .hero-wrap { padding: 72px 0 28px; }
                  .hero-grid {
                    display: grid;
                    grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
                    gap: 24px;
                    align-items: stretch;
                  }
                  .hero-card,
                  .summary-card,
                  .section-card,
                  .offer-card {
                    background: rgba(16, 19, 26, 0.62);
                    border: 1px solid rgba(255,255,255,0.08);
                    backdrop-filter: blur(12px);
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
                  }
                  .eyebrow {
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    font-size: 0.78rem;
                    font-weight: 700;
                  }
                  .hero-title {
                    font-size: clamp(2.4rem, 5vw, 4.4rem);
                    line-height: 0.98;
                    letter-spacing: -0.04em;
                    font-weight: 800;
                    margin: 0;
                  }
                  .hero-title .accent { color: var(--accent-2); }
                  .hero-copy {
                    color: var(--text-muted);
                    font-size: 1.05rem;
                    line-height: 1.7;
                    max-width: 760px;
                  }
                  .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
                  .btn-accent {
                    background: linear-gradient(135deg, #7c5cff, #5a8cff);
                    color: #fff;
                    border: none;
                    box-shadow: 0 14px 28px rgba(124, 92, 255, 0.26);
                  }
                  .btn-accent:hover, .btn-accent:focus { color: #fff; background: linear-gradient(135deg, #6f51ff, #4e7bff); }
                  .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 14px;
                  }
                  .stat-label {
                    color: var(--text-muted);
                    font-size: 0.74rem;
                    text-transform: uppercase;
                    letter-spacing: 0.04em;
                    font-weight: 700;
                  }
                  .stat-value {
                    font-size: 1.45rem;
                    font-weight: 800;
                    letter-spacing: -0.03em;
                  }
                  .section-title {
                    font-size: 1.55rem;
                    font-weight: 700;
                    letter-spacing: -0.03em;
                    margin: 0;
                  }
                  .section-copy {
                    color: var(--text-muted);
                    line-height: 1.75;
                    margin: 0;
                  }
                  .offer-grid {
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 18px;
                  }
                  .offer-card h3 {
                    font-size: 1rem;
                    font-weight: 700;
                    margin-bottom: 10px;
                  }
                  .offer-card p {
                    color: var(--text-muted);
                    margin: 0;
                    line-height: 1.65;
                  }
                  .chip-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                  }
                  .chip {
                    border: 1px solid rgba(255,255,255,0.10);
                    background: rgba(255,255,255,0.04);
                    border-radius: 999px;
                    padding: 0.4rem 0.8rem;
                    color: var(--text-muted);
                    font-size: 0.78rem;
                  }
                  .divider-line {
                    height: 1px;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
                    margin: 10px 0 18px;
                  }
                  @media (max-width: 1080px) {
                    .hero-grid,
                    .offer-grid,
                    .stat-grid { grid-template-columns: 1fr; }
                  }
                </style>
            </head>
            <body class="astra-body">
                <header class="navbar navbar-expand-lg navbar-glass sticky-top">
                    <div class="container-fluid px-3">
                        <a class="navbar-brand d-flex align-items-center gap-2" href="{{ url_for('index') }}">
                            <span class="brand-dot"></span>
                            <span class="fw-semibold">Astra Tradebot</span>
                            <span class="badge text-bg-secondary ms-1">Realtime</span>
                        </a>
                        <div class="d-flex align-items-center gap-2">
                            {% if auth_enabled and session_has_access %}
                            <a href="{{ url_for('dashboard') }}" class="btn btn-sm btn-outline-secondary">
                                <i class="bi bi-grid-1x2"></i>
                                <span class="d-none d-sm-inline ms-1">Open Dashboard</span>
                            </a>
                            {% else %}
                            <a href="{{ url_for('login', next=url_for('dashboard')) }}" class="btn btn-sm btn-outline-secondary">
                                <i class="bi bi-box-arrow-in-right"></i>
                                <span class="d-none d-sm-inline ms-1">Login</span>
                            </a>
                            {% endif %}
                        </div>
                    </div>
                </header>
                <main class="container-fluid px-3">
                    <div class="site-shell">
                        <section class="hero-wrap">
                            <div class="hero-grid">
                                <section class="hero-card p-4 p-lg-5">
                                    <div class="eyebrow mb-3">Astra • Quantitative research • Realtime operations</div>
                                    <h1 class="hero-title mb-4">Production-facing monitoring and <span class="accent">systematic execution infrastructure</span>.</h1>
                                    <p class="hero-copy mb-4">
                                        Astra develops Tradebot, a proprietary stack for research, deployment control,
                                        live monitoring, and systematic execution in digital asset markets.
                                    </p>
                                    <div class="hero-actions">
                                        {% if auth_enabled and session_has_access %}
                                        <a href="{{ url_for('dashboard') }}" class="btn btn-accent btn-lg">
                                            <i class="bi bi-grid-1x2 me-2"></i>Open Dashboard
                                        </a>
                                        {% else %}
                                        <a href="{{ url_for('login', next=url_for('dashboard')) }}" class="btn btn-accent btn-lg">
                                            <i class="bi bi-box-arrow-in-right me-2"></i>Login to Dashboard
                                        </a>
                                        {% endif %}
                                    </div>
                                </section>
                                <aside class="summary-card p-4 p-lg-4">
                                    <div class="eyebrow mb-3">Runtime profile</div>
                                    <div class="stat-grid mb-3">
                                        <div>
                                            <div class="stat-label">Modes</div>
                                            <div class="stat-value">Live / Paper</div>
                                        </div>
                                        <div>
                                            <div class="stat-label">Surface</div>
                                            <div class="stat-value">Realtime</div>
                                        </div>
                                        <div>
                                            <div class="stat-label">Auth</div>
                                            <div class="stat-value">Local</div>
                                        </div>
                                    </div>
                                    <div class="divider-line"></div>
                                    <p class="section-copy mb-3">
                                        This environment is designed for operational supervision, position visibility,
                                        trade inspection, and controlled runtime observation.
                                    </p>
                                    <div class="chip-row">
                                        <span class="chip">Live monitoring</span>
                                        <span class="chip">Execution controls</span>
                                        <span class="chip">Position visibility</span>
                                        <span class="chip">Quant + engineering</span>
                                    </div>
                                </aside>
                            </div>
                        </section>
                        <section class="section-card p-4 p-lg-5 mb-4">
                            <div class="eyebrow mb-2">Operational overview</div>
                            <h2 class="section-title mb-3">The realtime dashboard is the production-facing surface of Tradebot.</h2>
                            <p class="section-copy mb-0">
                                It centralizes positions, equity, allocation, signals, trades, system health, and supporting diagnostics for controlled live or paper operation.
                            </p>
                        </section>
                        <section class="offer-grid pb-4">
                            <article class="offer-card p-4">
                                <div class="eyebrow mb-2">Visibility</div>
                                <h3>Portfolio state</h3>
                                <p>Open positions, allocation, equity history, PnL, and execution footprint in a single view.</p>
                            </article>
                            <article class="offer-card p-4">
                                <div class="eyebrow mb-2">Monitoring</div>
                                <h3>Runtime supervision</h3>
                                <p>Feed delay, process health, system telemetry, and realtime diagnostics for operational confidence.</p>
                            </article>
                            <article class="offer-card p-4">
                                <div class="eyebrow mb-2">Discipline</div>
                                <h3>Controlled access</h3>
                                <p>In-app login, local user base, admin controls, and owner-level authority for restricted collaboration.</p>
                            </article>
                        </section>
                    </div>
                </main>
            </body>
            </html>
            """,
            shared_css=shared_css,
            session_has_access=session_has_access(auth_cfg, session.get(auth_cfg.session_key)),
        )

    @app.get("/dashboard")
    def dashboard() -> Any:
        st = store.get().to_dict()
        return render_template("index.html", initial_state=st)

    @app.get("/api/state")
    def api_state() -> Any:
        st = store.get().to_dict()
        meta = st.get("meta") or {}
        system = meta.get("system") or {}
        if not system:
            tail = _read_sysmon_tail(sysmon_path, limit=1)
            if tail:
                meta = dict(meta)
                meta["system"] = tail[-1]
                st["meta"] = meta
        return jsonify(st)

    @app.post("/api/update")
    def api_update() -> Any:
        """
        Endpoint pra integração futura com o robô:
        POST JSON com summary/positions/trades/allocation/meta.
        """
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload_must_be_object"}), 400
        try:
            store.update_from_payload(payload)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    @app.get("/api/health")
    def api_health() -> Any:
        return jsonify({"ok": True})

    @app.get("/api/config")
    def api_config() -> Any:
        return jsonify(
            {
                "demo": bool(demo),
                "refresh_sec": float(refresh_sec),
                "server": "realtime_dashboard",
            }
        )

    @app.get("/api/system")
    def api_system() -> Any:
        try:
            limit = int(request.args.get("limit", "300") or 300)
        except Exception:
            limit = 300
        limit = max(1, min(2000, limit))
        data = _read_sysmon_tail(sysmon_path, limit=limit)
        return jsonify({"ok": True, "data": data})

    @app.get("/api/ohlc_window")
    def api_ohlc_window() -> Any:
        """
        Retorna OHLC+EMA para um símbolo.
        Params:
        - symbol (obrigatório)
        - end_ms (epoch ms, opcional; default=agora)
        - lookback_min (minutos, opcional; default=30, max=1440)
        """
        import time as _time
        _t0 = _time.time()
        print(f"[DEBUG] ohlc_window called: {request.args}")
        
        if load_ohlc_1m_series is None or pd is None:
            print("[DEBUG] ohlc_loader_unavailable")
            return jsonify({"ok": False, "error": "ohlc_loader_unavailable"}), 500
        sym = (request.args.get("symbol") or "").upper().strip()
        if not sym:
            print("[DEBUG] symbol_required")
            return jsonify({"ok": False, "error": "symbol_required"}), 400
        try:
            end_ms = int(request.args.get("end_ms") or int(pd.Timestamp.utcnow().value // 1_000_000))
        except Exception:
            end_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
        try:
            lookback_min = min(1440, max(1, int(request.args.get("lookback_min") or 30)))
        except Exception:
            lookback_min = 30
        print(f"[DEBUG] symbol={sym}, end_ms={end_ms}, lookback_min={lookback_min}")
        start_ms = end_ms - lookback_min * 60_000
        
        # Calculate needed days based on start_ms relative to now
        now_ms = int(_time.time() * 1000)
        age_ms = now_ms - start_ms
        days_needed = (age_ms / 86400_000.0) + 0.5 # margin
        
        # Carregar apenas o necessário + 1 dia extra
        days = min(7, max(1, int(days_needed))) # Increased limit to 7 days, min 1
        
        print(f"[DEBUG] loading days={days} for sym={sym}")
        df = load_ohlc_1m_series(sym, days, remove_tail_days=0)
        if df is None or df.empty:
            return jsonify({"ok": False, "error": "no_data"}), 404
        df = df[(df.index.view("int64") // 1_000_000 >= start_ms) & (df.index.view("int64") // 1_000_000 <= end_ms)]
        if df.empty:
            # fallback: usa os últimos candles disponíveis
            df = load_ohlc_1m_series(sym, days, remove_tail_days=0)
            if df is None or df.empty:
                return jsonify({"ok": False, "error": "no_data_in_range"}), 404
            df = df.sort_index().tail(min(120, lookback_min))
        df = df.sort_index()
        # Limita a 500 candles para performance
        if len(df) > 500:
            df = df.tail(500)
        if (DEFAULT_TRADE_CONTRACT is not None) and (exit_ema_span_from_window is not None):
            span = exit_ema_span_from_window(DEFAULT_TRADE_CONTRACT, 60)
            offset = float(getattr(DEFAULT_TRADE_CONTRACT, "exit_ema_init_offset_pct", 0.0) or 0.0)
        else:
            span = 55
            offset = 0.0
        payload = [
            {
                "ts": int(ts.value // 1_000_000),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for ts, r in df.iterrows()
        ]
        _elapsed = _time.time() - _t0
        print(f"[DEBUG] ohlc_window returning {len(payload)} candles in {_elapsed:.3f}s")
        return jsonify(
            {
                "ok": True,
                "symbol": sym,
                "data": payload,
                "ema_span": int(span),
                "ema_offset_pct": float(offset),
            }
        )

    # Só pra evitar "unused variable" e deixar explícito:
    _ = asdict  # noqa: F841
    _ = _bool_env  # noqa: F841
    _ = demo_gen  # noqa: F841

    return app, store
