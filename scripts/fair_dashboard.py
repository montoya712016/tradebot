import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template, render_template_string, jsonify, abort, send_file, session, request, redirect, url_for
import threading
import time
import subprocess
import urllib.request
import json
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass, field
from collections import deque

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_IMPORT_ROOT = SCRIPT_DIR.parent
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from modules.realtime.auth import (
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
from modules.realtime.remote_control import RemoteControlManager
from modules.realtime.site_oos_assets import build_site_snapshot

# Reuse existing ngrok logic
@dataclass
class NgrokConfig:
    downloads_dir: Path = Path(r"C:\Users\NovoLucas\Downloads")
    domain: str = "astra-assistent.ngrok.app"
    port: int = 5060
    authtoken: str = os.getenv("WF_NGROK_AUTHTOKEN", "")

    def build_command(self) -> list[str]:
        exe = self.downloads_dir / "ngrok.exe"
        cmd = [
            str(exe), "http",
            "--domain", self.domain,
        ]
        if self.authtoken:
            cmd += ["--authtoken", self.authtoken]
        cmd.append(str(self.port))
        return cmd

class NgrokManager:
    def __init__(self, config: NgrokConfig):
        self.config = config
        self._proc = None
    
    def start(self):
        try:
            urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1)
            print("[OK] ngrok already running.")
            return
        except:
            pass
        print(f"[INFO] Starting ngrok tunnel to https://{self.config.domain}")
        self._proc = subprocess.Popen(self.config.build_command(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def stop(self):
        if self._proc:
            self._proc.terminate()

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
EQUITY_METRIC_CACHE: dict[str, dict] = {}
SHARED_DASHBOARD_CSS = (REPO_ROOT / "modules" / "realtime" / "static" / "astra_shared.css")
REMOTE_CONTROL = RemoteControlManager(
    repo_root=REPO_ROOT,
    storage_dir=REPO_ROOT / "local" / "remote_control",
)
app = Flask(
    __name__,
    template_folder=str(REPO_ROOT / "modules" / "realtime" / "templates"),
    static_folder=str(REPO_ROOT / "modules" / "realtime" / "static"),
    static_url_path="/static",
)
app.secret_key = os.getenv("ASTRA_DASHBOARD_SECRET_KEY", "astra-dashboard-dev")
AUTH_CFG = load_dashboard_auth_config(
    user_envs=("WF_DASHBOARD_USER", "ASTRA_DASHBOARD_USER", "DASHBOARD_USER", "NGROK_BASIC_USER"),
    pass_envs=("WF_DASHBOARD_PASS", "ASTRA_DASHBOARD_PASS", "DASHBOARD_PASS", "NGROK_BASIC_PASS"),
    session_key="astra_fair_dashboard_auth",
    default_db_relpath=str(REPO_ROOT / "local" / "dashboard_users.json"),
)
ensure_user_store(AUTH_CFG)


def _fair_root() -> Path:
    raw = str(os.getenv("WF_FAIR_ROOT", "fair_wf_explore_v6") or "fair_wf_explore_v6").strip()
    return REPO_ROOT / "data" / "generated" / raw


def _shared_dashboard_css_text() -> str:
    try:
        return SHARED_DASHBOARD_CSS.read_text(encoding="utf-8")
    except Exception:
        return ""


def _require_assistant_admin() -> str:
    current_user = session.get(AUTH_CFG.session_key)
    if not session_is_admin(AUTH_CFG, current_user):
        abort(403)
    return str(current_user or "")


@app.context_processor
def inject_auth_context() -> dict:
    current_user = session.get(AUTH_CFG.session_key)
    return {
        "auth_enabled": AUTH_CFG.enabled,
        "auth_is_admin": session_is_admin(AUTH_CFG, current_user),
        "auth_is_owner": session_is_owner(AUTH_CFG, current_user),
    }


@app.before_request
def require_dashboard_login():
    if not AUTH_CFG.enabled:
        return None
    endpoint = (request.endpoint or "").strip()
    if endpoint in {"index", "login", "register", "logout", "static"}:
        return None
    if session_has_access(AUTH_CFG, session.get(AUTH_CFG.session_key)):
        return None
    session.pop(AUTH_CFG.session_key, None)
    next_url = normalize_next_url(request.full_path if request.query_string else request.path)
    return redirect(url_for("login", next=next_url))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_CFG.enabled:
        return redirect(url_for("dashboard"))
    next_url = normalize_next_url(request.values.get("next"), fallback=url_for("dashboard"))
    error = ""
    username = ""
    if request.method == "POST":
        username = str(request.form.get("username", "") or "").strip()
        password = str(request.form.get("password", "") or "")
        ok, error = verify_user_login(AUTH_CFG, username, password)
        if ok:
            session.clear()
            session[AUTH_CFG.session_key] = str(get_user(AUTH_CFG, username).get("username"))
            return redirect(next_url)
    return render_template(
        "login.html",
        title="Astra — Fair Explore Login",
        badge="Fair Explore",
        eyebrow="Acesso protegido",
        heading="Entrar no Fair Explore",
        subtitle="Use suas credenciais para acompanhar milestones, trials e artefatos do explore.",
        error=error,
        next_url=next_url,
        username=username,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if not AUTH_CFG.enabled:
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
            AUTH_CFG,
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
        title="Astra — Criar conta Fair Explore",
        badge="Fair Explore",
        eyebrow="Cadastro",
        heading="Criar conta para o Fair Explore",
        subtitle="Seu cadastro ficará pendente até a liberação manual do acesso pela administração.",
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
def logout():
    session.pop(AUTH_CFG.session_key, None)
    return redirect(url_for("index"))


@app.get("/admin/users")
def admin_users():
    if not session_is_admin(AUTH_CFG, session.get(AUTH_CFG.session_key)):
        return redirect(url_for("dashboard"))
    users = list_users(AUTH_CFG)
    flash_success = session.pop("admin_flash_success", "")
    flash_error = session.pop("admin_flash_error", "")
    return render_template(
        "admin_users.html",
        title="Astra — Usuários Fair Explore",
        badge="Fair Explore Admin",
        eyebrow="Administração",
        heading="Usuários do Fair Explore",
        subtitle="Gerencie aprovações e bloqueios de acesso ao dashboard do explore.",
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
def admin_users_action():
    current_user = str(session.get(AUTH_CFG.session_key, "") or "")
    is_admin = session_is_admin(AUTH_CFG, current_user)
    is_owner = session_is_owner(AUTH_CFG, current_user)
    if not is_admin:
        return redirect(url_for("dashboard"))
    username = str(request.form.get("username", "") or "").strip()
    action = str(request.form.get("action", "") or "").strip().lower()
    target = get_user(AUTH_CFG, username)
    if target and target.get("is_owner") and not is_owner:
        ok, msg = False, "A conta owner só pode ser alterada pelo próprio owner."
        session["admin_flash_error"] = msg
        return redirect(url_for("admin_users"))
    if action == "enable":
        ok, msg = set_user_enabled(AUTH_CFG, username, True)
    elif action == "disable":
        if username == current_user and is_owner:
            ok, msg = False, "O owner não pode desabilitar a própria conta."
        else:
            ok, msg = set_user_enabled(AUTH_CFG, username, False)
    elif action == "make_admin":
        ok, msg = set_user_admin(AUTH_CFG, username, True)
    elif action == "remove_admin":
        admins = [u for u in list_users(AUTH_CFG) if u.get("is_admin")]
        if username == current_user:
            ok, msg = False, "Você não pode remover seu próprio papel de admin."
        elif len(admins) <= 1:
            ok, msg = False, "Não é possível remover o último admin."
        else:
            ok, msg = set_user_admin(AUTH_CFG, username, False)
    elif action == "make_owner":
        if not is_owner:
            ok, msg = False, "Só o owner pode promover outro owner."
        else:
            ok, msg = set_user_owner(AUTH_CFG, username, True)
    elif action == "remove_owner":
        owners = [u for u in list_users(AUTH_CFG) if u.get("is_owner")]
        if not is_owner:
            ok, msg = False, "Só o owner pode remover outro owner."
        elif username == current_user:
            ok, msg = False, "Você não pode remover o próprio papel de owner."
        elif len(owners) <= 1:
            ok, msg = False, "Não é possível remover o último owner."
        else:
            ok, msg = set_user_owner(AUTH_CFG, username, False)
    elif action == "delete":
        if username == current_user:
            ok, msg = False, "Você não pode remover a própria conta em uso."
        else:
            ok, msg = delete_user(AUTH_CFG, username)
    else:
        ok, msg = False, "Ação inválida."
    session["admin_flash_success" if ok else "admin_flash_error"] = msg
    return redirect(url_for("admin_users"))

REFRESH_PARAM_FIELDS = [
    "label_profit_thr",
    "exit_ema_span_min",
    "exit_ema_init_offset_pct",
]
TRAIN_PARAM_FIELDS = [
    "entry_ratio_neg_per_pos",
    "calib_tail_blend",
    "calib_tail_boost",
    "top_metric_qs",
    "top_metric_min_count",
]
TRIAL_PARAM_FIELDS = [
    "tau_entry",
    "total_exposure",
    "max_trade_exposure",
    "min_trade_exposure",
    "exposure_multiplier",
]
PARAM_LABELS = {
    "label_profit_thr": "Label Profit Thr",
    "exit_ema_span_min": "Exit EMA Span Min",
    "exit_ema_init_offset_pct": "Exit EMA Init Offset %",
    "entry_ratio_neg_per_pos": "Entry Ratio Neg/Pos",
    "calib_tail_blend": "Calib Tail Blend",
    "calib_tail_boost": "Calib Tail Boost",
    "top_metric_qs": "Top Metric QS",
    "top_metric_min_count": "Top Metric Min Count",
    "tau_entry": "Tau Entry",
    "max_positions": "Max Positions",
    "total_exposure": "Total Exposure",
    "max_trade_exposure": "Max Trade Exposure",
    "min_trade_exposure": "Min Trade Exposure",
    "exposure_multiplier": "Exposure Multiplier",
}

def _to_float(v, default=0.0):
    try:
        out = float(v)
        if not np.isfinite(out):
            return default
        return out
    except:
        return default


def _clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    if isinstance(value, (int, bool)):
        return value
    text = str(value).strip()
    return text or None


def _equity_metrics_from_bt_dir(bt_out_dir: str) -> dict[str, float]:
    csv_path = Path(str(bt_out_dir or "").strip()) / "portfolio_equity.csv"
    cache_key = str(csv_path.resolve()) if csv_path.exists() else str(csv_path)
    cached = EQUITY_METRIC_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    metrics = {
        "month_pos_frac": 0.0,
        "month_worst": 0.0,
        "max_neg_month_streak": 0.0,
        "underwater_frac": 1.0,
        "worst_rolling_90d": 0.0,
    }
    try:
        if csv_path.exists():
            eq = pd.read_csv(csv_path, index_col=0)
            eq.index = pd.to_datetime(eq.index)
            if "equity" in eq.columns and not eq.empty:
                eq_s = pd.to_numeric(eq["equity"], errors="coerce").dropna()
                if not eq_s.empty:
                    month_ret = eq_s.resample("ME").last().pct_change().dropna()
                    peaks = eq_s.cummax()
                    underwater = (eq_s < peaks).astype(float)
                    max_streak = 0
                    cur_streak = 0
                    for value in month_ret.to_numpy(dtype=float):
                        if value < 0.0:
                            cur_streak += 1
                            max_streak = max(max_streak, cur_streak)
                        else:
                            cur_streak = 0
                    rolling_90d = (eq_s / eq_s.shift(90)) - 1.0
                    metrics = {
                        "month_pos_frac": float((month_ret > 0.0).mean()) if len(month_ret) else 0.0,
                        "month_worst": float(month_ret.min()) if len(month_ret) else 0.0,
                        "max_neg_month_streak": float(max_streak),
                        "underwater_frac": float(underwater.mean()) if len(underwater) else 1.0,
                        "worst_rolling_90d": float(rolling_90d.min()) if rolling_90d.notna().any() else 0.0,
                    }
    except Exception:
        pass
    EQUITY_METRIC_CACHE[cache_key] = dict(metrics)
    return metrics


def _collect_params(row_dict: dict, fields: list[str]) -> list[dict]:
    params = []
    for field in fields:
        value = _clean_value(row_dict.get(field))
        if value is None:
            continue
        params.append({
            "key": field,
            "label": PARAM_LABELS.get(field, field),
            "value": value,
        })
    return params


def _build_param_layers(df_all: pd.DataFrame, backtest_row: dict) -> dict:
    label_id = str(backtest_row.get("label_id") or "").strip()
    model_id = str(backtest_row.get("model_id") or "").strip()
    backtest_id = str(backtest_row.get("backtest_id") or "").strip()

    refresh_row = {}
    train_row = {}
    if label_id:
        refresh_match = df_all[
            (df_all.get("stage").astype(str) == "refresh")
            & (df_all.get("label_id").astype(str) == label_id)
            & (df_all.get("status").astype(str) == "ok")
        ]
        if not refresh_match.empty:
            refresh_row = refresh_match.iloc[-1].to_dict()
    if label_id and model_id:
        train_match = df_all[
            (df_all.get("stage").astype(str) == "train")
            & (df_all.get("label_id").astype(str) == label_id)
            & (df_all.get("model_id").astype(str) == model_id)
            & (df_all.get("status").astype(str) == "ok")
        ]
        if not train_match.empty:
            train_row = train_match.iloc[-1].to_dict()

    return {
        "refresh": {
            "title": "Refresh",
            "meta": {"label_id": label_id},
            "items": _collect_params(refresh_row, REFRESH_PARAM_FIELDS),
        },
        "train": {
            "title": "Train",
            "meta": {
                "label_id": label_id,
                "model_id": model_id,
                "train_id": _clean_value(train_row.get("train_id")) or f"{label_id}/{model_id}",
                "train_run_dir": _clean_value(train_row.get("train_run_dir")),
            },
            "items": _collect_params(train_row, TRAIN_PARAM_FIELDS),
        },
        "trial": {
            "title": "Trial",
            "meta": {
                "backtest_id": backtest_id,
                "train_id": _clean_value(backtest_row.get("train_id")),
                "bt_out_dir": _clean_value(backtest_row.get("bt_out_dir")),
                "phase": _clean_value(backtest_row.get("phase")),
                "generation": _clean_value(backtest_row.get("generation")),
                "source_cluster_id": _clean_value(backtest_row.get("source_cluster_id")),
                "source_backtest_id": _clean_value(backtest_row.get("source_backtest_id")),
            },
            "items": _collect_params(backtest_row, TRIAL_PARAM_FIELDS),
        },
    }

def calc_score(row):
    ret = max(0.0, _to_float(row.get("ret_pct", 0.0)))
    dd = _to_float(row.get("max_dd", 0.0))
    clusters = max(0.0, _to_float(row.get("clusters", 0.0)))
    worst_90d = _to_float(row.get("worst_rolling_90d", 0.0))
    worst_trade = _to_float(row.get("worst_trade", 0.0))
    if ret <= 0:
        return 0.0
    quality = np.log1p(ret * 40.0) / np.power(dd + 0.04, 1.15)
    activity = np.clip(clusters / (clusters + 20.0), 0.15, 0.95)
    tail_90d = 1.0 / (1.0 + 12.0 * max(0.0, -worst_90d))
    tail_trade = 1.0 / (1.0 + 18.0 * max(0.0, -worst_trade))
    return float(quality * activity * tail_90d * tail_trade)

def get_step_data(step):
    step_dir = _fair_root() / f"step_{step}d"
    csv_path = step_dir / "explore_runs.csv"
    if not csv_path.exists():
        return {"status": "Pending", "rows": [], "summary": {}}
    
    try:
        # Safe read for Windows (avoid lock when explore.py is writing)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = Path(tmp.name)
            shutil.copy2(csv_path, tmp_path)
        
        df = pd.read_csv(tmp_path)
        try: os.remove(tmp_path)
        except: pass
        df_all = df.copy()
        df = df[(df["stage"] == "backtest") & (df["status"] == "ok")]
        if df.empty:
            return {"status": "Running...", "rows": [], "summary": {}}
        if "phase" not in df.columns:
            df["phase"] = "broad"
        if "generation" not in df.columns:
            df["generation"] = 1
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(1).astype(int)
        if "max_neg_month_streak" not in df.columns:
            df["max_neg_month_streak"] = np.nan
        if "underwater_frac" not in df.columns:
            df["underwater_frac"] = np.nan
        if "worst_rolling_90d" not in df.columns:
            df["worst_rolling_90d"] = np.nan
        if "worst_trade" not in df.columns:
            df["worst_trade"] = np.nan
        missing_mask = (
            df["max_neg_month_streak"].isna()
            | df["underwater_frac"].isna()
            | df["worst_rolling_90d"].isna()
        )
        if missing_mask.any():
            enriched = df.loc[missing_mask, "bt_out_dir"].apply(_equity_metrics_from_bt_dir)
            for col in ("month_pos_frac", "month_worst", "max_neg_month_streak", "underwater_frac", "worst_rolling_90d"):
                if col not in df.columns:
                    df[col] = np.nan
                df.loc[missing_mask, col] = enriched.apply(lambda x: x.get(col, np.nan))
        df["score"] = df.apply(calc_score, axis=1)
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df = df.sort_values("score", ascending=False)
        
        total_trials = len(df)
        rows = df.head(100).to_dict("records")
        for r in rows:
            # Fix paths for artifact serving
            if "equity_html" in r and r["equity_html"]:
                p = Path(r["equity_html"])
                try: r["rel_html"] = p.relative_to(REPO_ROOT).as_posix()
                except: r["rel_html"] = None
            r["params"] = _build_param_layers(df_all, r)

        best = df.iloc[0]
        broad_ok = int((df["phase"].astype(str) == "broad").sum())
        refine_ok = int((df["phase"].astype(str) == "refine").sum())
        latest_generation = int(df["generation"].max()) if not df.empty else 0
        summary = {
            "best_score": round(best["score"], 2),
            "best_ret": round(_to_float(best.get("ret_pct"), 0.0) * 100.0, 1),
            "best_dd": round(_to_float(best.get("max_dd"), 0.0) * 100.0, 1),
            "best_wr": round(_to_float(best.get("win_rate"), 0.0) * 100.0, 1),
            "total_ok": len(df),
            "total_count": total_trials,
            "broad_ok": broad_ok,
            "refine_ok": refine_ok,
            "latest_generation": latest_generation,
        }
        
        return {
            "status": "Finished" if (step_dir / ".finished").exists() else "Running...",
            "rows": rows,
            "summary": summary
        }
    except Exception as e:
        print(f"[fair_dashboard][error] step={step} failed: {type(e).__name__}: {e}", flush=True)
        return {"status": "Error", "rows": [], "summary": {}}

@app.route("/api/data")
def api_data():
    steps = [1440, 1260, 1080, 900, 720, 540, 360, 180]
    return jsonify({step: get_step_data(step) for step in steps})

@app.route("/")
def index():
    contact_email = os.getenv("ASTRA_CONTACT_EMAIL", "astraquantlab@gmail.com").strip() or "astraquantlab@gmail.com"
    try:
        oos_snapshot = build_site_snapshot(_fair_root())
    except Exception:
        oos_snapshot = None
    return render_template(
        "landing.html",
        title="Astra Tradebot",
        runtime_badge="Fair Explore",
        brand_logo=url_for("static", filename="branding/astra-tradebot-symbol-light.svg"),
        dashboard_url=url_for("dashboard"),
        login_url=url_for("login", next=url_for("dashboard")),
        register_url=url_for("register", next=url_for("dashboard")),
        contact_email=contact_email,
        contact_focus="Systematic strategy, controlled deployment, and private partnerships",
        contact_href=f"mailto:{contact_email}?subject=Astra%20Tradebot%20Conversation",
        session_has_access=session_has_access(AUTH_CFG, session.get(AUTH_CFG.session_key)),
        hero_eyebrow="Astra • private capital • systematic execution",
        hero_brand_line="Production profile • controlled crypto strategy",
        hero_brand_copy=(
            "Built for capital that values discipline, controlled drawdown, and a process that can be defended under scrutiny."
        ),
        hero_prefix="Systematic crypto returns with",
        hero_accent="institutional-style risk control",
        hero_suffix=".",
        hero_copy=(
            "Astra presents a systematic crypto sleeve shaped through walk-forward validation, controlled deployment, "
            "and an explicit bias toward steadier out-of-sample behavior."
        ),
        performance_title=(oos_snapshot.performance_title if oos_snapshot else "Walk-forward OOS performance"),
        performance_metrics=(
            oos_snapshot.performance_metrics if oos_snapshot else [
                {"label": "Net Return", "value": "—", "sub": "Unavailable."},
                {"label": "Max Drawdown", "value": "—", "sub": "Unavailable."},
                {"label": "Profit Factor", "value": "—", "sub": "Unavailable."},
                {"label": "Hit Rate", "value": "—", "sub": "Unavailable."},
            ]
        ),
        performance_visual_html=(oos_snapshot.visual_html if oos_snapshot else None),
        performance_visual_title=(oos_snapshot.visual_title if oos_snapshot else "Stitched OOS equity"),
        performance_visual_caption=(oos_snapshot.visual_caption if oos_snapshot else ""),
        performance_note=(
            oos_snapshot.performance_note if oos_snapshot else
            "The strategy combines controlled upside participation, contained drawdown, and a production process designed "
            "to remain resilient across adverse digital-asset regimes."
        ),
        stats=(
            oos_snapshot.stats if oos_snapshot else [
                {"label": "Research steps", "value": "7"},
                {"label": "Feature preset", "value": "Core80"},
                {"label": "Refreshes / step", "value": "56"},
                {"label": "Backtests / retrain", "value": "21"},
            ]
        ),
        summary_copy=(
            oos_snapshot.summary_copy if oos_snapshot else
            "The current production candidate is designed around one objective: preserve upside while keeping the path "
            "to that return commercially defensible."
        ),
        chips=[
            "Walk-forward validated",
            "Out-of-sample tested",
            "Controlled drawdown",
            "Private-capital profile",
            "Execution discipline",
        ],
        system_title="Astra is built around robustness first, not narrative-first performance.",
        system_copy=(
            "The public message is simple: returns matter, but only when they come from a repeatable process with "
            "visible risk control, disciplined model review, and serious operating standards."
        ),
        info_cards=[
            {
                "eyebrow": "Return profile",
                "title": "Upside with control",
                "copy": "The strategy aims for meaningful participation in crypto upside without relying on violent drawdowns or unstable behavior.",
            },
            {
                "eyebrow": "Risk discipline",
                "title": "Commercially defensible",
                "copy": "What matters is not just headline return, but whether the path, drawdown, and operating profile are investable for serious capital.",
            },
            {
                "eyebrow": "Partnership",
                "title": "Selective collaboration",
                "copy": "Positioned for private conversations where systematic performance, process quality, and operating discipline all matter.",
            },
        ],
        pipeline=[
            {
                "step": "01",
                "title": "Every candidate is tested out of sample",
                "copy": "The strategy is reviewed across sequential historical windows rather than sold on a single optimized period.",
            },
            {
                "step": "02",
                "title": "Risk is treated as part of the product",
                "copy": "Drawdown control, portfolio construction, and deployment behavior are treated as core design constraints, not afterthoughts.",
            },
            {
                "step": "03",
                "title": "The process is engineered to be auditable",
                "copy": "Selection, review, and monitoring sit inside one controlled environment so the research story stays tied to the operating reality.",
            },
        ],
    )
    shared_css = _shared_dashboard_css_text()
    html = """
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
                --surface: rgba(16, 19, 26, 0.66);
                --surface-light: rgba(255, 255, 255, 0.08);
                --accent: #7c5cff;
                --accent-2: #00ffb3;
            }
            body {
                color: var(--text-main);
                font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            .site-shell { max-width: 1260px; margin: 0 auto; }
            .hero-wrap { padding: 56px 0 24px; }
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
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.22);
            }
            .eyebrow {
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.72rem;
                font-weight: 700;
            }
            .hero-title {
                font-size: clamp(2rem, 4.8vw, 3.5rem);
                line-height: 1.0;
                letter-spacing: -0.04em;
                font-weight: 800;
                margin: 0;
            }
            .hero-title .accent { color: var(--accent-2); }
            .hero-copy {
                color: var(--text-muted);
                font-size: 0.98rem;
                line-height: 1.65;
                max-width: 760px;
            }
            .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
            .btn-accent {
                background: linear-gradient(135deg, rgba(124, 92, 255, 0.18), rgba(76, 58, 140, 0.34));
                color: #fff;
                border: 1px solid rgba(124, 92, 255, 0.34);
                backdrop-filter: blur(10px);
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.08),
                    0 10px 24px rgba(0, 0, 0, 0.22);
            }
            .btn-accent:hover, .btn-accent:focus {
                color: #fff;
                background: linear-gradient(135deg, rgba(124, 92, 255, 0.24), rgba(88, 66, 164, 0.42));
                border-color: rgba(124, 92, 255, 0.46);
            }
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
                font-size: 1.3rem;
                font-weight: 800;
                letter-spacing: -0.03em;
            }
            .section-title {
                font-size: 1.35rem;
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
                font-size: 0.94rem;
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
                font-size: 0.74rem;
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
            @media (max-width: 767px) {
                .hero-wrap { padding: 38px 0 20px; }
                .hero-copy { font-size: 0.92rem; }
                .hero-actions { gap: 8px; }
                .hero-actions .btn {
                    width: auto;
                    max-width: 100%;
                    justify-content: center;
                    flex: 0 1 auto;
                    align-self: flex-start;
                    white-space: nowrap;
                }
            }
        </style>
    </head>
    <body class="astra-body">
        <header class="navbar navbar-expand-lg navbar-glass sticky-top">
            <div class="container-fluid px-3">
                <a class="navbar-brand d-flex align-items-center gap-2" href="{{ url_for('index') }}">
                    <span class="brand-lockup">
                        <img class="brand-symbol" src="{{ url_for('static', filename='branding/astra-tradebot-symbol-light.svg') }}" alt="Astra symbol" />
                        <span class="brand-copy">
                            <span class="brand-kicker">Astra Tradebot</span>
                            <span class="brand-title">Astra</span>
                        </span>
                    </span>
                    <span class="badge text-bg-secondary ms-1">Astra</span>
                </a>
                <div class="d-flex align-items-center gap-2">
                    {% if auth_enabled and session_has_access %}
                    <a href="{{ url_for('dashboard') }}" class="btn btn-sm btn-outline-secondary">
                        <i class="bi bi-grid-1x2"></i>
                        <span class="d-none d-sm-inline ms-1">Dashboard</span>
                    </a>
                    {% endif %}
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
                            <div class="eyebrow mb-3">Astra • Quantitative research • Systematic execution</div>
                            <h1 class="hero-title mb-4">Systematic crypto research and <span class="accent">proprietary trading infrastructure</span>.</h1>
                            <p class="hero-copy mb-4">
                                Astra develops Tradebot, a proprietary stack built around walk-forward research,
                                controlled deployment, and production-grade engineering for digital asset markets.
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
                                <a href="#system-overview" class="btn btn-outline-secondary btn-lg">
                                    <i class="bi bi-diagram-3 me-2"></i>System Overview
                                </a>
                            </div>
                        </section>

                        <aside class="summary-card p-4 p-lg-4">
                            <div class="eyebrow mb-3">Current stack</div>
                            <div class="stat-grid mb-3">
                                <div>
                                    <div class="stat-label">Research steps</div>
                                    <div class="stat-value">7</div>
                                </div>
                                <div>
                                    <div class="stat-label">Edge params</div>
                                    <div class="stat-value">4</div>
                                </div>
                                <div>
                                    <div class="stat-label">Refreshes / step</div>
                                    <div class="stat-value">49</div>
                                </div>
                            </div>
                            <div class="divider-line"></div>
                            <p class="section-copy mb-3">
                                The current production candidate is being refined through independent step exploration,
                                out-of-sample validation, and dashboard-based operational review.
                            </p>
                            <div class="chip-row">
                                <span class="chip">Walk-forward design</span>
                                <span class="chip">OOS validation</span>
                                <span class="chip">Risk-aware deployment</span>
                                <span class="chip">Quant + engineering</span>
                            </div>
                        </aside>
                    </div>
                </section>

                <section id="system-overview" class="section-card p-4 p-lg-5 mb-4">
                    <div class="eyebrow mb-2">System overview</div>
                    <h2 class="section-title mb-3">Tradebot is Astra’s internal research and execution stack.</h2>
                    <p class="section-copy mb-0">
                        It combines market data ingestion, feature engineering, labeling workflows, walk-forward training,
                        portfolio backtesting, out-of-sample evaluation, and live-ready monitoring in a single controlled environment.
                    </p>
                </section>

                <section class="offer-grid pb-4">
                    <article class="offer-card p-4">
                        <div class="eyebrow mb-2">Research</div>
                        <h3>Methodological discipline</h3>
                        <p>Walk-forward model design, parameter exploration, and out-of-sample validation aimed at reducing narrative bias and backtest illusion.</p>
                    </article>
                    <article class="offer-card p-4">
                        <div class="eyebrow mb-2">Infrastructure</div>
                        <h3>Engineering-first stack</h3>
                        <p>Data pipeline, training flow, portfolio backtests, dashboards, auth, and operational tooling built for serious iteration and deployment.</p>
                    </article>
                    <article class="offer-card p-4">
                        <div class="eyebrow mb-2">Partnership</div>
                        <h3>Selective collaboration</h3>
                        <p>Structured for private conversations with family offices, small funds, and aligned partners seeking quant and systematic engineering capability.</p>
                    </article>
                </section>
            </div>
        </main>
    </body>
    </html>
    """
    return render_template_string(html, shared_css=shared_css, session_has_access=session_has_access(AUTH_CFG, session.get(AUTH_CFG.session_key)))


@app.route("/dashboard")
def dashboard():
    # Show milestones from oldest to newest (to match orchestrator start)
    steps = [1440, 1260, 1080, 900, 720, 540, 360, 180]
    all_data = {step: get_step_data(step) for step in steps}
    
    # Auto-select the first step that is 'Running' or the first one if all 'Pending'
    active_step = steps[0]
    for s in steps:
        if all_data[s]["status"] == "Running...":
            active_step = s
            break
        if all_data[s]["status"] == "Finished":
            # If we find a finished one, we might want to keep looking for a running one
            active_step = s
    
    # Calculate global overview
    finished_steps = [s for s, d in all_data.items() if d["status"] == "Finished"]
    avg_score = round(np.mean([all_data[s]["summary"]["best_score"] for s in finished_steps]), 2) if finished_steps else 0
    
    shared_css = _shared_dashboard_css_text()

    html = """
    <!DOCTYPE html>
    <html lang="pt-br" data-bs-theme="dark">
    <head>
        <title>Fair Explore Dashboard</title>
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
                --surface: rgba(16, 19, 26, 0.68);
                --surface-light: rgba(255, 255, 255, 0.08);
                --accent: #7c5cff;
                --accent-2: #00ffb3;
                --accent-glow: rgba(124, 92, 255, 0.18);
                --text-main: #f8fafc;
                --text-muted: rgba(255, 255, 255, 0.60);
                --green: #00ffb3;
                --orange: #ffc85c;
                --red: #ff5c7a;
            }
            body {
                color: var(--text-main);
                line-height: 1.5;
                overflow-y: scroll;
                font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            .page-shell { max-width: 1320px; margin: 0 auto; }
            .hero-card { margin-bottom: 1rem; }
            .hero-meta { color: var(--text-muted); font-size: 0.88rem; }
            .hero-pill {
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: rgba(255, 255, 255, 0.04);
                border-radius: 999px;
                padding: 0.35rem 0.75rem;
                color: var(--text-muted);
                font-size: 0.72rem;
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
            }
            .hero-pill strong { color: var(--text-main); font-weight: 600; }
            .brand-line { color: var(--text-muted); font-size: 0.76rem; letter-spacing: 0.05em; text-transform: uppercase; }
            .hero-title { font-size: clamp(1.55rem, 2.8vw, 2.15rem); font-weight: 700; margin: 0; }
            .hero-title .accent { color: var(--accent-2); }
            .status-pill {
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: rgba(255, 255, 255, 0.04);
                border-radius: 999px;
                padding: 0.45rem 0.8rem;
            }
            .navbar-brand {
                text-decoration: none;
                color: var(--text-main);
            }
            .navbar-brand:hover { color: var(--text-main); }
            
            .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
            .kpi-card { padding: 13px; border-radius: 13px; }
            .kpi-label { color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }
            .kpi-value { font-size: 1.24rem; font-weight: 700; color: var(--text-main); font-family: 'JetBrains Mono', monospace; }

            .tabs-shell { margin-bottom: 1rem; }
            .tabs { display: flex; gap: 10px; justify-content: center; margin-bottom: 0; padding: 14px 16px 10px; border-radius: 12px; overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; scrollbar-color: rgba(148, 163, 184, 0.55) rgba(255,255,255,0.05); }
            .tabs::-webkit-scrollbar { height: 10px; }
            .tabs::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 999px; }
            .tabs::-webkit-scrollbar-thumb { background: linear-gradient(90deg, rgba(148, 163, 184, 0.55), rgba(124, 92, 255, 0.45)); border-radius: 999px; border: 2px solid rgba(22, 28, 45, 0.95); }
            .tabs::-webkit-scrollbar-thumb:hover { background: linear-gradient(90deg, rgba(148, 163, 184, 0.7), rgba(124, 92, 255, 0.6)); }
            .tab { padding: 9px 14px; border-radius: 10px; cursor: pointer; color: var(--text-muted); font-weight: 600; font-size: 0.84rem; transition: all 0.2s; white-space: nowrap; display: flex; align-items: center; border: 1px solid transparent; }
            .tab:hover { background: rgba(255,255,255,0.05); color: var(--text-main); }
            .tab.active {
                background: linear-gradient(135deg, rgba(124, 92, 255, 0.18), rgba(76, 58, 140, 0.34));
                color: #f8fafc;
                border-color: rgba(124, 92, 255, 0.34);
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.08),
                    0 10px 24px rgba(0, 0, 0, 0.22);
            }
            .tab-status { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
            .tab-count { margin-left: 10px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--text-main); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }
            .tab.active .tab-count {
                background: rgba(124, 92, 255, 0.18);
                color: #fff;
                border: 1px solid rgba(124, 92, 255, 0.24);
            }

            .step-panel { display: none; }
            .step-panel.active { display: block; }

            .table-scroll { width: 100%; overflow-x: auto; overflow-y: hidden; -webkit-overflow-scrolling: touch; scrollbar-width: thin; scrollbar-color: rgba(148, 163, 184, 0.55) rgba(255,255,255,0.05); }
            .table-scroll::-webkit-scrollbar { height: 10px; }
            .table-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 999px; }
            .table-scroll::-webkit-scrollbar-thumb { background: linear-gradient(90deg, rgba(148, 163, 184, 0.55), rgba(124, 92, 255, 0.45)); border-radius: 999px; border: 2px solid rgba(22, 28, 45, 0.95); }
            table { width: 100%; border-collapse: separate; border-spacing: 0; background: transparent; border-radius: 12px; border: 1px solid var(--surface-light); margin-bottom: 50px; overflow: hidden; }
            th { background: rgba(255,255,255,0.05); padding: 11px 13px; text-align: left; color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; font-weight: 700; cursor: pointer; user-select: none; transition: color 0.2s; }
            th:hover { color: var(--accent); }
            th.sorted-asc::after { content: ' ↑'; color: var(--accent); }
            th.sorted-desc::after { content: ' ↓'; color: var(--accent); }
            
            td { padding: 11px 13px; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.84rem; background: rgba(16, 19, 26, 0.54); }
            tr.row-clickable { cursor: pointer; transition: background 0.1s; }
            tr.row-clickable:hover td { background: rgba(255,255,255,0.03); }
            tr.row-active td { background: var(--accent-glow) !important; border-left: 2px solid var(--accent); }
            
            .metric { font-family: 'JetBrains Mono', monospace; }
            .score-pill { background: var(--accent-glow); color: var(--accent); padding: 2px 8px; border-radius: 4px; font-weight: 700; }
            
            .iframe-row { display: none; background: transparent; }
            .detail-shell { padding: 18px; background: linear-gradient(180deg, rgba(7, 11, 19, 0.96), rgba(3, 6, 11, 0.98)); }
            .detail-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
            .detail-card { background: rgba(22, 28, 45, 0.96); border: 1px solid rgba(124, 92, 255, 0.15); border-radius: 12px; padding: 14px; min-width: 0; overflow: hidden; }
            .detail-title { color: var(--accent); font-size: 0.72rem; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.05em; }
            .detail-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
            .detail-chip { padding: 4px 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); color: var(--text-muted); font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; max-width: 100%; white-space: normal; overflow-wrap: anywhere; line-height: 1.35; }
            .param-list { display: grid; gap: 8px; }
            .param-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: start; border-bottom: 1px solid rgba(148, 163, 184, 0.08); padding-bottom: 6px; }
            .param-label { color: var(--text-muted); font-size: 0.74rem; min-width: 0; }
            .param-value { color: var(--text-main); font-size: 0.76rem; text-align: right; font-family: 'JetBrains Mono', monospace; min-width: 0; max-width: 100%; word-break: break-word; overflow-wrap: anywhere; }
            .param-empty { color: var(--text-muted); font-size: 0.76rem; font-style: italic; }
            .iframe-container { width: 100%; height: 620px; border: none; }
            
            .empty-state { padding: 72px; text-align: center; color: var(--text-muted); background: rgba(16, 19, 26, 0.58); border-radius: 12px; border: 1px solid var(--surface-light); }

            @media (max-width: 1100px) {
                .kpi-row { grid-template-columns: repeat(2, 1fr); }
                .detail-grid { grid-template-columns: 1fr; }
            }

            @media (max-width: 700px) {
                .kpi-row { grid-template-columns: 1fr; }
                .tabs { justify-content: flex-start; }
                .tabs { padding: 12px 12px 8px; }
                .tab { padding: 9px 14px; }
                .table-scroll { margin: 0 -6px 12px; padding: 0 6px 8px; }
                table { min-width: 720px; }
                .param-row { grid-template-columns: 1fr; }
                .param-value { text-align: left; }
                .hero-meta { font-size: 0.82rem; }
                .iframe-container { height: 460px; }
                .empty-state { padding: 36px 18px; }
            }
        </style>
    </head>
    <body class="astra-body">
        <header class="navbar navbar-expand-lg navbar-glass sticky-top">
            <div class="container-fluid px-3">
                <a class="navbar-brand d-flex align-items-center gap-2" href="{{ url_for('index') }}">
                    <span class="brand-lockup">
                        <img class="brand-symbol" src="{{ url_for('static', filename='branding/astra-tradebot-symbol-light.svg') }}" alt="Astra symbol" />
                        <span class="brand-copy">
                            <span class="brand-kicker">Astra Tradebot</span>
                            <span class="brand-title">Fair Explore</span>
                        </span>
                    </span>
                    <span class="badge text-bg-secondary ms-1">Fair Explore</span>
                </a>
                <div class="d-flex align-items-center gap-3">
                    <div class="d-none d-md-flex align-items-center gap-2 status-pill">
                        <span class="status-dot status-online"></span>
                        <small class="text-secondary">
                            <span id="status-dot" class="text-secondary-emphasis fw-semibold">dados ao vivo</span>
                            <span class="mx-1">•</span>
                            <span id="countdown">próxima atualização...</span>
                        </small>
                    </div>
                    {% if auth_enabled and auth_is_admin %}
                    <a href="{{ url_for('admin_users') }}" class="btn btn-sm btn-outline-secondary">
                        <i class="bi bi-people"></i>
                        <span class="d-none d-sm-inline ms-1">Usuários</span>
                    </a>
                    {% endif %}
                    {% if auth_enabled and auth_is_admin %}
                    <a href="{{ url_for('assistant') }}" class="btn btn-sm btn-outline-secondary">
                        <i class="bi bi-robot"></i>
                        <span class="d-none d-sm-inline ms-1">Assistant</span>
                    </a>
                    {% endif %}
                    {% if auth_enabled %}
                    <form method="post" action="{{ url_for('logout') }}" class="m-0">
                        <button class="btn btn-sm btn-outline-secondary" type="submit">
                            <i class="bi bi-box-arrow-right"></i>
                            <span class="d-none d-sm-inline ms-1">Sair</span>
                        </button>
                    </form>
                    {% endif %}
                </div>
            </div>
        </header>

        <main class="container-fluid px-3 py-3">
            <div class="page-shell">
                <section class="card card-glass hero-card">
                    <div class="card-body p-4">
                        <div class="d-flex flex-column flex-lg-row justify-content-between align-items-start gap-3">
                            <div>
                                <div class="brand-line mb-2">Milestone search • walk-forward fair protocol</div>
                                <h1 class="hero-title">Fair Explore <span class="accent">Command Center</span></h1>
                                <div class="hero-meta mt-2">Acompanhamento de milestones independentes, trials por step e artefatos de robustez em tempo real.</div>
                            </div>
                            <div class="d-flex flex-wrap gap-2 justify-content-lg-end">
                                <span class="hero-pill"><i class="bi bi-diagram-3"></i><strong>4</strong> parâmetros de edge</span>
                                <span class="hero-pill"><i class="bi bi-shield-check"></i>treino e risco fixos</span>
                                <span class="hero-pill"><i class="bi bi-arrow-repeat"></i>49 refreshes por step</span>
                            </div>
                        </div>
                    </div>
                </section>

                <div class="kpi-row">
                    <div class="kpi-card card card-glass">
                        <div class="card-body p-3">
                            <div class="kpi-label">Milestones finalizados</div>
                            <div id="kpi-finished" class="kpi-value">--</div>
                        </div>
                    </div>
                    <div class="kpi-card card card-glass">
                        <div class="card-body p-3">
                            <div class="kpi-label">Score histórico médio</div>
                            <div id="kpi-avg-score" class="kpi-value" style="color: var(--accent-2)">--</div>
                        </div>
                    </div>
                    <div class="kpi-card card card-glass">
                        <div class="card-body p-3">
                            <div class="kpi-label">Trials totais</div>
                            <div id="kpi-trials" class="kpi-value">--</div>
                        </div>
                    </div>
                    <div class="kpi-card card card-glass">
                        <div class="card-body p-3">
                            <div class="kpi-label">Modo do sistema</div>
                            <div class="kpi-value" style="color: var(--green)">AUTÔNOMO</div>
                        </div>
                    </div>
                </div>

                <section class="card card-glass tabs-shell">
                    <div class="tabs" id="step-tabs"></div>
                </section>
                <div id="main-content"></div>
            </div>
        </main>

        <footer class="container-fluid px-3 pb-3">
            <div class="page-shell text-center text-secondary small">
                Astra Tradebot • fair explore dashboard • identidade compartilhada com o bot live
            </div>
        </footer>

        <script
            src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
            integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
            crossorigin="anonymous"
        ></script>

        <script>
            const artifactBase = "{{ url_for('artifact', relpath='__REL__') }}";
            let state = {
                data: {},
                activeStep: localStorage.getItem('fair_activeStep') || '1440',
                sortKey: localStorage.getItem('fair_sortKey') || 'score',
                sortDir: localStorage.getItem('fair_sortDir') || 'desc',
                expandedPlotId: localStorage.getItem('fair_expandedPlotId') || null,
                lastUpdate: 0
            };

            const steps = [1440, 1260, 1080, 900, 720, 540, 360, 180];

            async function fetchData() {
                try {
                    const res = await fetch('{{ url_for("api_data") }}');
                    state.data = await res.json();
                    state.lastUpdate = Date.now();
                    document.getElementById('status-dot').textContent = 'dados ao vivo';
                    render();
                } catch(e) {
                    console.error("Dashboard Fetch Error:", e);
                    document.getElementById('status-dot').style.color = 'var(--red)';
                    document.getElementById('status-dot').textContent = 'erro offline';
                }
            }

            function asNumber(value, fallback = 0) {
                const n = Number(value);
                return Number.isFinite(n) ? n : fallback;
            }

            function setStep(step) {
                state.activeStep = String(step);
                localStorage.setItem('fair_activeStep', state.activeStep);
                render();
            }

            function setSort(key) {
                if (state.sortKey === key) {
                    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
                } else {
                    state.sortKey = key;
                    state.sortDir = 'desc';
                }
                localStorage.setItem('fair_sortKey', state.sortKey);
                localStorage.setItem('fair_sortDir', state.sortDir);
                render();
            }

            function compactTrialLabel(row) {
                const trainId = String(row.train_id || '');
                const backtestId = String(row.backtest_id || '');
                const labelMatch = trainId.match(/label_(\\d+)/i);
                const modelMatch = trainId.match(/model_(\\d+)/i);
                const btMatch = backtestId.match(/bt_(\\d+)/i);
                const labelNum = labelMatch ? String(Number(labelMatch[1])).padStart(2, '0') : '--';
                const modelNum = modelMatch ? String(Number(modelMatch[1])).padStart(2, '0') : '--';
                const btNum = btMatch ? String(Number(btMatch[1])).padStart(2, '0') : '--';
                return `${labelNum}/${modelNum}/${btNum}`;
            }

            function togglePlot(id) {
                state.expandedPlotId = (state.expandedPlotId === id) ? null : id;
                localStorage.setItem('fair_expandedPlotId', state.expandedPlotId);
                render();
            }

            function render() {
                updateKPIs();
                updateTabs();
                updatePanel();
            }

            function updateKPIs() {
                let finished = 0, trials = 0, scores = [];
                Object.keys(state.data).forEach(s => {
                    const d = state.data[s] || { rows: [], summary: {} };
                    if (d.status === 'Finished') {
                        finished++;
                        if (d.summary.best_score) scores.push(d.summary.best_score);
                    }
                    trials += (d.summary.total_count || d.rows.length);
                });
                document.getElementById('kpi-finished').textContent = `${finished}/8`;
                document.getElementById('kpi-trials').textContent = trials;
                const avg = scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2) : '0';
                document.getElementById('kpi-avg-score').textContent = avg;
            }

            function updateTabs() {
                const container = document.getElementById('step-tabs');
                container.innerHTML = steps.map(s => {
                    const d = state.data[s] || { status: 'Pending', summary: {} };
                    const color = d.status === 'Finished' ? 'var(--green)' : (d.status === 'Running...' ? 'var(--orange)' : 'var(--text-muted)');
                    const active = String(s) === state.activeStep ? 'active' : '';
                    const total = Number(d.summary?.total_count || d.summary?.total_ok || 0);
                    return `<div class="tab ${active}" onclick="setStep('${s}')">
                        <span class="tab-status" style="background: ${color}"></span>
                        ${s} days
                        <span class="tab-count">${total}</span>
                    </div>`;
                }).join('');
            }

            function formatParamValue(value) {
                if (value === null || value === undefined || value === '') return '—';
                if (typeof value === 'boolean') return value ? 'true' : 'false';
                if (typeof value === 'number') {
                    if (Number.isInteger(value)) return String(value);
                    return value.toFixed(6).replace(/0+$/, '').replace(/\\.$/, '');
                }
                return String(value);
            }

            function renderMetaChips(meta) {
                const entries = Object.entries(meta || {}).filter(([, value]) => value !== null && value !== undefined && value !== '');
                if (!entries.length) return '';
                return `<div class="detail-meta">${entries.map(([key, value]) => `<span class="detail-chip">${key}: ${formatParamValue(value)}</span>`).join('')}</div>`;
            }

            function renderParamCard(layer) {
                const items = Array.isArray(layer?.items) ? layer.items : [];
                return `
                    <div class="detail-card">
                        <div class="detail-title">${layer?.title || 'Layer'}</div>
                        ${renderMetaChips(layer?.meta || {})}
                        ${items.length ? `<div class="param-list">${items.map(item => `
                            <div class="param-row">
                                <div class="param-label">${item.label || item.key}</div>
                                <div class="param-value">${formatParamValue(item.value)}</div>
                            </div>
                        `).join('')}</div>` : `<div class="param-empty">No parameter payload found for this layer.</div>`}
                    </div>
                `;
            }

            function updatePanel() {
                const container = document.getElementById('main-content');
                const step = state.activeStep;
                const d = state.data[step] || { status: 'Pending', rows: [] };
                
                if (!d.rows || d.rows.length === 0) {
                    const icon = d.status === 'Running...' ? '⚙️' : '⌛';
                    const msg = d.status === 'Running...' ? 
                        `Exploration for <strong>${step} days</strong> is active.<br><small style="color: var(--text-muted)">Analyzing labels and training models. Results will appear here shortly.</small>` : 
                        `Exploration for <strong>${step} days</strong> has not started yet.`;
                    container.innerHTML = `<div class="empty-state"><div style="font-size: 2rem;">${icon}</div><div>${msg}</div></div>`;
                    return;
                }

                // Sort rows
                let rows = [...d.rows];
                rows.sort((a, b) => {
                    let valA = a[state.sortKey], valB = b[state.sortKey];
                    const numA = Number(valA);
                    const numB = Number(valB);
                    const bothNumeric = Number.isFinite(numA) && Number.isFinite(numB);
                    if (bothNumeric) {
                        return state.sortDir === 'desc' ? (numB - numA) : (numA - numB);
                    }
                    valA = String(valA ?? '');
                    valB = String(valB ?? '');
                    return state.sortDir === 'desc'
                        ? valB.localeCompare(valA, undefined, { numeric: true, sensitivity: 'base' })
                        : valA.localeCompare(valB, undefined, { numeric: true, sensitivity: 'base' });
                });

                // If table already exists and we have an expanded plot, try to update surgically
                const existingTable = document.getElementById('main-table');
                if (
                    existingTable &&
                    existingTable.getAttribute('data-step') === String(step) &&
                    existingTable.getAttribute('data-expanded-id') === String(state.expandedPlotId) &&
                    existingTable.getAttribute('data-sort-key') === String(state.sortKey) &&
                    existingTable.getAttribute('data-sort-dir') === String(state.sortDir)
                ) {
                    rows.forEach(r => {
                        const id = `${step}-${r.backtest_id}-${r.train_id}`;
                        const scoreEl = document.getElementById(`score-${id}`);
                        if (scoreEl) {
                            scoreEl.textContent = asNumber(r.score).toFixed(4);
                            document.getElementById(`ret-${id}`).textContent = `+${(asNumber(r.ret_pct) * 100).toFixed(1)}%`;
                            document.getElementById(`dd-${id}`).textContent = `${(asNumber(r.max_dd) * 100).toFixed(2)}%`;
                            document.getElementById(`pf-${id}`).textContent = asNumber(r.profit_factor).toFixed(2);
                            document.getElementById(`win-${id}`).textContent = `${(asNumber(r.win_rate) * 100).toFixed(1)}%`;
                            document.getElementById(`trades-${id}`).textContent = r.trades;
                        }
                    });
                    // Skip full re-render to preserve iframe state (zoom)
                    return;
                }

                const table = `
                    <div class="table-scroll">
                    <table
                        id="main-table"
                        data-step="${step}"
                        data-expanded-id="${state.expandedPlotId}"
                        data-sort-key="${state.sortKey}"
                        data-sort-dir="${state.sortDir}"
                    >
                        <thead>
                            <tr>
                                <th onclick="setSort('backtest_id')">Trial</th>
                                <th style="text-align: right" onclick="setSort('score')" class="${state.sortKey==='score'?'sorted-'+state.sortDir:''}">Score</th>
                                <th style="text-align: right" onclick="setSort('ret_pct')" class="${state.sortKey==='ret_pct'?'sorted-'+state.sortDir:''}">Return %</th>
                                <th style="text-align: right" onclick="setSort('max_dd')" class="${state.sortKey==='max_dd'?'sorted-'+state.sortDir:''}">Max DD</th>
                                <th style="text-align: right" onclick="setSort('profit_factor')" class="${state.sortKey==='profit_factor'?'sorted-'+state.sortDir:''}">Profit Factor</th>
                                <th style="text-align: right" onclick="setSort('win_rate')" class="${state.sortKey==='win_rate'?'sorted-'+state.sortDir:''}">Win Rate</th>
                                <th style="text-align: right" onclick="setSort('trades')">Trades</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${rows.map((r, i) => {
                                const id = `${step}-${r.backtest_id}-${r.train_id}`;
                                const isExpanded = state.expandedPlotId === id;
                                const compactTrial = compactTrialLabel(r);
                                const fullTrial = `${r.train_id}/${r.backtest_id}`;
                                return `
                                    <tr class="row-clickable ${isExpanded?'row-active':''}" onclick="togglePlot('${id}')" id="row-${id}">
                                        <td class="metric text-muted" title="${fullTrial}">${compactTrial}</td>
                                        <td style="text-align: right"><span class="score-pill" id="score-${id}">${asNumber(r.score).toFixed(4)}</span></td>
                                        <td style="text-align: right; color: var(--green)" class="metric" id="ret-${id}">+${(asNumber(r.ret_pct) * 100).toFixed(1)}%</td>
                                        <td style="text-align: right; color: var(--red)" class="metric" id="dd-${id}">${(asNumber(r.max_dd) * 100).toFixed(2)}%</td>
                                        <td style="text-align: right" class="metric" id="pf-${id}">${asNumber(r.profit_factor).toFixed(2)}</td>
                                        <td style="text-align: right" class="metric" id="win-${id}">${(asNumber(r.win_rate) * 100).toFixed(1)}%</td>
                                        <td style="text-align: right" class="metric text-muted" id="trades-${id}">${r.trades}</td>
                                    </tr>
                                    <tr class="iframe-row" id="iframe-row-${id}" style="display: ${isExpanded?'table-row':'none'}">
                                        <td colspan="7" style="padding: 0;">
                                            ${isExpanded ? `
                                                <div class="detail-shell">
                                                    <div class="detail-grid">
                                                        ${renderParamCard(r.params?.refresh)}
                                                        ${renderParamCard(r.params?.train)}
                                                        ${renderParamCard(r.params?.trial)}
                                                    </div>
                                                    ${r.rel_html ? `<iframe class="iframe-container" src="${artifactBase.replace('__REL__', r.rel_html)}"></iframe>` : '<div class="empty-state" style="padding: 40px;">No plot artifact found for this trial.</div>'}
                                                </div>
                                            ` : ''}
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                    </div>
                `;
                container.innerHTML = table;
            }

            // Sync countdown
            setInterval(() => {
                const diff = 10 - Math.floor((Date.now() - state.lastUpdate) / 1000);
                document.getElementById('countdown').textContent = (diff > 0) ? `próxima atualização em ${diff}s` : 'atualizando...';
                if (diff <= 0) fetchData();
            }, 1000);

            // Initial load
            fetchData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html, shared_css=shared_css)


@app.get("/assistant")
def assistant():
    _require_assistant_admin()
    return render_template("fair_assistant.html", shared_css=_shared_dashboard_css_text())


@app.get("/api/assistant/capabilities")
def assistant_capabilities_api():
    _require_assistant_admin()
    return jsonify(REMOTE_CONTROL.capabilities())


@app.get("/api/assistant/conversations")
def assistant_conversations_api():
    _require_assistant_admin()
    return jsonify({"conversations": REMOTE_CONTROL.list_conversations(limit=30)})


@app.get("/api/assistant/conversations/<conversation_id>")
def assistant_conversation_detail_api(conversation_id):
    _require_assistant_admin()
    payload = REMOTE_CONTROL.get_conversation(conversation_id)
    if not payload:
        return jsonify({"ok": False, "error": "Conversa não encontrada."}), 404
    return jsonify({"ok": True, **payload})


@app.route("/api/assistant/jobs", methods=["GET", "POST"])
def assistant_jobs_api():
    current_user = _require_assistant_admin()
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt", "") or "")
        model = str(payload.get("model", "") or "")
        reasoning_effort = str(payload.get("reasoning_effort", "") or "")
        access_mode = str(payload.get("access_mode", "") or "")
        conversation_id = str(payload.get("conversation_id", "") or "")
        try:
            job = REMOTE_CONTROL.create_codex_job(
                prompt=prompt,
                created_by=current_user,
                model=model,
                reasoning_effort=reasoning_effort,
                access_mode=access_mode,
                conversation_id=conversation_id,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify({"ok": True, "job": job})
    return jsonify({"jobs": REMOTE_CONTROL.list_jobs(limit=30)})


@app.get("/api/assistant/jobs/<job_id>")
def assistant_job_detail_api(job_id):
    _require_assistant_admin()
    job = REMOTE_CONTROL.get_job(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job não encontrado."}), 404
    return jsonify({"ok": True, "job": job, "transcript": REMOTE_CONTROL.transcript(job_id)})


@app.get("/api/assistant/jobs/<job_id>/log")
def assistant_job_log_api(job_id):
    _require_assistant_admin()
    job = REMOTE_CONTROL.get_job(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job não encontrado."}), 404
    return jsonify({"ok": True, "log": REMOTE_CONTROL.filtered_log(job_id)})


@app.post("/api/assistant/jobs/<job_id>/cancel")
def assistant_job_cancel_api(job_id):
    _require_assistant_admin()
    try:
        job = REMOTE_CONTROL.cancel_job(job_id)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "job": job})

@app.route("/artifact/<path:relpath>")
def artifact(relpath):
    target = (REPO_ROOT / relpath).resolve()
    # Safety check
    if not str(target).startswith(str(REPO_ROOT)):
        abort(403)
    if not target.exists():
        abort(404)
    mime, _ = mimetypes.guess_type(str(target))
    return send_file(target, mimetype=mime or "application/octet-stream")

if __name__ == "__main__":
    # Start ngrok only in the main process (not the reloader child)
    ng_mgr = None
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        ng_cfg = NgrokConfig()
        ng_mgr = NgrokManager(ng_cfg)
        ng_mgr.start()
    
    try:
        app.run(host="127.0.0.1", port=5060, debug=True)
    finally:
        if ng_mgr:
            ng_mgr.stop()
