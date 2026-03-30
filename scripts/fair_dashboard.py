import os
import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, jsonify, abort, send_file
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

# Reuse existing ngrok logic
@dataclass
class NgrokConfig:
    downloads_dir: Path = Path(r"C:\Users\NovoLucas\Downloads")
    domain: str = "astra-assistent.ngrok.app"
    port: int = 5060
    username: str = "astra"
    password: str = "Peixe_2017."
    authtoken: str = os.getenv("WF_NGROK_AUTHTOKEN", "")

    def build_command(self) -> list[str]:
        exe = self.downloads_dir / "ngrok.exe"
        cmd = [
            str(exe), "http",
            "--domain", self.domain,
            "--basic-auth", f"{self.username}:{self.password}",
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

app = Flask(__name__)

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
EQUITY_METRIC_CACHE: dict[str, dict] = {}


def _fair_root() -> Path:
    raw = str(os.getenv("WF_FAIR_ROOT", "fair_wf_explore_v4") or "fair_wf_explore_v4").strip()
    return REPO_ROOT / "data" / "generated" / raw

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
    "max_positions",
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
    try: return float(v)
    except: return default


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


def _read_results_csv_compat(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        header = [str(x).strip() for x in header]
        compat_header = list(header)
        if "exposure_multiplier" not in compat_header:
            try:
                insert_at = compat_header.index("month_mean")
            except ValueError:
                insert_at = len(compat_header)
            compat_header.insert(insert_at, "exposure_multiplier")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) == len(compat_header):
                rows.append(dict(zip(compat_header, row)))
            elif len(row) == len(header):
                rows.append(dict(zip(header, row)))
            elif len(row) < len(header):
                padded = row + [""] * (len(header) - len(row))
                rows.append(dict(zip(header, padded)))
            else:
                trimmed = row[: len(compat_header)]
                rows.append(dict(zip(compat_header, trimmed)))
    return pd.DataFrame(rows)


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
    ret = _to_float(row.get("ret_pct", 0.0)) * 100.0
    dd = _to_float(row.get("max_dd", 0.0))
    pf = _to_float(row.get("profit_factor", 1.0))
    trades = _to_float(row.get("trades", 100.0))
    month_pos = _to_float(row.get("month_pos_frac", 0.0))
    streak = _to_float(row.get("max_neg_month_streak", 0.0))
    underwater = _to_float(row.get("underwater_frac", 1.0))
    worst_90d = _to_float(row.get("worst_rolling_90d", 0.0))
    if ret <= 0 or dd > 0.30: return 0.0
    dd_penalty = np.exp(-8.0 * dd) * np.exp(-18.0 * max(0.0, dd - 0.12))
    smoothed_ret = np.sqrt(ret) * 9.0
    trade_mult = min(1.0, trades / 120.0)
    pf_mult = min(1.25, max(0.85, pf))
    month_support = 0.80 + 0.20 * np.clip(month_pos, 0.0, 1.0)
    streak_penalty = np.exp(-0.55 * max(0.0, streak))
    underwater_penalty = np.exp(-2.25 * np.clip(underwater, 0.0, 1.0))
    regime_penalty = np.exp(-7.5 * max(0.0, -worst_90d))
    return smoothed_ret * dd_penalty * trade_mult * pf_mult * month_support * streak_penalty * underwater_penalty * regime_penalty

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
        
        df = _read_results_csv_compat(tmp_path)
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
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Fair Universe Dashboard v3.0</title>
        <meta charset="UTF-8">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0b0f19;
                --surface: #161c2d;
                --surface-light: #232d45;
                --accent: #38bdf8;
                --accent-glow: rgba(56, 189, 248, 0.2);
                --text-main: #f1f5f9;
                --text-muted: #94a3b8;
                --green: #4ade80;
                --orange: #fbbf24;
                --red: #f87171;
            }
            body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text-main); margin: 0; padding: 20px; line-height: 1.5; overflow-y: scroll; }
            .container { max-width: 1400px; margin: 0 auto; }
            header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            h1 { font-size: 1.8rem; font-weight: 700; margin: 0; color: var(--accent); }
            
            .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
            .kpi-card { background: var(--surface); padding: 15px; border-radius: 10px; border: 1px solid var(--surface-light); }
            .kpi-label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; }
            .kpi-value { font-size: 1.4rem; font-weight: 700; color: var(--text-main); font-family: 'JetBrains Mono', monospace; }

            .tabs { display: flex; gap: 10px; margin-bottom: 20px; background: var(--surface); padding: 14px 16px 10px; border-radius: 12px; border: 1px solid var(--surface-light); overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; scrollbar-color: rgba(148, 163, 184, 0.55) rgba(255,255,255,0.05); }
            .tabs::-webkit-scrollbar { height: 10px; }
            .tabs::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 999px; }
            .tabs::-webkit-scrollbar-thumb { background: linear-gradient(90deg, rgba(148, 163, 184, 0.55), rgba(56, 189, 248, 0.45)); border-radius: 999px; border: 2px solid rgba(22, 28, 45, 0.95); }
            .tabs::-webkit-scrollbar-thumb:hover { background: linear-gradient(90deg, rgba(148, 163, 184, 0.7), rgba(56, 189, 248, 0.6)); }
            .tab { padding: 10px 16px; border-radius: 10px; cursor: pointer; color: var(--text-muted); font-weight: 600; font-size: 0.9rem; transition: all 0.2s; white-space: nowrap; display: flex; align-items: center; border: 1px solid transparent; }
            .tab:hover { background: var(--surface-light); color: var(--text-main); }
            .tab.active { background: var(--accent); color: var(--bg); box-shadow: 0 10px 24px rgba(56, 189, 248, 0.18); }
            .tab-status { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
            .tab-count { margin-left: 10px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--text-main); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }
            .tab.active .tab-count { background: rgba(11, 15, 25, 0.16); color: var(--bg); }

            .step-panel { display: none; }
            .step-panel.active { display: block; }

            table { width: 100%; border-collapse: separate; border-spacing: 0; background: var(--surface); border-radius: 12px; border: 1px solid var(--surface-light); margin-bottom: 50px; }
            th { background: var(--surface-light); padding: 12px 15px; text-align: left; color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; cursor: pointer; user-select: none; transition: color 0.2s; }
            th:hover { color: var(--accent); }
            th.sorted-asc::after { content: ' ↑'; color: var(--accent); }
            th.sorted-desc::after { content: ' ↓'; color: var(--accent); }
            
            td { padding: 12px 15px; border-bottom: 1px solid var(--surface-light); font-size: 0.9rem; }
            tr.row-clickable { cursor: pointer; transition: background 0.1s; }
            tr.row-clickable:hover td { background: rgba(255,255,255,0.03); }
            tr.row-active td { background: var(--accent-glow) !important; border-left: 2px solid var(--accent); }
            
            .metric { font-family: 'JetBrains Mono', monospace; }
            .score-pill { background: var(--accent-glow); color: var(--accent); padding: 2px 8px; border-radius: 4px; font-weight: 700; }
            
            .iframe-row { display: none; background: #000; }
            .detail-shell { padding: 18px; background: linear-gradient(180deg, rgba(7, 11, 19, 0.96), rgba(3, 6, 11, 0.98)); }
            .detail-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
            .detail-card { background: rgba(22, 28, 45, 0.96); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 12px; padding: 14px; min-width: 0; overflow: hidden; }
            .detail-title { color: var(--accent); font-size: 0.78rem; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.04em; }
            .detail-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
            .detail-chip { padding: 4px 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); color: var(--text-muted); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; max-width: 100%; white-space: normal; overflow-wrap: anywhere; line-height: 1.35; }
            .param-list { display: grid; gap: 8px; }
            .param-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: start; border-bottom: 1px solid rgba(148, 163, 184, 0.08); padding-bottom: 6px; }
            .param-label { color: var(--text-muted); font-size: 0.78rem; min-width: 0; }
            .param-value { color: var(--text-main); font-size: 0.8rem; text-align: right; font-family: 'JetBrains Mono', monospace; min-width: 0; max-width: 100%; word-break: break-word; overflow-wrap: anywhere; }
            .param-empty { color: var(--text-muted); font-size: 0.8rem; font-style: italic; }
            .iframe-container { width: 100%; height: 700px; border: none; }
            
            .empty-state { padding: 100px; text-align: center; color: var(--text-muted); background: var(--surface); border-radius: 12px; border: 1px solid var(--surface-light); }

            @media (max-width: 1100px) {
                .kpi-row { grid-template-columns: repeat(2, 1fr); }
                .detail-grid { grid-template-columns: 1fr; }
            }

            @media (max-width: 700px) {
                .kpi-row { grid-template-columns: 1fr; }
                .tabs { padding: 12px 12px 8px; }
                .tab { padding: 9px 14px; }
                .param-row { grid-template-columns: 1fr; }
                .param-value { text-align: left; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>Fair Universe Dashboard</h1>
                    <div style="color: var(--text-muted); font-size: 0.9rem;">Multi-Generation Robustness Monitor</div>
                </div>
                <div style="text-align: right">
                    <div id="status-dot" style="color: var(--green); font-weight: 700; font-size: 0.8rem;">● LIVE DATA</div>
                    <div id="countdown" style="color: var(--text-muted); font-size: 0.75rem;">Next update in ...</div>
                </div>
            </header>

            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">Milestones Finished</div>
                    <div id="kpi-finished" class="kpi-value">--</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Historical Score</div>
                    <div id="kpi-avg-score" class="kpi-value" style="color: var(--accent)">--</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Fair Trials</div>
                    <div id="kpi-trials" class="kpi-value">--</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">System Mode</div>
                    <div class="kpi-value" style="color: var(--green)">AUTONOMOUS</div>
                </div>
            </div>

            <div class="tabs" id="step-tabs"></div>
            <div id="main-content"></div>

            <footer style="margin-top: 30px; border-top: 1px solid var(--surface-light); padding: 20px 0; display: flex; justify-content: space-between; color: var(--text-muted); font-size: 0.8rem;">
                <div>Fair Universe Protocol &bull; Independent Milestone Optimization</div>
                <div>Ngrok: https://astra-assistent.ngrok.app</div>
            </footer>
        </div>

        <script>
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
                    const res = await fetch('/api/data');
                    state.data = await res.json();
                    state.lastUpdate = Date.now();
                    render();
                } catch(e) {
                    console.error("Dashboard Fetch Error:", e);
                    document.getElementById('status-dot').style.color = 'var(--red)';
                    document.getElementById('status-dot').textContent = '● ERROR (Offline)';
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
                    <table
                        id="main-table"
                        data-step="${step}"
                        data-expanded-id="${state.expandedPlotId}"
                        data-sort-key="${state.sortKey}"
                        data-sort-dir="${state.sortDir}"
                    >
                        <thead>
                            <tr>
                                <th onclick="setSort('generation')">Gen</th>
                                <th onclick="setSort('phase')">Phase</th>
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
                                return `
                                    <tr class="row-clickable ${isExpanded?'row-active':''}" onclick="togglePlot('${id}')" id="row-${id}">
                                        <td class="metric text-muted">G${r.generation || 1}</td>
                                        <td class="metric text-muted">${r.phase || 'broad'}</td>
                                        <td class="metric" style="color: var(--text-muted)">${r.backtest_id}/${r.train_id}</td>
                                        <td style="text-align: right"><span class="score-pill" id="score-${id}">${asNumber(r.score).toFixed(4)}</span></td>
                                        <td style="text-align: right; color: var(--green)" class="metric" id="ret-${id}">+${(asNumber(r.ret_pct) * 100).toFixed(1)}%</td>
                                        <td style="text-align: right; color: var(--red)" class="metric" id="dd-${id}">${(asNumber(r.max_dd) * 100).toFixed(2)}%</td>
                                        <td style="text-align: right" class="metric" id="pf-${id}">${asNumber(r.profit_factor).toFixed(2)}</td>
                                        <td style="text-align: right" class="metric" id="win-${id}">${(asNumber(r.win_rate) * 100).toFixed(1)}%</td>
                                        <td style="text-align: right" class="metric text-muted" id="trades-${id}">${r.trades}</td>
                                    </tr>
                                    <tr class="iframe-row" id="iframe-row-${id}" style="display: ${isExpanded?'table-row':'none'}">
                                        <td colspan="9" style="padding: 0;">
                                            ${isExpanded ? `
                                                <div class="detail-shell">
                                                    <div class="detail-grid">
                                                        ${renderParamCard(r.params?.refresh)}
                                                        ${renderParamCard(r.params?.train)}
                                                        ${renderParamCard(r.params?.trial)}
                                                    </div>
                                                    ${r.rel_html ? `<iframe class="iframe-container" src="/artifact/${r.rel_html}"></iframe>` : '<div class="empty-state" style="padding: 40px;">No plot artifact found for this trial.</div>'}
                                                </div>
                                            ` : ''}
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                `;
                container.innerHTML = table;
            }

            // Sync countdown
            setInterval(() => {
                const diff = 10 - Math.floor((Date.now() - state.lastUpdate) / 1000);
                document.getElementById('countdown').textContent = (diff > 0) ? `Next update in ${diff}s` : 'Updating...';
                if (diff <= 0) fetchData();
            }, 1000);

            // Initial load
            fetchData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

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
