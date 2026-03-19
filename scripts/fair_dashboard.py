import os
import sys
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
FAIR_ROOT = REPO_ROOT / "data" / "generated" / "fair_wf_explore"

def _to_float(v, default=0.0):
    try: return float(v)
    except: return default

def calc_score(row):
    ret = _to_float(row.get("ret_pct", 0.0)) * 100.0
    dd = _to_float(row.get("max_dd", 0.0))
    pf = _to_float(row.get("profit_factor", 1.0))
    trades = _to_float(row.get("trades", 100.0))
    if ret <= 0: return 0.0
    dd_penalty = np.exp(-25.0 * (dd ** 2))
    smoothed_ret = np.sqrt(ret) * 10
    trade_mult = min(1.0, trades / 100.0)
    return smoothed_ret * dd_penalty * trade_mult * min(3.0, pf)

def get_step_data(step):
    step_dir = FAIR_ROOT / f"step_{step}d"
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
        df = df[(df["stage"] == "backtest") & (df["status"] == "ok")]
        if df.empty:
            return {"status": "Running...", "rows": [], "summary": {}}
            
        df["score"] = df.apply(calc_score, axis=1)
        import numpy as np
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
        
        best = df.iloc[0]
        summary = {
            "best_score": round(best["score"], 2),
            "best_ret": round(best["ret_pct"] * 100, 1),
            "best_dd": round(best["max_dd"] * 100, 1),
            "best_wr": round(best["win_rate"] * 100, 1),
            "total_ok": len(df),
            "total_count": total_trials
        }
        
        return {
            "status": "Finished" if (step_dir / ".finished").exists() else "Running...",
            "rows": rows,
            "summary": summary
        }
    except Exception as e:
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

            .tabs { display: flex; gap: 10px; margin-bottom: 20px; background: var(--surface); padding: 10px; border-radius: 10px; border: 1px solid var(--surface-light); overflow-x: auto; }
            .tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; color: var(--text-muted); font-weight: 600; font-size: 0.9rem; transition: all 0.2s; white-space: nowrap; display: flex; align-items: center; }
            .tab:hover { background: var(--surface-light); color: var(--text-main); }
            .tab.active { background: var(--accent); color: var(--bg); }
            .tab-status { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }

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
            .iframe-container { width: 100%; height: 700px; border: none; }
            
            .empty-state { padding: 100px; text-align: center; color: var(--text-muted); background: var(--surface); border-radius: 12px; border: 1px solid var(--surface-light); }
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
                    const d = state.data[s] || { status: 'Pending' };
                    const color = d.status === 'Finished' ? 'var(--green)' : (d.status === 'Running...' ? 'var(--orange)' : 'var(--text-muted)');
                    const active = String(s) === state.activeStep ? 'active' : '';
                    return `<div class="tab ${active}" onclick="setStep('${s}')">
                        <span class="tab-status" style="background: ${color}"></span>
                        ${s} days
                    </div>`;
                }).join('');
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
                    if (state.sortDir === 'desc') return valB - valA;
                    return valA - valB;
                });

                // If table already exists and we have an expanded plot, try to update surgically
                const existingTable = document.getElementById('main-table');
                if (existingTable && existingTable.getAttribute('data-step') === String(step) && existingTable.getAttribute('data-expanded-id') === String(state.expandedPlotId)) {
                    rows.forEach(r => {
                        const id = `${step}-${r.backtest_id}-${r.train_id}`;
                        const scoreEl = document.getElementById(`score-${id}`);
                        if (scoreEl) {
                            scoreEl.textContent = r.score.toFixed(4);
                            document.getElementById(`ret-${id}`).textContent = `+${(r.ret_pct * 100).toFixed(1)}%`;
                            document.getElementById(`dd-${id}`).textContent = `${(r.max_dd * 100).toFixed(2)}%`;
                            document.getElementById(`pf-${id}`).textContent = r.profit_factor.toFixed(2);
                            document.getElementById(`win-${id}`).textContent = `${(r.win_rate * 100).toFixed(1)}%`;
                            document.getElementById(`trades-${id}`).textContent = r.trades;
                        }
                    });
                    // Skip full re-render to preserve iframe state (zoom)
                    return;
                }

                const table = `
                    <table id="main-table" data-step="${step}" data-expanded-id="${state.expandedPlotId}">
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
                                return `
                                    <tr class="row-clickable ${isExpanded?'row-active':''}" onclick="togglePlot('${id}')" id="row-${id}">
                                        <td class="metric" style="color: var(--text-muted)">${r.backtest_id}/${r.train_id}</td>
                                        <td style="text-align: right"><span class="score-pill" id="score-${id}">${r.score.toFixed(4)}</span></td>
                                        <td style="text-align: right; color: var(--green)" class="metric" id="ret-${id}">+${(r.ret_pct * 100).toFixed(1)}%</td>
                                        <td style="text-align: right; color: var(--red)" class="metric" id="dd-${id}">${(r.max_dd * 100).toFixed(2)}%</td>
                                        <td style="text-align: right" class="metric" id="pf-${id}">${r.profit_factor.toFixed(2)}</td>
                                        <td style="text-align: right" class="metric" id="win-${id}">${(r.win_rate * 100).toFixed(1)}%</td>
                                        <td style="text-align: right" class="metric text-muted" id="trades-${id}">${r.trades}</td>
                                    </tr>
                                    <tr class="iframe-row" id="iframe-row-${id}" style="display: ${isExpanded?'table-row':'none'}">
                                        <td colspan="7" style="padding: 0;">
                                            ${isExpanded ? `<iframe class="iframe-container" src="/artifact/${r.rel_html}"></iframe>` : ''}
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
