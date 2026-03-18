(() => {
  const refreshSec = Number(window.__REFRESH_SEC__ || 6);
  let timer = null;
  
  let currentState = null;
  let expandedRows = new Set(); // Track expanded bt_ids
  let currentSortCol = "score";
  let currentSortAsc = false;

  let isRefreshing = false;
  let loopActive = true;
  let runsLimit = 50;

  function calcScore(row) {
    const ret = toFloat(row.ret_pct, 0.0);
    const dd = toFloat(row.max_dd, 0.0);
    const win = toFloat(row.win_rate, 0.0);
    const pf = toFloat(row.profit_factor, 1.0);
    const trades = toFloat(row.trades, 0.0);
    
    // Extracted consistency metrics (defaulting to win rate if not present)
    const monthPos = toFloat(row.month_pos_frac, win);
    const semPos = toFloat(row.semester_pos_frac, win);

    if (ret <= 0) return ret; // Penalize immediately if negative

    // 1. Exponential Decay on SQUARED drawdown
    const dd_penalty = Math.exp(-25.0 * Math.pow(dd, 2));

    // 2. Square Root of the return to prevent outlier domination
    const smoothed_ret = Math.sqrt(ret) * 10;

    const consistency_mult = 0.2 + monthPos + semPos;
    const trade_mult = Math.min(1.0, trades / 100.0);
    const pf_mult = Math.min(3.0, pf);

    return smoothed_ret * dd_penalty * consistency_mult * trade_mult * pf_mult;
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setOnline(isOnline) {
    const dot = document.getElementById("status-dot");
    if (!dot) return;
    dot.classList.remove("status-online", "status-offline");
    dot.classList.add(isOnline ? "status-online" : "status-offline");
  }

  function toFloat(v, dflt = null) {
    if (v === null || v === undefined) return dflt;
    const n = Number(v);
    return Number.isFinite(n) ? n : dflt;
  }

  function fmtPct(v, digits = 2) {
    if (!Number.isFinite(v)) return "--";
    const s = v.toFixed(digits) + "%";
    return v > 0 ? "+" + s : s;
  }

  function fmtNum(v, digits = 2) {
    if (!Number.isFinite(v)) return "--";
    return v.toFixed(digits);
  }

  function fmtId(row) {
    const t = row.train_id || "--";
    const b = row.backtest_id || "--";
    return t + "/" + b;
  }

  function renderTable(rows) {
    const tbody = document.getElementById("runs-tbody");
    if (!tbody) return;

    // Create a new fresh tbody content
    const fragment = document.createDocumentFragment();

    rows.forEach((r, idx) => {
        const t_id = r.train_id || "--";
        const b_id = r.backtest_id || "--";
        const status = (r.status || "").toLowerCase();
        
        const eqEnd = parseFloat(r.eq_end) || 0;
        const maxDd = parseFloat(r.max_dd) || 0;
        const winR = parseFloat(r.win_rate) || 0;
        const pf = parseFloat(r.profit_factor) || 0;
        const trds = parseInt(r.trades) || 0;
        const tauE = r.tau_entry || "--";
        
        const score = calcScore(r);
        const rowId = fmtId(r);
        const isExpanded = expandedRows.has(rowId);

        const tr = document.createElement("tr");
        tr.style.cursor = "pointer"; // indicate clickability
        
        // Using exactly 10 columns matching the <thead>
        tr.innerHTML = `
            <td class="mono">${t_id}</td>
            <td class="mono">${b_id}</td>
            <td>
              <span class="badge ${status === "ok" ? "ok" : "err"}">
                ${status || "--"}
              </span>
            </td>
            <td class="right">
                <span class="pill outline" style="color:var(--c-accent)">
                    ${score.toFixed(4)}
                </span>
            </td>
            <td class="right ${eqEnd - 1 > 0 ? "value-pos" : eqEnd - 1 < 0 ? "value-neg" : ""}">
                ${fmtPct((eqEnd - 1) * 100.0)}
            </td>
            <td class="right" style="color:var(--c-danger)">
                ${(maxDd * 100).toFixed(2)}%
            </td>
            <td class="right">${fmtNum(pf)}</td>
            <td class="right">${fmtPct(winR * 100.0)}</td>
            <td class="right">${trds}</td>
            <td class="right">${tauE}</td>
        `;

        tr.addEventListener("click", () => {
            if (expandedRows.has(rowId)) {
                expandedRows.delete(rowId);
            } else {
                expandedRows.add(rowId);
            }
            if (currentState) renderState(currentState, true); 
        });

        fragment.appendChild(tr);

        // Sub-row for expanded iframe
        if (isExpanded && r.equity_html) {
            const subTr = document.createElement("tr");
            subTr.style.backgroundColor = "rgba(0, 0, 0, 0.4)";
            
            // Generate path to HTML file using the server's /artifact/ endpoint
            const htmlPath = "/artifact/" + encodeURI(r.equity_html);
            
            // subTr spans all columns (10 standard table columns)
            subTr.innerHTML = `
                <td colspan="10" style="padding: 0;">
                    <div style="width: 100%; border-bottom: 1px solid rgba(255,255,255,.08); background: #0b0d12; overflow: hidden; display: flex; flex-direction: column;">
                        <iframe src="${htmlPath}" title="artifact preview" 
                                style="width: 100%; height: 600px; border: none; background: transparent;">
                        </iframe>
                    </div>
                </td>
            `;
            fragment.appendChild(subTr);
        }
    });

    tbody.innerHTML = "";
    tbody.appendChild(fragment);

    if (!rows.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="10" class="muted">No data yet.</td>`;
      tbody.appendChild(tr);
    }
  }

  function renderLogs(logs) {
    const loopEl = document.getElementById("log-loop");
    const dashEl = document.getElementById("log-dash");
    const loopLines = logs || [];
    if (loopEl) {
      loopEl.textContent = loopLines.length ? loopLines.join("") : "No logs yet.";
    }
    // dashEl is not used in the new log structure, but keep it for robustness if HTML still has it
    if (dashEl) {
      dashEl.textContent = "Dashboard logs are not available in this view.";
    }
    setText("log-count", String(loopLines.length));
  }

  function renderState(data, userAction = false) {
    currentState = data; // Update global state reference
    const meta = data.meta || {};
    const summary = data.summary || {};

    setText("meta-rows", String(meta.rows_total || 0));
    setText("meta-status", meta.last_status || "--");
    setText("meta-csv", meta.csv_path || "--");
    setText("last-update", meta.last_end_utc || meta.updated_at_utc || "--");

    setText("kpi-total-trains", String(summary.total_trains || 0));
    setText("kpi-total-backtests", String(summary.total_backtests || 0));
    setText("kpi-ok", String(summary.ok_backtests || 0));
    setText("kpi-err", String(summary.err_backtests || 0));

    const bestRet = summary.best_ret || {};
    const bestPf = summary.best_pf || {};
    const bestWin = summary.best_win || {};
    const bestDd = summary.best_dd || {};

    setText("kpi-best-ret", fmtPct(toFloat(bestRet.ret_pct, 0)));
    setText("kpi-best-ret-id", fmtId(bestRet));
    setText("kpi-best-pf", fmtNum(toFloat(bestPf.profit_factor, 0)));
    setText("kpi-best-pf-id", fmtId(bestPf));
    setText("kpi-best-win", fmtPct(toFloat(bestWin.win_rate, 0) * 100.0));
    setText("kpi-best-win-id", fmtId(bestWin));
    setText("kpi-best-dd", fmtPct(toFloat(bestDd.max_dd, 0) * 100.0));
    setText("kpi-best-dd-id", fmtId(bestDd));

    // Handle runs logic
    const allRuns = Array.isArray(data.rows) ? data.rows : [];
    const backtests = allRuns.filter(r => (r.stage || "").toLowerCase() === "backtest");

    backtests.sort((a, b) => {
        let valA, valB;

        if (currentSortCol === "score") {
            valA = calcScore(a);
            valB = calcScore(b);
        } else if (currentSortCol === "status" || currentSortCol === "train_id" || currentSortCol === "backtest_id") {
            valA = (a[currentSortCol] || "").toLowerCase();
            valB = (b[currentSortCol] || "").toLowerCase();
        } else if (currentSortCol === "ret_pct") {
            // Prefer eq_end if available (as used in table rendering)
            valA = a.eq_end !== undefined ? toFloat(a.eq_end, 1) - 1 : toFloat(a.ret_pct, 0);
            valB = b.eq_end !== undefined ? toFloat(b.eq_end, 1) - 1 : toFloat(b.ret_pct, 0);
        } else {
            valA = toFloat(a[currentSortCol], 0);
            valB = toFloat(b[currentSortCol], 0);
        }

        if (valA < valB) return currentSortAsc ? -1 : 1;
        if (valA > valB) return currentSortAsc ? 1 : -1;

        // tie-breaker: score
        return calcScore(b) - calcScore(a);
    });

    const backtestsToShow = backtests.slice(0, runsLimit);
    
    setText("runs-count", `${backtestsToShow.length} of ${backtests.length}`);
    
    // Only re-render the physical DOM table if the user explicitly clicked something (sort/expand)
    // OR if no plots are currently expanded. This prevents the auto-refresh from destroying the iframe state.
    const shouldRenderTable = userAction || expandedRows.size === 0;
    if (shouldRenderTable) {
        renderTable(backtestsToShow);
    }

    const subtitle = document.getElementById("table-subtitle");
    if (subtitle) {
        if (expandedRows.size > 0 && !userAction) {
            subtitle.innerHTML = 'Latest rows from random_runs.csv <span style="color:var(--c-warning); margin-left:8px; font-weight:600;">⏸️ Table frozen while plot is open</span>';
        } else if (expandedRows.size === 0) {
            subtitle.innerHTML = 'Latest rows from random_runs.csv';
        }
    }

    // Limit load more button visibility
    const btnLoadMoreRuns = document.getElementById("btn-load-more-runs");
    if (btnLoadMoreRuns) {
        btnLoadMoreRuns.style.display = runsLimit >= backtests.length ? "none" : "inline-block";
    }

    // Render logs
    renderLogs(data.loop_log);
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("http_" + res.status);
    return await res.json();
  }

  async function refreshOnce() {
    if (isRefreshing) return;
    isRefreshing = true;
    try {
      const data = await fetchJson("/api/state");
      renderState(data, false); // false = not a user action (auto refresh)
      setOnline(true);
    } catch (e) {
      setOnline(false);
      console.error("Failed to fetch state:", e);
    } finally {
      isRefreshing = false;
    }
  }

  function startPolling() {
    stopPolling();
    timer = setInterval(() => {
      if (loopActive) refreshOnce();
    }, Math.max(1000, refreshSec * 1000));
  }

  function stopPolling() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function wireUI() {
    const btnRefresh = document.getElementById("btn-refresh");
    if (btnRefresh) btnRefresh.addEventListener("click", refreshOnce);

    const btnPause = document.getElementById("btn-pause-refresh");
    if (btnPause) {
        btnPause.addEventListener("click", () => {
            loopActive = !loopActive;
            btnPause.className = loopActive ? "btn ghost" : "btn primary";
            btnPause.textContent = loopActive ? "Pause Auto-Refresh" : "Resume Auto-Refresh";
        });
    }

    const btnLoadMoreRuns = document.getElementById("btn-load-more-runs");
    if (btnLoadMoreRuns) {
        btnLoadMoreRuns.addEventListener("click", () => {
            runsLimit += 50;
            if (currentState) renderState(currentState, true);
        });
    }

    // Attach row sorting logic to headers
    document.querySelectorAll("th[data-sort]").forEach(th => {
        th.style.cursor = "pointer";
        th.addEventListener("click", () => {
            const col = th.getAttribute("data-sort");
            if (currentSortCol === col) {
                currentSortAsc = !currentSortAsc; // toggle direction
            } else {
                currentSortCol = col;
                // Default descending for most metrics (Score, Return, PF, WinRate)
                // Default ascending for Drawdown (lower is better) or strings
                currentSortAsc = (col === "max_dd" || col === "status" || col === "train_id" || col === "backtest_id") ? true : false;
            }

            // Update arrow icons
            document.querySelectorAll("th[data-sort] .sort-icon").forEach(icon => {
                icon.textContent = "";
            });
            const icon = th.querySelector(".sort-icon");
            if (icon) {
                icon.textContent = currentSortAsc ? "▲" : "▼";
            }

            // Re-render table with new sorting without refetching
            if (currentState) renderState(currentState, true);
        });
    });
  }

  wireUI();
  if (window.__INITIAL_STATE__) {
    renderState(window.__INITIAL_STATE__);
    setOnline(true);
  }
  refreshOnce();
  startPolling();
})();
