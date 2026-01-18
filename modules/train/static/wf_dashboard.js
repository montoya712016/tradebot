(() => {
  const refreshSec = Number(window.__REFRESH_SEC__ || 6);
  let paused = false;
  let timer = null;

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
    tbody.innerHTML = "";

    for (const row of rows) {
      const ret = toFloat(row.ret_pct);
      const dd = toFloat(row.max_dd);
      const pf = toFloat(row.profit_factor);
      const win = toFloat(row.win_rate);
      const trades = row.trades || "0";
      const tauE = row.tau_entry || "--";
      const status = (row.status || "").toLowerCase();

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">${row.train_id || "--"}</td>
        <td class="mono">${row.backtest_id || "--"}</td>
        <td>
          <span class="badge ${status === "ok" ? "ok" : "err"}">
            ${status || "--"}
          </span>
        </td>
        <td class="right ${ret > 0 ? "value-pos" : ret < 0 ? "value-neg" : ""}">
          ${fmtPct(ret || 0)}
        </td>
        <td class="right">${fmtPct((dd || 0) * 100.0)}</td>
        <td class="right">${fmtNum(pf || 0)}</td>
        <td class="right">${fmtPct((win || 0) * 100.0)}</td>
        <td class="right">${trades}</td>
        <td class="right">${tauE}</td>
      `;
      tbody.appendChild(tr);
    }

    if (!rows.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="9" class="muted">No data yet.</td>`;
      tbody.appendChild(tr);
    }
  }

  function renderGallery(images) {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;
    grid.innerHTML = "";
    for (const item of images) {
      const ret = toFloat(item.ret_pct, 0);
      const dd = toFloat(item.max_dd, 0);
      const pf = toFloat(item.profit_factor, 0);
      const card = document.createElement("div");
      card.className = "gallery-card";
      card.innerHTML = `
        <img src="/img/${item.img}" alt="equity" loading="lazy" />
        <div class="gallery-body">
          <div class="gallery-title mono">${item.train_id}/${item.backtest_id}</div>
          <div class="gallery-meta">
            <span class="${ret > 0 ? "value-pos" : ret < 0 ? "value-neg" : ""}">
              ${fmtPct(ret)}
            </span>
            <span>DD ${fmtPct(dd * 100.0)}</span>
            <span>PF ${fmtNum(pf)}</span>
          </div>
        </div>
      `;
      grid.appendChild(card);
    }
    if (!images.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No equity images yet.";
      grid.appendChild(empty);
    }
  }

  function renderLogs(logs) {
    const loopEl = document.getElementById("log-loop");
    const dashEl = document.getElementById("log-dash");
    const loopLines = (logs && logs.loop) || [];
    const dashLines = (logs && logs.dash) || [];
    if (loopEl) {
      loopEl.textContent = loopLines.length ? loopLines.join("") : "No logs yet.";
    }
    if (dashEl) {
      dashEl.textContent = dashLines.length ? dashLines.join("") : "No logs yet.";
    }
    setText("log-count", String(loopLines.length + dashLines.length));
  }

  function renderState(state) {
    const meta = state.meta || {};
    const summary = state.summary || {};
    const rows = state.rows || [];
    const images = state.images || [];
    const logs = state.logs || {};

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

    const backtests = rows.filter((r) => (r.stage || "").toLowerCase() === "backtest");
    setText("runs-count", String(backtests.length));
    renderTable(backtests.slice(-30).reverse());

    setText("img-count", String(images.length));
    renderGallery(images);
    renderLogs(logs);
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("http_" + res.status);
    return await res.json();
  }

  async function refreshOnce() {
    try {
      const st = await fetchJson("/api/state");
      renderState(st);
      setOnline(true);
    } catch (e) {
      setOnline(false);
    }
  }

  function startPolling() {
    stopPolling();
    timer = setInterval(() => {
      if (!paused) refreshOnce();
    }, Math.max(1000, refreshSec * 1000));
  }

  function stopPolling() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function wireUI() {
    const btnRefresh = document.getElementById("btn-refresh");
    if (btnRefresh) btnRefresh.addEventListener("click", refreshOnce);
    const btnPause = document.getElementById("btn-pause");
    if (btnPause) {
      btnPause.addEventListener("click", () => {
        paused = !paused;
        btnPause.textContent = paused ? "Resume" : "Pause";
      });
    }
  }

  wireUI();
  if (window.__INITIAL_STATE__) {
    renderState(window.__INITIAL_STATE__);
    setOnline(true);
  }
  refreshOnce();
  startPolling();
})();
