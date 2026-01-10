/* global Chart */

(function () {
  const fmtUSD = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });

  const fmtNum = new Intl.NumberFormat("en-US", { maximumFractionDigits: 6 });

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setBadge(id, text, type) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = "badge text-bg-" + (type || "secondary");
  }

  function pnlClass(v) {
    if (v == null || isNaN(v)) return "";
    if (v > 0) return "pnl-pos";
    if (v < 0) return "pnl-neg";
    return "";
  }

  function fmtPnl(v) {
    if (v == null || isNaN(v)) return "—";
    const s = fmtUSD.format(v);
    return (v > 0 ? "+" : "") + s;
  }

  function fmtPct(v) {
    if (v == null || isNaN(v)) return "—";
    const s = v.toFixed(2) + "%";
    return (v > 0 ? "+" : "") + s;
  }

  function utcTimeShort(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toISOString().slice(11, 19);
    } catch (e) {
      return String(iso);
    }
  }

  let paused = false;
  let timer = null;
  let chart = null;

  function ensureChart(labels, values) {
    const canvas = document.getElementById("allocation-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const colors = [
      "#7c5cff",
      "#00ffb3",
      "#ffc400",
      "#ff5c7a",
      "#00a3ff",
      "#9cff00",
      "#ff8a00",
      "#b17cff",
    ];

    if (chart) {
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.update();
      return;
    }

    chart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: values,
            backgroundColor: labels.map((_, i) => colors[i % colors.length]),
            borderColor: "rgba(255,255,255,0.12)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "rgba(255,255,255,0.75)" },
          },
          tooltip: {
            callbacks: {
              label: function (ctx2) {
                const v = ctx2.parsed;
                return " " + fmtUSD.format(v);
              },
            },
          },
        },
        cutout: "62%",
      },
    });
  }

  function setOnline(isOnline) {
    const dot = document.getElementById("status-dot");
    if (!dot) return;
    dot.classList.remove("status-online", "status-offline");
    dot.classList.add(isOnline ? "status-online" : "status-offline");
  }

  function renderState(state) {
    const summary = (state && state.summary) || {};
    const positions = (state && state.positions) || [];
    const trades = (state && state.recent_trades) || [];
    const allocation = (state && state.allocation) || {};
    const meta = (state && state.meta) || {};

    setText("mode-badge", meta.mode || "—");
    setText("last-update", utcTimeShort(state.generated_at_utc || summary.updated_at_utc));

    const equity = Number(summary.equity_usd || 0);
    const cash = Number(summary.cash_usd || 0);
    const exposure = Number(summary.exposure_usd || 0);
    const unreal = Number(summary.unrealized_pnl_usd || 0);
    const real = Number(summary.realized_pnl_usd || 0);
    const exposurePct = equity !== 0 ? (exposure / equity) * 100.0 : 0;

    setText("kpi-equity", fmtUSD.format(equity));
    setText("kpi-cash", fmtUSD.format(cash));
    setText("kpi-exposure", fmtUSD.format(exposure));
    setText("kpi-exposure-pct", exposurePct.toFixed(1) + "%");

    const unrealEl = document.getElementById("kpi-unreal");
    if (unrealEl) {
      unrealEl.textContent = fmtPnl(unreal);
      unrealEl.className = "kpi-value " + pnlClass(unreal);
    }
    setText("kpi-real", fmtPnl(real));

    setText("kpi-symbols", String(Object.keys(allocation).length || 0));
    setText("kpi-positions", String(positions.length || 0));

    setText("alloc-total", fmtUSD.format(Object.values(allocation).reduce((a, b) => a + Number(b || 0), 0)));
    setText("positions-badge", String(positions.length || 0));
    setText("trades-badge", String(trades.length || 0));

    // positions table
    const pt = document.getElementById("positions-tbody");
    if (pt) {
      pt.innerHTML = "";
      for (const p of positions) {
        const tr = document.createElement("tr");
        const pnl = Number(p.pnl_usd || 0);
        tr.innerHTML = `
          <td class="fw-semibold">${p.symbol || "—"}</td>
          <td><span class="badge text-bg-secondary">${p.side || "—"}</span></td>
          <td class="text-end">${fmtNum.format(Number(p.qty || 0))}</td>
          <td class="text-end">${fmtUSD.format(Number(p.entry_price || 0))}</td>
          <td class="text-end">${fmtUSD.format(Number(p.mark_price || 0))}</td>
          <td class="text-end">${fmtUSD.format(Number(p.notional_usd || 0))}</td>
          <td class="text-end ${pnlClass(pnl)}">
            <div>${fmtPnl(pnl)}</div>
            <div class="small text-secondary">${fmtPct(Number(p.pnl_pct || 0))}</div>
          </td>
        `;
        pt.appendChild(tr);
      }
      if (!positions.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="7" class="text-center text-secondary py-4">Sem posições abertas</td>`;
        pt.appendChild(tr);
      }
    }

    // trades table
    const tt = document.getElementById("trades-tbody");
    if (tt) {
      tt.innerHTML = "";
      for (const t of trades.slice(0, 20)) {
        const pnl = t.pnl_usd == null ? null : Number(t.pnl_usd);
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="text-secondary">${utcTimeShort(t.ts_utc)}</td>
          <td class="fw-semibold">${t.symbol || "—"}</td>
          <td><span class="badge text-bg-secondary">${t.action || "—"}</span></td>
          <td class="text-end">${fmtNum.format(Number(t.qty || 0))}</td>
          <td class="text-end">${fmtUSD.format(Number(t.price || 0))}</td>
          <td class="text-end ${pnlClass(pnl == null ? 0 : pnl)}">${pnl == null ? "—" : fmtPnl(pnl)}</td>
        `;
        tt.appendChild(tr);
      }
      if (!trades.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="6" class="text-center text-secondary py-4">Sem trades recentes</td>`;
        tt.appendChild(tr);
      }
    }

    // allocation chart
    const labels = Object.keys(allocation);
    const values = labels.map((k) => Number(allocation[k] || 0));
    ensureChart(labels, values);
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
      setBadge("health-badge", "OK", "success");
    } catch (e) {
      setOnline(false);
      setBadge("health-badge", "OFF", "danger");
    }
  }

  function startPolling() {
    stopPolling();
    timer = setInterval(function () {
      if (!paused) refreshOnce();
    }, 2000);
  }

  function stopPolling() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-bs-theme") || "dark";
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-bs-theme", next);
    try {
      localStorage.setItem("astra-theme", next);
    } catch (e) {}
  }

  function wireUI() {
    const btnTheme = document.getElementById("theme-toggle");
    if (btnTheme) btnTheme.addEventListener("click", toggleTheme);

    const btnRefresh = document.getElementById("btn-refresh");
    if (btnRefresh) btnRefresh.addEventListener("click", refreshOnce);

    const btnPause = document.getElementById("btn-pause");
    if (btnPause) {
      btnPause.addEventListener("click", function () {
        paused = !paused;
        btnPause.innerHTML = paused
          ? '<i class="bi bi-play-circle me-1"></i>Retomar'
          : '<i class="bi bi-pause-circle me-1"></i>Pausar';
      });
    }
  }

  // bootstrap
  wireUI();
  if (window.__INITIAL_STATE__) {
    renderState(window.__INITIAL_STATE__);
    setOnline(true);
    setBadge("health-badge", "OK", "success");
  }
  refreshOnce();
  startPolling();
})();

