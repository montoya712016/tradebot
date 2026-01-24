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
  let eqChart = null;
  let selectedTrade = null;
  let selectedSignal = null;
  let lastFeed = { last_ts_ms: null, ws_msgs: 0, symbols: 0 };
  let signalsCache = [];
  let signalsShown = 20;

  function tsMsToIso(ts) {
    if (!ts || isNaN(ts)) return "—";
    try {
      return new Date(Number(ts)).toISOString();
    } catch (e) {
      return String(ts);
    }
  }

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

  function ensureEquityChart(labels, values) {
    const canvas = document.getElementById("equity-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const borderColor = "rgba(0, 255, 179, 0.9)";
    const fillColor = "rgba(0, 255, 179, 0.12)";

    if (eqChart) {
      eqChart.data.labels = labels;
      eqChart.data.datasets[0].data = values;
      eqChart.update();
      return;
    }

    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Equity (USDT)",
            data: values,
            borderColor,
            backgroundColor: fillColor,
            tension: 0.2,
            fill: true,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx2) {
                const v = ctx2.parsed.y;
                return " " + fmtUSD.format(v);
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: "rgba(255,255,255,0.7)" },
          },
          y: {
            ticks: {
              color: "rgba(255,255,255,0.7)",
              callback: (v) => fmtUSD.format(v),
            },
          },
        },
      },
    });
  }

  function renderTradeChart(symbol, data, entryPrice, exitPrice, side, scoreVal) {
    const el = document.getElementById("trade-chart");
    if (!el || !data || !data.length) return;
    const ts = data.map((d) => new Date(d.ts).toISOString());
    const hasScoreSeries = data.some((d) => d.score != null && !isNaN(d.score));
    const scoreValNum = scoreVal != null && !isNaN(scoreVal) ? Number(scoreVal) : null;
    const scoreSeries = hasScoreSeries
      ? data.map((d) => Number(d.score || 0))
      : scoreValNum != null
      ? Array(ts.length).fill(scoreValNum)
      : null;

    const fig = [
      {
        type: "candlestick",
        x: ts,
        open: data.map((d) => d.open),
        high: data.map((d) => d.high),
        low: data.map((d) => d.low),
        close: data.map((d) => d.close),
        increasing: { line: { color: "#00ffb3" } },
        decreasing: { line: { color: "#ff5c7a" } },
        showlegend: false,
        name: symbol,
        xaxis: "x",
        yaxis: "y",
      },
      {
        type: "scatter",
        mode: "lines",
        x: ts,
        y: data.map((d) => d.ema),
        line: { color: "#ffd166", width: 2 },
        name: "EMA",
        xaxis: "x",
        yaxis: "y",
      },
    ];

    if (scoreSeries && scoreSeries.length) {
      fig.push({
        type: "scatter",
        mode: "lines",
        x: ts,
        y: scoreSeries,
        line: { color: "#61c0ff", width: 2 },
        name: "Score",
        xaxis: "x2",
        yaxis: "y2",
      });
    }

    const shapes = [];
    if (entryPrice) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: entryPrice,
        y1: entryPrice,
        line: { color: side && side.toUpperCase() === "SHORT" ? "#ff5c7a" : "#00ffb3", dash: "dot" },
      });
    }
    if (exitPrice) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: exitPrice,
        y1: exitPrice,
        line: { color: "#ccc", dash: "dot" },
      });
    }

    const layout = {
      grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
      margin: { l: 50, r: 20, t: 10, b: 70 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: {
        showgrid: false,
        zeroline: false,
        color: "rgba(255,255,255,0.7)",
        rangeslider: { visible: false },
        domain: [0, 1],
        anchor: "y",
      },
      yaxis: {
        showgrid: true,
        gridcolor: "rgba(255,255,255,0.08)",
        color: "rgba(255,255,255,0.7)",
        domain: [0.35, 1],
      },
      xaxis2: {
        showgrid: false,
        zeroline: false,
        color: "rgba(255,255,255,0.7)",
        anchor: "y2",
        domain: [0, 1],
        matches: "x",
      },
      yaxis2: {
        showgrid: true,
        gridcolor: "rgba(255,255,255,0.08)",
        color: "rgba(255,255,255,0.7)",
        domain: [0, 0.25],
        title: "Score",
        rangemode: "tozero",
        zeroline: true,
      },
      shapes,
    };
    Plotly.newPlot(el, fig, layout, { responsive: true, displayModeBar: false });
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
    const equityHistory = (state && state.equity_history) || [];
    const feed = meta.feed || {};
    lastFeed = {
      last_ts_ms: feed.last_ts_ms || null,
      ws_msgs: feed.ws_msgs || 0,
      symbols: feed.symbols || 0,
    };
    signalsCache = meta.signals || [];
    // sempre recomeça mostrando 20 quando o snapshot muda
    signalsShown = Math.min(20, signalsCache.length || 20);
    // se o sinal selecionado não estiver mais na lista, limpa detalhes/plot
    if (
      selectedSignal &&
      !signalsCache.find((s) => s.symbol === selectedSignal.symbol && String(s.ts_ms) === String(selectedSignal.ts_ms))
    ) {
      selectedSignal = null;
      clearTradeChart();
      resetTradeDetail();
    }

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
    setText("signals-badge", String(signalsCache.length || 0));
    if (equityHistory.length) {
      const last = equityHistory[equityHistory.length - 1];
      setText("equity-badge", utcTimeShort(last.ts_utc || state.generated_at_utc));
    } else {
      setText("equity-badge", "—");
    }

    // positions table
    const pt = document.getElementById("positions-tbody");
    if (pt) {
      pt.innerHTML = "";
      for (const p of positions) {
        const tr = document.createElement("tr");
        tr.dataset.symbol = p.symbol || "";
        tr.dataset.entryTsUtc = p.entry_ts_utc || "";
        tr.dataset.side = p.side || "";
        tr.dataset.qty = p.qty || "";
        tr.dataset.price = p.entry_price || "";
        tr.dataset.exitPrice = p.mark_price || "";
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
      for (const t of trades.slice(0, 50)) {
        const pnl = t.pnl_usd == null ? null : Number(t.pnl_usd);
        const tr = document.createElement("tr");
        tr.dataset.symbol = t.symbol || "";
        tr.dataset.entryTsUtc = t.entry_ts_utc || t.ts_utc || "";
        tr.dataset.exitTsUtc = t.exit_ts_utc || "";
        tr.dataset.side = t.side || t.action || "";
        tr.dataset.qty = t.qty || "";
        tr.dataset.price = t.price || "";
        tr.dataset.exitPrice = t.exit_price || "";
        tr.dataset.pnlUsd = t.pnl_usd || "";
        tr.dataset.pnlPct = t.pnl_pct || "";
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

    // equity chart
    const eqLabels = equityHistory.map((p) => utcTimeShort(p.ts_utc));
    const eqValues = equityHistory.map((p) => Number(p.equity_usd || 0));
    ensureEquityChart(eqLabels, eqValues);

    // feed status
    const lastTsIso = feed.last_ts_ms ? tsMsToIso(feed.last_ts_ms) : "—";
    setText("feed-last", lastTsIso === "—" ? "—" : utcTimeShort(lastTsIso));
    const delay = feed.delay_sec == null ? null : Number(feed.delay_sec);
    const delayStr = delay == null ? "—" : delay.toFixed(1) + "s";
    setText("feed-delay", delayStr);
    setText("feed-ws-msgs", String(feed.ws_msgs || 0));
    setText("feed-symbols", String(feed.symbols || 0));
    const badge = document.getElementById("feed-delay-badge");
    if (badge) {
      const cls = delay != null && delay < 10 ? "success" : delay != null && delay < 60 ? "warning" : "danger";
      badge.className = "badge text-bg-" + cls;
      badge.textContent = delayStr;
    }

    // signals ranking
    renderSignals();
  }

  function renderSignals() {
    const stt = document.getElementById("signals-tbody");
    if (!stt) return;
    stt.innerHTML = "";
    const slice = signalsCache.slice(0, signalsShown);
    for (const s of slice) {
      const tr = document.createElement("tr");
      tr.dataset.symbol = s.symbol || "";
      tr.dataset.price = s.price || "";
      tr.dataset.tsMs = s.ts_ms || "";
      tr.dataset.score = s.score || "";
      tr.innerHTML = `
        <td class="fw-semibold">${s.symbol || "—"}</td>
        <td class="text-end">${Number(s.score || 0).toFixed(4)}</td>
        <td class="text-end">${fmtUSD.format(Number(s.price || 0))}</td>
        <td class="text-end">${s.ts_ms ? utcTimeShort(tsMsToIso(s.ts_ms)) : "—"}</td>
      `;
      stt.appendChild(tr);
    }
    if (!slice.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="4" class="text-center text-secondary py-3">Sem sinais no momento</td>`;
      stt.appendChild(tr);
    }
  }

  function clearTradeChart() {
    const el = document.getElementById("trade-chart");
    if (el) el.innerHTML = "";
  }

  function resetTradeDetail() {
    setText("trade-detail-badge", "Selecione um trade ou símbolo");
    setText("trade-info-symbol", "—");
    setText("trade-info-side", "—");
    setText("trade-info-qty", "—");
    setText("trade-info-entry", "—");
    setText("trade-info-exit", "—");
    setText("trade-info-pnl", "—");
    setText("trade-info-time", "—");
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
      // se houver trade selecionado, tenta re-renderizar detalhes com dados mais atuais
      if (selectedTrade) {
        await loadTradeDetail(selectedTrade, false);
      } else if (selectedSignal) {
        await loadSignalDetail(selectedSignal);
      }
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
    }, 1000);
  }

  function stopPolling() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  async function loadTradeDetail(info, updateSelection = true) {
    if (updateSelection) {
      selectedTrade = info;
      selectedSignal = null;
    }
    setText("trade-detail-badge", info.symbol || "—");
    setText("trade-info-symbol", info.symbol || "—");
    setText("trade-info-side", info.side || "—");
    setText("trade-info-qty", info.qty ? fmtNum.format(Number(info.qty)) : "—");
    setText("trade-info-entry", info.price ? fmtUSD.format(Number(info.price)) : "—");
    setText("trade-info-exit", info.exit_price ? fmtUSD.format(Number(info.exit_price)) : "—");
    setText("trade-info-pnl", info.pnl_usd ? fmtPnl(Number(info.pnl_usd)) : "—");
    setText("trade-info-time", info.entry_ts || info.exit_ts || "—");

    const endMs = info.exit_ts
      ? Date.parse(info.exit_ts)
      : info.entry_ts
      ? Date.parse(info.entry_ts) + 15 * 60 * 1000
      : Date.now();
    try {
      const res = await fetchJson(
        `/api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${endMs}`
      );
      if (res && res.ok && res.data) {
        renderTradeChart(
          info.symbol,
          res.data,
          info.price ? Number(info.price) : null,
          info.exit_price ? Number(info.exit_price) : null,
          info.side || "",
          info.score != null ? Number(info.score) : null
        );
      }
    } catch (e) {
      // ignora falha de chart
    }
  }

  async function loadSignalDetail(info) {
    selectedSignal = info;
    selectedTrade = null;
    setText("trade-detail-badge", info.symbol || "—");
    setText("trade-info-symbol", info.symbol || "—");
    setText("trade-info-side", "—");
    setText("trade-info-qty", "—");
    setText("trade-info-entry", info.price ? fmtUSD.format(Number(info.price)) : "—");
    setText("trade-info-exit", "—");
    setText("trade-info-pnl", "—");
    setText("trade-info-time", info.ts_ms ? utcTimeShort(tsMsToIso(info.ts_ms)) : "—");
    try {
      const endMs = info.ts_ms || Date.now();
      let res = await fetchJson(
        `/api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${endMs}&lookback_min=60`
      );
      if (!res || !res.ok || !res.data || !res.data.length) {
        res = await fetchJson(
          `/api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${Date.now()}&lookback_min=120`
        );
      }
      if (res && res.ok && res.data) {
        renderTradeChart(info.symbol, res.data, null, null, "", info.score != null ? Number(info.score) : null);
      }
    } catch (e) {
      // ignora falha de chart
    }
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

    const tt = document.getElementById("trades-tbody");
    if (tt) {
      tt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        const info = {
          symbol: row.dataset.symbol,
          side: row.dataset.side || row.dataset.action || "",
          qty: row.dataset.qty || "",
          entry_ts: row.dataset.entryTsUtc || row.dataset.ts || "",
          exit_ts: row.dataset.exitTsUtc || "",
          price: row.dataset.price || "",
          exit_price: row.dataset.exitPrice || "",
          pnl_usd: row.dataset.pnlUsd || "",
          pnl_pct: row.dataset.pnlPct || "",
        };
        await loadTradeDetail(info);
      });
    }

    const pt = document.getElementById("positions-tbody");
    if (pt) {
      pt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        const info = {
          symbol: row.dataset.symbol,
          side: row.dataset.side || "",
          qty: row.dataset.qty || "",
          entry_ts: row.dataset.entryTsUtc || "",
          price: row.dataset.price || "",
          exit_price: row.dataset.exitPrice || "",
        };
        await loadTradeDetail(info);
      });
    }

    const stt = document.getElementById("signals-tbody");
    if (stt) {
      stt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        const info = {
          symbol: row.dataset.symbol,
          price: row.dataset.price || "",
          ts_ms: row.dataset.tsMs || "",
          score: row.dataset.score || "",
        };
        await loadSignalDetail(info);
      });
    }

    const btnMore = document.getElementById("signals-load-more");
    if (btnMore) {
      btnMore.addEventListener("click", () => {
        signalsShown = Math.min(signalsCache.length, signalsShown + 20);
        renderSignals();
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

  // atualiza delay do feed a cada segundo usando o last_ts_ms atual
  setInterval(function () {
    if (!lastFeed.last_ts_ms) return;
    const nowSec = Date.now() / 1000;
    const delay = Math.max(0, nowSec - Number(lastFeed.last_ts_ms) / 1000);
    const delayStr = delay.toFixed(1) + "s";
    setText("feed-delay", delayStr);
    const badge = document.getElementById("feed-delay-badge");
    if (badge) {
      const cls = delay < 10 ? "success" : delay < 60 ? "warning" : "danger";
      badge.className = "badge text-bg-" + cls;
      badge.textContent = delayStr;
    }
  }, 1000);
})();
