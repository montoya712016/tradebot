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
  let selectedLookbackMin = 60;
  let selectedEqRange = "1d";
  let lastChartUpdateMs = 0;
  let lastChartKey = "";
  let lastChartDataTs = 0;

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

  function filterEquity(history, range) {
    if (!history || !history.length) return { labels: [], values: [] };
    const nowMs = Date.now();
    let minMs = 0;
    const DAY = 24 * 60 * 60 * 1000;
    if (range === "1d") minMs = nowMs - DAY;
    else if (range === "1w") minMs = nowMs - 7 * DAY;
    else if (range === "1m") minMs = nowMs - 30 * DAY;
    else if (range === "1y") minMs = nowMs - 365 * DAY;
    const filtered = history.filter((p) => {
      if (!p.ts_utc) return false;
      const ms = Date.parse(p.ts_utc);
      return isFinite(ms) && ms >= minMs;
    });
    const arr = filtered.length ? filtered : history;
    return {
      labels: arr.map((p) => utcTimeShort(p.ts_utc)),
      values: arr.map((p) => Number(p.equity_usd || 0)),
    };
  }

  function setEqRangeButtons(range) {
    selectedEqRange = range;
    document.querySelectorAll(".eq-range-btn").forEach((btn) => {
      const val = btn.dataset.range;
      if (val === selectedEqRange) btn.classList.add("active");
      else btn.classList.remove("active");
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

  function renderTradeChart(symbol, data, entryPrice, exitPrice, side, opts = {}) {
    const el = document.getElementById("trade-chart");
    if (!el || !data || !data.length) return;
    lastChartDataTs = Math.max(...data.map((d) => Number(d.ts) || 0), 0);
    const ts = data.map((d) => new Date(d.ts).toISOString());
    const lows = data.map((d) => Number(d.low));
    const highs = data.map((d) => Number(d.high));
    const minLow = Math.min(...lows);
    const maxHigh = Math.max(...highs);
    const span = Math.max(1e-9, maxHigh - minLow);
    const pad =
      span === 0
        ? Math.max(Math.abs(maxHigh) * 0.02, 0.0001)
        : Math.max(span * 0.1, Math.abs(maxHigh) * 0.001); // padding para não achatar

    const tickFmt =
      maxHigh >= 100000
        ? ",.0f"
        : maxHigh >= 10000
        ? ",.1f"
        : maxHigh >= 1
        ? ",.4f"
        : ".6f";

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
      },
    ];

    if (entryPrice != null && !isNaN(entryPrice)) {
      const span = Number(opts.emaSpan || 0);
      const offsetPct = Number(opts.emaOffsetPct || 0);
      const entryTsMs = opts.entryTsMs ? Number(opts.entryTsMs) : null;
      const exitTsMs = opts.exitTsMs ? Number(opts.exitTsMs) : null;
      if (entryTsMs == null) {
        // sem ts de entrada não traçamos EMA para evitar série contínua errada
        // eslint-disable-next-line no-unused-expressions
        entryPrice = null;
      }
      const alpha = span > 0 ? 2.0 / (span + 1) : 0.0;
      const emaSeries = new Array(data.length).fill(null);
      let startIdx = 0;
      if (entryTsMs) {
        const foundIdx = data.findIndex((d) => Number(d.ts) >= entryTsMs);
        if (foundIdx < 0) {
          // entrada fora do range -> não plota EMA
          entryPrice = null;
        } else {
          startIdx = foundIdx;
        }
      }
      if (span > 0 && entryPrice != null) {
        let emaVal = Number(entryPrice) * (1.0 - offsetPct);
        for (let i = startIdx; i < data.length; i++) {
          if (exitTsMs && Number(data[i].ts) > exitTsMs) break;
          if (i === startIdx) {
            emaSeries[i] = emaVal;
            continue;
          }
          emaVal = emaVal + alpha * (Number(data[i].close) - emaVal);
          emaSeries[i] = emaVal;
        }
      }
      fig.push({
        type: "scatter",
        mode: "lines",
        x: ts,
        y: emaSeries,
        line: { color: "#ffd166", width: 2 },
        name: "EMA",
        showlegend: false,
        connectgaps: false,
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
      margin: { l: 50, r: 20, t: 10, b: 50 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: {
        showgrid: false,
        zeroline: false,
        color: "rgba(255,255,255,0.7)",
        rangeslider: { visible: false },
      },
      yaxis: {
        showgrid: true,
        gridcolor: "rgba(255,255,255,0.08)",
        color: "rgba(255,255,255,0.7)",
        autorange: false,
        range: [minLow - pad, maxHigh + pad],
        tickformat: tickFmt,
        hoverformat: tickFmt,
        fixedrange: false,
      },
      shapes,
      legend: {
        orientation: "h",
        y: -0.25,
        x: 0,
        font: { color: "rgba(255,255,255,0.7)" },
      },
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
    window.__LAST_STATE__ = state;
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
    // ajusta limite apenas se ficou maior que o tamanho atual
    if (signalsShown > signalsCache.length) {
      signalsShown = Math.min(signalsShown, signalsCache.length || 20);
    }
    // se um sinal estiver selecionado, sincroniza com os valores mais recentes
    if (selectedSignal) {
      const updated = signalsCache.find((s) => s.symbol === selectedSignal.symbol);
      if (updated) {
        const changed =
          Number(updated.score || 0) !== Number(selectedSignal.score || 0) ||
          Number(updated.ts_ms || 0) !== Number(selectedSignal.ts_ms || 0) ||
          Number(updated.price || 0) !== Number(selectedSignal.price || 0);
        selectedSignal = updated;
        if (changed) {
          loadSignalDetail(selectedSignal, true);
        }
      }
    }
    // se o sinal selecionado não estiver mais na lista, limpa detalhes/plot
    if (
      selectedSignal &&
      !signalsCache.find((s) => s.symbol === selectedSignal.symbol)
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
    const eqFiltered = filterEquity(equityHistory, selectedEqRange);
    ensureEquityChart(eqFiltered.labels, eqFiltered.values);

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
      tr.dataset.range24h = s.range_24h_pct || "";
      tr.dataset.range1h = s.range_1h_pct || "";
      tr.innerHTML = `
        <td class="fw-semibold">${s.symbol || "—"}</td>
        <td class="text-end">${Number(s.score || 0).toFixed(4)}</td>
        <td class="text-end">${fmtUSD.format(Number(s.price || 0))}</td>
        <td class="text-end">${s.range_24h_pct != null ? Number(s.range_24h_pct).toFixed(2) + "%" : "—"}</td>
        <td class="text-end">${s.range_1h_pct != null ? Number(s.range_1h_pct).toFixed(2) + "%" : "—"}</td>
        <td class="text-end">${s.ts_ms ? utcTimeShort(tsMsToIso(s.ts_ms)) : "—"}</td>
      `;
      stt.appendChild(tr);
    }
    if (!slice.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" class="text-center text-secondary py-3">Sem sinais no momento</td>`;
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
    setText("trade-info-score", "—");
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
      // refresh do gráfico só se passou tempo mínimo
      maybeRefreshChart();
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

  async function loadTradeDetail(info, updateSelection = true, force = false) {
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
    setText("trade-info-score", info.score != null ? Number(info.score).toFixed(4) : "—");
    setText("trade-info-time", info.entry_ts || info.exit_ts || "—");

    const endMsBase = info.exit_ts
      ? Date.parse(info.exit_ts)
      : info.entry_ts
      ? Date.parse(info.entry_ts) + 15 * 60 * 1000
      : Date.now();
    // sempre inclui o instante atual para garantir que o range abranja o Ãºltimo candle
    const endMs = Math.max(Date.now(), endMsBase || 0, Number(lastFeed.last_ts_ms || 0));
    const chartKey = `${info.symbol}|${selectedLookbackMin}|${info.entry_ts || ""}|${info.exit_ts || ""}`;
    const nowMs = Date.now();
    if (!force && chartKey === lastChartKey && nowMs - lastChartUpdateMs < 2000 && lastFeed.last_ts_ms <= lastChartDataTs) {
      return; // evita redesenho constante (mantém tooltip)
    }
    try {
      const res = await fetchJson(
        `/api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${endMs}&lookback_min=${selectedLookbackMin}`
      );
      if (res && res.ok && res.data) {
        renderTradeChart(
          info.symbol,
          res.data,
          info.price ? Number(info.price) : null,
          info.exit_price ? Number(info.exit_price) : null,
          info.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs: info.entry_ts ? Date.parse(info.entry_ts) : null,
            exitTsMs: info.exit_ts ? Date.parse(info.exit_ts) : null,
          }
        );
        lastChartKey = chartKey;
        lastChartUpdateMs = Date.now();
      }
    } catch (e) {
      // ignora falha de chart
    }
  }

  async function loadSignalDetail(info, force = false) {
    selectedSignal = info;
    selectedTrade = null;
    let detail = { ...info };
    // tenta enriquecer com posição aberta ou último trade para habilitar EMA
    const st = window.__LAST_STATE__ || {};
    const positions = st.positions || [];
    const trades = st.trades || [];
    const openPos = positions.find((p) => (p.symbol || "").toUpperCase() === (info.symbol || "").toUpperCase());
    const lastTrade = trades
      .filter((t) => (t.symbol || "").toUpperCase() === (info.symbol || "").toUpperCase())
      .sort((a, b) => Date.parse(b.entry_ts_utc || b.ts_utc || 0) - Date.parse(a.entry_ts_utc || a.ts_utc || 0))[0];
    if (openPos) {
      detail.side = openPos.side || detail.side;
      detail.qty = openPos.qty || detail.qty;
      detail.price = openPos.entry_price || detail.price;
      detail.entry_ts = openPos.entry_ts_utc || detail.entry_ts;
      detail.exit_price = null;
      detail.exit_ts = null;
    } else if (lastTrade) {
      detail.side = lastTrade.side || lastTrade.action || detail.side;
      detail.qty = lastTrade.qty || detail.qty;
      detail.price = lastTrade.entry_price || lastTrade.price || detail.price;
      detail.entry_ts = lastTrade.entry_ts_utc || lastTrade.ts_utc || detail.entry_ts;
      detail.exit_price = lastTrade.exit_price || detail.exit_price;
      detail.exit_ts = lastTrade.exit_ts_utc || detail.exit_ts;
      detail.pnl_usd = lastTrade.pnl_usd || detail.pnl_usd;
    }

    setText("trade-detail-badge", detail.symbol || "—");
    setText("trade-info-symbol", detail.symbol || "—");
    setText("trade-info-side", detail.side || "—");
    setText("trade-info-qty", detail.qty ? fmtNum.format(Number(detail.qty)) : "—");
    setText("trade-info-entry", detail.price ? fmtUSD.format(Number(detail.price)) : "—");
    setText("trade-info-exit", detail.exit_price ? fmtUSD.format(Number(detail.exit_price)) : "—");
    setText("trade-info-pnl", detail.pnl_usd ? fmtPnl(Number(detail.pnl_usd)) : "—");
    setText("trade-info-score", detail.score != null ? Number(detail.score).toFixed(4) : "—");
    const tsLabel = detail.entry_ts || detail.exit_ts || (detail.ts_ms ? utcTimeShort(tsMsToIso(detail.ts_ms)) : "—");
    setText("trade-info-time", tsLabel);

    const chartKey = `${detail.symbol}|${selectedLookbackMin}|signal|${detail.entry_ts || detail.ts_ms || ""}|${detail.exit_ts || ""}`;
    const nowMs = Date.now();
    if (!force && chartKey === lastChartKey && nowMs - lastChartUpdateMs < 2000 && lastFeed.last_ts_ms <= lastChartDataTs) {
      return;
    }
    try {
      const endMsCandidates = [
        detail.exit_ts ? Date.parse(detail.exit_ts) : 0,
        detail.ts_ms || 0,
        Number(lastFeed.last_ts_ms || 0),
        Date.now(),
      ].filter(Boolean);
      const endMs = Math.max(...endMsCandidates, Date.now());
      let res = await fetchJson(
        `/api/ohlc_window?symbol=${encodeURIComponent(detail.symbol)}&end_ms=${endMs}&lookback_min=${selectedLookbackMin}`
      );
      if (!res || !res.ok || !res.data || !res.data.length) {
        res = await fetchJson(
          `/api/ohlc_window?symbol=${encodeURIComponent(detail.symbol)}&end_ms=${Date.now()}&lookback_min=${Math.max(120, selectedLookbackMin)}`
        );
      }
      if (res && res.ok && res.data) {
        const entryTsMs = detail.entry_ts ? Date.parse(detail.entry_ts) : null;
        const useEntryPrice = entryTsMs ? (detail.price ? Number(detail.price) : null) : null;
        renderTradeChart(
          detail.symbol,
          res.data,
          useEntryPrice,
          detail.exit_price ? Number(detail.exit_price) : null,
          detail.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs,
            exitTsMs: detail.exit_ts ? Date.parse(detail.exit_ts) : null,
          }
        );
        lastChartKey = chartKey;
        lastChartUpdateMs = Date.now();
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

  function setLookbackButtons(activeMin) {
    selectedLookbackMin = Number(activeMin);
    document.querySelectorAll(".lookback-btn").forEach((btn) => {
      const val = Number(btn.dataset.min || 0);
      if (val === selectedLookbackMin) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
    // recarrega detalhe atual com novo lookback
    if (selectedTrade) {
      loadTradeDetail(selectedTrade, false, true);
    } else if (selectedSignal) {
      loadSignalDetail(selectedSignal, true);
    }
  }

  function maybeRefreshChart() {
    const nowMs = Date.now();
    const stale = nowMs - lastChartUpdateMs > 5000;
    if (selectedTrade) {
      const force =
        stale || Number(lastFeed.last_ts_ms || 0) > Number(lastChartDataTs || 0);
      loadTradeDetail(selectedTrade, false, force);
    } else if (selectedSignal) {
      const force =
        stale || Number(lastFeed.last_ts_ms || 0) > Number(lastChartDataTs || 0);
      loadSignalDetail(selectedSignal, force);
    }
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

    document.querySelectorAll(".lookback-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = btn.dataset.min || "60";
        setLookbackButtons(Number(val));
      });
    });

    document.querySelectorAll(".eq-range-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = btn.dataset.range || "1d";
        setEqRangeButtons(val);
        // re-render equity chart com novo range
        if (window.__LAST_STATE__) {
          const hist = (window.__LAST_STATE__.equity_history || []).slice();
          const eqFiltered = filterEquity(hist, selectedEqRange);
          ensureEquityChart(eqFiltered.labels, eqFiltered.values);
        }
      });
    });
  }

  // bootstrap
  wireUI();
  // garante lookback default marcado
  setLookbackButtons(selectedLookbackMin);
  setEqRangeButtons(selectedEqRange);
  if (window.__INITIAL_STATE__) {
    window.__LAST_STATE__ = window.__INITIAL_STATE__;
    renderState(window.__INITIAL_STATE__);
    setOnline(true);
    setBadge("health-badge", "OK", "success");
  }
  refreshOnce();
  startPolling();

  // o delay do feed jÃ¡ Ã© atualizado em renderState; evitar segundo loop para nÃ£o ficar "piscando"
})();
