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

  function apiUrl(path) {
    const clean = String(path || "").replace(/^\/+/, "");
    const base = window.location.pathname.replace(/\/[^/]*$/, "/");
    return base.replace(/\/+$/, "/") + clean;
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

  function fmtSec(v, digits = 1) {
    if (v == null || isNaN(v)) return "-";
    return Number(v).toFixed(digits) + "s";
  }

  function fmtSecInt(v) {
    if (v == null || isNaN(v)) return "-";
    return String(Math.max(0, Math.round(Number(v)))) + "s";
  }

  function tradeSideLabel(raw) {
    if (!raw) return "â€”";
    const s = String(raw).toUpperCase();
    if (s === "BUY") return "LONG";
    if (s === "SELL") return "SHORT";
    return s;
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

  function fmtDuration(entryIso, exitIso) {
    if (!entryIso || !exitIso) return "—";
    try {
      const entryMs = Date.parse(entryIso);
      const exitMs = Date.parse(exitIso);
      if (isNaN(entryMs) || isNaN(exitMs)) return "—";
      const diffMs = exitMs - entryMs;
      if (diffMs < 0) return "—";
      const totalSeconds = Math.floor(diffMs / 1000);
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = totalSeconds % 60;
      if (hours > 0) {
        return `${hours}h ${minutes}m`;
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
      } else {
        return `${seconds}s`;
      }
    } catch (e) {
      return "—";
    }
  }

  // Atualiza badge de latência em tempo real (chamado a cada segundo)
  function updateLatencyBadge() {
    const badge = document.getElementById("feed-delay-badge");
    const delayEl = document.getElementById("feed-delay");

    if (!badge) return;

    // Calcula delay baseado no last_ts_ms armazenado (tick em tempo real)
    let delay = null;
    if (lastFeed.last_ts_ms && lastFeed.last_ts_ms > 0) {
      const nowMs = Date.now();
      // Ajuste: candle fecha ~60s após o timestamp base
      delay = (nowMs - lastFeed.last_ts_ms) / 1000 - 60;
    }

    const delayVal = delay != null ? Math.max(0, delay) : null;
    const delayStr = fmtSecInt(delayVal);
    badge.textContent = delayStr;
    if (delayEl) delayEl.textContent = delayStr;

    // Cores: verde < 60s, amarelo 60-70s, vermelho > 70s
    if (delayVal != null) {
      const cls = delayVal < 60 ? "success" : delayVal <= 70 ? "warning" : "danger";
      badge.className = "badge text-bg-" + cls;
    } else {
      badge.className = "badge text-bg-secondary";
    }
  }


  let paused = false;
  let timer = null;
  let chart = null;
  let eqChart = null;
  let selectedTrade = null;
  let selectedSignal = null;
  let selectedPosition = null;
  let lastFeed = { last_ts_ms: null, ws_msgs: 0, symbols: 0, delay_sec: null };
  let signalsCache = [];
  let signalsShown = 20;
  let tradesCache = [];
  let tradesShown = 50;
  let selectedLookbackMin = 60;
  let selectedLookbackMinTrades = 60;
  let selectedLookbackMinPositions = 60;
  let selectedEqRange = "1d";
  let lastChartUpdateMs = 0;
  let lastChartKey = "";
  let lastChartDataTs = 0;
  let activeTab = "overview";
  let latencyTimer = null; // Timer para atualizar latência em tempo real
  let pendingChartPayload = null;
  let pendingTradesChartPayload = null;
  let pendingPositionsChartPayload = null;
  let lastTradesChartUpdateMs = 0;
  let lastTradesChartKey = "";
  let lastTradesChartDataTs = 0;
  let lastPositionsChartUpdateMs = 0;
  let lastPositionsChartKey = "";
  let lastPositionsChartDataTs = 0;
  let sysCpuChart = null;
  let sysGpuChart = null;
  let sysTempChart = null;
  let sysHistory = [];
  let sysTimer = null;

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
    const HOUR = 60 * 60 * 1000;
    const MIN = 60 * 1000;

    // Define intervalo de agregação baseado no range
    // Objetivo: ~200-500 pontos no gráfico
    let aggregationMs = MIN; // default 1min
    if (range === "1d") {
      minMs = nowMs - DAY;
      aggregationMs = 5 * MIN; // 5min -> ~288 pontos/dia
    } else if (range === "1w") {
      minMs = nowMs - 7 * DAY;
      aggregationMs = 30 * MIN; // 30min -> ~336 pontos/semana
    } else if (range === "1m") {
      minMs = nowMs - 30 * DAY;
      aggregationMs = 2 * HOUR; // 2h -> ~360 pontos/mês
    } else if (range === "1y") {
      minMs = nowMs - 365 * DAY;
      aggregationMs = DAY; // 1day -> ~365 pontos/ano
    } else {
      // "all" - usa todos os dados, agrega baseado na quantidade
      const totalPoints = history.length;
      if (totalPoints > 1000) aggregationMs = DAY;
      else if (totalPoints > 500) aggregationMs = 4 * HOUR;
      else if (totalPoints > 200) aggregationMs = HOUR;
      else aggregationMs = MIN;
    }

    // Filtra por range de tempo
    const filtered = history.filter((p) => {
      if (!p.ts_utc) return false;
      const ms = Date.parse(p.ts_utc);
      return isFinite(ms) && ms >= minMs;
    });
    const arr = filtered.length ? filtered : history;

    // Agrega os dados por bucket de tempo
    const buckets = new Map();
    for (const p of arr) {
      const ms = Date.parse(p.ts_utc);
      if (!isFinite(ms)) continue;
      const bucketKey = Math.floor(ms / aggregationMs) * aggregationMs;
      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, { sum: 0, count: 0, lastTs: p.ts_utc });
      }
      const bucket = buckets.get(bucketKey);
      bucket.sum += Number(p.equity_usd || 0);
      bucket.count += 1;
      bucket.lastTs = p.ts_utc;
    }

    // Converte buckets para arrays ordenados
    const sortedKeys = Array.from(buckets.keys()).sort((a, b) => a - b);
    const labels = [];
    const values = [];
    for (const key of sortedKeys) {
      const bucket = buckets.get(key);
      const avgEquity = bucket.count > 0 ? bucket.sum / bucket.count : 0;
      // Formata label baseado no range
      const d = new Date(key);
      let label;
      if (range === "1y" || aggregationMs >= DAY) {
        label = d.toISOString().slice(5, 10); // MM-DD
      } else if (aggregationMs >= HOUR) {
        label = d.toISOString().slice(5, 16).replace("T", " "); // MM-DD HH:mm
      } else {
        label = d.toISOString().slice(11, 16); // HH:mm
      }
      labels.push(label);
      values.push(avgEquity);
    }

    return { labels, values };
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

  function ensureSysChart(canvasId, label, color, labels, values, existing) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return existing;
    const ctx = canvas.getContext("2d");
    if (!ctx) return existing;
    if (existing) {
      existing.data.labels = labels;
      existing.data.datasets[0].data = values;
      existing.update();
      return existing;
    }
    return new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label,
            data: values,
            borderColor: color,
            backgroundColor: "rgba(255,255,255,0.04)",
            tension: 0.2,
            fill: true,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "rgba(255,255,255,0.6)" } },
          y: { ticks: { color: "rgba(255,255,255,0.6)" } },
        },
      },
    });
  }

  const SYS_WINDOW_MS = 5 * 60 * 1000;

  function updateSystemHistory(sys) {
    if (!sys || !sys.ts_utc) return;
    const last = sysHistory.length ? sysHistory[sysHistory.length - 1] : null;
    if (last && last.ts_utc === sys.ts_utc) return;
    sysHistory.push(sys);
    const nowMs = Date.now();
    sysHistory = sysHistory.filter((r) => {
      const ms = Date.parse(r.ts_utc || "");
      return isFinite(ms) && nowMs - ms <= SYS_WINDOW_MS;
    });
  }

  function renderSystem(sys) {
    if (!sys || !sys.ts_utc) return;
    updateSystemHistory(sys);

    const cpu = sys.cpu_pct != null ? Number(sys.cpu_pct) : null;
    const gpu = sys.gpu_pct != null ? Number(sys.gpu_pct) : null;
    const temp = sys.temp_c != null ? Number(sys.temp_c) : null;

    setText("sys-cpu", cpu != null ? cpu.toFixed(1) + "%" : "—");
    setText("sys-gpu", gpu != null ? gpu.toFixed(1) + "%" : "—");
    setText("sys-temp", temp != null ? temp.toFixed(1) + "C" : "—");
    setText("sys-sample", sys.ts_utc ? utcTimeShort(sys.ts_utc) : "—");
    setText("sys-badge", sys.ts_utc ? utcTimeShort(sys.ts_utc) : "—");

    const labels = sysHistory.map((r) => utcTimeShort(r.ts_utc));
    const cpuVals = sysHistory.map((r) => Number(r.cpu_pct || 0));
    const gpuVals = sysHistory.map((r) => Number(r.gpu_pct || 0));
    const tempVals = sysHistory.map((r) => Number(r.temp_c || 0));

    sysCpuChart = ensureSysChart("sys-cpu-chart", "CPU", "#00ffb3", labels, cpuVals, sysCpuChart);
    sysGpuChart = ensureSysChart("sys-gpu-chart", "GPU", "#7c5cff", labels, gpuVals, sysGpuChart);
    sysTempChart = ensureSysChart("sys-temp-chart", "Temp", "#ff5c7a", labels, tempVals, sysTempChart);
  }

  async function loadSystemHistory() {
    try {
      const res = await fetchJson(apiUrl("api/system?limit=300"));
      if (!res || !res.ok || !Array.isArray(res.data) || !res.data.length) return;
      const nowMs = Date.now();
      sysHistory = res.data.filter((r) => {
        const ms = Date.parse(r.ts_utc || "");
        return isFinite(ms) && nowMs - ms <= SYS_WINDOW_MS;
      });
      const last = sysHistory[sysHistory.length - 1];
      renderSystem(last);
    } catch (e) {
      // ignore
    }
  }

  async function pollSystemOnce() {
    try {
      const res = await fetchJson(apiUrl("api/system?limit=1"), 4000);
      if (!res || !res.ok || !Array.isArray(res.data) || !res.data.length) return;
      const last = res.data[res.data.length - 1];
      renderSystem(last);
    } catch (e) {
      // ignore
    }
  }

  function startSystemPolling() {
    if (sysTimer) clearInterval(sysTimer);
    sysTimer = setInterval(pollSystemOnce, 1000);
  }

  function isChartVisible() {
    const el = document.getElementById("trade-chart");
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function flushPendingChart() {
    if (!pendingChartPayload || !isChartVisible()) return;
    const payload = pendingChartPayload;
    pendingChartPayload = null;
    renderTradeChart(
      payload.symbol,
      payload.data,
      payload.entryPrice,
      payload.exitPrice,
      payload.side,
      payload.opts
    );
  }

  function renderTradeChart(symbol, data, entryPrice, exitPrice, side, opts = {}) {
    const el = document.getElementById("trade-chart");
    if (!el || !data || !data.length) return false;
    if (!isChartVisible()) {
      pendingChartPayload = { symbol, data, entryPrice, exitPrice, side, opts };
      return false;
    }
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
    if (opts.entryTsMs) {
      const xIso = new Date(Number(opts.entryTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#00ffb3", width: 1, dash: "dash" },
      });
    }
    if (opts.exitTsMs) {
      const xIso = new Date(Number(opts.exitTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#ff5c7a", width: 1, dash: "dash" },
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
    return true;
  }

  // ========== TRADES CHART (closed trades - linha verde compra, vermelha venda) ==========
  function isTradesChartVisible() {
    const el = document.getElementById("trade-chart-trades");
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function renderTradesChart(symbol, data, entryPrice, exitPrice, side, opts = {}) {
    const el = document.getElementById("trade-chart-trades");
    if (!el || !data || !data.length) return false;
    if (!isTradesChartVisible()) {
      pendingTradesChartPayload = { symbol, data, entryPrice, exitPrice, side, opts };
      return false;
    }
    lastTradesChartDataTs = Math.max(...data.map((d) => Number(d.ts) || 0), 0);
    const ts = data.map((d) => new Date(d.ts).toISOString());
    const lows = data.map((d) => Number(d.low));
    const highs = data.map((d) => Number(d.high));
    const minLow = Math.min(...lows);
    const maxHigh = Math.max(...highs);
    const span = Math.max(1e-9, maxHigh - minLow);
    const pad = span === 0 ? Math.max(Math.abs(maxHigh) * 0.02, 0.0001) : Math.max(span * 0.1, Math.abs(maxHigh) * 0.001);
    const tickFmt = maxHigh >= 100000 ? ",.0f" : maxHigh >= 10000 ? ",.1f" : maxHigh >= 1 ? ",.4f" : ".6f";

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

    // EMA from entry to exit
    if (entryPrice != null && !isNaN(entryPrice)) {
      const emaSpan = Number(opts.emaSpan || 55);
      const offsetPct = Number(opts.emaOffsetPct || 0);
      const entryTsMs = opts.entryTsMs ? Number(opts.entryTsMs) : null;
      const exitTsMs = opts.exitTsMs ? Number(opts.exitTsMs) : null;
      const alpha = emaSpan > 0 ? 2.0 / (emaSpan + 1) : 0.0;
      const emaSeries = new Array(data.length).fill(null);
      let startIdx = 0;
      if (entryTsMs) {
        const foundIdx = data.findIndex((d) => Number(d.ts) >= entryTsMs);
        if (foundIdx >= 0) startIdx = foundIdx;
      }
      if (emaSpan > 0) {
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
    // Linha VERDE para compra (entry)
    if (entryPrice != null && !isNaN(entryPrice)) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: entryPrice,
        y1: entryPrice,
        line: { color: "#00ffb3", width: 2, dash: "dot" },
      });
    }
    // Linha VERMELHA para venda (exit)
    if (exitPrice != null && !isNaN(exitPrice)) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: exitPrice,
        y1: exitPrice,
        line: { color: "#ff5c7a", width: 2, dash: "dot" },
      });
    }
    if (opts.entryTsMs) {
      const xIso = new Date(Number(opts.entryTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#00ffb3", width: 1, dash: "dash" },
      });
    }
    if (opts.exitTsMs) {
      const xIso = new Date(Number(opts.exitTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#ff5c7a", width: 1, dash: "dash" },
      });
    }

    const layout = {
      margin: { l: 50, r: 20, t: 10, b: 50 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: { showgrid: false, zeroline: false, color: "rgba(255,255,255,0.7)", rangeslider: { visible: false } },
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
    };
    Plotly.newPlot(el, fig, layout, { responsive: true, displayModeBar: false });
    return true;
  }

  // ========== POSITIONS CHART (realtime - linha verde entrada, cinza preço atual) ==========
  function isPositionsChartVisible() {
    const el = document.getElementById("position-chart");
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function renderPositionsChart(symbol, data, entryPrice, markPrice, side, opts = {}) {
    const el = document.getElementById("position-chart");
    if (!el || !data || !data.length) return false;
    if (!isPositionsChartVisible()) {
      pendingPositionsChartPayload = { symbol, data, entryPrice, markPrice, side, opts };
      return false;
    }
    lastPositionsChartDataTs = Math.max(...data.map((d) => Number(d.ts) || 0), 0);
    const ts = data.map((d) => new Date(d.ts).toISOString());
    const lows = data.map((d) => Number(d.low));
    const highs = data.map((d) => Number(d.high));
    const minLow = Math.min(...lows);
    const maxHigh = Math.max(...highs);
    const span = Math.max(1e-9, maxHigh - minLow);
    const pad = span === 0 ? Math.max(Math.abs(maxHigh) * 0.02, 0.0001) : Math.max(span * 0.1, Math.abs(maxHigh) * 0.001);
    const tickFmt = maxHigh >= 100000 ? ",.0f" : maxHigh >= 10000 ? ",.1f" : maxHigh >= 1 ? ",.4f" : ".6f";

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

    // EMA from entry (ongoing, no exit)
    if (entryPrice != null && !isNaN(entryPrice)) {
      const emaSpan = Number(opts.emaSpan || 55);
      const offsetPct = Number(opts.emaOffsetPct || 0);
      const entryTsMs = opts.entryTsMs ? Number(opts.entryTsMs) : null;
      const alpha = emaSpan > 0 ? 2.0 / (emaSpan + 1) : 0.0;
      const emaSeries = new Array(data.length).fill(null);
      let startIdx = 0;
      if (entryTsMs) {
        const foundIdx = data.findIndex((d) => Number(d.ts) >= entryTsMs);
        if (foundIdx >= 0) startIdx = foundIdx;
      }
      if (emaSpan > 0) {
        let emaVal = Number(entryPrice) * (1.0 - offsetPct);
        for (let i = startIdx; i < data.length; i++) {
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
    // Linha VERDE para entrada
    if (entryPrice != null && !isNaN(entryPrice)) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: entryPrice,
        y1: entryPrice,
        line: { color: "#00ffb3", width: 2, dash: "dot" },
      });
    }
    // Linha CINZA para preço atual (ainda não vendeu)
    if (markPrice != null && !isNaN(markPrice)) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: markPrice,
        y1: markPrice,
        line: { color: "#888888", width: 2, dash: "dash" },
      });
    }
    if (opts.entryTsMs) {
      const xIso = new Date(Number(opts.entryTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#00ffb3", width: 1, dash: "dash" },
      });
    }

    const layout = {
      margin: { l: 50, r: 20, t: 10, b: 50 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: { showgrid: false, zeroline: false, color: "rgba(255,255,255,0.7)", rangeslider: { visible: false } },
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
    };
    Plotly.newPlot(el, fig, layout, { responsive: true, displayModeBar: false });
    return true;
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
    let system = meta.system || {};
    lastFeed = {
      last_ts_ms: feed.last_ts_ms || null,
      ws_msgs: feed.ws_msgs || 0,
      symbols: feed.symbols || 0,
      delay_sec: feed.delay_sec != null ? Number(feed.delay_sec) : null,
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

    // positions table - skip re-render if expanded row exists (avoids flicker)
    const pt = document.getElementById("positions-tbody");
    if (pt && !pt.querySelector(".expanded-chart-row")) {
      pt.innerHTML = "";
      for (const p of positions) {
        const tr = document.createElement("tr");
        tr.dataset.symbol = p.symbol || "";
        tr.dataset.entryTsUtc = p.entry_ts_utc || "";
        tr.dataset.side = p.side || "";
        tr.dataset.qty = p.qty || "";
        tr.dataset.price = p.entry_price || "";
        tr.dataset.exitPrice = p.mark_price || "";
        tr.dataset.pnlUsd = p.pnl_usd || "";
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

    tradesCache = trades;
    const filteredTrades = tradesCache.filter(t => {
      const status = (t.status || "").toUpperCase();
      const hasExit = !!t.exit_ts_utc;
      return status === "CLOSED" || hasExit;
    });
    if (tradesShown > filteredTrades.length) {
      tradesShown = Math.min(tradesShown, filteredTrades.length || 50);
    }

    // trades table - skip re-render if expanded row exists (avoids flicker)
    const tt = document.getElementById("trades-tbody");
    if (tt && !tt.querySelector(".expanded-chart-row")) {
      tt.innerHTML = "";
      // Filtra apenas trades FECHADOS - tem status=CLOSED ou tem exit_ts_utc
      for (const t of filteredTrades.slice(0, tradesShown)) {
        const pnl = t.pnl_usd == null ? null : Number(t.pnl_usd);
        const duration = fmtDuration(t.entry_ts_utc || t.ts_utc, t.exit_ts_utc);
        const tr = document.createElement("tr");
        tr.dataset.symbol = t.symbol || "";
        tr.dataset.entryTsUtc = t.entry_ts_utc || t.ts_utc || "";
        tr.dataset.exitTsUtc = t.exit_ts_utc || "";
        tr.dataset.side = t.side || t.action || "";
        tr.dataset.qty = t.qty || "";
        tr.dataset.price = t.price || t.entry_price || "";
        tr.dataset.entryPrice = t.entry_price || t.price || "";
        tr.dataset.exitPrice = t.exit_price || "";
        tr.dataset.pnlUsd = t.pnl_usd || "";
        tr.dataset.pnlPct = t.pnl_pct || "";

        const entryPriceVal = Number(t.entry_price || t.price || 0);
        const sideLabel = tradeSideLabel(t.side || t.action);

        tr.innerHTML = `
          <td class="text-secondary small">${utcTimeShort(t.entry_ts_utc || t.ts_utc)}</td>
          <td class="fw-bold">${t.symbol || "—"}</td>
          <td><span class="badge text-bg-secondary">${sideLabel}</span></td>
          <td class="text-end text-nowrap text-secondary small">${duration}</td>
          <td class="text-end text-nowrap">${fmtNum.format(Number(t.qty || 0))}</td>
          <td class="text-end text-nowrap">${fmtUSD.format(entryPriceVal)}</td>
          <td class="text-end fw-bold ${pnlClass(pnl == null ? 0 : pnl)}">${pnl == null ? "—" : fmtPnl(pnl)}</td>
        `;
        tt.appendChild(tr);
      }
      if (!filteredTrades.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="7" class="text-center text-secondary py-4">Sem trades recentes</td>`;
        tt.appendChild(tr);
      }
    }
    const btnMoreTrades = document.getElementById("trades-load-more");
    if (btnMoreTrades) {
      const canShowMore = filteredTrades.length > tradesShown;
      btnMoreTrades.disabled = !canShowMore;
      btnMoreTrades.classList.toggle("disabled", !canShowMore);
      btnMoreTrades.textContent = canShowMore ? "Carregar mais" : "Sem mais trades";
    }

    // allocation chart
    const labels = Object.keys(allocation);
    const values = labels.map((k) => Number(allocation[k] || 0));
    ensureChart(labels, values);

    // equity chart
    const eqFiltered = filterEquity(equityHistory, selectedEqRange);
    ensureEquityChart(eqFiltered.labels, eqFiltered.values);

    // feed status - armazena last_ts_ms para cálculo em tempo real
    if (feed.last_ts_ms) {
      lastFeed.last_ts_ms = feed.last_ts_ms;
    }
    if (feed.delay_sec != null && !isNaN(feed.delay_sec)) {
      lastFeed.delay_sec = Number(feed.delay_sec);
    }
    lastFeed.ws_msgs = feed.ws_msgs || 0;
    lastFeed.symbols = feed.symbols || 0;

    const lastTsIso = lastFeed.last_ts_ms ? tsMsToIso(lastFeed.last_ts_ms) : "—";
    setText("feed-last", lastTsIso === "—" ? "—" : utcTimeShort(lastTsIso));
    setText("feed-ws-msgs", String(lastFeed.ws_msgs || 0));
    setText("feed-symbols", String(lastFeed.symbols || 0));

    const runtime = meta.runtime || {};
    const stageLabel = runtime.stage ? String(runtime.stage) : "-";
    const stageDetail = runtime.detail ? ` (${runtime.detail})` : "";
    setText("proc-stage", stageLabel === "-" ? stageLabel : stageLabel + stageDetail);
    const cycleSec = runtime.last_cycle_sec != null ? Number(runtime.last_cycle_sec) : null;
    const procSec = runtime.last_cycle_proc_sec != null ? Number(runtime.last_cycle_proc_sec) : null;
    const shownProc = procSec != null ? procSec : cycleSec;
    setText("proc-cycle", fmtSecInt(shownProc));
    setText("proc-last", runtime.last_cycle_ts_utc ? utcTimeShort(runtime.last_cycle_ts_utc) : "-");

    // Atualiza latência imediatamente
    updateLatencyBadge();

    // sistema
    if ((!system || !system.ts_utc) && sysHistory.length) {
      system = sysHistory[sysHistory.length - 1];
    }
    renderSystem(system);

    // signals ranking
    renderSignals();
  }

  function renderSignals() {
    const stt = document.getElementById("signals-tbody");
    if (!stt) return;

    // Skip re-render if expanded row exists (avoids flicker)
    if (stt.querySelector(".expanded-chart-row")) return;

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
    pendingChartPayload = null;
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

  // ========== EXPANDABLE ROWS - Gráfico inline dentro da tabela ==========
  let expandedRowId = null;  // ID da linha atualmente expandida
  let expandedRowLookback = 60;  // Lookback atual da linha expandida
  let expandedRowInfo = null;  // Info da linha expandida para re-render
  let expandedRowType = null;  // Tipo da tabela (trades, positions, signals)

  function closeExpandedRow() {
    // Fecha TODAS as linhas expandidas em todas as tabelas
    document.querySelectorAll(".expanded-chart-row").forEach((row) => row.remove());
    document.querySelectorAll("tr.row-expanded").forEach((tr) => tr.classList.remove("row-expanded"));
    expandedRowId = null;
    expandedRowInfo = null;
    expandedRowType = null;
  }

  function createExpandedRow(parentRow, colSpan, symbol) {
    closeExpandedRow();  // Fecha TODAS as rows expandidas

    const chartId = "inline-chart-" + symbol.replace(/[^a-zA-Z0-9]/g, "");
    const tr = document.createElement("tr");
    tr.className = "expanded-chart-row";
    tr.dataset.symbol = symbol;
    tr.innerHTML = `
      <td colspan="${colSpan}" style="padding: 0; background: rgba(0,0,0,0.2);">
        <div class="p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div class="d-flex align-items-center gap-2">
              <span class="fw-semibold">${symbol}</span>
              <div class="btn-group btn-group-sm" role="group">
                <button type="button" class="btn btn-outline-secondary inline-lookback-btn active" data-min="60">1h</button>
                <button type="button" class="btn btn-outline-secondary inline-lookback-btn" data-min="240">4h</button>
                <button type="button" class="btn btn-outline-secondary inline-lookback-btn" data-min="720">12h</button>
                <button type="button" class="btn btn-outline-secondary inline-lookback-btn" data-min="1440">24h</button>
              </div>
            </div>
            <button type="button" class="btn btn-sm btn-outline-secondary inline-close-btn">
              <i class="bi bi-x-lg"></i> Fechar
            </button>
          </div>
          <div id="${chartId}" style="height: 300px; width: 100%;"></div>
          <div class="d-flex gap-4 mt-2 small text-secondary" id="${chartId}-info">
            <span>Carregando...</span>
          </div>
        </div>
      </td>
    `;

    // Insere após a linha pai
    parentRow.after(tr);
    parentRow.classList.add("row-expanded");
    expandedRowId = symbol;
    expandedRowLookback = 60;

    // Evento para fechar
    tr.querySelector(".inline-close-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      closeExpandedRow();
    });

    return { tr, chartId };
  }

  function setupLookbackButtons(tr, chartId, fetchAndRender) {
    console.log("[DEBUG] setupLookbackButtons called, chartId:", chartId);
    const buttons = tr.querySelectorAll(".inline-lookback-btn");
    console.log("[DEBUG] Found", buttons.length, "lookback buttons");
    buttons.forEach((btn, idx) => {
      console.log("[DEBUG] Setting up button", idx, "data-min:", btn.dataset.min);
      btn.addEventListener("click", async (e) => {
        console.log("[DEBUG] Lookback button clicked, lookback:", btn.dataset.min);
        e.stopPropagation();
        tr.querySelectorAll(".inline-lookback-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const lookback = Number(btn.dataset.min || 60);
        console.log("[DEBUG] Lookback value:", lookback, "chartId:", chartId);
        expandedRowLookback = lookback;
        const el = document.getElementById(chartId);
        console.log("[DEBUG] Chart element found:", !!el);
        if (el) el.innerHTML = '<div class="text-center py-4 text-secondary">Carregando...</div>';
        console.log("[DEBUG] Calling fetchAndRender...");
        try {
          await fetchAndRender(lookback);
          console.log("[DEBUG] fetchAndRender completed");
        } catch (err) {
          console.error("[DEBUG] fetchAndRender error:", err);
        }
      });
    });
  }

  function renderInlineChart(chartId, symbol, data, entryPrice, exitPrice, side, opts = {}) {
    console.log("[DEBUG] renderInlineChart called, chartId:", chartId, "data length:", data?.length);
    const el = document.getElementById(chartId);
    if (!el || !data || !data.length) {
      console.warn("[DEBUG] renderInlineChart returning false: missing el or data");
      return false;
    }

    const ts = data.map((d) => new Date(d.ts).toISOString());
    const opens = data.map((d) => Number(d.open));
    const highs = data.map((d) => Number(d.high));
    const lows = data.map((d) => Number(d.low));
    const closes = data.map((d) => Number(d.close));
    const minLow = Math.min(...lows);
    const maxHigh = Math.max(...highs);

    console.log("[DEBUG] Chart Range: MinLow=", minLow, "MaxHigh=", maxHigh, "FirstTS=", ts[0], "LastTS=", ts[ts.length - 1]);

    if (!isFinite(minLow) || !isFinite(maxHigh)) {
      console.error("[DEBUG] Invalid data range (NaN or Inf)");
      el.innerHTML = '<div class="text-center py-4 text-danger">Dados inválidos (NaN)</div>';
      return false;
    }

    const span = Math.max(1e-9, maxHigh - minLow);
    const pad = span === 0 ? Math.max(Math.abs(maxHigh) * 0.02, 0.0001) : Math.max(span * 0.1, Math.abs(maxHigh) * 0.001);
    const tickFmt = maxHigh >= 100000 ? ",.0f" : maxHigh >= 10000 ? ",.1f" : maxHigh >= 1 ? ",.4f" : ".6f";

    const fig = [
      {
        type: "candlestick",
        x: ts,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: { line: { color: "#00ffb3" } },
        decreasing: { line: { color: "#ff5c7a" } },
        showlegend: false,
        name: symbol,
        hoverinfo: "x+text",
        text: data.map((d) => `O: ${Number(d.open).toFixed(6)}<br>H: ${Number(d.high).toFixed(6)}<br>L: ${Number(d.low).toFixed(6)}<br>C: ${Number(d.close).toFixed(6)}`),
      },
    ];

    const shapes = [];
    const showTradeLines = opts.tableType === "trades" || opts.tableType === "positions";

    console.log("[DEBUG] Trade Lines check: show=", showTradeLines, "entry=", entryPrice, "exit=", exitPrice, "emaSpan=", opts.emaSpan);

    // EMA - apenas para trades e positions
    // Relaxed check: allow EMA even if entryPrice is 0 or weird, as long as it's not null/NaN
    if (showTradeLines && entryPrice != null && !isNaN(entryPrice) && opts.emaSpan) {
      const emaSpan = Number(opts.emaSpan || 55);
      const offsetPct = Number(opts.emaOffsetPct || 0);
      const entryTsMs = opts.entryTsMs ? Number(opts.entryTsMs) : null;
      const exitTsMs = opts.exitTsMs ? Number(opts.exitTsMs) : null;
      const alpha = emaSpan > 0 ? 2.0 / (emaSpan + 1) : 0.0;
      const emaSeries = new Array(data.length).fill(null);
      let startIdx = 0;
      if (entryTsMs) {
        const foundIdx = data.findIndex((d) => Number(d.ts) >= entryTsMs);
        if (foundIdx >= 0) startIdx = foundIdx;
      }
      if (emaSpan > 0) {
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
        hoverinfo: "skip",
      });
    }

    // Linha de entrada (verde) - apenas para trades e positions
    if (showTradeLines && entryPrice != null && !isNaN(entryPrice)) {
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: entryPrice,
        y1: entryPrice,
        line: { color: "#00ffb3", width: 2, dash: "dot" },
      });
    }

    // Linha de saída/preço atual - apenas para trades e positions
    if (showTradeLines && exitPrice != null && !isNaN(exitPrice)) {
      const isClosed = opts.tableType === "trades";
      shapes.push({
        type: "line",
        x0: ts[0],
        x1: ts[ts.length - 1],
        y0: exitPrice,
        y1: exitPrice,
        line: { color: isClosed ? "#ff5c7a" : "#888888", width: 2, dash: isClosed ? "dot" : "dash" },
      });
    }
    if (showTradeLines && opts.entryTsMs) {
      const xIso = new Date(Number(opts.entryTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#00ffb3", width: 1, dash: "dash" },
      });
    }
    if (showTradeLines && opts.exitTsMs && opts.tableType === "trades") {
      const xIso = new Date(Number(opts.exitTsMs)).toISOString();
      shapes.push({
        type: "line",
        x0: xIso,
        x1: xIso,
        y0: minLow - pad,
        y1: maxHigh + pad,
        line: { color: "#ff5c7a", width: 1, dash: "dash" },
      });
    }

    const layout = {
      margin: { l: 55, r: 20, t: 10, b: 40 },
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
      hoverlabel: {
        bgcolor: "rgba(0,0,0,0.8)",
        font: { color: "#fff", size: 12 },
      },
      hovermode: "x unified",
    };

    console.log("[DEBUG] Calling Plotly.newPlot...");
    try {
      // Usar newPlot para garantir que destruimos o anterior corretamente ao limpar innerHTML
      el.innerHTML = "";
      Plotly.newPlot(el, fig, layout, { responsive: true, displayModeBar: false });
      console.log("[DEBUG] Plotly.newPlot completed");
    } catch (plotlyErr) {
      console.error("[DEBUG] Plotly.newPlot error:", plotlyErr);
      el.innerHTML = '<div class="text-center py-4 text-danger">Erro Plotly</div>';
      return false;
    }
    return true;
  }

  async function loadInlineChart(row, info, tableType) {
    const colSpan = row.cells.length;
    const symbol = info.symbol || "—";

    // Verifica se já está expandido neste símbolo - fecha
    if (expandedRowId === symbol) {
      closeExpandedRow();
      return;
    }

    const { tr, chartId } = createExpandedRow(row, colSpan, symbol);
    expandedRowInfo = info;
    expandedRowType = tableType;

    const fetchAndRender = async (lookback) => {
      console.log("[DEBUG] fetchAndRender called, lookback:", lookback, "symbol:", symbol);
      const entryTsMs = info.entry_ts ? Date.parse(info.entry_ts) : null;
      const exitTsMs = info.exit_ts ? Date.parse(info.exit_ts) : null;
      const nowMs = Date.now();

      // Para trades fechados: o gráfico termina no exit + 5min de margem
      // Para trades abertos: termina no momento atual
      const isClosed = !!exitTsMs;
      const endMs = isClosed ? exitTsMs + 5 * 60 * 1000 : nowMs;

      // Calcula lookback baseado na duração real do trade (se fechado)
      let effectiveLookback = lookback;
      if (isClosed && entryTsMs) {
        // Para trades fechados, limita o lookback à duração do trade + margem
        const tradeDurationMin = (exitTsMs - entryTsMs) / 60000;
        const maxLookback = Math.ceil(tradeDurationMin * 1.3) + 10; // 30% margem + 10min buffer antes
        effectiveLookback = Math.min(lookback, Math.max(maxLookback, 30)); // mínimo 30min
      }

      console.log("[DEBUG] endMs:", endMs, "entryTsMs:", entryTsMs, "exitTsMs:", exitTsMs, "effectiveLookback:", effectiveLookback);

      const el = document.getElementById(chartId);
      const infoEl = document.getElementById(chartId + "-info");
      console.log("[DEBUG] el found:", !!el, "infoEl found:", !!infoEl);

      const url = apiUrl(`api/ohlc_window?symbol=${encodeURIComponent(symbol)}&end_ms=${endMs}&lookback_min=${effectiveLookback}`);
      console.log("[DEBUG] Fetching URL:", url);

      try {
        const res = await fetchJson(url);
        console.log("[DEBUG] API response ok:", res?.ok, "data length:", res?.data?.length);
        if (res && res.ok && res.data && res.data.length) {
          const entryPrice = info.price ? Number(info.price) : (info.entry_price ? Number(info.entry_price) : null);
          const exitPrice = info.exit_price ? Number(info.exit_price) : (info.mark_price ? Number(info.mark_price) : null);

          const rendered = renderInlineChart(
            chartId,
            symbol,
            res.data,
            entryPrice,
            exitPrice,
            info.side || "",
            {
              emaSpan: res.ema_span,
              emaOffsetPct: res.ema_offset_pct,
              entryTsMs,
              exitTsMs,
              tableType,
            }
          );

          if (!rendered && el) {
            el.innerHTML = '<div class="text-center py-4 text-warning">Erro ao renderizar gráfico</div>';
          }

          // Atualiza info - diferente para cada tipo
          if (infoEl) {
            if (tableType === "trades") {
              const entryStr = entryPrice ? fmtUSD.format(entryPrice) : "—";
              const exitStr = exitPrice ? fmtUSD.format(exitPrice) : "—";
              const pnlStr = info.pnl_usd ? fmtPnl(Number(info.pnl_usd)) : "—";
              infoEl.innerHTML = `
                <span>Entrada: <strong>${entryStr}</strong></span>
                <span>Saída: <strong>${exitStr}</strong></span>
                <span>PnL: <strong class="${pnlClass(Number(info.pnl_usd || 0))}">${pnlStr}</strong></span>
              `;
            } else if (tableType === "positions") {
              const entryStr = entryPrice ? fmtUSD.format(entryPrice) : "—";
              const markStr = exitPrice ? fmtUSD.format(exitPrice) : "—";
              const pnlStr = info.pnl_usd ? fmtPnl(Number(info.pnl_usd)) : "—";
              infoEl.innerHTML = `
                <span>Entrada: <strong>${entryStr}</strong></span>
                <span>Atual: <strong>${markStr}</strong></span>
                <span>PnL: <strong class="${pnlClass(Number(info.pnl_usd || 0))}">${pnlStr}</strong></span>
              `;
            } else {
              // signals - apenas preço e score
              const priceStr = info.price ? fmtUSD.format(Number(info.price)) : "—";
              const scoreStr = info.score ? Number(info.score).toFixed(4) : "—";
              infoEl.innerHTML = `
                <span>Preço: <strong>${priceStr}</strong></span>
                <span>Score: <strong>${scoreStr}</strong></span>
              `;
            }
          }
        } else {
          if (el) el.innerHTML = '<div class="text-center py-4 text-warning">Sem dados OHLC para este período</div>';
          if (infoEl) infoEl.innerHTML = `<span class="text-warning">Sem dados disponíveis</span>`;
        }
      } catch (e) {
        console.error("Erro ao carregar gráfico:", e);
        if (el) el.innerHTML = '<div class="text-center py-4 text-danger">Erro ao carregar dados</div>';
        if (infoEl) infoEl.innerHTML = `<span class="text-danger">Erro: ${e.message}</span>`;
      }
    };

    // Calcule melhor lookback inicial baseado na duração
    let initialLookback = 60;
    const eTs = info.entry_ts ? Date.parse(info.entry_ts) : null;
    if (eTs) {
      const xTs = info.exit_ts ? Date.parse(info.exit_ts) : Date.now();
      const diffMin = (xTs - eTs) / 60000;
      // Margem de segurança de 10%
      const needed = diffMin * 1.1;
      if (needed > 720) initialLookback = 1440;
      else if (needed > 240) initialLookback = 720;
      else if (needed > 60) initialLookback = 240;
    }
    expandedRowLookback = initialLookback;

    // Atualiza botões
    tr.querySelectorAll(".inline-lookback-btn").forEach(btn => {
      const bMin = Number(btn.dataset.min);
      if (bMin === initialLookback) btn.classList.add("active");
      else btn.classList.remove("active");
    });

    // Configura eventos de lookback
    setupLookbackButtons(tr, chartId, fetchAndRender);

    // Carrega inicialmente
    await fetchAndRender(expandedRowLookback);
  }



  async function fetchJson(url, timeoutMs = 30000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { cache: "no-store", signal: controller.signal });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error("http_" + res.status);
      return await res.json();
    } catch (e) {
      clearTimeout(timeoutId);
      if (e.name === "AbortError") {
        throw new Error("timeout");
      }
      throw e;
    }
  }

  async function refreshOnce() {
    try {
      const st = await fetchJson(apiUrl("api/state"));
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

    const entryTsMs = info.entry_ts ? Date.parse(info.entry_ts) : null;
    const exitTsMs = info.exit_ts ? Date.parse(info.exit_ts) : null;
    const signalTsMs = entryTsMs;
    const feedTsMs = Number(lastFeed.last_ts_ms || 0);
    const nowMs = Date.now();

    const anchorA = entryTsMs || signalTsMs || nowMs;
    const anchorB = exitTsMs || signalTsMs || nowMs;
    const baseDurationMs = Math.abs(anchorB - anchorA);
    const baseDurationMin = Math.max(10, Math.ceil(baseDurationMs / 60000) + 10);
    const lookbackUsed = Math.max(selectedLookbackMin, baseDurationMin, 30);

    const endMsCandidates = [
      nowMs,
      feedTsMs,
      exitTsMs || null,
      signalTsMs || null,
      entryTsMs ? entryTsMs + 15 * 60 * 1000 : null,
    ].filter(Boolean);
    const endMs = Math.max(...endMsCandidates);

    const chartKey = `${info.symbol}|${lookbackUsed}|${entryTsMs || ""}|${exitTsMs || ""}|trade`;
    const chartEmpty = !lastChartKey || !lastChartDataTs;
    const tooFresh =
      !force && chartKey === lastChartKey && nowMs - lastChartUpdateMs < 2000 && feedTsMs <= lastChartDataTs && !chartEmpty;
    if (tooFresh) return;

    async function fetchWindow(endMsVal, lb) {
      return fetchJson(
        apiUrl(`api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${endMsVal}&lookback_min=${lb}`)
      );
    }

    try {
      let res = await fetchWindow(endMs, lookbackUsed);
      if (!res || !res.ok || !res.data || !res.data.length) {
        res = await fetchWindow(Date.now(), Math.max(120, lookbackUsed * 2));
      }
      if (res && res.ok && res.data && res.data.length) {
        const rendered = renderTradeChart(
          info.symbol,
          res.data,
          info.price ? Number(info.price) : null,
          info.exit_price ? Number(info.exit_price) : null,
          info.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs,
            exitTsMs,
          }
        );
        if (rendered) {
          lastChartKey = chartKey;
          lastChartUpdateMs = Date.now();
        }
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

    const entryTsMs = detail.entry_ts ? Date.parse(detail.entry_ts) : null;
    const exitTsMs = detail.exit_ts ? Date.parse(detail.exit_ts) : null;
    const signalTsMs = detail.ts_ms ? Number(detail.ts_ms) : entryTsMs;
    const feedTsMs = Number(lastFeed.last_ts_ms || 0);
    const nowMs = Date.now();

    const anchorA = entryTsMs || signalTsMs || nowMs;
    const anchorB = exitTsMs || signalTsMs || nowMs;
    const baseDurationMs = Math.abs(anchorB - anchorA);
    const baseDurationMin = Math.max(10, Math.ceil(baseDurationMs / 60000) + 10);
    const lookbackUsed = Math.max(selectedLookbackMin, baseDurationMin, 30);

    const endMsCandidates = [
      nowMs,
      feedTsMs,
      exitTsMs || null,
      signalTsMs || null,
      entryTsMs ? entryTsMs + 15 * 60 * 1000 : null,
    ].filter(Boolean);
    const endMs = Math.max(...endMsCandidates);

    const chartKey = `${detail.symbol}|${lookbackUsed}|signal|${entryTsMs || signalTsMs || ""}|${exitTsMs || ""}`;
    const chartEmpty = !lastChartKey || !lastChartDataTs;
    const tooFresh =
      !force && chartKey === lastChartKey && nowMs - lastChartUpdateMs < 2000 && feedTsMs <= lastChartDataTs && !chartEmpty;
    if (tooFresh) return;

    async function fetchWindow(endMsVal, lb) {
      return fetchJson(
        apiUrl(`api/ohlc_window?symbol=${encodeURIComponent(detail.symbol)}&end_ms=${endMsVal}&lookback_min=${lb}`)
      );
    }

    try {
      let res = await fetchWindow(endMs, lookbackUsed);
      if (!res || !res.ok || !res.data || !res.data.length) {
        res = await fetchWindow(Date.now(), Math.max(120, lookbackUsed * 2));
      }
      if (res && res.ok && res.data && res.data.length) {
        const useEntryPrice = entryTsMs ? (detail.price ? Number(detail.price) : null) : null;
        const rendered = renderTradeChart(
          detail.symbol,
          res.data,
          useEntryPrice,
          exitTsMs ? Number(detail.exit_price || detail.last_price || null) : null,
          detail.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs,
            exitTsMs,
          }
        );
        if (rendered) {
          lastChartKey = chartKey;
          lastChartUpdateMs = Date.now();
        }
      }
    } catch (e) {
      // ignora falha de chart
    }
  }

  // ========== LOAD TRADE DETAIL FOR TRADES TAB ==========
  async function loadTradeDetailForTrades(info, force = false) {
    selectedTrade = info;
    setText("trade-detail-badge-trades", info.symbol || "—");
    setText("trade-info-symbol-trades", info.symbol || "—");
    setText("trade-info-side-trades", info.side || "—");
    setText("trade-info-qty-trades", info.qty ? fmtNum.format(Number(info.qty)) : "—");
    setText("trade-info-entry-trades", info.price ? fmtUSD.format(Number(info.price)) : "—");
    setText("trade-info-exit-trades", info.exit_price ? fmtUSD.format(Number(info.exit_price)) : "—");
    setText("trade-info-pnl-trades", info.pnl_usd ? fmtPnl(Number(info.pnl_usd)) : "—");

    const entryTsMs = info.entry_ts ? Date.parse(info.entry_ts) : null;
    const exitTsMs = info.exit_ts ? Date.parse(info.exit_ts) : null;
    const feedTsMs = Number(lastFeed.last_ts_ms || 0);
    const nowMs = Date.now();

    const anchorA = entryTsMs || nowMs;
    const anchorB = exitTsMs || nowMs;
    const baseDurationMs = Math.abs(anchorB - anchorA);
    const baseDurationMin = Math.max(10, Math.ceil(baseDurationMs / 60000) + 10);
    const lookbackUsed = Math.max(selectedLookbackMinTrades, baseDurationMin, 30);

    // Fim do período = exitTsMs + margem se existir, senão now
    const endMs = exitTsMs ? exitTsMs + 5 * 60 * 1000 : nowMs;

    const chartKey = `${info.symbol}|${lookbackUsed}|${entryTsMs || ""}|${exitTsMs || ""}|trades`;
    const chartEmpty = !lastTradesChartKey || !lastTradesChartDataTs;
    const tooFresh =
      !force && chartKey === lastTradesChartKey && nowMs - lastTradesChartUpdateMs < 2000 && !chartEmpty;
    if (tooFresh) return;

    try {
      const res = await fetchJson(
        apiUrl(`api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${endMs}&lookback_min=${lookbackUsed}`)
      );
      if (res && res.ok && res.data && res.data.length) {
        const rendered = renderTradesChart(
          info.symbol,
          res.data,
          info.price ? Number(info.price) : null,
          info.exit_price ? Number(info.exit_price) : null,
          info.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs,
            exitTsMs,
          }
        );
        if (rendered) {
          lastTradesChartKey = chartKey;
          lastTradesChartUpdateMs = Date.now();
        }
      }
    } catch (e) {
      // ignora falha
    }
  }

  // ========== LOAD POSITION DETAIL (REALTIME) ==========
  async function loadPositionDetail(info, force = false) {
    selectedPosition = info;
    setText("position-detail-badge", info.symbol || "—");
    setText("position-info-symbol", info.symbol || "—");
    setText("position-info-side", info.side || "—");
    setText("position-info-qty", info.qty ? fmtNum.format(Number(info.qty)) : "—");
    setText("position-info-entry", info.entry_price ? fmtUSD.format(Number(info.entry_price)) : "—");
    setText("position-info-mark", info.mark_price ? fmtUSD.format(Number(info.mark_price)) : "—");
    setText("position-info-pnl", info.pnl_usd ? fmtPnl(Number(info.pnl_usd)) : "—");

    const entryTsMs = info.entry_ts ? Date.parse(info.entry_ts) : null;
    const feedTsMs = Number(lastFeed.last_ts_ms || 0);
    const nowMs = Date.now();

    const lookbackUsed = Math.max(selectedLookbackMinPositions, 60);

    const chartKey = `${info.symbol}|${lookbackUsed}|${entryTsMs || ""}|positions`;
    const chartEmpty = !lastPositionsChartKey || !lastPositionsChartDataTs;
    const stale = nowMs - lastPositionsChartUpdateMs > 5000;
    const tooFresh =
      !force && !stale && chartKey === lastPositionsChartKey && feedTsMs <= lastPositionsChartDataTs && !chartEmpty;
    if (tooFresh) return;

    try {
      const res = await fetchJson(
        apiUrl(`api/ohlc_window?symbol=${encodeURIComponent(info.symbol)}&end_ms=${nowMs}&lookback_min=${lookbackUsed}`)
      );
      if (res && res.ok && res.data && res.data.length) {
        const rendered = renderPositionsChart(
          info.symbol,
          res.data,
          info.entry_price ? Number(info.entry_price) : null,
          info.mark_price ? Number(info.mark_price) : null,
          info.side || "",
          {
            emaSpan: res.ema_span,
            emaOffsetPct: res.ema_offset_pct,
            entryTsMs,
          }
        );
        if (rendered) {
          lastPositionsChartKey = chartKey;
          lastPositionsChartUpdateMs = Date.now();
        }
      }
    } catch (e) {
      // ignora falha
    }
  }


  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-bs-theme") || "dark";
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-bs-theme", next);
    try {
      localStorage.setItem("astra-theme", next);
    } catch (e) { }
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
    if (!isChartVisible()) return;
    if (selectedTrade) {
      const force =
        stale || Number(lastFeed.last_ts_ms || 0) > Number(lastChartDataTs || 0);
      loadTradeDetail(selectedTrade, false, force);
    } else if (selectedSignal) {
      const force =
        stale || Number(lastFeed.last_ts_ms || 0) > Number(lastChartDataTs || 0);
      loadSignalDetail(selectedSignal, force);
    }
    // Atualiza gráfico de posição se visível e selecionado
    if (selectedPosition && isPositionsChartVisible()) {
      const stalePos = nowMs - lastPositionsChartUpdateMs > 5000;
      const forcePos =
        stalePos || Number(lastFeed.last_ts_ms || 0) > Number(lastPositionsChartDataTs || 0);
      if (forcePos) {
        // Atualiza mark_price da posição selecionada
        const st = window.__LAST_STATE__ || {};
        const positions = st.positions || [];
        const updated = positions.find((p) => (p.symbol || "").toUpperCase() === (selectedPosition.symbol || "").toUpperCase());
        if (updated) {
          selectedPosition.mark_price = updated.mark_price || selectedPosition.mark_price;
          selectedPosition.pnl_usd = updated.pnl_usd || selectedPosition.pnl_usd;
        }
        loadPositionDetail(selectedPosition, true);
      }
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

    document.querySelectorAll(".tab-trigger").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.tab || "overview";
        showTab(tab);
      });
    });

    const tt = document.getElementById("trades-tbody");
    if (tt) {
      tt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        // Ignora clique na linha expandida
        if (row.classList.contains("expanded-chart-row")) return;
        const info = {
          symbol: row.dataset.symbol,
          side: row.dataset.side || row.dataset.action || "",
          qty: row.dataset.qty || "",
          entry_ts: row.dataset.entryTsUtc || row.dataset.ts || "",
          exit_ts: row.dataset.exitTsUtc || "",
          price: row.dataset.price || row.dataset.entryPrice || "",
          exit_price: row.dataset.exitPrice || "",
          pnl_usd: row.dataset.pnlUsd || "",
        };
        console.log("[DEBUG] Clicked trade:", info);
        await loadInlineChart(row, info, "trades");
      });
    }

    const pt = document.getElementById("positions-tbody");
    if (pt) {
      pt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        if (row.classList.contains("expanded-chart-row")) return;
        const info = {
          symbol: row.dataset.symbol,
          side: row.dataset.side || "",
          qty: row.dataset.qty || "",
          entry_ts: row.dataset.entryTsUtc || "",
          price: row.dataset.price || "",
          entry_price: row.dataset.price || "",
          mark_price: row.dataset.exitPrice || "",
          pnl_usd: row.dataset.pnlUsd || "",
        };
        await loadInlineChart(row, info, "positions");
      });
    }

    const stt = document.getElementById("signals-tbody");
    if (stt) {
      stt.addEventListener("click", async (ev) => {
        const row = ev.target.closest("tr");
        if (!row || !row.dataset.symbol) return;
        if (row.classList.contains("expanded-chart-row")) return;
        const info = {
          symbol: row.dataset.symbol,
          price: row.dataset.price || "",
          ts_ms: row.dataset.tsMs || "",
          score: row.dataset.score || "",
        };
        await loadInlineChart(row, info, "signals");
      });
    }

    const btnMore = document.getElementById("signals-load-more");
    if (btnMore) {
      btnMore.addEventListener("click", () => {
        signalsShown = Math.min(signalsCache.length, signalsShown + 20);
        renderSignals();
      });
    }
    const btnMoreTrades = document.getElementById("trades-load-more");
    if (btnMoreTrades) {
      btnMoreTrades.addEventListener("click", () => {
        const total = tradesCache.length || 0;
        tradesShown = Math.min(total, tradesShown + 50);
        renderState(window.__LAST_STATE__ || {});
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

    // Lookback buttons para trades
    document.querySelectorAll(".lookback-btn-trades").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = Number(btn.dataset.min || 60);
        selectedLookbackMinTrades = val;
        document.querySelectorAll(".lookback-btn-trades").forEach((b) => {
          b.classList.toggle("active", Number(b.dataset.min || 0) === val);
        });
        if (selectedTrade) loadTradeDetailForTrades(selectedTrade, true);
      });
    });

    // Lookback buttons para positions
    document.querySelectorAll(".lookback-btn-positions").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = Number(btn.dataset.min || 60);
        selectedLookbackMinPositions = val;
        document.querySelectorAll(".lookback-btn-positions").forEach((b) => {
          b.classList.toggle("active", Number(b.dataset.min || 0) === val);
        });
        if (selectedPosition) loadPositionDetail(selectedPosition, true);
      });
    });
  }

  function showTab(tab) {
    activeTab = tab;
    const all = [
      { key: "overview", cls: ".tab-pane-overview" },
      { key: "trades", cls: ".tab-pane-trades" },
      { key: "signals", cls: ".tab-pane-signals" },
      { key: "monitor", cls: ".tab-pane-monitor" },
    ];
    all.forEach((t) => {
      document.querySelectorAll(t.cls).forEach((el) => {
        if (t.key === activeTab) el.classList.remove("d-none");
        else el.classList.add("d-none");
      });
    });
    document.querySelectorAll(".tab-trigger").forEach((btn) => {
      const isActive = (btn.dataset.tab || "") === activeTab;
      btn.classList.toggle("active", isActive);
    });
    document.querySelectorAll(".inline-hidden").forEach((el) => {
      el.classList.add("d-none");
    });
    if (activeTab === "monitor") {
      const charts = [sysCpuChart, sysGpuChart, sysTempChart];
      charts.forEach((c) => {
        if (!c) return;
        try {
          c.resize();
        } catch (e) {}
      });
    }
    const scheduleChartFlush = (attempt = 0) => {
      const el = document.getElementById("trade-chart");
      if (!el) return;
      if (isChartVisible()) {
        flushPendingChart();
        try {
          Plotly.Plots.resize(el);
        } catch (e) { }
        return;
      }
      if (attempt >= 12) return;
      requestAnimationFrame(() => {
        setTimeout(() => scheduleChartFlush(attempt + 1), 120);
      });
    };
    scheduleChartFlush();
  }

  // bootstrap
  wireUI();
  // garante lookback default marcado
  setLookbackButtons(selectedLookbackMin);
  setEqRangeButtons(selectedEqRange);
  showTab(activeTab);
  if (window.__INITIAL_STATE__) {
    window.__LAST_STATE__ = window.__INITIAL_STATE__;
    renderState(window.__INITIAL_STATE__);
    setOnline(true);
    setBadge("health-badge", "OK", "success");
  }
  loadSystemHistory();
  startSystemPolling();
  refreshOnce();
  startPolling();

  // Timer separado para atualizar latência a cada segundo
  if (latencyTimer) clearInterval(latencyTimer);
  latencyTimer = setInterval(updateLatencyBadge, 1000);

  // o delay do feed já é atualizado em renderState e também pelo timer de latência
})();
