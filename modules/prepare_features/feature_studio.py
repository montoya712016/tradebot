# -*- coding: utf-8 -*-
"""
Feature Studio HTML renderer (shared by stocks/crypto).
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import webbrowser

from plotly.io import to_json


def render_feature_studio(
    *,
    fig,
    panels: list[str],
    panel_to_group: dict[str, str],
    trace_meta: list[dict],
    groups: dict[str, list[dict]],
    title: str = "Feature Studio",
    out_path: str | Path | None = None,
    open_browser: bool = True,
) -> Path:
    """
    Renderiza o HTML interativo do Feature Studio.
    - fig: plotly Figure com todas as séries (mesmo que iniciem ocultas)
    - panels: lista de painéis na ordem das linhas da figura
    - panel_to_group: mapeia painel -> grupo lógico (ex.: "atr" -> "atr")
    - trace_meta: lista de dicts [{i,name,panel,group,type,yaxis}, ...]
    - groups: {group_name: [trace_meta...]}
    """
    out_dir = Path(out_path or (Path(__file__).resolve().parents[2] / "data" / "generated" / "plots"))
    if out_path and Path(out_path).suffix:
        out_dir = Path(out_path).expanduser().resolve()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir / "feature_studio.html"
    if out_dir.suffix == "":
        out_dir = out_dir / "feature_studio.html"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #101720;
      --panel-2: #0f1a26;
      --accent: #56d4ff;
      --accent-2: #f9c784;
      --text: #e6edf3;
      --muted: #8b98a5;
      --stroke: #1f2a36;
      --success: #7ee787;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 800px at 10% -10%, #132033 0%, #0b0f14 45%, #0b0f14 100%);
      min-height: 100vh;
      overflow: hidden;
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      height: 100vh;
      padding: 18px;
    }}
    .sidebar {{
      background: linear-gradient(145deg, var(--panel), var(--panel-2));
      border: 1px solid var(--stroke);
      border-radius: 18px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      animation: slideIn 0.6s ease;
    }}
    .title {{
      font-weight: 700;
      font-size: 20px;
      letter-spacing: 0.4px;
      margin-bottom: 6px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 16px;
    }}
    .control-list {{
      overflow-y: auto;
      padding-right: 6px;
    }}
    .feature-card {{
      border: 1px solid var(--stroke);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 10px;
      background: rgba(9, 13, 19, 0.5);
      animation: fadeUp 0.4s ease;
    }}
    .feature-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .feature-label {{
      font-weight: 600;
      font-size: 14px;
    }}
    .badge {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--stroke);
      color: var(--accent);
    }}
    .feature-select {{
      width: 100%;
      margin-top: 8px;
      background: #0b1118;
      border: 1px solid var(--stroke);
      color: var(--text);
      border-radius: 12px;
      padding: 8px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      min-height: 56px;
      max-height: 160px;
      overflow-y: auto;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }}
    .feature-select::-webkit-scrollbar {{
      width: 8px;
    }}
    .feature-select::-webkit-scrollbar-thumb {{
      background: linear-gradient(180deg, rgba(86,212,255,0.45), rgba(86,212,255,0.08));
      border-radius: 10px;
      border: 1px solid #0b1118;
      box-shadow: inset 0 0 4px rgba(0,0,0,0.35);
    }}
    .feature-select::-webkit-scrollbar-track {{
      background: #0b1118;
      border-radius: 10px;
      box-shadow: inset 0 0 4px rgba(0,0,0,0.45);
    }}
    .feature-select:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .hidden {{ display: none; }}
    .toggle {{
      position: relative;
      width: 44px;
      height: 24px;
      border-radius: 999px;
      background: #1b2633;
      cursor: pointer;
      border: 1px solid var(--stroke);
      transition: background 0.2s ease;
    }}
    .toggle input {{ display: none; }}
    .toggle span {{
      position: absolute;
      top: 2px;
      left: 2px;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #5a6b7a;
      transition: transform 0.2s ease, background 0.2s ease;
    }}
    .toggle input:checked + span {{
      transform: translateX(20px);
      background: var(--accent);
    }}
    .plot-area {{
      background: #0b0f14;
      border: 1px solid var(--stroke);
      border-radius: 18px;
      padding: 8px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      animation: fadeIn 0.6s ease;
    }}
    #plot {{ width: 100%; height: 100%; }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes slideIn {{
      from {{ opacity: 0; transform: translateX(-16px); }}
      to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 980px) {{
      body {{ overflow: auto; }}
      .app {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      .plot-area {{
        min-height: 540px;
      }}
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="title">{title}</div>
      <div class="subtitle">Toggle indicators and pick windows without rerunning.</div>
      <div class="control-list" id="controls"></div>
    </aside>
    <div class="plot-area">
      <div id="plot"></div>
    </div>
  </div>
  <script>
    const fig = {to_json(fig, validate=False)};
    const panels = {json.dumps(panels)};
    const panelToGroup = {json.dumps(panel_to_group)};
    const traces = {json.dumps(trace_meta)};
    const groups = {json.dumps(groups)};

    const plotDiv = document.getElementById("plot");
    plotDiv.style.height = "calc(100vh - 80px)";
    plotDiv.style.minHeight = "680px";
    // guarda cópias base para reconstruir sob demanda
    const baseData = fig.data || [];
    const baseLayout = fig.layout || {{}};
    Plotly.newPlot(plotDiv, baseData, baseLayout, {{responsive: true}});
    Plotly.Plots.resize(plotDiv);

    const groupState = {{}};
    for (const key of Object.keys(groups)) {{
      groupState[key] = {{ enabled: false, selected: [] }};
    }}

    function axisName(prefix, idx) {{
      return idx === 1 ? prefix : prefix + idx;
    }}

    function rebuildPlot() {{
      const data = [];
      let row = 1;
      const priceTraces = traces.filter((t) => t.group === "price");
      if (priceTraces.length === 0) return;

      const cloneTrace = (idx, targetRow) => {{
        const t = JSON.parse(JSON.stringify(baseData[idx]));
        const yName = axisName("y", targetRow);
        const xName = axisName("x", targetRow);
        t.yaxis = yName;
        t.xaxis = xName;
        t.visible = true;
        return t;
      }};

      priceTraces.forEach((tr) => data.push(cloneTrace(tr.i, row)));

      for (const [group, list] of Object.entries(groups)) {{
        const state = groupState[group];
        if (!state.enabled) continue;
        const selected = state.selected;
        const enabledTraces = list.filter(
          (t) => selected.length === 0 || selected.includes(t.name || "")
        );
        if (!enabledTraces.length) continue;
        row += 1;
        enabledTraces.forEach((tr) => data.push(cloneTrace(tr.i, row)));
      }}

      const nRows = row;
      const gap = 0.02;
      const usable = 1.0 - gap * (nRows - 1);
      const h = usable / nRows;
      const newLayout = JSON.parse(JSON.stringify(baseLayout || {{}}));

      Object.keys(newLayout)
        .filter((k) => /^(x|y)axis\\d*$/.test(k))
        .forEach((k) => delete newLayout[k]);

      for (let r = 1; r <= nRows; r++) {{
        const start = 1.0 - r * h - (r - 1) * gap;
        const end = start + h;
        const yName = axisName("yaxis", r);
        const xName = axisName("xaxis", r);
        newLayout[yName] = Object.assign(
          {{}},
          fig.layout[yName] || {{}},
          {{ domain: [start, end], anchor: xName, visible: true }}
        );
        newLayout[xName] = Object.assign(
          {{}},
          fig.layout[xName] || {{}},
          {{ domain: [0, 1], anchor: yName, visible: true, showticklabels: r === nRows }}
        );
      }}

      Plotly.react(plotDiv, data, newLayout, {{responsive: true}});
    }}

    function makeFeatureCard(groupName, entries) {{
      const card = document.createElement("div");
      card.className = "feature-card";
      const row = document.createElement("div");
      row.className = "feature-row";
      const label = document.createElement("div");
      label.className = "feature-label";
      label.textContent = groupName.replace(/_/g, " ");
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = entries.length + " series";
      const toggle = document.createElement("label");
      toggle.className = "toggle";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = false;
      const knob = document.createElement("span");
      toggle.appendChild(checkbox);
      toggle.appendChild(knob);

      row.appendChild(label);
      row.appendChild(badge);
      row.appendChild(toggle);
      card.appendChild(row);

      const select = document.createElement("select");
      select.className = "feature-select hidden";
      select.multiple = true;
      const opts = Array.from(new Set(entries.map((t) => t.name).filter(Boolean))).sort();
      opts.forEach((name) => {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }});
      select.disabled = true;
      card.appendChild(select);

      const actions = document.createElement("div");
      actions.style.display = "flex";
      actions.style.gap = "6px";
      actions.style.marginTop = "6px";
      const btnAll = document.createElement("button");
      btnAll.textContent = "All";
      btnAll.style.flex = "1";
      btnAll.style.padding = "4px 6px";
      btnAll.style.borderRadius = "8px";
      btnAll.style.border = "1px solid var(--stroke)";
      btnAll.style.background = "rgba(86,212,255,0.12)";
      btnAll.style.color = "var(--text)";
      btnAll.style.cursor = "pointer";
      btnAll.onclick = () => {{
        Array.from(select.options).forEach((o) => (o.selected = true));
        groupState[groupName].selected = Array.from(select.selectedOptions).map((o) => o.value);
        rebuildPlot();
      }};
      const btnReset = document.createElement("button");
      btnReset.textContent = "Reset";
      btnReset.style.flex = "1";
      btnReset.style.padding = "4px 6px";
      btnReset.style.borderRadius = "8px";
      btnReset.style.border = "1px solid var(--stroke)";
      btnReset.style.background = "rgba(255,255,255,0.06)";
      btnReset.style.color = "var(--text)";
      btnReset.style.cursor = "pointer";
      btnReset.onclick = () => {{
        Array.from(select.options).forEach((o) => (o.selected = false));
        groupState[groupName].selected = [];
        rebuildPlot();
      }};
      actions.appendChild(btnAll);
      actions.appendChild(btnReset);
      card.appendChild(actions);

      checkbox.addEventListener("change", () => {{
        const on = checkbox.checked;
        groupState[groupName].enabled = on;
        select.disabled = !on;
        if (on) {{
          select.classList.remove("hidden");
        }} else {{
          select.classList.add("hidden");
        }}
        rebuildPlot();
      }});
      select.addEventListener("change", () => {{
        groupState[groupName].selected = Array.from(select.selectedOptions).map((o) => o.value);
        rebuildPlot();
      }});
      return card;
    }}

    const controls = document.getElementById("controls");
    Object.keys(groups).sort().forEach((group) => {{
      controls.appendChild(makeFeatureCard(group, groups[group]));
    }});
    rebuildPlot();
  </script>
</body>
</html>
"""
    out_dir.write_text(html, encoding="utf-8")
    if open_browser:
        try:
            webbrowser.open(out_dir.as_uri())
        except Exception:
            pass
    return out_dir


__all__ = ["render_feature_studio"]
