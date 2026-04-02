from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class SiteOosSnapshot:
    performance_title: str
    performance_metrics: list[dict[str, str]]
    performance_note: str
    stats: list[dict[str, str]]
    summary_copy: str
    visual_html: str | None
    visual_static_path: str | None
    visual_title: str
    visual_caption: str


def _read_equity_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty or len(df.columns) < 2:
        raise RuntimeError(f"invalid equity csv: {csv_path}")
    dt_col = df.columns[0]
    eq_col = "equity" if "equity" in df.columns else df.columns[1]
    dt = pd.to_datetime(df[dt_col], errors="coerce")
    eq = pd.to_numeric(df[eq_col], errors="coerce")
    ser = pd.Series(eq.values, index=dt).dropna()
    ser = ser[~ser.index.isna()].sort_index()
    if ser.empty:
        raise RuntimeError(f"empty equity series: {csv_path}")
    return ser


def _max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    dd = (equity / peaks) - 1.0
    return float(abs(dd.min())) if len(dd) else 0.0


def _resolve_selected_metric_rows(report_dir: Path, segments_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with segments_csv.open("r", encoding="utf-8", newline="") as fh:
        for seg in csv.DictReader(fh):
            step = str(seg.get("source_step", "") or "").strip()
            label_id = str(seg.get("label_id", "") or "").strip()
            model_id = str(seg.get("model_id", "") or "").strip()
            backtest_id = str(seg.get("backtest_id", "") or "").strip()
            if not step or not label_id or not model_id or not backtest_id:
                continue
            step_csv = report_dir.parent / f"step_{step}d" / "explore_runs.csv"
            if not step_csv.exists():
                continue
            with step_csv.open("r", encoding="utf-8", newline="") as sfh:
                for row in csv.DictReader(sfh):
                    if (
                        str(row.get("label_id", "")).strip() == label_id
                        and str(row.get("model_id", "")).strip() == model_id
                        and str(row.get("backtest_id", "")).strip() == backtest_id
                    ):
                        rows.append(row)
                        break
    return rows


def _weighted_average(rows: list[dict[str, str]], value_key: str, weight_key: str = "trades") -> float | None:
    total_w = 0.0
    total_v = 0.0
    for row in rows:
        try:
            v = float(row.get(value_key, "") or "")
            w = float(row.get(weight_key, "") or "")
        except Exception:
            continue
        if not math.isfinite(v) or not math.isfinite(w) or w <= 0:
            continue
        total_v += v * w
        total_w += w
    if total_w <= 0:
        return None
    return total_v / total_w


def _build_summary_plotly_html(equity: pd.Series) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            line=dict(color="#EEE8DF", width=3.6, shape="hv"),
            hovertemplate="%{x|%Y-%m-%d}<br>Equity %{y:.2f}<extra></extra>",
            name="Equity",
        )
    )
    fig.add_hline(
        y=1.0,
        line_width=1,
        line_dash="dot",
        line_color="rgba(255,255,255,0.14)",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=14, r=10, t=8, b=8),
        height=420,
        showlegend=False,
        font=dict(color="#dbe2ec", family='Inter, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#98a4b5", size=11),
        ticks="",
        showline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.035)",
        zeroline=False,
        tickfont=dict(color="#98a4b5", size=11),
        ticks="",
        showline=False,
        title_text="",
        fixedrange=True,
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )


def build_site_snapshot(fair_root: Path) -> SiteOosSnapshot:
    report_dir = fair_root / "robustness_report"
    equity_csv = report_dir / "walkforward_oos_equity_reuse.csv"
    segments_csv = report_dir / "walkforward_oos_segments_reuse.csv"
    if not equity_csv.exists() or not segments_csv.exists():
        raise FileNotFoundError(f"missing robustness artifacts in {report_dir}")

    equity = _read_equity_series(equity_csv)
    ret_total = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
    max_dd = _max_drawdown(equity)
    date_span = f"{equity.index[0].date()} to {equity.index[-1].date()}"

    selected_rows = _resolve_selected_metric_rows(report_dir, segments_csv)
    pf = _weighted_average(selected_rows, "profit_factor")
    hit = _weighted_average(selected_rows, "win_rate")
    trades = sum(int(float(r.get("trades", "0") or "0")) for r in selected_rows if str(r.get("trades", "")).strip())

    with segments_csv.open("r", encoding="utf-8", newline="") as fh:
        seg_rows = list(csv.DictReader(fh))
    semesters = len(seg_rows)
    avg_sem_ret = sum(float(r.get("ret_total", "0") or 0.0) for r in seg_rows) / semesters if semesters else 0.0

    return SiteOosSnapshot(
        performance_title="Validated performance profile",
        performance_metrics=[
            {
                "label": "Net Return",
                "value": f"{ret_total * 100:+.1f}%",
                "sub": "Net return delivered across the validated out-of-sample window.",
            },
            {
                "label": "Max Drawdown",
                "value": f"{max_dd * 100:.1f}%",
                "sub": "Peak capital drawdown observed across the same out-of-sample profile.",
            },
            {
                "label": "Profit Factor",
                "value": f"{(pf if pf is not None else 0.0):.2f}",
                "sub": "Trade-weighted efficiency across the selected validated sleeves.",
            },
            {
                "label": "Hit Rate",
                "value": f"{(hit if hit is not None else 0.0) * 100:.0f}%",
                "sub": "Winning-trade rate across the stitched validated profile.",
            },
        ],
        performance_note=(
            f"The visual below shows the out-of-sample path of the validated strategy profile from {date_span}."
        ),
        stats=[
            {"label": "OOS semesters", "value": str(semesters)},
            {"label": "Avg return / semester", "value": f"{avg_sem_ret * 100:+.1f}%"},
            {"label": "Selected OOS trades", "value": f"{trades:d}"},
            {"label": "Report source", "value": f"WF {fair_root.name} reuse"},
        ],
        summary_copy=(
            "These numbers come directly from the stored out-of-sample report and are presented here as evidence of a "
            "strategy built for durability, not just attractive in-sample optics."
        ),
        visual_html=_build_summary_plotly_html(equity),
        visual_static_path=None,
        visual_title="Validated OOS equity path",
        visual_caption=f"Walk-forward reuse mode • {date_span}",
    )
