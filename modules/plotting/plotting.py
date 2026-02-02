# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Plotting helpers (plotly).
Centralized for stocks and crypto.
"""

from typing import Any
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import DEFAULT_PLOTLY_COLORS
except Exception as e:  # pragma: no cover
    raise RuntimeError("Este modulo requer plotly (pip install plotly).") from e

DARK_TEMPLATE = "plotly_dark"
DARK_BG = "#0d1117"
DARK_GRID = "#30363d"
DARK_FONT = "#c9d1d9"
FEATURE_PALETTE = [
    "#58a6ff",
    "#ffa657",
    "#7ee787",
    "#ff7b72",
    "#d2a8ff",
    "#39c5bb",
    "#f2cc60",
    "#a371f7",
    "#f97583",
    "#79c0ff",
]
ENTRY_PALETTE = [
    "#c9d1ff",  # light lavender
    "#b0c9ff",
    "#97c0ff",
    "#7fb6ff",
    "#69abff",
    "#559fff",
    "#4491ff",
    "#3a7de6",
    "#3169cc",
    "#2956b3",  # darker for longer windows
]
LONG_PALETTE = [
    "#7ee787",  # light green
    "#4fd18b",
    "#2ea043",
    "#238636",
    "#196c2e",  # darker green
]
SHORT_PALETTE = [
    "#ff7b72",  # light red
    "#f85149",
    "#e5534b",
    "#cc3d3d",
    "#8e2f2f",  # darker red
]

try:
    from prepare_features import pf_config as cfg  # type: ignore[import]
except Exception:
    try:
        from prepare_features import pf_config as cfg  # type: ignore[import]
    except Exception:
        cfg = None  # type: ignore[assignment]


def _get_env_int(name: str) -> int | None:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_series(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    try:
        return df[col].to_numpy(dtype=float, copy=False)
    except Exception:
        return None


def _color_for_name(name: str) -> str:
    try:
        s = str(name)
        if s.startswith("sniper_long_") or s.startswith("sniper_short_"):
            is_short = "short" in s
            import re

            m = re.search(r"(?:_|w)(\\d+)m", s)
            w = int(m.group(1)) if m else 0
            windows = [30, 60, 120, 240, 360]
            try:
                idx = windows.index(w)
            except Exception:
                idx = 0
            palette = SHORT_PALETTE if is_short else LONG_PALETTE
            idx = max(0, min(idx, len(palette) - 1))
            return palette[idx]
        if s.startswith("p_entry_w") and s[9:].isdigit():
            # map window minutes to a light->dark palette
            w = int(s[9:])
            windows = [30, 60, 120, 240, 360]
            try:
                idx = windows.index(w)
            except Exception:
                idx = 0
            idx = max(0, min(idx, len(ENTRY_PALETTE) - 1))
            return ENTRY_PALETTE[idx]
        import hashlib

        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        palette = FEATURE_PALETTE or DEFAULT_PLOTLY_COLORS
        idx = int(h[:8], 16) % len(palette)
        return palette[idx]
    except Exception:
        return (FEATURE_PALETTE or DEFAULT_PLOTLY_COLORS)[0]


def _apply_gap_nan(arr: np.ndarray | None, gap_mask: np.ndarray | None) -> np.ndarray | None:
    if arr is None or gap_mask is None:
        return arr
    try:
        out = np.array(arr, dtype=float, copy=True)
        out[gap_mask] = np.nan
        return out
    except Exception:
        return arr


def _add_line(
    fig,
    *,
    row: int,
    x,
    y,
    name: str,
    dash: str | None = None,
    gap_threshold: timedelta | None = None,
):
    if y is None:
        return
    if gap_threshold is None:
        gap_threshold = timedelta(hours=2)
    x_arr = pd.to_datetime(x).to_numpy()
    y_arr = np.asarray(y, dtype=float)
    gap_threshold64 = np.timedelta64(gap_threshold)
    if len(x_arr) < 2:
        fig.add_trace(
            go.Scatter(x=x_arr, y=y_arr, name=name, mode="lines", line=dict(dash=dash) if dash else None),
            row=row,
            col=1,
        )
        return

    line_color = _color_for_name(name)
    start = 0
    first = True
    for i in range(1, len(x_arr)):
        gap = (x_arr[i] - x_arr[i - 1]) >= gap_threshold64
        nan_break = np.isnan(y_arr[i]) or np.isnan(y_arr[i - 1])
        if gap or nan_break:
            if i - start > 1:
                fig.add_trace(
                    go.Scatter(
                        x=x_arr[start:i],
                        y=y_arr[start:i],
                        name=name,
                        mode="lines",
                        line=dict(color=line_color, dash=dash) if dash else dict(color=line_color),
                        showlegend=first,
                        legendgroup=name,
                    ),
                    row=row,
                    col=1,
                )
                first = False
            start = i
    if len(x_arr) - start > 1:
        fig.add_trace(
            go.Scatter(
                x=x_arr[start:],
                y=y_arr[start:],
                name=name,
                mode="lines",
                line=dict(color=line_color, dash=dash) if dash else dict(color=line_color),
                showlegend=first,
                legendgroup=name,
            ),
            row=row,
            col=1,
        )


def _add_markers(fig, *, row: int, x, y, name: str, color: str, symbol: str):
    fig.add_trace(
        go.Scatter(x=x, y=y, name=name, mode="markers", marker=dict(color=color, symbol=symbol, size=6)),
        row=row,
        col=1,
    )


def _build_panels(flags: dict[str, Any]) -> list[str]:
    panels = ["candles"]
    if flags.get("shitidx"):
        panels.append("shitidx")
    if flags.get("keltner"):
        panels.extend(["keltner_width", "keltner_center", "keltner_pos", "keltner_squeeze"])
    if flags.get("atr"):
        panels.append("atr")
    if flags.get("rsi"):
        panels.append("rsi")
    if flags.get("slope"):
        panels.append("slope")
    if flags.get("vol"):
        panels.append("vol")
    if flags.get("ci"):
        panels.append("ci")
    if flags.get("cum_logret"):
        panels.append("logret")
    if flags.get("cci"):
        panels.append("cci")
    if flags.get("adx"):
        panels.append("adx")
    if flags.get("time_since"):
        panels.extend(["pctmm", "timesince"])
    if flags.get("zlog"):
        panels.append("zlog")
    if flags.get("slope_reserr"):
        panels.append("slope_reserr")
    if flags.get("vol_ratio"):
        panels.append("vol_ratio")
    if flags.get("regime"):
        panels.append("regime")
    if flags.get("liquidity"):
        panels.append("liquidity")
    if flags.get("rev_speed"):
        panels.append("rev_speed")
    if flags.get("vol_z"):
        panels.append("vol_z")
    if flags.get("shadow"):
        panels.append("shadow")
    if flags.get("range_ratio"):
        panels.append("range_ratio")
    if flags.get("runs"):
        panels.append("runs")
    if flags.get("hh_hl"):
        panels.append("hh_hl")
    if flags.get("ema_cross"):
        panels.append("ema_conf")
    if flags.get("breakout"):
        panels.append("breakout")
    if flags.get("mom_short"):
        panels.append("mom_short")
    if flags.get("wick_stats"):
        panels.append("wick_stats")
    if flags.get("label"):
        panels.extend(["long_short_weights", "long_short_returns"])
    return panels


def plot_all(
    df: pd.DataFrame,
    flags: dict[str, Any],
    u_threshold: float = 0.0,  # unused, kept for compatibility
    *,
    candle_sec: int = 60,
    plot_candles: bool = True,
    grey_zone: float | None = None,  # unused, kept for compatibility
    show: bool = True,
    save_path: str | None = None,
    max_points: int | None = None,
    max_candles: int | None = None,
    mark_gaps: bool = True,
    show_price_ema: bool = False,
    price_ema_span: int | None = None,
):
    """
    Plot main chart with optional indicator panels using plotly.
    """
    if df is None or df.empty:
        return None

    if max_points is None:
        max_points = _get_env_int("PF_PLOT_MAX_POINTS")
    if max_candles is None:
        max_candles = _get_env_int("PF_PLOT_MAX_CANDLES")

    if max_points and len(df) > int(max_points):
        step = int(np.ceil(len(df) / float(max_points)))
        df = df.iloc[::step].copy()

    panels = _build_panels(flags or {})
    ratios = [3 if p == "candles" else 1 for p in panels]
    total = float(sum(ratios))
    heights = [r / total for r in ratios]

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
    )

    x = df.index
    gap_mask = None
    if mark_gaps:
        gap_threshold = timedelta(hours=2)
        gap_threshold64 = np.timedelta64(gap_threshold)
        gap_mask = np.zeros(len(x), dtype=bool)
        for i in range(1, len(x)):
            if (x[i] - x[i - 1]) >= gap_threshold64:
                gap_mask[i] = True

    line_df = df
    if gap_mask is not None and gap_mask.any():
        line_df = df.copy()
        try:
            num_cols = line_df.select_dtypes(include=[np.number]).columns
            line_df.loc[gap_mask, num_cols] = np.nan
        except Exception:
            pass
    row_map = {p: i + 1 for i, p in enumerate(panels)}

    # Candles or close line
    if plot_candles and (max_candles is None or len(df) <= int(max_candles)):
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="candles",
            ),
            row=row_map["candles"],
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=x, y=_apply_gap_nan(line_df["close"].to_numpy(dtype=float, copy=False), gap_mask), name="close", mode="lines"),
            row=row_map["candles"],
            col=1,
        )

    # Optional EMA overlay for price (uses close)
    if show_price_ema:
        try:
            span = int(price_ema_span) if price_ema_span else 30
            if isinstance(df.index, pd.DatetimeIndex):
                day_keys = df.index.normalize()
                ema = (
                    df["close"]
                    .groupby(day_keys)
                    .transform(lambda s: s.ewm(span=span, adjust=False).mean())
                    .to_numpy(dtype=float, copy=False)
                )
            else:
                ema = df["close"].ewm(span=span, adjust=False).mean().to_numpy(dtype=float, copy=False)
            ema = _apply_gap_nan(ema, gap_mask)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ema,
                    name=f"ema_{span}",
                    mode="lines",
                    line=dict(color="rgba(255,140,0,0.9)", width=1.0),
                ),
                row=row_map["candles"],
                col=1,
            )
        except Exception:
            pass

    # markers (if present)
    for col, color, symbol in (
        ("exit_entry_top_preview", "green", "triangle-up"),
        ("exit_entry_bot_preview", "red", "triangle-down"),
        ("exit_sell_top_preview", "green", "x"),
        ("exit_sell_bot_preview", "red", "x"),
    ):
        if col in df.columns:
            sel = df[col].astype(bool)
            if sel.any():
                _add_markers(
                    fig,
                    row=row_map["candles"],
                    x=df.index[sel],
                    y=df.loc[sel, "close"],
                    name=col,
                    color=color,
                    symbol=symbol,
                )

    if cfg is not None:
        # shitidx
        if "shitidx" in row_map:
            for s, l in cfg.EMA_PAIRS:
                col = f"shitidx_pct_{s}_{l}"
                _add_line(
                    fig,
                    row=row_map["shitidx"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"{s}/{l}",
                    gap_threshold=gap_threshold,
                )

        # keltner
        if "keltner_width" in row_map:
            for m in cfg.KELTNER_WIDTH_MIN:
                col = f"keltner_halfwidth_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["keltner_width"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"HalfW{m}",
                    gap_threshold=gap_threshold,
                )
        if "keltner_center" in row_map:
            for m in cfg.KELTNER_CENTER_MIN:
                col = f"keltner_center_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["keltner_center"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"Center{m}",
                    gap_threshold=gap_threshold,
                )
        if "keltner_pos" in row_map:
            for m in cfg.KELTNER_POS_MIN:
                col = f"keltner_pos_{m}"
                _add_line(
                    fig,
                    row=row_map["keltner_pos"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"Pos{m}",
                    gap_threshold=gap_threshold,
                )
        if "keltner_squeeze" in row_map:
            for m in cfg.KELTNER_Z_MIN:
                col = f"keltner_width_z_{m}"
                _add_line(
                    fig,
                    row=row_map["keltner_squeeze"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"WidthZ{m}",
                    gap_threshold=gap_threshold,
                )

        # atr
        if "atr" in row_map:
            for m in cfg.ATR_MIN:
                col = f"atr_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["atr"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"ATR{m}",
                    gap_threshold=gap_threshold,
                )

        # rsi
        if "rsi" in row_map:
            for m in cfg.RSI_PRICE_MIN:
                col = f"rsi_price_{m}"
                _add_line(
                    fig,
                    row=row_map["rsi"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"RSI{m}",
                    gap_threshold=gap_threshold,
                )
            for span, m in cfg.RSI_EMA_PAIRS:
                col = f"rsi_ema{span}_{m}"
                _add_line(
                    fig,
                    row=row_map["rsi"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"RSI-EMA{span}_{m}",
                    dash="dash",
                    gap_threshold=gap_threshold,
                )

        # slope
        if "slope" in row_map:
            for m in cfg.SLOPE_MIN:
                col = f"slope_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["slope"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"Slope{m}",
                    gap_threshold=gap_threshold,
                )

        # vol
        if "vol" in row_map:
            for m in cfg.VOL_MIN:
                col = f"vol_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["vol"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"Vol{m}",
                    gap_threshold=gap_threshold,
                )

        # ci
        if "ci" in row_map:
            for m in cfg.CI_MIN:
                col = f"ci_{m}"
                _add_line(
                    fig,
                    row=row_map["ci"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"CI{m}",
                    gap_threshold=gap_threshold,
                )

        # logret
        if "logret" in row_map:
            for m in cfg.LOGRET_MIN:
                col = f"cum_ret_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["logret"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"CumRet{m}",
                    gap_threshold=gap_threshold,
                )

        # cci
        if "cci" in row_map:
            for m in cfg.CCI_MIN:
                col = f"cci_{m}"
                _add_line(
                    fig,
                    row=row_map["cci"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"CCI{m}",
                    gap_threshold=gap_threshold,
                )

        # adx
        if "adx" in row_map:
            for m in cfg.ADX_MIN:
                col = f"adx_{m}"
                _add_line(
                    fig,
                    row=row_map["adx"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"ADX{m}",
                    gap_threshold=gap_threshold,
                )

        # pctmm
        if "pctmm" in row_map:
            for m in cfg.MINMAX_MIN:
                a = f"pct_from_min_{m}"
                b = f"pct_from_max_{m}"
                _add_line(
                    fig,
                    row=row_map["pctmm"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["pctmm"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )

        # timesince
        if "timesince" in row_map:
            for m in cfg.MINMAX_MIN:
                a = f"time_since_min_{m}"
                b = f"time_since_max_{m}"
                _add_line(
                    fig,
                    row=row_map["timesince"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["timesince"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )

        # zlog
        if "zlog" in row_map:
            for m in cfg.ZLOG_MIN:
                col = f"zlog_{m}m"
                _add_line(
                    fig,
                    row=row_map["zlog"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"Zlog{m}m",
                    gap_threshold=gap_threshold,
                )

        # slope_reserr
        if "slope_reserr" in row_map:
            for m in cfg.SLOPE_RESERR_MIN:
                col = f"slope_reserr_pct_{m}"
                _add_line(
                    fig,
                    row=row_map["slope_reserr"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"ResErr{m}m",
                    gap_threshold=gap_threshold,
                )

        # vol_ratio
        if "vol_ratio" in row_map:
            for a, b in cfg.VOL_RATIO_PAIRS:
                col = f"vol_ratio_pct_{a}_{b}"
                _add_line(
                    fig,
                    row=row_map["vol_ratio"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=f"VolRatio {a}/{b}",
                    gap_threshold=gap_threshold,
                )

        # regime
        if "regime" in row_map:
            _add_line(
                fig,
                row=row_map["regime"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "log_volume_ema"), gap_mask),
                name="log_volume_ema",
                gap_threshold=gap_threshold,
            )
            _add_line(
                fig,
                row=row_map["regime"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "liquidity_ratio"), gap_mask),
                name="liquidity_ratio",
                gap_threshold=gap_threshold,
            )

        # liquidity
        if "liquidity" in row_map:
            liq_windows = getattr(cfg, "LIQUIDITY_MIN", None)
            if liq_windows:
                for m in liq_windows:
                    col = f"volume_to_range_ema{m}"
                    _add_line(
                        fig,
                        row=row_map["liquidity"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=col,
                        gap_threshold=gap_threshold,
                    )
            else:
                _add_line(
                    fig,
                    row=row_map["liquidity"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, "volume_to_range_ema1440"), gap_mask),
                    name="volume_to_range_ema1440",
                    gap_threshold=gap_threshold,
                )

        # rev_speed
        if "rev_speed" in row_map:
            for m in cfg.REV_WINDOWS:
                cu = f"rev_speed_up_{m}"
                cd = f"rev_speed_down_{m}"
                _add_line(
                    fig,
                    row=row_map["rev_speed"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, cu), gap_mask),
                    name=cu,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["rev_speed"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, cd), gap_mask),
                    name=cd,
                    gap_threshold=gap_threshold,
                )

        # vol_z
        if "vol_z" in row_map:
            _add_line(
                fig,
                row=row_map["vol_z"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "vol_z"), gap_mask),
                name="vol_z",
                gap_threshold=gap_threshold,
            )
            _add_line(
                fig,
                row=row_map["vol_z"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "signed_vol_z"), gap_mask),
                name="signed_vol_z",
                gap_threshold=gap_threshold,
            )

        # shadow
        if "shadow" in row_map:
            _add_line(
                fig,
                row=row_map["shadow"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "shadow_balance"), gap_mask),
                name="shadow_balance",
                gap_threshold=gap_threshold,
            )
            _add_line(
                fig,
                row=row_map["shadow"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "shadow_balance_raw"), gap_mask),
                name="shadow_balance_raw",
                dash="dot",
                gap_threshold=gap_threshold,
            )

        # range_ratio
        if "range_ratio" in row_map:
            rr_pairs = getattr(cfg, "RANGE_RATIO_PAIRS", None)
            if rr_pairs:
                for a, b in rr_pairs:
                    col = f"range_ratio_{a}_{b}"
                    _add_line(
                        fig,
                        row=row_map["range_ratio"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=col,
                        gap_threshold=gap_threshold,
                    )
            else:
                _add_line(
                    fig,
                    row=row_map["range_ratio"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, "range_ratio_60_1440"), gap_mask),
                    name="range_ratio_60_1440",
                    gap_threshold=gap_threshold,
                )

        # runs
        if "runs" in row_map:
            for m in cfg.RUN_WINDOWS_MIN:
                a = f"run_up_cnt_{m}"
                b = f"run_dn_cnt_{m}"
                _add_line(
                    fig,
                    row=row_map["runs"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["runs"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )
            _add_line(
                fig,
                row=row_map["runs"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "run_up_len"), gap_mask),
                name="run_up_len",
                dash="dash",
                gap_threshold=gap_threshold,
            )
            _add_line(
                fig,
                row=row_map["runs"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "run_dn_len"), gap_mask),
                name="run_dn_len",
                dash="dash",
                gap_threshold=gap_threshold,
            )

        # hh_hl
        if "hh_hl" in row_map:
            for m in cfg.HHHL_WINDOWS_MIN:
                a = f"hh_cnt_{m}"
                b = f"hl_cnt_{m}"
                _add_line(
                    fig,
                    row=row_map["hh_hl"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["hh_hl"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )

        # ema_conf
        if "ema_conf" in row_map:
            for s in cfg.EMA_CONFIRM_SPANS_MIN:
                a = f"bars_above_ema_{s}"
                b = f"bars_below_ema_{s}"
                c = f"bars_since_cross_{s}"
                _add_line(
                    fig,
                    row=row_map["ema_conf"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["ema_conf"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["ema_conf"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, c), gap_mask),
                    name=c,
                    gap_threshold=gap_threshold,
                )

        # breakout
        if "breakout" in row_map:
            for n in cfg.BREAK_LOOKBACK_MIN:
                a = f"break_high_{n}"
                b = f"break_low_{n}"
                c = f"bars_since_bhigh_{n}"
                d = f"bars_since_blow_{n}"
                _add_line(
                    fig,
                    row=row_map["breakout"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, a), gap_mask),
                    name=a,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["breakout"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, b), gap_mask),
                    name=b,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["breakout"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, c), gap_mask),
                    name=c,
                    gap_threshold=gap_threshold,
                )
                _add_line(
                    fig,
                    row=row_map["breakout"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, d), gap_mask),
                    name=d,
                    gap_threshold=gap_threshold,
                )

        # mom_short
        if "mom_short" in row_map:
            for a, b in cfg.SLOPE_DIFF_PAIRS_MIN:
                col = f"slope_diff_{a}_{b}"
                _add_line(
                    fig,
                    row=row_map["mom_short"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=col,
                    gap_threshold=gap_threshold,
                )

        # wick_stats
        if "wick_stats" in row_map:
            for m in cfg.WICK_MEAN_WINDOWS_MIN:
                col = f"wick_lower_mean_{m}"
                _add_line(
                    fig,
                    row=row_map["wick_stats"],
                    x=x,
                    y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                    name=col,
                    gap_threshold=gap_threshold,
                )
            _add_line(
                fig,
                row=row_map["wick_stats"],
                x=x,
                y=_apply_gap_nan(_safe_series(line_df, "wick_lower_streak"), gap_mask),
                name="wick_lower_streak",
                dash="dash",
                gap_threshold=gap_threshold,
            )

        # long_short_weights
        if "long_short_weights" in row_map:
            for col in df.columns:
                if col.startswith("sniper_short_weight_") and col.endswith("m"):
                    suf = str(col).replace("sniper_short_weight_", "")
                    name = f"sniper_short_weight_{suf}"
                    _add_line(
                        fig,
                        row=row_map["long_short_weights"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=name,
                        gap_threshold=gap_threshold,
                    )
                elif col.startswith("sniper_long_weight_") and col.endswith("m"):
                    suf = str(col).replace("sniper_long_weight_", "")
                    name = f"sniper_long_weight_{suf}"
                    _add_line(
                        fig,
                        row=row_map["long_short_weights"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=name,
                        gap_threshold=gap_threshold,
                    )
            # labels sao marcadas no candle via linhas verticais; aqui mantemos apenas weights
        # long_short_returns
        if "long_short_returns" in row_map:
            for col in df.columns:
                if col.startswith("sniper_short_ret_pct_") and col.endswith("m"):
                    suf = str(col).replace("sniper_short_ret_pct_", "")
                    name = f"sniper_short_ret_pct_{suf}"
                    _add_line(
                        fig,
                        row=row_map["long_short_returns"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=name,
                        gap_threshold=gap_threshold,
                    )
                elif col.startswith("sniper_long_ret_pct_") and col.endswith("m"):
                    suf = str(col).replace("sniper_long_ret_pct_", "")
                    name = f"sniper_long_ret_pct_{suf}"
                    _add_line(
                        fig,
                        row=row_map["long_short_returns"],
                        x=x,
                        y=_apply_gap_nan(_safe_series(line_df, col), gap_mask),
                        name=name,
                        gap_threshold=gap_threshold,
                    )
    else:
        # fallback: detect columns by prefix (evita dependência circular com pf_config)
        def _plot_prefix(panel: str, prefixes: tuple[str, ...]):
            if panel not in row_map:
                return
            for col in df.columns:
                if any(col.startswith(p) for p in prefixes):
                    _add_line(
                        fig,
                        row=row_map[panel],
                        x=x,
                        y=_safe_series(df, col),
                        name=col,
                        gap_threshold=gap_threshold,
                    )

        _plot_prefix("atr", ("atr_pct_",))
        _plot_prefix("rsi", ("rsi_price_", "rsi_ema"))
        _plot_prefix("slope", ("slope_pct_",))
        _plot_prefix("vol", ("vol_pct_",))
        _plot_prefix("ci", ("ci_",))
        _plot_prefix("logret", ("cum_ret_pct_", "cum_logret_", "logret_"))
        _plot_prefix("keltner_width", ("keltner_halfwidth_pct_",))
        _plot_prefix("keltner_center", ("keltner_center_pct_",))
        _plot_prefix("keltner_pos", ("keltner_pos_",))
        _plot_prefix("keltner_squeeze", ("keltner_width_z_",))
        _plot_prefix("cci", ("cci_",))
        _plot_prefix("adx", ("adx_",))
        _plot_prefix("pctmm", ("pct_from_min_", "pct_from_max_"))
        _plot_prefix("timesince", ("time_since_min_", "time_since_max_"))
        _plot_prefix("zlog", ("zlog_",))
        _plot_prefix("slope_reserr", ("slope_reserr_",))
        _plot_prefix("vol_ratio", ("vol_ratio_",))
        _plot_prefix("regime", ("log_volume_ema", "liquidity_ratio"))
        _plot_prefix("liquidity", ("volume_to_range_ema", "liquidity_"))
        _plot_prefix("rev_speed", ("rev_speed_",))
        _plot_prefix("vol_z", ("vol_z", "signed_vol_z"))
        _plot_prefix("shadow", ("shadow_",))
        _plot_prefix("range_ratio", ("range_ratio_",))
        _plot_prefix("runs", ("run_up_cnt_", "run_dn_cnt_", "run_up_len", "run_dn_len"))
        _plot_prefix("hh_hl", ("hh_cnt_", "hl_cnt_"))
        _plot_prefix("ema_conf", ("bars_above_ema_", "bars_below_ema_", "bars_since_cross_"))
        _plot_prefix("breakout", ("break_high_", "break_low_", "bars_since_bhigh_", "bars_since_blow_"))
        _plot_prefix("mom_short", ("slope_diff_",))
        _plot_prefix("wick_stats", ("wick_lower_", "wick_upper_", "wick_"))
        _plot_prefix("long_short_weights", ("sniper_long_weight_", "sniper_short_weight_"))
        _plot_prefix("long_short_returns", ("sniper_long_ret_pct_", "sniper_short_ret_pct_"))

    # Label markers on candles (sniper_long_label_* == 1 / sniper_short_label_* == 1)
    shapes = []
    if "candles" in row_map:
        try:
            y0 = float(np.nanmin(df["low"].to_numpy(dtype=float, copy=False)))
            y1 = float(np.nanmax(df["high"].to_numpy(dtype=float, copy=False)))
            label_cols = [c for c in df.columns if c.startswith("sniper_long_label_") and c.endswith("m")]
            if label_cols:
                for col in label_cols:
                    suffix = str(col).replace("sniper_long_label_", "")
                    weight_col = f"sniper_long_weight_{suffix}"
                    line_color = _color_for_name(weight_col)
                    sel = df[col].to_numpy(copy=False) == 1
                    if np.any(sel):
                        for i in np.where(sel)[0].tolist():
                            shapes.append(
                                dict(
                                    type="line",
                                    xref="x",
                                    yref="y1",
                                    x0=x[i],
                                    x1=x[i],
                                    y0=y0,
                                    y1=y1,
                                    line=dict(color=line_color, width=1, dash="dash"),
                                    layer="above",
                                )
                            )
            label_cols_short = [c for c in df.columns if c.startswith("sniper_short_label_") and c.endswith("m")]
            if label_cols_short:
                for col in label_cols_short:
                    suffix = str(col).replace("sniper_short_label_", "")
                    weight_col = f"sniper_short_weight_{suffix}"
                    line_color = _color_for_name(weight_col)
                    sel = df[col].to_numpy(copy=False) == 1
                    if np.any(sel):
                        for i in np.where(sel)[0].tolist():
                            shapes.append(
                                dict(
                                    type="line",
                                    xref="x",
                                    yref="y1",
                                    x0=x[i],
                                    x1=x[i],
                                    y0=y0,
                                    y1=y1,
                                    line=dict(color=line_color, width=1, dash="dot"),
                                    layer="above",
                                )
                            )
        except Exception:
            pass

    # Shade market-closed gaps
    if mark_gaps and gap_mask is not None:
        y0 = float(np.nanmin(df["low"].to_numpy(dtype=float, copy=False)))
        y1 = float(np.nanmax(df["high"].to_numpy(dtype=float, copy=False)))
        for i in range(1, len(x)):
            if gap_mask[i]:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y1",
                        x0=x[i - 1],
                        x1=x[i],
                        y0=y0,
                        y1=y1,
                        fillcolor="rgba(255,215,0,0.25)",
                        line=dict(width=0),
                        layer="below",
                    )
                )

    fig.update_layout(
        title="plot",
        xaxis_rangeslider_visible=False,
        template=DARK_TEMPLATE,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=DARK_FONT),
        shapes=shapes,
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID)

    if save_path:
        from plotly.io import write_html

        write_html(fig, file=str(save_path), auto_open=bool(show), include_plotlyjs="cdn")
    elif show:
        fig.show()

    return fig


def plot_backtest_single(
    df: pd.DataFrame,
    *,
    trades,
    equity: np.ndarray,
    p_entry: np.ndarray,
    p_entry_map: dict[str, np.ndarray] | None = None,
    p_entry_long_map: dict[str, np.ndarray] | None = None,
    p_entry_short_map: dict[str, np.ndarray] | None = None,
    p_danger: np.ndarray,
    entry_sig: np.ndarray,
    entry_sig_long: np.ndarray | None = None,
    entry_sig_short: np.ndarray | None = None,
    danger_sig: np.ndarray,
    tau_entry: float,
    tau_entry_long: float | None = None,
    tau_entry_short: float | None = None,
    tau_danger: float,
    title: str = "backtest",
    save_path: str | None = None,
    show: bool = True,
    ema_exit: np.ndarray | None = None,
    mark_gaps: bool = True,
    gap_hours: float = 2.0,
    plot_probs: bool = True,
    plot_signals: bool = True,
    plot_candles: bool = True,
    modebar_undo: bool = True,
) -> go.Figure:
    """
    Plot resultado de backtest single-symbol com candles, probs, sinais e equity.
    """
    if df is None or df.empty:
        raise ValueError("df vazio para plot.")

    idx = pd.to_datetime(df.index)
    gap_mask = None
    if mark_gaps:
        gth = timedelta(hours=float(gap_hours))
        gth64 = np.timedelta64(gth)
        gap_mask = np.zeros(len(idx), dtype=bool)
        for i in range(1, len(idx)):
            if (idx[i] - idx[i - 1]) >= gth64:
                gap_mask[i] = True
    panels = ["candles"]
    if plot_probs:
        panels.append("probs")
    if plot_signals:
        panels.append("signals")
    panels.append("equity")
    ratios = []
    for p in panels:
        if p == "candles":
            ratios.append(3.0 if (plot_probs or plot_signals) else 2.5)
        elif p == "probs":
            ratios.append(1.2)
        elif p == "signals":
            ratios.append(0.8)
        elif p == "equity":
            ratios.append(1.2)
    total = float(sum(ratios))
    heights = [r / total for r in ratios]
    row_map = {p: i + 1 for i, p in enumerate(panels)}

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
    )

    # Candles ou linha de close + EMA exit
    if plot_candles:
        fig.add_trace(
            go.Candlestick(
                x=idx,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="candles",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=_apply_gap_nan(df["close"].to_numpy(dtype=float, copy=False), gap_mask),
                name="close",
                mode="lines",
                line=dict(color="#e6edf3", width=1.0),
            ),
            row=1,
            col=1,
        )
    if ema_exit is not None and len(ema_exit) == len(df):
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=_apply_gap_nan(np.asarray(ema_exit, dtype=float), gap_mask),
                name="ema_exit",
                mode="lines",
                line=dict(color="#ffa657", width=1),
            ),
            row=1,
            col=1,
        )

    # Trades (faixas + marcadores)
    shapes = []
    for t in trades:
        try:
            et = pd.to_datetime(t.entry_ts)
            xt = pd.to_datetime(t.exit_ts)
        except Exception:
            continue
        side = str(getattr(t, "side", "") or "long").lower()
        if side == "short":
            color = "rgba(248,81,73,0.25)"
        else:
            color = "rgba(63,185,80,0.25)"
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y1",
                x0=et,
                x1=xt,
                y0=float(df["low"].min()),
                y1=float(df["high"].max()),
                fillcolor=color,
                line=dict(width=0),
                layer="below",
            )
        )
        # entrada/saida
        side = str(getattr(t, "side", "") or "long").lower()
        if side == "short":
            markers = ((et, "triangle-down", "#ff7b72"), (xt, "triangle-up", "#3fb950"))
        else:
            markers = ((et, "triangle-up", "#3fb950"), (xt, "triangle-down", "#f85149"))
        for ts, marker, mcolor in markers:
            try:
                pos = int(idx.get_indexer([ts], method="nearest")[0])
            except Exception:
                continue
            if pos < 0 or pos >= len(df):
                continue
            fig.add_trace(
                go.Scatter(
                    x=[ts],
                    y=[float(df.iloc[pos]["close"])],
                    mode="markers",
                    marker=dict(symbol=marker, color=mcolor, size=7),
                    name="entry" if marker == "triangle-up" else "exit",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Probabilidades
    if plot_probs:
        if p_entry_long_map or p_entry_short_map:
            for name, arr in (p_entry_long_map or {}).items():
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=_apply_gap_nan(np.asarray(arr, dtype=float), gap_mask),
                        name=f"p_entry_long_{name}",
                        mode="lines",
                        line=dict(color="#3fb950", width=1),
                    ),
                    row=row_map["probs"],
                    col=1,
                )
            for name, arr in (p_entry_short_map or {}).items():
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=_apply_gap_nan(np.asarray(arr, dtype=float), gap_mask),
                        name=f"p_entry_short_{name}",
                        mode="lines",
                        line=dict(color="#ff7b72", width=1),
                    ),
                    row=row_map["probs"],
                    col=1,
                )
        elif p_entry_map:
            for name, arr in p_entry_map.items():
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=_apply_gap_nan(np.asarray(arr, dtype=float), gap_mask),
                        name=f"p_entry_{name} (lighter=shorter, darker=longer)",
                        mode="lines",
                        line=dict(color=_color_for_name(f"p_entry_{name}"), width=1),
                    ),
                    row=row_map["probs"],
                    col=1,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=_apply_gap_nan(np.asarray(p_entry, dtype=float), gap_mask),
                    name="p_entry",
                    mode="lines",
                    line=dict(color="#58a6ff", width=1),
                ),
                row=row_map["probs"],
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=[float(tau_entry_long if tau_entry_long is not None else tau_entry)] * len(idx),
                name="tau_entry_long",
                mode="lines",
                line=dict(color="#3fb950", dash="dash", width=1),
            ),
            row=row_map["probs"],
            col=1,
        )
        if tau_entry_short is not None:
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=[float(tau_entry_short)] * len(idx),
                    name="tau_entry_short",
                    mode="lines",
                    line=dict(color="#ff7b72", dash="dash", width=1),
                ),
                row=row_map["probs"],
                col=1,
            )
        if np.any(np.asarray(p_danger, dtype=float)):
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=_apply_gap_nan(np.asarray(p_danger, dtype=float), gap_mask),
                    name="p_danger",
                    mode="lines",
                    line=dict(color="#ff7b72", width=1),
                ),
                row=row_map["probs"],
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=[float(tau_danger)] * len(idx),
                    name="tau_danger",
                    mode="lines",
                    line=dict(color="#ff7b72", dash="dash", width=1),
                ),
                row=row_map["probs"],
                col=1,
            )

    # Sinais (0/1)
    if plot_signals:
        if entry_sig_long is not None or entry_sig_short is not None:
            if entry_sig_long is not None:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=_apply_gap_nan(entry_sig_long.astype(float), gap_mask),
                        name="entry_long",
                        mode="lines",
                        line=dict(color="#3fb950", width=1),
                        fill=None,
                    ),
                    row=row_map["signals"],
                    col=1,
                )
            if entry_sig_short is not None:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=_apply_gap_nan(entry_sig_short.astype(float), gap_mask),
                        name="entry_short",
                        mode="lines",
                        line=dict(color="#ff7b72", width=1),
                        fill=None,
                    ),
                    row=row_map["signals"],
                    col=1,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=_apply_gap_nan(entry_sig.astype(float), gap_mask),
                    name="entry_ok",
                    mode="lines",
                    line=dict(color="#58a6ff", width=1),
                    fill=None,
                ),
                row=row_map["signals"],
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=_apply_gap_nan(danger_sig.astype(float), gap_mask),
                name="danger",
                mode="lines",
                line=dict(color="#ff7b72", width=1),
            ),
            row=row_map["signals"],
            col=1,
        )
        fig.update_yaxes(range=[-0.1, 1.1], row=row_map["signals"], col=1)

    # Equity
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=_apply_gap_nan(np.asarray(equity, dtype=float), gap_mask),
            name="equity",
            mode="lines",
            line=dict(color="#58a6ff", width=1.5),
        ),
        row=row_map["equity"],
        col=1,
    )

    # Faixas de mercado fechado (gaps > gap_hours)
    if mark_gaps and gap_mask is not None:
        y0 = float(np.nanmin(df["low"].to_numpy(dtype=float, copy=False)))
        y1 = float(np.nanmax(df["high"].to_numpy(dtype=float, copy=False)))
        for i in range(1, len(idx)):
            if gap_mask[i]:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y1",
                        x0=idx[i - 1],
                        x1=idx[i],
                        y0=y0,
                        y1=y1,
                        fillcolor="rgba(255,215,0,0.25)",
                        line=dict(width=0),
                        layer="below",
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template=DARK_TEMPLATE,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=DARK_FONT),
        shapes=shapes,
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID)

    config = None
    if modebar_undo:
        # Add a reset view button (acts as undo for pan/zoom).
        config = {"displaylogo": False, "modeBarButtonsToAdd": ["resetScale2d"]}

    if save_path:
        from plotly.io import write_html

        Path(save_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        write_html(fig, file=str(save_path), auto_open=bool(show), include_plotlyjs="cdn", config=config)
    elif show:
        fig.show(config=config)

    return fig
