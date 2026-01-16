# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from .pf_config import (
    EMA_PAIRS,
    KELTNER_WIDTH_MIN, KELTNER_CENTER_MIN, KELTNER_POS_MIN, KELTNER_Z_MIN,
    ATR_MIN, RSI_PRICE_MIN, RSI_EMA_PAIRS, SLOPE_MIN, VOL_MIN, CI_MIN,
    LOGRET_MIN, CCI_MIN, ADX_MIN, MINMAX_MIN, ZLOG_MIN, SLOPE_RESERR_MIN,
    VOL_RATIO_PAIRS, REV_WINDOWS,
    # novos
    RUN_WINDOWS_MIN, HHHL_WINDOWS_MIN, EMA_CONFIRM_SPANS_MIN, BREAK_LOOKBACK_MIN,
    SLOPE_DIFF_PAIRS_MIN, WICK_MEAN_WINDOWS_MIN,
)

def _safe_legend(ax, **kwargs) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        if "loc" not in kwargs:
            kwargs["loc"] = "upper left"
        ax.legend(**kwargs)


def plot_all(
    df: pd.DataFrame,
    flags: dict,
    u_threshold: float = 0.0,
    *,
    candle_sec: int = 60,
    plot_candles: bool = True,
    grey_zone: float | None = None,
    show: bool = True,
    max_points: int | None = None,
    max_candles: int | None = None,
):
    """Plot principal com paineis opcionais baseados em flags.

    - u_threshold/grey_zone: reservado (painéis de utilidade removidos)
    - candle_sec: largura do candle em segundos quando plotando velas
    - plot_candles: True=velas; False=linha de close
    - show: chama plt.show() ao final
    """
    panels = ["candles"]
    if flags.get("shitidx"):    panels.append("shitidx")
    if flags.get("keltner"):
        panels.extend(["keltner_width", "keltner_center", "keltner_pos", "keltner_squeeze"])
    if flags.get("atr"):        panels.append("atr")
    if flags.get("rsi"):        panels.append("rsi")
    if flags.get("slope"):      panels.append("slope")
    if flags.get("vol"):        panels.append("vol")
    if flags.get("ci"):         panels.append("ci")
    if flags.get("cum_logret"): panels.append("logret")
    if flags.get("cci"):        panels.append("cci")
    if flags.get("adx"):        panels.append("adx")
    if flags.get("time_since"): panels.extend(["pctmm", "timesince"])
    if flags.get("zlog"):         panels.append("zlog")
    if flags.get("slope_reserr"): panels.append("slope_reserr")
    if flags.get("vol_ratio"):    panels.append("vol_ratio")
    if flags.get("regime"):       panels.append("regime")
    if flags.get("liquidity"):    panels.append("liquidity")
    if flags.get("rev_speed"):    panels.append("rev_speed")
    if flags.get("vol_z"):        panels.append("vol_z")
    if flags.get("shadow"):       panels.append("shadow")
    if flags.get("range_ratio"):  panels.append("range_ratio")
    # novos paineis
    if flags.get("runs"):        panels.append("runs")
    if flags.get("hh_hl"):       panels.append("hh_hl")
    if flags.get("ema_cross"):   panels.append("ema_conf")
    if flags.get("breakout"):    panels.append("breakout")
    if flags.get("mom_short"):   panels.append("mom_short")
    if flags.get("wick_stats"):  panels.append("wick_stats")
    if flags.get("label"):
        sniper_cols = {
            "sniper_entry_label",
            "sniper_exit_code",
            "sniper_danger_label",
        }
        if any(col in df.columns for col in sniper_cols):
            panels.extend(["sniper_entry", "sniper_danger"])

    if max_points is None:
        v = os.getenv("PF_PLOT_MAX_POINTS", "").strip()
        max_points = int(v) if v else None
    if max_candles is None:
        v = os.getenv("PF_PLOT_MAX_CANDLES", "").strip()
        max_candles = int(v) if v else None

    if max_points and len(df) > int(max_points):
        step = int(np.ceil(len(df) / float(max_points)))
        df = df.iloc[::step].copy()

    ratios = [3] + [1] * (len(panels) - 1)
    fig, axs = plt.subplots(
        len(panels), 1,
        figsize=(14, 2.5 * sum(ratios)),
        sharex=True,
        gridspec_kw={'hspace': .20, 'height_ratios': ratios},
        constrained_layout=False,
    )
    if len(panels) == 1:
        axs = [axs]

    x_vals = mdates.date2num(df.index.to_numpy())

    for ax, panel in zip(axs, panels):
        if panel == "candles":
            if plot_candles:
                if max_candles and len(df) > int(max_candles):
                    plot_candles = False
            if plot_candles:
                w = candle_sec / 86400
                for ts, row in df.iterrows():
                    t = mdates.date2num(ts)
                    c = 'g' if row.close >= row.open else 'r'
                    ax.plot([t, t], [row.low, row.high], 'k', lw=.7)
                    ax.add_patch(Rectangle(
                        (t - w/2, min(row.open, row.close)),
                        w, abs(row.close - row.open),
                        ec='k', fc=c, lw=.5
                    ))
                if "gap_after" in df.columns:
                    gap_ts = df.index[df["gap_after"].astype(bool)]
                    if len(gap_ts) > 0:
                        x_gap = mdates.date2num(gap_ts.to_pydatetime())
                        for x in x_gap:
                            ax.axvline(x, color="gold", linestyle="--", linewidth=1.0, alpha=0.9)
                        ax.plot([], [], color="gold", linestyle="--", linewidth=1.0, label="gap_after")
                # marcadores adicionais
                try:
                    if "exit_entry_top_preview" in df.columns:
                        ts_et = df.index[df["exit_entry_top_preview"].astype(bool)]
                        if len(ts_et) > 0:
                            ax.plot(mdates.date2num(ts_et.to_pydatetime()), df.loc[ts_et, "close"],
                                    marker='^', linestyle='None', color='green', markersize=5, label='entry_top')
                    if "exit_entry_bot_preview" in df.columns:
                        ts_eb = df.index[df["exit_entry_bot_preview"].astype(bool)]
                        if len(ts_eb) > 0:
                            ax.plot(mdates.date2num(ts_eb.to_pydatetime()), df.loc[ts_eb, "close"],
                                    marker='v', linestyle='None', color='red', markersize=5, label='entry_bot')
                    if "exit_sell_top_preview" in df.columns:
                        ts_st = df.index[df["exit_sell_top_preview"].astype(bool)]
                        if len(ts_st) > 0:
                            ax.plot(mdates.date2num(ts_st.to_pydatetime()), df.loc[ts_st, "close"],
                                    marker='x', linestyle='None', color='green', markersize=5, label='sell_top')
                    if "exit_sell_bot_preview" in df.columns:
                        ts_sb = df.index[df["exit_sell_bot_preview"].astype(bool)]
                        if len(ts_sb) > 0:
                            ax.plot(mdates.date2num(ts_sb.to_pydatetime()), df.loc[ts_sb, "close"],
                                    marker='x', linestyle='None', color='red', markersize=5, label='sell_bot')
                except Exception:
                    pass
                ax.set_title("Candles")
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    seen = set(); h2, l2 = [], []
                    for h, l in zip(handles, labels):
                        if l not in seen:
                            seen.add(l); h2.append(h); l2.append(l)
                    ax.legend(h2, l2, loc="upper left")
            else:
                ax.plot(x_vals, df["close"], color="k", linewidth=0.8, label="close")
                # marcadores adicionais
                try:
                    if "exit_entry_top_preview" in df.columns:
                        ts_et = df.index[df["exit_entry_top_preview"].astype(bool)]
                        if len(ts_et) > 0:
                            ax.plot(mdates.date2num(ts_et.to_pydatetime()), df.loc[ts_et, "close"],
                                    marker='^', linestyle='None', color='green', markersize=5, label='entry_top')
                    if "exit_entry_bot_preview" in df.columns:
                        ts_eb = df.index[df["exit_entry_bot_preview"].astype(bool)]
                        if len(ts_eb) > 0:
                            ax.plot(mdates.date2num(ts_eb.to_pydatetime()), df.loc[ts_eb, "close"],
                                    marker='v', linestyle='None', color='red', markersize=5, label='entry_bot')
                    if "exit_sell_top_preview" in df.columns:
                        ts_st = df.index[df["exit_sell_top_preview"].astype(bool)]
                        if len(ts_st) > 0:
                            ax.plot(mdates.date2num(ts_st.to_pydatetime()), df.loc[ts_st, "close"],
                                    marker='x', linestyle='None', color='green', markersize=5, label='sell_top')
                    if "exit_sell_bot_preview" in df.columns:
                        ts_sb = df.index[df["exit_sell_bot_preview"].astype(bool)]
                        if len(ts_sb) > 0:
                            ax.plot(mdates.date2num(ts_sb.to_pydatetime()), df.loc[ts_sb, "close"],
                                    marker='x', linestyle='None', color='red', markersize=5, label='sell_bot')
                except Exception:
                    pass
                ax.set_title("Close")
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    seen = set(); h2, l2 = [], []
                    for h, l in zip(handles, labels):
                        if l not in seen:
                            seen.add(l); h2.append(h); l2.append(l)
                    ax.legend(h2, l2, loc="upper left")

        elif panel=="shitidx":
            for s, l in EMA_PAIRS:
                col = f"shitidx_pct_{s}_{l}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"{s}/{l}")
            ax.set_title("Shitcoin Index (ΔEMA %)")
            _safe_legend(ax, loc='upper left'); ax.grid()

        elif panel == "keltner_width":
            for m in KELTNER_WIDTH_MIN:
                col = f"keltner_halfwidth_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"HalfW{m}")
            ax.set_title("Keltner Half-Width (%)")
            _safe_legend(ax); ax.grid()

        elif panel == "keltner_center":
            for m in KELTNER_CENTER_MIN:
                col = f"keltner_center_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"Center{m}")
            ax.axhline(0, ls="--", c="gray", lw=0.8)
            ax.set_title("Keltner Center Offset (%)")
            _safe_legend(ax); ax.grid()

        elif panel == "keltner_pos":
            for m in KELTNER_POS_MIN:
                col = f"keltner_pos_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"Pos{m}")
            ax.axhline(0,   ls="--", c="gray", lw=0.8)
            ax.axhline(0.5, ls="--", c="gray", lw=0.8)
            ax.axhline(1.0, ls="--", c="gray", lw=0.8)
            ax.set_ylim(-0.5, 1.5)
            ax.set_title("Keltner Position (0..1; breakout fora)")
            _safe_legend(ax); ax.grid()

        elif panel == "keltner_squeeze":
            any_line = False
            for m in KELTNER_Z_MIN:
                col = f"keltner_width_z_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"WidthZ{m}")
                    any_line = True
            ax.axhline(0,  ls="--", c="gray", lw=0.8)
            ax.axhline(1,  ls="--", c="gray", lw=0.6)
            ax.axhline(-1, ls="--", c="gray", lw=0.6)
            ax.axhline(2,  ls=":",  c="gray", lw=0.6)
            ax.axhline(-2, ls=":",  c="gray", lw=0.6)
            ax.set_title("Keltner Width Z-Score (squeeze)")
            if any_line: _safe_legend(ax)
            ax.grid()

        elif panel=="atr":
            for m in ATR_MIN:
                col = f"atr_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"ATR{m}")
            ax.set_title("ATR (%)"); _safe_legend(ax); ax.grid()

        elif panel == "rsi":
            for m in RSI_PRICE_MIN:
                col = f"rsi_price_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"RSI{m}")
            for span, m in RSI_EMA_PAIRS:
                col = f"rsi_ema{span}_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], '--', label=f"RSI-EMA{span}_{m}")
            ax.axhline(70, ls='--', c='gray'); ax.axhline(30, ls='--', c='gray')
            ax.set_title("RSI"); _safe_legend(ax); ax.grid()

        elif panel == "slope":
            for m in SLOPE_MIN:
                col = f"slope_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"Slope{m}")
            ax.set_title("Slope (%)")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel == "vol":
            for m in VOL_MIN:
                col = f"vol_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"Vol{m}")
            ax.set_title("Volatility (%)")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel=="ci":
            for m in CI_MIN:
                col = f"ci_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"CI{m}")
            ax.set_title("Choppiness"); _safe_legend(ax); ax.grid()

        elif panel=="logret":
            for m in LOGRET_MIN:
                col = f"cum_ret_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"CumRet{m}")
            ax.set_title("Retorno Acumulado (%)"); _safe_legend(ax); ax.grid()

        elif panel=="cci":
            for m in CCI_MIN:
                col = f"cci_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"CCI{m}")
            ax.axhline(100,ls='--',c='gray'); ax.axhline(-100,ls='--',c='gray')
            ax.set_title("CCI"); _safe_legend(ax); ax.grid()

        elif panel=="adx":
            for m in ADX_MIN:
                col = f"adx_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"ADX{m}")
            ax.set_title("ADX"); _safe_legend(ax); ax.grid()

        elif panel=="pctmm":
            for m in MINMAX_MIN:
                a = f"pct_from_min_{m}"; b = f"pct_from_max_{m}"
                if a in df.columns: ax.plot(x_vals, df[a],  label=f"from_min_{m}")
                if b in df.columns: ax.plot(x_vals, df[b],  label=f"from_max_{m}")
            ax.set_title("% from min/max"); _safe_legend(ax); ax.grid()

        elif panel=="timesince":
            for m in MINMAX_MIN:
                a = f"time_since_min_{m}"; b = f"time_since_max_{m}"
                if a in df.columns: ax.plot(x_vals, df[a],  label=f"since_min_{m}")
                if b in df.columns: ax.plot(x_vals, df[b],  label=f"since_max_{m}")
            ax.set_title("Candles since min/max"); _safe_legend(ax); ax.grid()

        elif panel == "zlog":
            for m in ZLOG_MIN:
                col = f"zlog_{m}m"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"Zlog{m}m")
            ax.set_title("Z-Score do log(preço)"); _safe_legend(ax); ax.grid()

        elif panel == "slope_reserr":
            for m in SLOPE_RESERR_MIN:
                col = f"slope_reserr_pct_{m}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"ResErr{m}m")
            ax.set_title("Desvio dos resíduos (OLS) — %"); _safe_legend(ax); ax.grid()

        elif panel == "vol_ratio":
            for a, b in VOL_RATIO_PAIRS:
                col = f"vol_ratio_pct_{a}_{b}"
                if col in df.columns:
                    ax.plot(x_vals, df[col], label=f"VolRatio {a}/{b} (%)")
            ax.axhline(0, ls='--', c='gray', lw=0.8)
            ax.set_title("Razão de Volatilidades (Excesso %)"); _safe_legend(ax); ax.grid()

        elif panel == "regime":
            if "log_volume_ema" in df.columns:
                ax.plot(x_vals, df["log_volume_ema"], label="log_volume_ema", color="tab:blue", alpha=0.9)
            if "liquidity_ratio" in df.columns:
                ax.plot(x_vals, df["liquidity_ratio"], label="liquidity_ratio", color="tab:green", alpha=0.7)
            ax.set_title("Regime de Mercado")
            ax.grid(); _safe_legend(ax, loc="upper left")

        elif panel == "liquidity":
            if "volume_to_range_ema1440" in df.columns:
                ax.plot(x_vals, df["volume_to_range_ema1440"], color="tab:orange", alpha=0.95, label="volume_to_range_ema1440")
            ax.set_title("Liquidez")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel == "rev_speed":
            for m in [30, 60, 120]:
                cu = f"rev_speed_up_{m}"; cd = f"rev_speed_down_{m}"
                if cu in df.columns: ax.plot(x_vals, df[cu], label=cu)
                if cd in df.columns: ax.plot(x_vals, df[cd], label=cd)
            ax.set_title("Reversal Speed (pct/candle): up vs down")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel == "vol_z":
            if "vol_z" in df.columns:
                ax.plot(x_vals, df["vol_z"], color="tab:blue", alpha=0.9, label="vol_z")
            if "signed_vol_z" in df.columns:
                ax.plot(x_vals, df["signed_vol_z"], color="tab:green", alpha=0.7, label="signed_vol_z")
            ax.set_title("Volume Z-score (log-volume)")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel == "shadow":
            if "shadow_balance" in df.columns:
                ax.plot(x_vals, df["shadow_balance"], color="tab:red", alpha=0.9, label="shadow_balance(ema10)")
            if "shadow_balance_raw" in df.columns:
                ax.plot(x_vals, df["shadow_balance_raw"], color="tab:pink", alpha=0.25, label="shadow_balance_raw")
            ax.axhline(0, ls='--', c='gray', lw=0.8)
            ax.set_title("Shadow Balance (upper - lower) / range")
            _safe_legend(ax, loc="upper left"); ax.grid()

        elif panel == "range_ratio":
            if "range_ratio_60_1440" in df.columns:
                ax.plot(x_vals, df["range_ratio_60_1440"], color="tab:olive", label="range_ratio_60_1440")
            ax.axhline(1.0, ls='--', c='gray', lw=0.8)
            ax.set_title("Range Ratio (60 vs 1440)")
            _safe_legend(ax, loc="upper left"); ax.grid()

        # ── novos paineis ──────────────────────────────────────────────────────
        elif panel == "runs":
            any_line = False
            for m in RUN_WINDOWS_MIN:
                a = f"run_up_cnt_{m}"; b = f"run_dn_cnt_{m}"
                if a in df.columns: ax.plot(x_vals, df[a], label=a); any_line = True
                if b in df.columns: ax.plot(x_vals, df[b], label=b); any_line = True
            if "run_up_len" in df.columns:
                ax.plot(x_vals, df["run_up_len"], '--', label="run_up_len")
            if "run_dn_len" in df.columns:
                ax.plot(x_vals, df["run_dn_len"], '--', label="run_dn_len")
            ax.set_title("Runs (contagem e consecutivos)")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "hh_hl":
            any_line = False
            for m in HHHL_WINDOWS_MIN:
                a = f"hh_cnt_{m}"; b = f"hl_cnt_{m}"
                if a in df.columns: ax.plot(x_vals, df[a], label=a); any_line = True
                if b in df.columns: ax.plot(x_vals, df[b], label=b); any_line = True
            ax.set_title("Higher High / Higher Low (contagem)")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "ema_conf":
            any_line = False
            for s in EMA_CONFIRM_SPANS_MIN:
                a = f"bars_above_ema_{s}"; b = f"bars_below_ema_{s}"; c = f"bars_since_cross_{s}"
                if a in df.columns: ax.plot(x_vals, df[a], label=a); any_line = True
                if b in df.columns: ax.plot(x_vals, df[b], label=b); any_line = True
                if c in df.columns: ax.plot(x_vals, df[c], label=c); any_line = True
            ax.set_title("EMA Confirmação (barras acima/abaixo e desde cruzamento)")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "breakout":
            any_line = False
            for n in BREAK_LOOKBACK_MIN:
                a = f"break_high_{n}"; b = f"break_low_{n}"
                c = f"bars_since_bhigh_{n}"; d = f"bars_since_blow_{n}"
                if a in df.columns: ax.plot(x_vals, df[a], label=a); any_line = True
                if b in df.columns: ax.plot(x_vals, df[b], label=b); any_line = True
                if c in df.columns: ax.plot(x_vals, df[c], label=c); any_line = True
                if d in df.columns: ax.plot(x_vals, df[d], label=d); any_line = True
            ax.set_title("Breakouts e tempo desde rompimento")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "mom_short":
            any_line = False
            for a, b in SLOPE_DIFF_PAIRS_MIN:
                col = f"slope_diff_{a}_{b}"
                if col in df.columns: ax.plot(x_vals, df[col], label=col); any_line = True
            ax.axhline(0, ls='--', c='gray', lw=0.8)
            ax.set_title("Momentum curto vs longo (diferença de slope)")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "wick_stats":
            any_line = False
            for m in WICK_MEAN_WINDOWS_MIN:
                col = f"wick_lower_mean_{m}"
                if col in df.columns: ax.plot(x_vals, df[col], label=col); any_line = True
            if "wick_lower_streak" in df.columns:
                ax.plot(x_vals, df["wick_lower_streak"], '--', label="wick_lower_streak")
            ax.set_title("Pavio inferior (média e sequência)")
            if any_line: _safe_legend(ax, loc="upper left")
            ax.grid()

        elif panel == "sniper_entry":
            plotted = False
            if "sniper_entry_label" in df.columns:
                ax.step(
                    x_vals,
                    df["sniper_entry_label"].astype(float),
                    where="mid",
                    color="tab:green",
                    lw=1.0,
                    label="entry label (lucro minimo no horizonte)",
                )
                plotted = True
                ax.set_ylim(-0.2, 1.2)
            if "sniper_exit_code" in df.columns:
                codes = df["sniper_exit_code"].to_numpy()
                mask = np.isfinite(codes) & (codes != 0)
                if mask.any():
                    code_map = {
                        1: ("Exit score", "tab:blue", 1.05),
                        -1: ("SL", "tab:red", -0.15),
                        -3: ("Timeout", "tab:gray", 0.25),
                        -4: ("Muito curto", "tab:purple", 0.75),
                    }
                    for code_val, (label, color, yval) in code_map.items():
                        m = mask & (codes == code_val)
                        if m.any():
                            ax.scatter(
                                x_vals[m],
                                np.full(m.sum(), yval),
                                color=color,
                                s=20,
                                marker='o',
                                alpha=0.8,
                                label=f"exit={label}",
                            )
                            plotted = True
            ax.set_title("Sniper — Label & Exit Code")
            if plotted:
                _safe_legend(ax, loc="upper left", ncol=2)
            ax.grid()

        elif panel == "sniper_danger":
            if "sniper_danger_label" in df.columns:
                ax.step(
                    x_vals,
                    df["sniper_danger_label"].astype(float),
                    where="mid",
                    color="tab:red",
                    lw=1.0,
                    label="Danger (queda forte em pouco tempo)",
                )
                ax.set_ylim(-0.2, 1.2)
                _safe_legend(ax, loc="upper left")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "sniper_danger_label não encontrado",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                )
            ax.set_title("Sniper – Danger label")
            ax.grid()



    try:
        for ax in axs:
            ax.title.set_pad(30)
    except Exception:
        pass
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')
    try:
        fig.subplots_adjust(hspace=0.25, top=0.96, bottom=0.06)
    except Exception:
        pass
    if show:
        plt.show()
    else:
        plt.close(fig)
