# -*- coding: utf-8 -*-
"""
Wrapper to run prepare_features for crypto (debug/visual).
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if p.name.lower() == "tradebot":
            root = p
            break
    if root:
        for cand in (root / "modules", root):
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)


_add_repo_paths()

from modules.prepare_features.prepare_features import run_from_flags_dict
from modules.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
from modules.prepare_features import pf_config as cfg
from modules.prepare_features import features as featmod
from modules.prepare_features.plotting import plot_all
from modules.prepare_features.feature_studio import render_feature_studio
from crypto.trade_contract import DEFAULT_TRADE_CONTRACT as CRYPTO_CONTRACT
from modules.prepare_features.prepare_features import (
    DEFAULT_SYMBOL,
    DEFAULT_DAYS,
    DEFAULT_REMOVE_TAIL_DAYS,
    DEFAULT_CANDLE_SEC,
    DEFAULT_U_THRESHOLD,
    DEFAULT_GREY_ZONE,
)


# Exemplo pronto (compatível com o estilo antigo) — pode importar direto como FLAGS
FLAGS_CRYPTO: dict[str, bool] = {
    "shitidx": False,
    "atr": False,
    "rsi": False,
    "slope": False,
    "vol": False,
    "ci": False,
    "cum_logret": False,
    "keltner": False,
    "cci": False,
    "adx": False,
    "time_since": False,
    "zlog": False,
    "slope_reserr": False,
    "vol_ratio": False,
    "regime": False,
    "liquidity": False,
    "rev_speed": False,
    "vol_z": False,
    "shadow": False,
    "range_ratio": False,
    "runs": False,
    "hh_hl": False,
    "ema_cross": False,
    "breakout": False,
    "mom_short": False,
    "wick_stats": False,
    "label": True,
    "plot_candles": True,
}


CFG_CRYPTO_WINDOWS = {
    "ATR_MIN": (15, 30, 60, 5760),
    "RSI_PRICE_MIN": (7, 14),
    "RSI_EMA_PAIRS": ((5, 9), (9, 14)),
    "SLOPE_MIN": (5, 10, 15, 30),
    "VOL_MIN": (60, 240, 720, 1440, 10080),
    "KELTNER_WIDTH_MIN": (30, 60),
    "KELTNER_CENTER_MIN": (60, 240, 720),
    "KELTNER_POS_MIN": (360, 2880),
    "KELTNER_Z_MIN": (),
    "ADX_MIN": (7, 15, 30, 120),
    "LOGRET_MIN": (240, 1440),
}

# ======== Config fixa (sem env) ========
SYMBOL = "XLMUSDT"
DAYS = 180
TAIL_DAYS = DEFAULT_REMOVE_TAIL_DAYS
CANDLE_SEC = DEFAULT_CANDLE_SEC

# Plot
PLOT_INTERACTIVE = True
PLOT_OUT = "data/generated/plots/crypto_prepare_features.html"
PLOT_DAYS = DAYS
PLOT_CANDLES = False


def _print_label_stats(df: pd.DataFrame) -> None:
    def _print_stats_for(col: str) -> None:
        if col not in df.columns:
            return
        arr = df[col].dropna().to_numpy(dtype=float)
        if arr.size == 0:
            print(f"[{col}] vazio", flush=True)
            return
        nz = float(np.mean(np.abs(arr) > 1e-12))
        print(f"\n[{col}]", flush=True)
        print(f"  mean : {arr.mean(): .6f}", flush=True)
        print(f"  std  : {arr.std(): .6f}", flush=True)
        print(f"  min  : {arr.min(): .6f}", flush=True)
        print(f"  max  : {arr.max(): .6f}", flush=True)
        print(f"  nz   : {nz:.2%}", flush=True)
        try:
            qs = [q / 100.0 for q in range(0, 101, 5)]
            qv = np.quantile(arr, qs)
            print("  pct  :", flush=True)
            line = []
            for q, v in zip(range(0, 101, 5), qv):
                line.append(f"p{q:02d}={v:+.6f}")
                if len(line) == 5:
                    print("    " + "  ".join(line), flush=True)
                    line = []
            if line:
                print("    " + "  ".join(line), flush=True)
        except Exception:
            pass

    print("\n[labels] resumo", flush=True)
    # Label assinado (retorno em escala bruta, clipado em +/- TIMING_LABEL_CLIP)
    _print_stats_for("timing_label")
    # Labels usados no treino dos regressors long/short (escala 0..100)
    _print_stats_for("timing_label_long")
    _print_stats_for("timing_label_short")
    # Novos labels (pipeline supervisionado)
    _print_stats_for("edge_label_long")
    _print_stats_for("edge_label_short")
    _print_stats_for("entry_gate_long")
    _print_stats_for("entry_gate_short")


def _apply_crypto_windows() -> None:
    for key, default_val in CFG_CRYPTO_WINDOWS.items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(default_val))
        if hasattr(featmod, key):
            setattr(featmod, key, tuple(default_val))


def _build_panels_from_flags(flags: dict[str, bool]) -> list[str]:
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
        panels.extend(["weights", "weights_side", "label", "timing_label"])
    return panels


def _plot_interactive_features(df: pd.DataFrame, flags: dict[str, bool], candle_sec: int, *, title: str = "Crypto Feature Studio") -> None:
    flags_all = dict(flags)
    flags_all["plot_candles"] = bool(flags.get("plot_candles", True))

    fig = plot_all(
        df,
        flags_all,
        candle_sec=int(candle_sec),
        plot_candles=bool(flags_all.get("plot_candles", True)),
        show=False,
        mark_gaps=True,
        show_price_ema=True,
        price_ema_span=30,
    )
    if fig is None:
        return

    for tr in fig.data:
        name = str(getattr(tr, "name", "") or "")
        if name not in {"candles", "close", "ema_30"}:
            tr.visible = False

    panels = _build_panels_from_flags(flags_all)
    panel_to_group = {
        "candles": "price",
        "shitidx": "shitidx",
        "keltner_width": "keltner",
        "keltner_center": "keltner",
        "keltner_pos": "keltner",
        "keltner_squeeze": "keltner",
        "atr": "atr",
        "rsi": "rsi",
        "slope": "slope",
        "vol": "vol",
        "ci": "ci",
        "logret": "cum_logret",
        "cci": "cci",
        "adx": "adx",
        "pctmm": "time_since",
        "timesince": "time_since",
        "zlog": "zlog",
        "slope_reserr": "slope_reserr",
        "vol_ratio": "vol_ratio",
        "regime": "regime",
        "liquidity": "liquidity",
        "rev_speed": "rev_speed",
        "vol_z": "vol_z",
        "shadow": "shadow",
        "range_ratio": "range_ratio",
        "runs": "runs",
        "hh_hl": "hh_hl",
        "ema_conf": "ema_cross",
        "breakout": "breakout",
        "mom_short": "mom_short",
        "wick_stats": "wick_stats",
        "weights": "edge_weight",
        "weights_side": "edge_weight",
        "label": "edge_label",
        "timing_label": "__hidden__",
    }

    trace_meta = []
    for i, tr in enumerate(fig.data):
        yaxis = getattr(tr, "yaxis", "y")
        if isinstance(yaxis, str) and yaxis.startswith("y"):
            try:
                row = int(yaxis[1:]) if len(yaxis) > 1 else 1
            except Exception:
                row = 1
        else:
            row = 1
        panel = panels[row - 1] if 0 <= row - 1 < len(panels) else "candles"
        group = panel_to_group.get(panel, panel)
        tname = str(getattr(tr, "name", "") or "")
        if panel == "label":
            if tname.startswith("edge_"):
                group = "edge_label"
            elif tname.startswith("entry_gate"):
                group = "entry_gate"
            else:
                group = "__hidden__"
        elif panel == "weights":
            if tname in {"edge_weight_long", "edge_weight_short", "timing_weight_long", "timing_weight_short"}:
                group = "edge_weight"
            elif tname in {"entry_gate_weight", "entry_gate_weight_long", "entry_gate_weight_short", "timing_weight"}:
                group = "entry_gate_weight"
            else:
                group = "__hidden__"
        elif panel == "weights_side":
            if tname in {"edge_weight_long", "edge_weight_short", "timing_weight_long", "timing_weight_short"}:
                group = "edge_weight"
            else:
                group = "__hidden__"
        elif panel == "timing_label":
            group = "__hidden__"
        trace_meta.append(
            {
                "i": i,
                "name": tname,
                "panel": panel,
                "group": group,
                "type": str(getattr(tr, "type", "")),
                "yaxis": yaxis,
            }
        )

    groups = {}
    for tr in trace_meta:
        if tr["group"] in {"price", "__hidden__"}:
            continue
        groups.setdefault(tr["group"], []).append(tr)
    # remove duplicatas por nome dentro de cada grupo (evita sobreplot por painÃ©is redundantes)
    for g, items in list(groups.items()):
        seen = set()
        dedup = []
        for it in items:
            key = str(it.get("name") or "")
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        groups[g] = dedup

    out_path = Path(PLOT_OUT).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_feature_studio(
        fig=fig,
        panels=panels,
        panel_to_group=panel_to_group,
        trace_meta=trace_meta,
        groups=groups,
        title=title,
        out_path=out_path,
        open_browser=True,
    )


def main() -> None:
    _apply_crypto_windows()
    sym = SYMBOL
    days = int(DAYS)
    tail = int(TAIL_DAYS)
    candle_sec = int(CANDLE_SEC)
    raw_1m = load_ohlc_1m_series(sym, int(days), remove_tail_days=int(tail))
    if raw_1m.empty:
        print("Sem dados retornados do MySQL.", flush=True)
        return
    df_ohlc = to_ohlc_from_1m(raw_1m, int(candle_sec))

    flags = dict(FLAGS_CRYPTO)
    flags.update({"label": True, "plot_candles": bool(PLOT_CANDLES)})
    feats_on = [k for k, v in flags.items() if v and k not in {"label", "plot_candles"}]
    if not feats_on:
        feats_on = ["label"]

    print(f"[crypto] símbolo={sym} dias={days} features_on={feats_on}", flush=True)

    # Calcula todas as features para habilitar os controles, mas inicia os painéis ocultos.
    contract = CRYPTO_CONTRACT
    df = run_from_flags_dict(
        df_ohlc,
        flags,
        plot=False,
        u_threshold=float(DEFAULT_U_THRESHOLD),
        grey_zone=DEFAULT_GREY_ZONE,
        show=False,
        trade_contract=contract,
        mark_gaps=True,
    )

    # estatistica dos labels (assinado + long/short)
    try:
        _print_label_stats(df)  
    except Exception as e:
        print(f"[labels] falhou ao imprimir estatisticas: {type(e).__name__}: {e}", flush=True)
    # opcional: filtra colunas para ficar apenas com as features selecionadas
    allow = getattr(cfg, "FEATURE_ALLOWLIST", None)
    if allow:
        allow_set = set(str(x) for x in allow)
        keep = []
        for c in df.columns:
            if c in {"open", "high", "low", "close", "volume"}:
                keep.append(c)
                continue
            if str(c).startswith(("timing_", "label_", "exit_", "edge_", "entry_gate_")):
                keep.append(c)
                continue
            if c in allow_set:
                keep.append(c)
        df = df.loc[:, keep]
    # remove colunas legadas que poluem o plot
    legacy = [c for c in df.columns if str(c).startswith("sniper_entry_")]
    legacy.extend([c for c in df.columns if str(c).startswith("sniper_") and str(c).endswith("_short")])
    if legacy:
        df = df.drop(columns=legacy, errors="ignore")
    plot_days = int(PLOT_DAYS)
    df_plot = df
    if plot_days > 0 and isinstance(df.index, pd.DatetimeIndex):
        cutoff = df.index.max() - pd.Timedelta(days=plot_days)
        df_plot = df.loc[df.index >= cutoff]
    if PLOT_INTERACTIVE:
        _plot_interactive_features(df_plot.copy(), flags, int(candle_sec), title=f"{sym} Feature Studio")
    else:
        plot_all(
            df_plot,
            flags,
            u_threshold=float(DEFAULT_U_THRESHOLD),
            candle_sec=int(candle_sec),
            plot_candles=bool(PLOT_CANDLES),
            grey_zone=DEFAULT_GREY_ZONE,
            show=True,
            save_path=PLOT_OUT,
            mark_gaps=True,
        )


if __name__ == "__main__":
    main()
