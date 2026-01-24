# -*- coding: utf-8 -*-
"""
Wrapper to run prepare_features for crypto (debug/visual).
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
import pandas as pd


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
    "shitidx": True,
    "atr": False,
    "rsi": True,
    "slope": False,
    "vol": False,
    "ci": False,
    "cum_logret": False,
    "keltner": False,
    "cci": False,
    "adx": True,
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
    "label": False,
    "plot_candles": True,
}


CFG_CRYPTO_WINDOWS = {
    "ATR_MIN": (30, 60, 5760),
    "RSI_PRICE_MIN": (14,),
    "RSI_EMA_PAIRS": ((9, 14),),
    "SLOPE_MIN": (15, 30),
    "VOL_MIN": (240, 720, 1440, 10080),
    "KELTNER_WIDTH_MIN": (30, 60),
    "KELTNER_CENTER_MIN": (60, 240, 720),
    "KELTNER_POS_MIN": (360, 2880),
    "KELTNER_Z_MIN": (),
    "ADX_MIN": (15, 30, 120),
    "CUM_LOGRET_MIN": (1440,),
}


def _parse_tuple_env(name: str) -> tuple[int, ...] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    out = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return tuple(out) if out else None


def _apply_crypto_windows() -> None:
    for key, default_val in CFG_CRYPTO_WINDOWS.items():
        env_val = _parse_tuple_env(f"PF_CRYPTO_{key}")
        val = env_val if env_val is not None else default_val
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(val))
        if hasattr(featmod, key):
            setattr(featmod, key, tuple(val))


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
        panels.append("entry_weights")
    return panels


def _plot_interactive_features(df: pd.DataFrame, flags: dict[str, bool], candle_sec: int, *, title: str = "Crypto Feature Studio") -> None:
    flags_all = dict(flags)
    for k in flags_all:
        if k != "plot_candles":
            flags_all[k] = True

    fig = plot_all(
        df,
        flags_all,
        candle_sec=int(candle_sec),
        plot_candles=True,
        show=False,
        mark_gaps=True,
        show_price_ema=True,
        price_ema_span=30,
    )
    if fig is None:
        return

    for tr in fig.data:
        name = str(getattr(tr, "name", "") or "")
        if name not in {"candles", "ema_30"}:
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
        "entry_weights": "label",
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
        trace_meta.append(
            {
                "i": i,
                "name": str(getattr(tr, "name", "") or ""),
                "panel": panel,
                "group": group,
                "type": str(getattr(tr, "type", "")),
                "yaxis": yaxis,
            }
        )

    groups = {}
    for tr in trace_meta:
        if tr["group"] == "price":
            continue
        groups.setdefault(tr["group"], []).append(tr)

    out_dir = Path(__file__).resolve().parents[1] / "data" / "generated" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_out = os.getenv("PF_CRYPTO_PLOT_OUT", "crypto_prepare_features.html")
    out_path = Path(env_out)
    if not out_path.is_absolute():
        out_path = (out_dir / out_path).resolve()

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
    sym = os.getenv("PF_CRYPTO_SYMBOL", DEFAULT_SYMBOL).strip().upper()
    days = int(os.getenv("PF_CRYPTO_DAYS", DEFAULT_DAYS) or DEFAULT_DAYS)
    tail = int(os.getenv("PF_CRYPTO_TAIL_DAYS", DEFAULT_REMOVE_TAIL_DAYS) or DEFAULT_REMOVE_TAIL_DAYS)
    candle_sec = int(os.getenv("PF_CRYPTO_CANDLE_SEC", DEFAULT_CANDLE_SEC) or DEFAULT_CANDLE_SEC)

    feat_list_raw = os.getenv("PF_CRYPTO_FEATURES", os.getenv("PF_CRYPTO_FEATURE", "")).strip()
    feats = [f.strip().lower() for f in feat_list_raw.replace(";", ",").split(",") if f.strip()]

    raw_1m = load_ohlc_1m_series(sym, int(days), remove_tail_days=int(tail))
    if raw_1m.empty:
        print("Sem dados retornados do MySQL.", flush=True)
        return
    df_ohlc = to_ohlc_from_1m(raw_1m, int(candle_sec))

    if feats:
        flags = {k: False for k in FLAGS_CRYPTO}
        flags.update({"label": False, "plot_candles": True})
        feats_on = []
        for ft in feats:
            if ft in flags:
                flags[ft] = True
                feats_on.append(ft)
        if not feats_on:
            flags["atr"] = True
            feats_on = ["atr"]
    else:
        flags = dict(FLAGS_CRYPTO)
        flags.update({"label": False, "plot_candles": True})
        feats_on = [k for k, v in flags.items() if v and k not in {"label", "plot_candles"}]
        if not feats_on:
            flags["atr"] = True
            feats_on = ["atr"]

    print(f"[crypto] símbolo={sym} dias={days} features_on={feats_on}", flush=True)

    # Calcula todas as features para habilitar os controles, mas inicia os painéis ocultos.
    flags_all = dict(flags)
    for k in flags_all:
        if k != "plot_candles":
            flags_all[k] = True

    df = run_from_flags_dict(
        df_ohlc,
        flags_all,
        plot=False,
        u_threshold=float(DEFAULT_U_THRESHOLD),
        grey_zone=DEFAULT_GREY_ZONE,
        show=False,
        trade_contract=CRYPTO_CONTRACT,
        mark_gaps=True,
    )
    plot_days = int(os.getenv("PF_CRYPTO_PLOT_DAYS", "30") or 30)
    df_plot = df
    if plot_days > 0 and isinstance(df.index, pd.DatetimeIndex):
        cutoff = df.index.max() - pd.Timedelta(days=plot_days)
        df_plot = df.loc[df.index >= cutoff]
    interactive = os.getenv("PF_CRYPTO_PLOT_INTERACTIVE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if interactive:
        _plot_interactive_features(df_plot.copy(), flags, int(candle_sec), title=f"{sym} Feature Studio")
    else:
        plot_all(
            df_plot,
            flags,
            candle_sec=int(candle_sec),
            plot_candles=True,
            show=True,
            mark_gaps=True,
            show_price_ema=True,
            price_ema_span=30,
        )
    try:
        print(f"OK: rows={len(df):,} | cols={len(df.columns):,}".replace(",", "."), flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
