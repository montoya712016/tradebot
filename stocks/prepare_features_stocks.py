# -*- coding: utf-8 -*-
"""
Prepare features for stocks (visual/debug).
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import numpy as np
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

from modules.prepare_features.features import make_features
from modules.prepare_features.labels import apply_trade_contract_labels
from modules.prepare_features.data import to_ohlc_from_1m
from modules.prepare_features.data_stocks import load_ohlc_1m_series_stock
from modules.prepare_features import pf_config as cfg
from modules.prepare_features import features as featmod
from modules.prepare_features.plotting import plot_all
from modules.prepare_features.feature_studio import render_feature_studio
from stocks.trade_contract import DEFAULT_TRADE_CONTRACT as STOCKS_CONTRACT
from modules.prepare_features.configs.stocks import (
    DEFAULT_SYMBOL,
    DEFAULT_DAYS,
    DEFAULT_REMOVE_TAIL_DAYS,
    DEFAULT_CANDLE_SEC,
)

# Exemplo pronto (compatível com o estilo antigo) — pode importar direto como FLAGS
FLAGS_STOCKS: dict[str, bool] = {
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
    "label": False,
    "plot_candles": True,
}

# Janelas padrão (pode sobrescrever via ENV, ex.: PF_STOCK_ATR_MIN=10,20)
CFG_STOCK_WINDOWS = {
    "EMA_WINDOWS": (20, 30, 60, 120),
    "EMA_PAIRS": ((20, 60), (30, 120)),
    "ATR_MIN": (5, 10, 20, 30, 60),
    "RSI_PRICE_MIN": (5, 10, 20, 30, 60),
    "RSI_EMA_PAIRS": ((5, 10), (9, 14)),
    "SLOPE_MIN": (5, 10, 20, 30),
    "VOL_MIN": (5, 10, 20, 30),
    "KELTNER_WIDTH_MIN": (20, 30),
    "KELTNER_CENTER_MIN": (20, 30),
    "KELTNER_POS_MIN": (15, 30),
    "KELTNER_Z_MIN": (20, 30, 60),
    "CCI_MIN": (10, 20, 30, 60),
    "ADX_MIN": (7, 14, 28),
    "CI_MIN": (10, 20, 30, 60),
    "LOGRET_MIN": (15, 30, 60, 120),
    "MINMAX_MIN": (15, 30, 60),
    "SLOPE_RESERR_MIN": (30, 60, 120),
    "ZLOG_MIN": (30, 60, 120),
    "REV_WINDOWS": (10, 20, 30, 60),
    "VOL_RATIO_PAIRS": ((10, 30), (20, 60)),
    "RUN_WINDOWS_MIN": (10, 20, 30),
    "HHHL_WINDOWS_MIN": (10, 20, 30),
    "EMA_CONFIRM_SPANS_MIN": (10, 20),
    "BREAK_LOOKBACK_MIN": (10, 20, 30),
    "SLOPE_DIFF_PAIRS_MIN": ((3, 10), (5, 15)),
    "WICK_MEAN_WINDOWS_MIN": (5, 10, 20),
    "LIQUIDITY_MIN": (30, 60, 120),
    "VOL_Z_SHORT_MIN": (5,),
    "VOL_Z_LONG_MIN": (30,),
    "RANGE_RATIO_PAIRS": ((10, 40), (20, 80)),
    "ROLLING_MINP_MODE": "progressive",
    "EMA_MINP_MODE": "progressive",
}


def _insert_daily_breaks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    # Adiciona uma única linha NaN logo após o último candle de cada dia para quebrar continuidade.
    day_keys = df.index.normalize()
    last_idx = df.groupby(day_keys, sort=False).tail(1).index
    gap_idx = (last_idx + pd.Timedelta(seconds=1)).difference(df.index)
    if gap_idx.empty:
        return df.copy()
    nan_block = pd.DataFrame(np.nan, index=gap_idx, columns=df.columns)
    out = pd.concat([df, nan_block]).sort_index()
    return out


def _maybe_fill_day_start(df: pd.DataFrame, cols: list[str]) -> None:
    if not cols or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return
    mode = os.getenv("PF_STOCK_FILL_DAY_START", "").strip().lower()
    if mode not in {"ffill", "bfill", "both"}:
        return
    day_keys = df.index.normalize()
    sel = [c for c in cols if c in df.columns]
    if not sel:
        return
    grp = df[sel].groupby(day_keys)
    if mode == "ffill":
        df[sel] = grp.transform("ffill")
    elif mode == "bfill":
        df[sel] = grp.transform("bfill")
    else:
        df[sel] = grp.transform("ffill").groupby(day_keys).transform("bfill")


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


def _apply_stock_windows() -> None:
    # aplica em ambos namespaces (modules.prepare_features e prepare_features)
    extra_cfg = None
    extra_featmod = None
    try:
        from prepare_features import pf_config as extra_cfg  # type: ignore
        from prepare_features import features as extra_featmod  # type: ignore
    except Exception:
        extra_cfg = None
        extra_featmod = None

    for key, default_val in CFG_STOCK_WINDOWS.items():
        if isinstance(default_val, str):
            env_val = os.getenv(f"PF_STOCK_{key}", "").strip()
            val = env_val if env_val else default_val
        else:
            env_val = _parse_tuple_env(f"PF_STOCK_{key}")
            val = env_val if env_val is not None else default_val
        if isinstance(default_val, str):
            if hasattr(cfg, key):
                setattr(cfg, key, str(val))
            if hasattr(featmod, key):
                setattr(featmod, key, str(val))
            if extra_cfg is not None and hasattr(extra_cfg, key):
                setattr(extra_cfg, key, str(val))
            if extra_featmod is not None and hasattr(extra_featmod, key):
                setattr(extra_featmod, key, str(val))
        else:
            if hasattr(cfg, key):
                setattr(cfg, key, tuple(val))
            if hasattr(featmod, key):
                setattr(featmod, key, tuple(val))
            if extra_cfg is not None and hasattr(extra_cfg, key):
                setattr(extra_cfg, key, tuple(val))
            if extra_featmod is not None and hasattr(extra_featmod, key):
                setattr(extra_featmod, key, tuple(val))


def _add_time_features(df: pd.DataFrame) -> None:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return
    idx = df.index
    day_keys = idx.normalize()
    minutes = (idx.hour * 60 + idx.minute).astype(np.int32)
    df["minute_of_day"] = minutes
    df["minute_of_day_sin"] = np.sin((minutes / 1440.0) * 2.0 * np.pi)
    df["minute_of_day_cos"] = np.cos((minutes / 1440.0) * 2.0 * np.pi)
    bars_since_open = df.groupby(day_keys).cumcount().astype(np.int32)
    df["bars_since_open"] = bars_since_open
    day_sizes = df.groupby(day_keys)["bars_since_open"].transform("max").astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        prog = bars_since_open.to_numpy(np.float32, copy=False) / np.maximum(1.0, day_sizes.to_numpy(np.float32, copy=False))
    df["session_progress"] = prog.astype(np.float32, copy=False)


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


def _plot_interactive_features(df: pd.DataFrame, flags: dict[str, bool], candle_sec: int, *, title: str = "Stocks Feature Studio") -> None:
    # evita SettingWithCopy em slices
    df = df.copy()
    flags_all = dict(flags)
    for k in flags_all:
        if k != "plot_candles":
            flags_all[k] = True

    # Garante que as colunas das features existam (mesmo que os flags originais estivessem desligados).
    need_cols = any(flags_all.get(k, False) for k in flags_all if k not in {"plot_candles", "label"})
    if need_cols:
        try:
            make_features(df, flags_all, verbose=False)
        except Exception:
            pass

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
    env_out = os.getenv("PF_STOCK_PLOT_OUT", "stocks_prepare_features.html")
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
    _apply_stock_windows()
    # overrides via ENV (ex.: PF_STOCK_SYMBOL=AAPL PF_STOCK_DAYS=90)
    sym = os.getenv("PF_STOCK_SYMBOL", "COIN").strip().upper()
    days = int(os.getenv("PF_STOCK_DAYS", "0") or 0)
    tail = int(os.getenv("PF_STOCK_TAIL_DAYS", DEFAULT_REMOVE_TAIL_DAYS) or DEFAULT_REMOVE_TAIL_DAYS)
    candle_sec = int(os.getenv("PF_STOCK_CANDLE_SEC", DEFAULT_CANDLE_SEC) or DEFAULT_CANDLE_SEC)
    feat_list_raw = os.getenv("PF_STOCK_FEATURES", os.getenv("PF_STOCK_FEATURE", "")).strip()
    feats = [f.strip().lower() for f in feat_list_raw.replace(";", ",").split(",") if f.strip()]
    label_env = os.getenv("PF_STOCK_LABEL", "").strip().lower()
    if label_env in {"1", "true", "yes", "y", "on"}:
        label_on = True
    elif label_env in {"0", "false", "no", "n", "off"}:
        label_on = False
    else:
        label_on = bool(FLAGS_STOCKS.get("label", False))

    raw_1m = load_ohlc_1m_series_stock(sym, int(days), remove_tail_days=int(tail))
    if raw_1m.empty:
        # tenta carregar tudo que existir (sem filtro de dias) para facilitar debug
        raw_1m = load_ohlc_1m_series_stock(sym, 0, remove_tail_days=int(tail))

    if raw_1m.empty:
        print(f"Sem dados retornados do MySQL para {sym}.", flush=True)
        return
    df_ohlc = to_ohlc_from_1m(raw_1m, int(candle_sec))

    # Se PF_STOCK_FEATURES/FEATURE for definido, liga apenas esses. Senao, usa FLAGS_STOCKS do arquivo.
    if feats:
        flags = {k: False for k in FLAGS_STOCKS}
        flags.update({"label": label_on, "plot_candles": True, "feature_timings": True})
        feats_on = []
        for ft in feats:
            if ft in flags:
                flags[ft] = True
                feats_on.append(ft)
        if not feats_on:
            print("[stocks] aviso: nenhuma feature valida informada; seguindo sem indicadores.", flush=True)
    else:
        flags = dict(FLAGS_STOCKS)
        flags.update({"label": label_on, "plot_candles": True, "feature_timings": True})
        feats_on = [k for k, v in flags.items() if v and k not in {"label", "plot_candles"}]
    print(f"[stocks] símbolo={sym} dias={days} features_on={feats_on}", flush=True)

    df = df_ohlc.copy()
    feature_keys = [k for k in flags.keys() if k not in {"label", "plot_candles"}]
    features_on = any(bool(flags.get(k, False)) for k in feature_keys)
    if features_on:
        df_day = _insert_daily_breaks(df)
        make_features(df_day, flags, verbose=True)
        new_cols = [c for c in df_day.columns if c not in df.columns]
        if new_cols:
            df[new_cols] = df_day.loc[df.index, new_cols].to_numpy()
            _maybe_fill_day_start(df, new_cols)
    if bool(flags.get("label", True)):
        apply_trade_contract_labels(
            df,
            contract=STOCKS_CONTRACT,
            candle_sec=int(candle_sec),
        )
    _add_time_features(df)
    plot_days = int(os.getenv("PF_STOCK_PLOT_DAYS", "30") or 30)
    df_plot = df
    if plot_days > 0 and isinstance(df.index, pd.DatetimeIndex):
        cutoff = df.index.max() - pd.Timedelta(days=plot_days)
        df_plot = df.loc[df.index >= cutoff]
    interactive = os.getenv("PF_STOCK_PLOT_INTERACTIVE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if interactive:
        _plot_interactive_features(df_plot, flags, int(candle_sec), title=f"{sym} Feature Studio")
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
        cols = [c for c in df.columns if any(c.startswith(pref) for pref in feats_on)]
        if cols:
            print("Columns geradas:", ", ".join(cols[:12]), flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
