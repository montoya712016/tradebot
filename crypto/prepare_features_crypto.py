# -*- coding: utf-8 -*-
"""
Wrapper to run prepare_features for crypto (debug/visual).
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
from time import perf_counter
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

from modules.prepare_features.prepare_features import run_from_flags_dict
from modules.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
from modules.prepare_features import pf_config as cfg
from modules.prepare_features import features as featmod
from modules.prepare_features.plotting import plot_all
from modules.prepare_features.feature_studio import render_feature_studio
from crypto.trade_contract import DEFAULT_TRADE_CONTRACT as CRYPTO_CONTRACT
from modules.prepare_features.prepare_features import (
    DEFAULT_CANDLE_SEC,
    DEFAULT_U_THRESHOLD,
    DEFAULT_GREY_ZONE,
)
from modules.prepare_features import labels as lblmod


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
    "eff": False,
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
    "EFF_MIN": (30, 60, 120, 240),
}


def _entry_only_mode() -> bool:
    pf_entry_only = os.getenv("PF_ENTRY_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    train_exit_on = os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1").strip().lower() in {"1", "true", "yes", "on"}
    return bool(pf_entry_only or (not train_exit_on))


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
    if flags.get("eff"):
        panels.append("eff")
    if flags.get("label"):
        panels.append("entry_weights")
        panels.append("oracle_equity")
        if not _entry_only_mode():
            panels.append("exit_regression")
    return panels


def _add_oracle_equity_for_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    if "sniper_oracle_equity" in df.columns:
        return df
    out = df.copy()
    lbl_col = "sniper_entry_label" if "sniper_entry_label" in out.columns else ""
    wait_col = "sniper_exit_wait_bars" if "sniper_exit_wait_bars" in out.columns else ""
    if (not lbl_col) or (not wait_col):
        suffixes = [str(int(w)) + "m" for w in (getattr(CRYPTO_CONTRACT, "entry_label_windows_minutes", []) or [])]
        for sfx in suffixes:
            lc = f"sniper_entry_label_{sfx}"
            wc = f"sniper_exit_wait_bars_{sfx}"
            if (not lbl_col) and (lc in out.columns):
                lbl_col = lc
            if (not wait_col) and (wc in out.columns):
                wait_col = wc
    if (not lbl_col) or (not wait_col):
        return out

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(np.float64, copy=False)
    lbl = pd.to_numeric(out[lbl_col], errors="coerce").fillna(0).to_numpy(np.float64, copy=False)
    wait = pd.to_numeric(out[wait_col], errors="coerce").fillna(0).to_numpy(np.float64, copy=False)
    n = int(len(out))
    eq = np.full(n, np.nan, dtype=np.float64)
    in_pos = np.zeros(n, dtype=np.float32)
    fees = float(getattr(CRYPTO_CONTRACT, "fee_pct_per_side", 0.0) or 0.0)
    slip = float(getattr(CRYPTO_CONTRACT, "slippage_pct", 0.0) or 0.0)
    cost_rt = 2.0 * (max(0.0, fees) + max(0.0, slip))

    equity = 1.0
    pos = False
    entry_px = np.nan
    exit_i = -1
    for i in range(n):
        if pos:
            in_pos[i] = 1.0
            if i >= exit_i:
                px_exit = close[i]
                if np.isfinite(entry_px) and entry_px > 0.0 and np.isfinite(px_exit) and px_exit > 0.0:
                    r = (px_exit / entry_px) - 1.0 - cost_rt
                    equity *= max(1e-9, 1.0 + float(r))
                pos = False
                entry_px = np.nan
                exit_i = -1
                in_pos[i] = 0.0
        if (not pos) and (lbl[i] > 0.5):
            px_entry = close[i]
            if np.isfinite(px_entry) and px_entry > 0.0:
                w = int(round(wait[i])) if np.isfinite(wait[i]) else 0
                if w < 1:
                    w = 1
                pos = True
                entry_px = float(px_entry)
                exit_i = int(min(n - 1, i + w))
                in_pos[i] = 1.0
        eq[i] = equity

    out["sniper_oracle_equity"] = pd.Series(eq.astype(np.float32), index=out.index)
    out["sniper_oracle_in_pos"] = pd.Series(in_pos.astype(np.float32), index=out.index)
    return out


def _add_dense_exit_target_for_plot(df: pd.DataFrame, candle_sec: int) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    try:
        windows = list(getattr(CRYPTO_CONTRACT, "entry_label_windows_minutes", []) or [])
        w_min = int(windows[0]) if windows else 360
        suffix = f"{w_min}m"
        dense_col = f"sniper_exit_span_target_dense_{suffix}"
        dense_q25_col = f"sniper_exit_span_q25_dense_{suffix}"
        dense_q50_col = f"sniper_exit_span_q50_dense_{suffix}"
        dense_q75_col = f"sniper_exit_span_q75_dense_{suffix}"
        if dense_col in df.columns and dense_q25_col in df.columns and dense_q75_col in df.columns:
            return df

        candle_seconds = max(1, int(candle_sec))
        close = df["close"].to_numpy(np.float64, copy=False)
        low = df["low"].to_numpy(np.float64, copy=False) if "low" in df.columns else close
        entry_all = np.ones(len(df), dtype=np.float32)

        exit_span_grid_min = lblmod._env_int_list(
            "PF_EXIT_SPAN_GRID_MIN",
            [
                15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240,
                255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480,
                495, 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720,
            ],
        )
        exit_min_hold_min = int(lblmod._env_int("PF_EXIT_LABEL_MIN_HOLD_MIN", 5))
        exit_confirm_bars = int(lblmod._env_int("PF_EXIT_LABEL_CONFIRM_BARS", 2))
        exit_ret_scale = float(lblmod._env_float("PF_EXIT_LABEL_RET_SCALE", 0.03))
        exit_edge_scale = float(lblmod._env_float("PF_EXIT_LABEL_EDGE_SCALE", 0.01))
        exit_softmax_temp = float(lblmod._env_float("PF_EXIT_LABEL_SOFTMAX_TEMP", 0.0025))
        exit_eff_blend = float(lblmod._env_float("PF_EXIT_LABEL_EFF_BLEND", 0.35))
        exit_window_tol_rel = float(lblmod._env_float("PF_EXIT_LABEL_WINDOW_TOL_REL", 0.25))
        exit_window_tol_abs = float(lblmod._env_float("PF_EXIT_LABEL_WINDOW_TOL_ABS", 0.0015))
        min_hold_bars = int(max(1, round((float(exit_min_hold_min) * 60.0) / float(candle_seconds))))
        spans_bars = np.array(
            [int(max(2, round((float(m) * 60.0) / float(candle_seconds)))) for m in exit_span_grid_min if int(m) > 0],
            dtype=np.int32,
        )
        if spans_bars.size == 0:
            spans_bars = np.array([int(max(2, round((60.0 * 60.0) / float(candle_seconds))))], dtype=np.int32)

        horizon_min_cfg = int(lblmod._env_int("PF_EXIT_LABEL_HORIZON_MIN", int(w_min)))
        exit_horizon_bars = int(max(1, round((float(horizon_min_cfg) * 60.0) / float(candle_seconds))))
        dense_q25, dense_q50, dense_q75, _dense_w, _dense_entropy = lblmod._exit_span_quantiles_numba(
            close,
            low,
            entry_all,
            int(exit_horizon_bars),
            int(min_hold_bars),
            int(exit_confirm_bars),
            spans_bars,
            float(getattr(CRYPTO_CONTRACT, "exit_ema_init_offset_pct", 0.0) or 0.0),
            float(getattr(CRYPTO_CONTRACT, "fee_pct_per_side", 0.0) or 0.0),
            float(getattr(CRYPTO_CONTRACT, "slippage_pct", 0.0) or 0.0),
            float(exit_ret_scale),
            float(exit_edge_scale),
            float(exit_softmax_temp),
            float(exit_eff_blend),
            float(exit_window_tol_rel),
            float(exit_window_tol_abs),
        )
        out = df.copy()
        out[dense_col] = pd.Series(dense_q50.astype(np.float32), index=out.index)
        out[dense_q25_col] = pd.Series(dense_q25.astype(np.float32), index=out.index)
        out[dense_q50_col] = pd.Series(dense_q50.astype(np.float32), index=out.index)
        out[dense_q75_col] = pd.Series(dense_q75.astype(np.float32), index=out.index)
        return out
    except Exception:
        return df


def _plot_interactive_features(df: pd.DataFrame, flags: dict[str, bool], candle_sec: int, *, title: str = "Crypto Feature Studio") -> None:
    t0 = perf_counter()
    dense_exit_on = (
        (os.getenv("PF_CRYPTO_PLOT_DENSE_EXIT", "0").strip().lower() not in {"0", "false", "no", "off"})
        and (not _entry_only_mode())
    )
    if dense_exit_on:
        df = _add_dense_exit_target_for_plot(df, int(candle_sec))
    df = _add_oracle_equity_for_labels(df)
    flags_all = dict(flags)
    for k in flags_all:
        if k != "plot_candles":
            flags_all[k] = True
    flags_all["oracle_equity"] = True

    show_entry_markers = os.getenv("PF_CRYPTO_SHOW_ENTRY_MARKERS", "1").strip().lower() not in {"0", "false", "no", "off"}
    fig = plot_all(
        df,
        flags_all,
        candle_sec=int(candle_sec),
        plot_candles=True,
        show=False,
        mark_gaps=True,
        show_price_ema=True,
        price_ema_span=30,
        show_label_markers=show_entry_markers,
    )
    if fig is None:
        return
    t_plot_all = perf_counter()

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
        "eff": "eff",
        "entry_weights": "label",
        "oracle_equity": "oracle",
        "exit_regression": "label",
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
    t_meta = perf_counter()

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
    t_render = perf_counter()
    print(
        f"[crypto] plot_timing(s): plot_all={t_plot_all - t0:.2f} meta={t_meta - t_plot_all:.2f} "
        f"render_html={t_render - t_meta:.2f} total={t_render - t0:.2f}",
        flush=True,
    )
    print(f"[crypto] plot salvo: {out_path}", flush=True)


def main() -> None:
    # Habilita resumo de performance das features neste wrapper (rows/cols/tempo total + tempos por feature).
    os.environ.setdefault("PF_LOG_SUMMARY", "1")
    # Pipeline padrao atual: entry-only (sem labels de exit).
    os.environ.setdefault("PF_ENTRY_ONLY", "1")
    # Label/weight alvo: label true para r_net > 0 com pouco peso para retornos marginais.
    os.environ.setdefault("PF_ENTRY_LABEL_NET_PROFIT_THR", "0.0")
    os.environ.setdefault("PF_ENTRY_WEIGHT_RET_SCALE_POS", "0.04")
    os.environ.setdefault("PF_ENTRY_WEIGHT_RET_SCALE_NEG", "0.03")
    os.environ.setdefault("PF_ENTRY_WEIGHT_RET_DEADZONE", "0.002")
    os.environ.setdefault("PF_ENTRY_WEIGHT_POS_GAIN", "6.0")
    os.environ.setdefault("PF_ENTRY_WEIGHT_NEG_GAIN", "5.0")
    os.environ.setdefault("PF_ENTRY_WEIGHT_POS_POWER", "2.6")
    os.environ.setdefault("PF_ENTRY_WEIGHT_NEG_POWER", "2.2")
    os.environ.setdefault("PF_ENTRY_WEIGHT_MODE", "ret_pct_abs")
    os.environ.setdefault("PF_ENTRY_WEIGHT_PCT_POWER", "1.0")

    _apply_crypto_windows()
    sym = 'XLMUSDT'
    days = 60
    tail = 0
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

    # Calcula todas as features para habilitar os controles no Feature Studio.
    flags_all = dict(flags)
    for k in flags_all:
        if k != "plot_candles":
            flags_all[k] = True
    flags_all["feature_timings"] = True

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
    _plot_days_env = os.getenv("PF_CRYPTO_PLOT_DAYS", "").strip()
    plot_days = int(_plot_days_env or str(days))
    print(f"[crypto] plot_days={plot_days}", flush=True)
    df_plot = df
    if plot_days > 0 and isinstance(df.index, pd.DatetimeIndex):
        cutoff = df.index.max() - pd.Timedelta(days=plot_days)
        df_plot = df.loc[df.index >= cutoff]
    try:
        n_entry = int(pd.to_numeric(df_plot.get("sniper_entry_label"), errors="coerce").fillna(0).sum()) if "sniper_entry_label" in df_plot.columns else 0
        print(f"[crypto] entry_count(plot,long): {n_entry}", flush=True)
        for wcol in (
            "sniper_entry_weight_360m",
            "sniper_entry_weight",
        ):
            if wcol in df_plot.columns:
                w = pd.to_numeric(df_plot[wcol], errors="coerce")
                print(
                    f"[crypto] {wcol}(plot): mean={float(w.mean()):.3f} p90={float(w.quantile(0.9)):.3f} max={float(w.max()):.3f}",
                    flush=True,
                )
                break
        if "sniper_exit_span_target" in df_plot.columns:
            xs = pd.to_numeric(df_plot["sniper_exit_span_target"], errors="coerce")
            xnn = int(xs.notna().sum())
            print(
                f"[crypto] sniper_exit_span_target(plot): p50={float(xs.quantile(0.5)):.1f} "
                f"p90={float(xs.quantile(0.9)):.1f} max={float(xs.max()):.1f} notna={xnn}",
                flush=True,
            )
            if n_entry > 0 and xnn <= n_entry:
                cov = float(xnn) / float(n_entry)
                print(f"[crypto] exit_target coverage_vs_entry_true: {cov:.3f}", flush=True)
            else:
                cov_rows = float(xnn) / float(max(1, len(df_plot)))
                print(f"[crypto] exit_target dense_coverage_rows: {cov_rows:.3f}", flush=True)
    except Exception:
        pass
    try:
        verbose_diag = os.getenv("PF_CRYPTO_VERBOSE_LABEL_DIAG", "").strip().lower() in {"1", "true", "yes", "on"}
        if verbose_diag:
            diag_full = dict(df.attrs.get("price_label_diag") or {})
            if diag_full:
                params = diag_full.get("params", {})
                if params:
                    print(f"[crypto] label_params: {params}", flush=True)
                if "single_long" in diag_full:
                    print(f"[crypto] label_diag[single_long]: {diag_full['single_long']}", flush=True)
    except Exception:
        pass
    interactive = os.getenv("PF_CRYPTO_PLOT_INTERACTIVE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if interactive:
        _plot_interactive_features(df_plot.copy(), flags, int(candle_sec), title=f"{sym} Feature Studio")
    else:
        fig = plot_all(
            df_plot,
            flags_all,
            candle_sec=int(candle_sec),
            plot_candles=True,
            show=True,
            mark_gaps=True,
            show_price_ema=True,
            price_ema_span=30,
        )
        try:
            if fig is None:
                print("[crypto][warn] plot_all retornou None", flush=True)
        except Exception:
            pass
    try:
        print(f"OK: rows={len(df):,} | cols={len(df.columns):,}".replace(",", "."), flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
