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
SAVE_REVERSAL_DEBUG = True


REV_LABEL_CFG = {
    # Escala de volatilidade.
    "atr_span": 48,
    # Multi-horizonte (horizon_bars, tp_atr, sl_atr), sem pivô/premove.
    "scenarios": (
        (4, 0.45, 0.35),
        (8, 0.80, 0.55),
        (16, 1.30, 0.90),
    ),
    # Confirmacao curta para evitar entrada ainda em queda/subida contra.
    "confirm_bars": 3,
    "confirm_move_atr_min": 0.35,
    "confirm_max_adverse_atr": 0.25,
    "first_bar_move_atr_min": 0.05,
    # Qualidade minima do padrao para virar gate.
    "confirm_score_atr_min": 0.60,
    "min_scenarios_pass": 3,
    "scenario_pass_frac_min": 1.00,
    "profit_edge_lambda": 1.00,
    "profit_edge_atr_min": 1.20,
    "profit_rr_min": 2.20,
    # Pesos
    "weight_min": 0.20,
    "weight_max": 4.00,
    "weight_pos_base": 1.10,
    "weight_pos_confirm_gain": 0.60,
    "weight_pos_scenario_gain": 0.55,
    "weight_hard_negative": 1.60,
}


def _compute_atr_abs(df: pd.DataFrame, span: int) -> np.ndarray:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=max(2, int(span)), adjust=False, min_periods=max(2, int(span))).mean()
    atr = atr.bfill().ffill()
    close_np = close.to_numpy(dtype=float, copy=False)
    atr_np = atr.to_numpy(dtype=float, copy=False)
    floor = np.maximum(1e-8, np.abs(close_np) * 1e-6)
    return np.maximum(atr_np, floor)


def _future_window_extrema(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """
    Para cada i, retorna max/min na janela futura [i+1 .. i+horizon].
    """
    h = max(1, int(horizon))
    rev = series.astype(float).iloc[::-1]
    fut_max = rev.shift(1).rolling(h, min_periods=1).max().iloc[::-1]
    fut_min = rev.shift(1).rolling(h, min_periods=1).min().iloc[::-1]
    return fut_max.astype(float), fut_min.astype(float)


def _triple_barrier_side(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_abs: np.ndarray,
    candidates: np.ndarray,
    side: str,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    timeout_ret_atr_min: float,
    edge_risk_lambda: float,
    edge_clip_atr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(close.size)
    edge = np.zeros(n, dtype=np.float32)
    gate = np.zeros(n, dtype=np.float32)
    outcome = np.full(n, np.nan, dtype=np.float32)
    bars_to_hit = np.full(n, np.nan, dtype=np.float32)
    mfe_atr = np.full(n, np.nan, dtype=np.float32)
    mae_atr = np.full(n, np.nan, dtype=np.float32)
    timeout_ret_atr = np.full(n, np.nan, dtype=np.float32)

    if side not in {"long", "short"}:
        raise ValueError("side invalido (use 'long' ou 'short')")

    for i in np.flatnonzero(candidates):
        px0 = float(close[i])
        atr = float(atr_abs[i])
        if (not np.isfinite(px0)) or (not np.isfinite(atr)) or px0 <= 0.0 or atr <= 0.0:
            continue
        end = min(n - 1, i + int(horizon_bars))
        if end <= i:
            continue

        if side == "long":
            tp_px = px0 + float(tp_atr) * atr
            sl_px = px0 - float(sl_atr) * atr
        else:
            tp_px = px0 - float(tp_atr) * atr
            sl_px = px0 + float(sl_atr) * atr

        first_touch = 0  # +1 tp primeiro, -1 sl primeiro, 0 timeout/ambiguo
        touch_bars = float(end - i)

        best_mfe = 0.0
        best_mae = 0.0

        for j in range(i + 1, end + 1):
            hj = float(high[j])
            lj = float(low[j])
            cj = float(close[j])
            if side == "long":
                best_mfe = max(best_mfe, (hj - px0) / atr)
                best_mae = max(best_mae, (px0 - lj) / atr)
                hit_tp = hj >= tp_px
                hit_sl = lj <= sl_px
                if hit_tp and hit_sl:
                    if cj > px0:
                        first_touch = 1
                    elif cj < px0:
                        first_touch = -1
                    else:
                        first_touch = 0
                    touch_bars = float(j - i)
                    break
                if hit_tp:
                    first_touch = 1
                    touch_bars = float(j - i)
                    break
                if hit_sl:
                    first_touch = -1
                    touch_bars = float(j - i)
                    break
            else:
                best_mfe = max(best_mfe, (px0 - lj) / atr)
                best_mae = max(best_mae, (hj - px0) / atr)
                hit_tp = lj <= tp_px
                hit_sl = hj >= sl_px
                if hit_tp and hit_sl:
                    if cj < px0:
                        first_touch = 1
                    elif cj > px0:
                        first_touch = -1
                    else:
                        first_touch = 0
                    touch_bars = float(j - i)
                    break
                if hit_tp:
                    first_touch = 1
                    touch_bars = float(j - i)
                    break
                if hit_sl:
                    first_touch = -1
                    touch_bars = float(j - i)
                    break

        ret_at_timeout = 0.0
        if side == "long":
            ret_at_timeout = (float(close[end]) - px0) / atr
        else:
            ret_at_timeout = (px0 - float(close[end])) / atr

        raw_edge = best_mfe - float(edge_risk_lambda) * best_mae
        if raw_edge < 0.0:
            raw_edge = 0.0
        edge01 = raw_edge / max(1e-9, float(edge_clip_atr))
        if edge01 > 1.0:
            edge01 = 1.0
        edge[i] = np.float32(100.0 * edge01)

        if first_touch > 0:
            gate[i] = np.float32(100.0)
        elif first_touch < 0:
            gate[i] = np.float32(0.0)
        else:
            gate[i] = np.float32(100.0 if ret_at_timeout >= float(timeout_ret_atr_min) else 0.0)

        outcome[i] = np.float32(first_touch)
        bars_to_hit[i] = np.float32(touch_bars)
        mfe_atr[i] = np.float32(best_mfe)
        mae_atr[i] = np.float32(best_mae)
        timeout_ret_atr[i] = np.float32(ret_at_timeout)

    return edge, gate, outcome, bars_to_hit, mfe_atr, mae_atr, timeout_ret_atr


def _apply_reversal_labels(df: pd.DataFrame) -> pd.DataFrame:
    need = {"high", "low", "close"}
    if not need.issubset(set(df.columns)):
        print("[rev-labels] colunas OHLC insuficientes; mantendo labels existentes.", flush=True)
        return df

    out = df.copy()

    legacy_cols = [
        "edge_label_long",
        "edge_label_short",
        "entry_gate_long",
        "entry_gate_short",
        "edge_weight_long",
        "edge_weight_short",
        "entry_gate_weight_long",
        "entry_gate_weight_short",
        "entry_gate_weight",
    ]
    for col in legacy_cols:
        if col in out.columns and f"{col}_legacy" not in out.columns:
            out[f"{col}_legacy"] = out[col]

    high_s = out["high"].astype(float)
    low_s = out["low"].astype(float)
    close_s = out["close"].astype(float)
    close = close_s.to_numpy(dtype=float, copy=False)
    n = int(len(out))
    if n <= 3:
        print("[rev-labels] poucas linhas; mantendo labels existentes.", flush=True)
        return out

    atr_abs = _compute_atr_abs(out, span=int(REV_LABEL_CFG["atr_span"]))
    safe_atr = pd.Series(np.maximum(atr_abs, 1e-9), index=out.index)
    out["rev_atr_abs"] = pd.Series(atr_abs.astype(np.float32), index=out.index)
    out["rev_atr_pct"] = pd.Series((atr_abs / np.maximum(np.abs(close), 1e-9)).astype(np.float32), index=out.index)

    raw_scenarios = REV_LABEL_CFG.get("scenarios", ())
    scenarios: list[tuple[int, float, float]] = []
    for row in raw_scenarios:
        try:
            h, tp, sl = row
            h_i = max(1, int(h))
            tp_f = float(tp)
            sl_f = float(sl)
            if tp_f > 0.0 and sl_f > 0.0:
                scenarios.append((h_i, tp_f, sl_f))
        except Exception:
            continue
    if not scenarios:
        scenarios = [(8, 0.80, 0.55)]
    scenarios = sorted(scenarios, key=lambda x: x[0])

    future_high_cache: dict[int, pd.Series] = {}
    future_low_cache: dict[int, pd.Series] = {}

    def _future_hl(horizon_bars: int) -> tuple[pd.Series, pd.Series]:
        h_i = max(1, int(horizon_bars))
        if h_i not in future_high_cache:
            fut_high, _ = _future_window_extrema(high_s, h_i)
            _, fut_low = _future_window_extrema(low_s, h_i)
            future_high_cache[h_i] = fut_high
            future_low_cache[h_i] = fut_low
        return future_high_cache[h_i], future_low_cache[h_i]

    confirm_bars = max(1, int(REV_LABEL_CFG.get("confirm_bars", 3)))
    confirm_move_thr = float(REV_LABEL_CFG.get("confirm_move_atr_min", 0.20))
    confirm_max_adv_thr = float(REV_LABEL_CFG.get("confirm_max_adverse_atr", 0.35))
    first_bar_move_thr = float(REV_LABEL_CFG.get("first_bar_move_atr_min", 0.00))

    fut_high_c, fut_low_c = _future_hl(confirm_bars)
    next_close = close_s.shift(-1)

    first_ret_long_s = (next_close - close_s) / safe_atr
    first_ret_short_s = (close_s - next_close) / safe_atr
    confirm_move_long_s = (fut_high_c - close_s) / safe_atr
    confirm_move_short_s = (close_s - fut_low_c) / safe_atr
    confirm_adv_long_s = (close_s - fut_low_c) / safe_atr
    confirm_adv_short_s = (fut_high_c - close_s) / safe_atr

    full_confirm = close_s.shift(-confirm_bars).notna() & next_close.notna()
    cand_long_s = (
        full_confirm
        & first_ret_long_s.notna()
        & confirm_move_long_s.notna()
        & confirm_adv_long_s.notna()
        & (first_ret_long_s >= first_bar_move_thr)
        & (confirm_move_long_s >= confirm_move_thr)
        & (confirm_adv_long_s <= confirm_max_adv_thr)
    )
    cand_short_s = (
        full_confirm
        & first_ret_short_s.notna()
        & confirm_move_short_s.notna()
        & confirm_adv_short_s.notna()
        & (first_ret_short_s >= first_bar_move_thr)
        & (confirm_move_short_s >= confirm_move_thr)
        & (confirm_adv_short_s <= confirm_max_adv_thr)
    )
    cand_long = cand_long_s.to_numpy(dtype=bool, copy=False)
    cand_short = cand_short_s.to_numpy(dtype=bool, copy=False)

    pass_any_long = np.zeros(n, dtype=bool)
    pass_any_short = np.zeros(n, dtype=bool)
    pass_count_long = np.zeros(n, dtype=np.int16)
    pass_count_short = np.zeros(n, dtype=np.int16)
    bars_l = np.full(n, np.nan, dtype=np.float32)
    bars_s = np.full(n, np.nan, dtype=np.float32)

    for h_i, tp_atr, sl_atr in scenarios:
        fut_high_h, fut_low_h = _future_hl(h_i)
        full_h = close_s.shift(-h_i).notna()
        move_up_s = (fut_high_h - close_s) / safe_atr
        move_dn_s = (close_s - fut_low_h) / safe_atr

        pass_l_s = (
            full_h
            & move_up_s.notna()
            & move_dn_s.notna()
            & (move_up_s >= float(tp_atr))
            & (move_dn_s < float(sl_atr))
        )
        pass_s_s = (
            full_h
            & move_up_s.notna()
            & move_dn_s.notna()
            & (move_dn_s >= float(tp_atr))
            & (move_up_s < float(sl_atr))
        )

        pass_l = pass_l_s.to_numpy(dtype=bool, copy=False)
        pass_s = pass_s_s.to_numpy(dtype=bool, copy=False)

        pass_any_long |= pass_l
        pass_any_short |= pass_s
        pass_count_long += pass_l.astype(np.int16, copy=False)
        pass_count_short += pass_s.astype(np.int16, copy=False)

        add_l = pass_l & np.isnan(bars_l)
        add_s = pass_s & np.isnan(bars_s)
        bars_l[add_l] = np.float32(h_i)
        bars_s[add_s] = np.float32(h_i)

    conf_score_long = (confirm_move_long_s - confirm_adv_long_s).to_numpy(dtype=float, copy=False)
    conf_score_short = (confirm_move_short_s - confirm_adv_short_s).to_numpy(dtype=float, copy=False)
    score_min = float(REV_LABEL_CFG.get("confirm_score_atr_min", 0.60))

    max_h = max(h_i for h_i, _, _ in scenarios)
    fut_high_max, fut_low_max = _future_hl(max_h)
    move_up_max = ((fut_high_max - close_s) / safe_atr).to_numpy(dtype=float, copy=False)
    move_dn_max = ((close_s - fut_low_max) / safe_atr).to_numpy(dtype=float, copy=False)

    n_scen = max(1, len(scenarios))
    min_pass = int(REV_LABEL_CFG.get("min_scenarios_pass", 3))
    min_pass = max(1, min(min_pass, n_scen))
    min_pass_frac = float(REV_LABEL_CFG.get("scenario_pass_frac_min", 1.00))
    pass_ratio_long = pass_count_long.astype(float) / float(n_scen)
    pass_ratio_short = pass_count_short.astype(float) / float(n_scen)

    edge_lambda = float(REV_LABEL_CFG.get("profit_edge_lambda", 1.00))
    edge_min = float(REV_LABEL_CFG.get("profit_edge_atr_min", 1.20))
    rr_min = float(REV_LABEL_CFG.get("profit_rr_min", 2.20))
    eps_rr = 1e-6
    profit_edge_long = move_up_max - edge_lambda * move_dn_max
    profit_edge_short = move_dn_max - edge_lambda * move_up_max
    profit_rr_long = move_up_max / np.maximum(move_dn_max, eps_rr)
    profit_rr_short = move_dn_max / np.maximum(move_up_max, eps_rr)

    gate_long = (
        cand_long
        & pass_any_long
        & (pass_count_long >= min_pass)
        & (pass_ratio_long >= min_pass_frac)
        & np.isfinite(conf_score_long)
        & np.isfinite(profit_edge_long)
        & np.isfinite(profit_rr_long)
        & (conf_score_long >= score_min)
        & (profit_edge_long >= edge_min)
        & (profit_rr_long >= rr_min)
    )
    gate_short = (
        cand_short
        & pass_any_short
        & (pass_count_short >= min_pass)
        & (pass_ratio_short >= min_pass_frac)
        & np.isfinite(conf_score_short)
        & np.isfinite(profit_edge_short)
        & np.isfinite(profit_rr_short)
        & (conf_score_short >= score_min)
        & (profit_edge_short >= edge_min)
        & (profit_rr_short >= rr_min)
    )

    both_gate = gate_long & gate_short
    if bool(np.any(both_gate)):
        qual_long = conf_score_long + 0.50 * profit_edge_long
        qual_short = conf_score_short + 0.50 * profit_edge_short
        prefer_long = qual_long >= qual_short
        gate_long[both_gate & (~prefer_long)] = False
        gate_short[both_gate & prefer_long] = False

    gate_l = np.where(gate_long, 100.0, 0.0).astype(np.float32, copy=False)
    gate_s = np.where(gate_short, 100.0, 0.0).astype(np.float32, copy=False)

    w_min = float(REV_LABEL_CFG["weight_min"])
    w_max = float(REV_LABEL_CFG["weight_max"])
    w_pos_base = float(REV_LABEL_CFG.get("weight_pos_base", 1.10))
    w_pos_confirm_gain = float(REV_LABEL_CFG.get("weight_pos_confirm_gain", 0.60))
    w_pos_scenario_gain = float(REV_LABEL_CFG.get("weight_pos_scenario_gain", 0.55))
    w_hard_negative = max(w_min, float(REV_LABEL_CFG.get("weight_hard_negative", 1.60)))

    den_confirm = max(1e-9, confirm_move_thr if confirm_move_thr > 0.0 else 0.10)
    den_first = max(1e-9, abs(first_bar_move_thr) + 0.10)
    first_ret_long = first_ret_long_s.to_numpy(dtype=float, copy=False)
    first_ret_short = first_ret_short_s.to_numpy(dtype=float, copy=False)
    confirm_move_long = confirm_move_long_s.to_numpy(dtype=float, copy=False)
    confirm_move_short = confirm_move_short_s.to_numpy(dtype=float, copy=False)

    confirm_strength_long = np.clip((confirm_move_long - confirm_move_thr) / den_confirm, 0.0, 3.0)
    confirm_strength_short = np.clip((confirm_move_short - confirm_move_thr) / den_confirm, 0.0, 3.0)
    first_strength_long = np.clip((first_ret_long - first_bar_move_thr) / den_first, 0.0, 3.0)
    first_strength_short = np.clip((first_ret_short - first_bar_move_thr) / den_first, 0.0, 3.0)

    weight_long = np.full(n, w_min, dtype=np.float32)
    weight_short = np.full(n, w_min, dtype=np.float32)
    pos_long = gate_long
    pos_short = gate_short

    if bool(np.any(pos_long)):
        raw_w_long = (
            w_pos_base
            + w_pos_confirm_gain * confirm_strength_long[pos_long]
            + w_pos_scenario_gain * pass_count_long[pos_long].astype(float)
            + 0.25 * first_strength_long[pos_long]
        )
        weight_long[pos_long] = np.clip(raw_w_long, w_min, w_max).astype(np.float32, copy=False)
    if bool(np.any(pos_short)):
        raw_w_short = (
            w_pos_base
            + w_pos_confirm_gain * confirm_strength_short[pos_short]
            + w_pos_scenario_gain * pass_count_short[pos_short].astype(float)
            + 0.25 * first_strength_short[pos_short]
        )
        weight_short[pos_short] = np.clip(raw_w_short, w_min, w_max).astype(np.float32, copy=False)

    hard_neg_long = cand_long & (~gate_long)
    hard_neg_short = cand_short & (~gate_short)
    if bool(np.any(hard_neg_long)):
        weight_long[hard_neg_long] = np.clip(
            np.maximum(weight_long[hard_neg_long], w_hard_negative),
            w_min,
            w_max,
        ).astype(np.float32, copy=False)
    if bool(np.any(hard_neg_short)):
        weight_short[hard_neg_short] = np.clip(
            np.maximum(weight_short[hard_neg_short], w_hard_negative),
            w_min,
            w_max,
        ).astype(np.float32, copy=False)

    mfe_l = move_up_max.astype(np.float32, copy=False)
    mae_l = move_dn_max.astype(np.float32, copy=False)
    mfe_s = move_dn_max.astype(np.float32, copy=False)
    mae_s = move_up_max.astype(np.float32, copy=False)
    timeout_close = close_s.shift(-max_h)
    ret_l = ((timeout_close - close_s) / safe_atr).to_numpy(dtype=np.float32, copy=False)
    ret_s = ((close_s - timeout_close) / safe_atr).to_numpy(dtype=np.float32, copy=False)

    for arr in (mfe_l, mae_l, mfe_s, mae_s):
        arr[~np.isfinite(arr)] = np.nan
        arr[arr < 0.0] = 0.0
    for arr in (ret_l, ret_s):
        arr[~np.isfinite(arr)] = np.nan

    bars_l = np.where(gate_long, bars_l, np.nan).astype(np.float32, copy=False)
    bars_s = np.where(gate_short, bars_s, np.nan).astype(np.float32, copy=False)

    outcome_l = np.full(n, np.nan, dtype=np.float32)
    outcome_s = np.full(n, np.nan, dtype=np.float32)
    outcome_l[gate_long] = np.float32(1.0)
    outcome_s[gate_short] = np.float32(1.0)
    outcome_l[cand_long & (~gate_long)] = np.float32(-1.0)
    outcome_s[cand_short & (~gate_short)] = np.float32(-1.0)

    out["rev_drop_atr"] = pd.Series(confirm_move_long.astype(np.float32, copy=False), index=out.index)
    out["rev_rise_atr"] = pd.Series(confirm_move_short.astype(np.float32, copy=False), index=out.index)
    out["rev_cand_long"] = pd.Series((cand_long.astype(np.float32) * 100.0), index=out.index)
    out["rev_cand_short"] = pd.Series((cand_short.astype(np.float32) * 100.0), index=out.index)
    out["rev_first_ret_atr_long"] = pd.Series(first_ret_long.astype(np.float32, copy=False), index=out.index)
    out["rev_first_ret_atr_short"] = pd.Series(first_ret_short.astype(np.float32, copy=False), index=out.index)
    out["rev_confirm_adv_atr_long"] = pd.Series(
        confirm_adv_long_s.to_numpy(dtype=np.float32, copy=False),
        index=out.index,
    )
    out["rev_confirm_adv_atr_short"] = pd.Series(
        confirm_adv_short_s.to_numpy(dtype=np.float32, copy=False),
        index=out.index,
    )
    out["rev_pass_count_long"] = pd.Series(pass_count_long.astype(np.float32, copy=False), index=out.index)
    out["rev_pass_count_short"] = pd.Series(pass_count_short.astype(np.float32, copy=False), index=out.index)
    out["rev_pass_ratio_long"] = pd.Series(pass_ratio_long.astype(np.float32, copy=False), index=out.index)
    out["rev_pass_ratio_short"] = pd.Series(pass_ratio_short.astype(np.float32, copy=False), index=out.index)
    out["rev_profit_edge_long"] = pd.Series(profit_edge_long.astype(np.float32, copy=False), index=out.index)
    out["rev_profit_edge_short"] = pd.Series(profit_edge_short.astype(np.float32, copy=False), index=out.index)
    out["rev_profit_rr_long"] = pd.Series(profit_rr_long.astype(np.float32, copy=False), index=out.index)
    out["rev_profit_rr_short"] = pd.Series(profit_rr_short.astype(np.float32, copy=False), index=out.index)

    out = out.drop(columns=["edge_label_long", "edge_label_short", "edge_weight_long", "edge_weight_short"], errors="ignore")
    out["entry_gate_long"] = pd.Series(gate_l, index=out.index)
    out["entry_gate_short"] = pd.Series(gate_s, index=out.index)
    out["entry_gate_weight_long"] = pd.Series(weight_long, index=out.index)
    out["entry_gate_weight_short"] = pd.Series(weight_short, index=out.index)
    out["entry_gate_weight"] = pd.Series(np.maximum(weight_long, weight_short), index=out.index)

    out["rev_outcome_long"] = pd.Series(outcome_l, index=out.index)
    out["rev_outcome_short"] = pd.Series(outcome_s, index=out.index)
    out["rev_bars_to_hit_long"] = pd.Series(bars_l, index=out.index)
    out["rev_bars_to_hit_short"] = pd.Series(bars_s, index=out.index)
    out["rev_mfe_atr_long"] = pd.Series(mfe_l, index=out.index)
    out["rev_mfe_atr_short"] = pd.Series(mfe_s, index=out.index)
    out["rev_mae_atr_long"] = pd.Series(mae_l, index=out.index)
    out["rev_mae_atr_short"] = pd.Series(mae_s, index=out.index)
    out["rev_timeout_ret_atr_long"] = pd.Series(ret_l, index=out.index)
    out["rev_timeout_ret_atr_short"] = pd.Series(ret_s, index=out.index)
    return out


def _print_reversal_stats(df: pd.DataFrame) -> None:
    if "rev_cand_long" not in df.columns:
        return
    cand_long = (df["rev_cand_long"].to_numpy(dtype=float, copy=False) > 0.0)
    cand_short = (df["rev_cand_short"].to_numpy(dtype=float, copy=False) > 0.0)
    gate_long = (df["entry_gate_long"].to_numpy(dtype=float, copy=False) >= 50.0) if "entry_gate_long" in df.columns else np.zeros(len(df), dtype=bool)
    gate_short = (df["entry_gate_short"].to_numpy(dtype=float, copy=False) >= 50.0) if "entry_gate_short" in df.columns else np.zeros(len(df), dtype=bool)

    n = float(max(1, len(df)))
    print("\n[rev-labels] resumo", flush=True)
    print(f"  candidatos long : {cand_long.sum():>6d} ({cand_long.sum()/n:.2%})", flush=True)
    print(f"  candidatos short: {cand_short.sum():>6d} ({cand_short.sum()/n:.2%})", flush=True)
    print(f"  gate long>=50   : {gate_long.sum():>6d} ({gate_long.sum()/n:.2%}) | cond cand={gate_long[cand_long].mean() if cand_long.any() else 0.0:.2%}", flush=True)
    print(f"  gate short>=50  : {gate_short.sum():>6d} ({gate_short.sum()/n:.2%}) | cond cand={gate_short[cand_short].mean() if cand_short.any() else 0.0:.2%}", flush=True)

def _save_reversal_debug(df: pd.DataFrame, symbol: str) -> None:
    if not SAVE_REVERSAL_DEBUG:
        return
    out_dir = Path("data/generated/labels").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = "".join(ch for ch in str(symbol) if ch.isalnum() or ch in {"_", "-"}).strip("_-") or "symbol"

    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "entry_gate_long",
        "entry_gate_short",
        "entry_gate_weight",
        "rev_atr_abs",
        "rev_atr_pct",
        "rev_drop_atr",
        "rev_rise_atr",
        "rev_first_ret_atr_long",
        "rev_first_ret_atr_short",
        "rev_confirm_adv_atr_long",
        "rev_confirm_adv_atr_short",
        "rev_pass_count_long",
        "rev_pass_count_short",
        "rev_pass_ratio_long",
        "rev_pass_ratio_short",
        "rev_profit_edge_long",
        "rev_profit_edge_short",
        "rev_profit_rr_long",
        "rev_profit_rr_short",
        "rev_cand_long",
        "rev_cand_short",
        "rev_outcome_long",
        "rev_outcome_short",
        "rev_bars_to_hit_long",
        "rev_bars_to_hit_short",
        "rev_mfe_atr_long",
        "rev_mfe_atr_short",
        "rev_mae_atr_long",
        "rev_mae_atr_short",
        "rev_timeout_ret_atr_long",
        "rev_timeout_ret_atr_short",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    all_path = out_dir / f"reversal_labels_debug_{safe_symbol}.parquet"
    df.loc[:, cols].to_parquet(all_path)

    mask = None
    if {"rev_cand_long", "rev_cand_short"}.issubset(df.columns):
        mask = (df["rev_cand_long"] > 0.0) | (df["rev_cand_short"] > 0.0)
    if mask is not None:
        events = df.loc[mask, cols]
        events_path = out_dir / f"reversal_events_{safe_symbol}.parquet"
        events.to_parquet(events_path)
        print(f"[rev-labels] debug salvo: {events_path} ({len(events)} eventos)", flush=True)
    print(f"[rev-labels] debug salvo: {all_path}", flush=True)


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
    # Labels auxiliares por lado (escala 0..100)
    _print_stats_for("timing_label_long")
    _print_stats_for("timing_label_short")
    # Label principal do classificador (pipeline reversao)
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
        "weights": "entry_gate_weight",
        "weights_side": "entry_gate_weight",
        "label": "entry_gate",
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
            if tname.startswith("entry_gate"):
                group = "entry_gate"
            else:
                group = "__hidden__"
        elif panel == "weights":
            if tname in {"entry_gate_weight_long", "entry_gate_weight_short"}:
                group = "entry_gate_weight"
            else:
                group = "__hidden__"
        elif panel == "weights_side":
            if tname in {"entry_gate_weight_long", "entry_gate_weight_short"}:
                group = "entry_gate_weight"
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

    # Substitui labels supervisionados por labels de reversao baseados em eventos.
    try:
        df = _apply_reversal_labels(df)
    except Exception as e:
        print(f"[rev-labels] falhou ao construir labels de reversao: {type(e).__name__}: {e}", flush=True)

    # estatistica dos labels (assinado + long/short)
    try:
        _print_label_stats(df)
        _print_reversal_stats(df)
    except Exception as e:
        print(f"[labels] falhou ao imprimir estatisticas: {type(e).__name__}: {e}", flush=True)

    try:
        _save_reversal_debug(df, sym)
    except Exception as e:
        print(f"[rev-labels] falhou ao salvar debug: {type(e).__name__}: {e}", flush=True)

    # opcional: filtra colunas para ficar apenas com as features selecionadas
    allow = getattr(cfg, "FEATURE_ALLOWLIST", None)
    if allow:
        allow_set = set(str(x) for x in allow)
        keep = []
        for c in df.columns:
            if c in {"open", "high", "low", "close", "volume"}:
                keep.append(c)
                continue
            if str(c).startswith(("timing_", "label_", "exit_", "edge_", "entry_gate_", "rev_")):
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
