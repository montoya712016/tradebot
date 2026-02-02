# -*- coding: utf-8 -*-
from typing import Tuple
import os
from pathlib import Path

import numpy as np, pandas as pd

if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = Path(__file__).resolve().parents[2].parent / "cache_sniper" / "numba"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache_dir)

from numba import njit

# ===== Entry confirmation (avoid buying into active downtrends) =====
CONFIRM_REQUIRE = True
CONFIRM_REQUIRE_UPCLOSE = True
CONFIRM_REQUIRE_ABOVE_EMA = True
CONFIRM_REQUIRE_EMA_UP = True
CONFIRM_EMA_DIV = 6  # span ~= window/6 (in minutes)
CONFIRM_MIN_SPAN = 5
CONFIRM_MIN_UP_PCT = 0.0005  # 0.05% acima do close anterior
INTRABAR_CONFIRM = True
INTRABAR_MIN_CLOSE_POS = 0.60  # close >= 60% do range (reversao confirmada)
UNCONFIRMED_WEIGHT_MULT = 2.0
SHORT_CONFIRM_REQUIRE = True

# Peso continuo por retorno (mais foco em negativos)
RET_WEIGHT_POS_MULT = 1.0
RET_WEIGHT_NEG_MULT = 1.0
RET_WEIGHT_CLIP = 5.0
RET_WEIGHT_MIN_SCALE = 0.005
RET_WEIGHT_DEADZONE = 0.003  # zona morta em torno do limiar (ex.: 0.3%)
RET_WEIGHT_POWER = 1.6  # >1 favorece extremos (mais peso em negativos fortes)
# Penalidade extra para retorno abaixo de 0 (quedas fortes)
RET_WEIGHT_NEG_ZERO_MULT = 4.0
RET_WEIGHT_NEG_ZERO_POWER = 2.0
RET_WEIGHT_NEG_ZERO_CLIP = 10.0
WEIGHT_LOG_COMPRESS = True
WEIGHT_LOG_K = 5.0

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT


@njit(cache=True)
def _simulate_entry_contract_numba(
    close: np.ndarray,
    horizon_bars: int,
    min_profit_pct: float,
    max_dd_pct: float,
    entry_weight_alpha: float,
    entry_weight_drop_mult: float,
    entry_weight_drop_min_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    labels = np.zeros(n, np.uint8)
    mae = np.zeros(n, np.float32)
    exit_code = np.zeros(n, np.int8)
    exit_wait = np.zeros(n, np.int32)
    weights = np.zeros(n, np.float32)
    rets = np.zeros(n, np.float32)
    if horizon_bars < 0:
        horizon_bars = 0
    if max_dd_pct < 0:
        max_dd_pct = 0.0

    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            mae[i] = 0.0
            exit_code[i] = 0
            exit_wait[i] = 0
            continue

        min_close = px0
        max_close = px0
        best_mae = 0.0
        label = 0
        code = 0
        exit_bar = i
        dipped_below_entry = False
        if horizon_bars <= 0:
            last_bar = n - 1
        else:
            last_bar = i + horizon_bars
            if last_bar >= n:
                last_bar = n - 1
        code = 0
        exit_bar = last_bar
        if last_bar <= i:
            r = 0.0
            rets[i] = float(r)
        else:
            sum_px = 0.0
            cnt = 0
            for j in range(i + 1, last_bar + 1):
                cl = close[j]
                if not np.isfinite(cl):
                    cl = close[j]
                if cl < min_close:
                    min_close = cl
                if cl > max_close:
                    max_close = cl
                cur_mae = (min_close / px0) - 1.0
                if cur_mae < best_mae:
                    best_mae = cur_mae
                if cl < px0:
                    dipped_below_entry = True
                sum_px += cl
                cnt += 1
            if cnt > 0:
                mean_px = sum_px / float(cnt)
            else:
                mean_px = px0
            r = (mean_px / px0) - 1.0
            rets[i] = float(r)
        scale = float(entry_weight_alpha) if float(entry_weight_alpha) > 1e-9 else 0.01
        w = 1.0
        if dipped_below_entry and float(entry_weight_drop_mult) > 0.0:
            if float(entry_weight_drop_min_pct) <= 0.0 or float(best_mae) <= -float(entry_weight_drop_min_pct):
                drop_pct = max(0.0, -float(best_mae)) * 100.0
                mult = 1.0 + (float(entry_weight_drop_mult) * float(drop_pct))
                if mult < 1.0:
                    mult = 1.0
                w = float(w) * float(mult)
        weights[i] = float(w)
        # Condicao 1: retorno medio futuro >= min_profit_pct
        # Condicao 2: nao pode cair abaixo do entry
        if (r >= min_profit_pct) and (not dipped_below_entry):
            label = 1
        else:
            label = 0

        labels[i] = label
        mae[i] = best_mae
        exit_code[i] = code
        exit_wait[i] = max(0, exit_bar - i)

    return labels, mae, exit_code, exit_wait, weights, rets


@njit(cache=True)
def _simulate_entry_contract_short_numba(
    close: np.ndarray,
    horizon_bars: int,
    min_profit_pct: float,
    max_dd_pct: float,
    entry_weight_alpha: float,
    entry_weight_drop_mult: float,
    entry_weight_drop_min_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    labels = np.zeros(n, np.uint8)
    mae = np.zeros(n, np.float32)
    exit_code = np.zeros(n, np.int8)
    exit_wait = np.zeros(n, np.int32)
    weights = np.zeros(n, np.float32)
    rets = np.zeros(n, np.float32)
    if horizon_bars < 0:
        horizon_bars = 0
    if max_dd_pct < 0:
        max_dd_pct = 0.0

    for i in range(n):
        px0 = close[i]
        if not np.isfinite(px0) or px0 <= 0.0:
            mae[i] = 0.0
            exit_code[i] = 0
            exit_wait[i] = 0
            continue

        min_close = px0
        max_close = px0
        best_mae = 0.0
        label = 0
        code = 0
        exit_bar = i
        dipped_above_entry = False
        if horizon_bars <= 0:
            last_bar = n - 1
        else:
            last_bar = i + horizon_bars
            if last_bar >= n:
                last_bar = n - 1

        if last_bar <= i:
            r = 0.0
            rets[i] = float(r)
        else:
            sum_px = 0.0
            cnt = 0
            for j in range(i + 1, last_bar + 1):
                cl = close[j]
                if not np.isfinite(cl):
                    cl = close[j]
                if cl < min_close:
                    min_close = cl
                if cl > max_close:
                    max_close = cl
                cur_mae = (max_close / px0) - 1.0
                if cur_mae > best_mae:
                    best_mae = cur_mae
                if cl > px0:
                    dipped_above_entry = True
                sum_px += cl
                cnt += 1
            if cnt > 0:
                mean_px = sum_px / float(cnt)
            else:
                mean_px = px0
            r = (px0 / mean_px) - 1.0
            rets[i] = float(r)

        scale = float(entry_weight_alpha) if float(entry_weight_alpha) > 1e-9 else 0.01
        w = 1.0
        if dipped_above_entry and float(entry_weight_drop_mult) > 0.0:
            if float(entry_weight_drop_min_pct) <= 0.0 or float(best_mae) >= float(entry_weight_drop_min_pct):
                drop_pct = max(0.0, float(best_mae)) * 100.0
                mult = 1.0 + (float(entry_weight_drop_mult) * float(drop_pct))
                if mult < 1.0:
                    mult = 1.0
                w = float(w) * float(mult)
        weights[i] = float(w)
        # Condicao 1: retorno medio futuro >= min_profit_pct
        # Condicao 2: nao pode subir acima do entry
        if (r >= min_profit_pct) and (not dipped_above_entry):
            label = 1
        else:
            label = 0

        labels[i] = label
        mae[i] = best_mae
        exit_code[i] = code
        exit_wait[i] = max(0, exit_bar - i)

    return labels, mae, exit_code, exit_wait, weights, rets


def apply_trade_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
) -> pd.DataFrame:
    """
    Gera labels binarios baseados no contrato fixo (sniper).
    Entry_label usa retorno medio futuro (mean close) e bloqueia se o DD ultrapassa o limite.

    Novas colunas:
        - sniper_long_label (uint8)  [janela unica para compat]
        - sniper_mae_pct (float32)
        - sniper_exit_code (int8)
        - sniper_exit_wait_bars (int32)
        - sniper_long_label_{Xm} (uint8) para cada janela (ex.: 30m/120m/360m)
        - sniper_long_weight_{Xm} (float32) para cada janela
        - sniper_mae_pct_{Xm} (float32)
        - sniper_long_ret_pct_{Xm} (float32)
        - sniper_short_label_{Xm} (uint8)
        - sniper_short_weight_{Xm} (float32)
        - sniper_short_ret_pct_{Xm} (float32)
        - sniper_mae_pct_short_{Xm} (float32)
        - sniper_exit_code_{Xm} (int8)
        - sniper_exit_wait_bars_{Xm} (int32)
        - sniper_long_weight_{Xm} (float32)
    """
    if contract is None:
        contract = DEFAULT_TRADE_CONTRACT

    candle_seconds = candle_sec or contract.timeframe_sec
    candle_seconds = max(1, int(candle_seconds))

    close = df["close"].to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)

    # labels (retorno medio futuro e DD por janela)
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    profits = list(getattr(contract, "entry_label_min_profit_pcts", []) or [])
    dd_limits = list(getattr(contract, "entry_label_max_dd_pcts", []) or [])
    # VALIDATE_ENTRY_WINDOWS
    if len(windows) < 1 or len(profits) < 1 or len(windows) != len(profits):
        raise ValueError("entry_label_windows_minutes e entry_label_min_profit_pcts devem ter o mesmo tamanho (>=1)")
    if not dd_limits:
        dd_limits = list(profits)
    if len(dd_limits) != len(windows):
        raise ValueError("entry_label_max_dd_pcts deve ter o mesmo tamanho de entry_label_windows_minutes")
    labels_by_win = []
    # gap detection (stocks: evita overnight)
    forbid_gap = bool(getattr(contract, "forbid_exit_on_gap", False))
    gap_hours = float(getattr(contract, "gap_hours_forbidden", 0.0) or 0.0)
    gap_next = None
    if forbid_gap and gap_hours > 0:
        idx = pd.to_datetime(df.index)
        gth = pd.Timedelta(hours=gap_hours)
        n = len(idx)
        gap_next = np.full(n, -1, dtype=np.int32)
        next_gap = -1
        for i in range(n - 1, 0, -1):
            if idx[i] - idx[i - 1] >= gth:
                next_gap = i
            gap_next[i - 1] = next_gap
    for w_min, pmin, ddmax in zip(windows, profits, dd_limits):
        hb = int(max(1, round((float(w_min) * 60.0) / float(candle_seconds))))
        entry_label, mae_pct, exit_code, exit_wait, weight, ret = _simulate_entry_contract_numba(
            close,
            int(hb),
            float(pmin),
            float(ddmax),
            float(getattr(contract, "entry_label_weight_alpha", 1.0) or 1.0),
            float(getattr(contract, "entry_label_weight_drop_mult", 1.0) or 1.0),
            float(getattr(contract, "entry_label_weight_drop_min_pct", 0.0) or 0.0),
        )
        entry_label_s, mae_pct_s, exit_code_s, exit_wait_s, weight_s, ret_s = _simulate_entry_contract_short_numba(
            close,
            int(hb),
            float(pmin),
            float(ddmax),
            float(getattr(contract, "entry_label_weight_alpha", 1.0) or 1.0),
            float(getattr(contract, "entry_label_weight_drop_mult", 1.0) or 1.0),
            float(getattr(contract, "entry_label_weight_drop_min_pct", 0.0) or 0.0),
        )
        if gap_next is not None:
            exit_bar = np.arange(exit_wait.size, dtype=np.int64) + exit_wait.astype(np.int64)
            hit_gap = (gap_next >= 0) & (gap_next <= exit_bar)
            if hit_gap.any():
                entry_label = entry_label.copy()
                exit_code = exit_code.copy()
                entry_label[hit_gap] = 0
                exit_code[hit_gap] = -4
                entry_label_s = entry_label_s.copy()
                entry_label_s[hit_gap] = 0
        # Ajuste de peso continuo por retorno (mais foco em negativos)
        if ret is not None and len(ret) == len(weight):
            scale = max(float(RET_WEIGHT_MIN_SCALE), float(getattr(contract, "entry_label_weight_alpha", 1.0) or 1.0))
            margin = ret - float(pmin)
            dist = np.maximum(0.0, np.abs(margin) - float(RET_WEIGHT_DEADZONE)) / float(scale)
            dist = np.minimum(float(RET_WEIGHT_CLIP), dist)
            curve = np.power(dist, float(RET_WEIGHT_POWER))
            pos = curve * (margin > 0.0)
            # so pesa negativos "de verdade" (ret < 0); abaixo do limiar mas positivo ~0
            neg = curve * ((margin < 0.0) & (ret < 0.0))
            # extra: retorno abaixo de 0 recebe peso mais alto (queda forte)
            neg_zero = np.maximum(0.0, -ret) / float(scale)
            neg_zero = np.minimum(float(RET_WEIGHT_NEG_ZERO_CLIP), neg_zero)
            neg_zero = np.power(neg_zero, float(RET_WEIGHT_NEG_ZERO_POWER))
            weight = weight.copy()
            weight *= (
                1.0
                + (float(RET_WEIGHT_POS_MULT) * pos)
                + (float(RET_WEIGHT_NEG_MULT) * neg)
                + (float(RET_WEIGHT_NEG_ZERO_MULT) * neg_zero)
            )
        if ret_s is not None and len(ret_s) == len(weight_s):
            scale = max(float(RET_WEIGHT_MIN_SCALE), float(getattr(contract, "entry_label_weight_alpha", 1.0) or 1.0))
            margin = ret_s - float(pmin)
            dist = np.maximum(0.0, np.abs(margin) - float(RET_WEIGHT_DEADZONE)) / float(scale)
            dist = np.minimum(float(RET_WEIGHT_CLIP), dist)
            curve = np.power(dist, float(RET_WEIGHT_POWER))
            pos = curve * (margin > 0.0)
            neg = curve * ((margin < 0.0) & (ret_s < 0.0))
            neg_zero = np.maximum(0.0, -ret_s) / float(scale)
            neg_zero = np.minimum(float(RET_WEIGHT_NEG_ZERO_CLIP), neg_zero)
            neg_zero = np.power(neg_zero, float(RET_WEIGHT_NEG_ZERO_POWER))
            weight_s = weight_s.copy()
            weight_s *= (
                1.0
                + (float(RET_WEIGHT_POS_MULT) * pos)
                + (float(RET_WEIGHT_NEG_MULT) * neg)
                + (float(RET_WEIGHT_NEG_ZERO_MULT) * neg_zero)
            )

        if CONFIRM_REQUIRE:
            span = max(int(CONFIRM_MIN_SPAN), int(round(float(w_min) / float(CONFIRM_EMA_DIV))))
            close_s = pd.Series(close, index=df.index)
            ema_fast = close_s.ewm(span=int(span), adjust=False).mean()
            confirm_mask = pd.Series(True, index=df.index)
            if CONFIRM_REQUIRE_UPCLOSE:
                prev = close_s.shift(1)
                min_up = prev * float(1.0 + CONFIRM_MIN_UP_PCT)
                confirm_mask &= close_s > min_up
            if CONFIRM_REQUIRE_ABOVE_EMA:
                confirm_mask &= close_s > ema_fast
            if CONFIRM_REQUIRE_EMA_UP:
                confirm_mask &= ema_fast.diff() > 0
            if INTRABAR_CONFIRM and ("high" in df.columns) and ("low" in df.columns):
                rng = np.maximum(1e-12, high - low)
                close_pos = (close - low) / rng
                confirm_mask &= close_pos >= float(INTRABAR_MIN_CLOSE_POS)
            confirm_np = confirm_mask.to_numpy(dtype=bool, copy=False)
            if np.any(~confirm_np):
                if UNCONFIRMED_WEIGHT_MULT > 1.0:
                    weight = weight.copy()
                    weight[~confirm_np] = weight[~confirm_np] * float(UNCONFIRMED_WEIGHT_MULT)
        if SHORT_CONFIRM_REQUIRE:
            span = max(int(CONFIRM_MIN_SPAN), int(round(float(w_min) / float(CONFIRM_EMA_DIV))))
            close_s = pd.Series(close, index=df.index)
            ema_fast = close_s.ewm(span=int(span), adjust=False).mean()
            confirm_mask_s = pd.Series(True, index=df.index)
            if CONFIRM_REQUIRE_UPCLOSE:
                prev = close_s.shift(1)
                max_dn = prev * float(1.0 - CONFIRM_MIN_UP_PCT)
                confirm_mask_s &= close_s < max_dn
            if CONFIRM_REQUIRE_ABOVE_EMA:
                confirm_mask_s &= close_s < ema_fast
            if CONFIRM_REQUIRE_EMA_UP:
                confirm_mask_s &= ema_fast.diff() < 0
            if INTRABAR_CONFIRM and ("high" in df.columns) and ("low" in df.columns):
                rng = np.maximum(1e-12, high - low)
                close_pos = (close - low) / rng
                confirm_mask_s &= close_pos <= float(1.0 - INTRABAR_MIN_CLOSE_POS)
            confirm_np_s = confirm_mask_s.to_numpy(dtype=bool, copy=False)
            if np.any(~confirm_np_s):
                if UNCONFIRMED_WEIGHT_MULT > 1.0:
                    weight_s = weight_s.copy()
                    weight_s[~confirm_np_s] = weight_s[~confirm_np_s] * float(UNCONFIRMED_WEIGHT_MULT)

        if WEIGHT_LOG_COMPRESS:
            # compressao logaritmica simples (ajuste via WEIGHT_LOG_K)
            wpos = np.maximum(0.0, weight) * float(WEIGHT_LOG_K)
            weight = np.log1p(wpos).astype(np.float32, copy=False)
            wpos_s = np.maximum(0.0, weight_s) * float(WEIGHT_LOG_K)
            weight_s = np.log1p(wpos_s).astype(np.float32, copy=False)
        suffix = f"{int(w_min)}m"
        ret_pct = None
        if ret is not None and len(ret) == len(weight):
            try:
                ret_pct = (ret.astype(np.float32, copy=False) * 100.0).astype(np.float32, copy=False)
            except Exception:
                ret_pct = (ret * 100.0).astype(np.float32, copy=False)
        ret_pct_s = None
        if ret_s is not None and len(ret_s) == len(weight_s):
            try:
                ret_pct_s = (ret_s.astype(np.float32, copy=False) * 100.0).astype(np.float32, copy=False)
            except Exception:
                ret_pct_s = (ret_s * 100.0).astype(np.float32, copy=False)
        df[f"sniper_long_label_{suffix}"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
        df[f"sniper_mae_pct_{suffix}"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
        if ret_pct is not None:
            df[f"sniper_long_ret_pct_{suffix}"] = pd.Series(ret_pct, index=df.index)
        df[f"sniper_short_label_{suffix}"] = pd.Series(entry_label_s.astype(np.uint8), index=df.index)
        df[f"sniper_mae_pct_short_{suffix}"] = pd.Series(mae_pct_s.astype(np.float32), index=df.index)
        if ret_pct_s is not None:
            df[f"sniper_short_ret_pct_{suffix}"] = pd.Series(ret_pct_s, index=df.index)
        df[f"sniper_exit_code_{suffix}"] = pd.Series(exit_code.astype(np.int8), index=df.index)
        df[f"sniper_exit_wait_bars_{suffix}"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
        df[f"sniper_long_weight_{suffix}"] = pd.Series(weight.astype(np.float32), index=df.index)
        df[f"sniper_short_weight_{suffix}"] = pd.Series(weight_s.astype(np.float32), index=df.index)
        labels_by_win.append((suffix, entry_label, mae_pct, exit_code, exit_wait, weight, ret_pct))

    # compat: usa a primeira janela para colunas legadas
    suffix, entry_label, mae_pct, exit_code, exit_wait, weight, ret_pct = labels_by_win[0]
    df["sniper_long_label"] = pd.Series(entry_label.astype(np.uint8), index=df.index)
    df["sniper_mae_pct"] = pd.Series(mae_pct.astype(np.float32), index=df.index)
    if ret_pct is not None and f"sniper_long_ret_pct_{suffix}" in df.columns:
        df["sniper_long_ret_pct"] = df[f"sniper_long_ret_pct_{suffix}"].astype(np.float32)
    df["sniper_exit_code"] = pd.Series(exit_code.astype(np.int8), index=df.index)
    df["sniper_exit_wait_bars"] = pd.Series(exit_wait.astype(np.int32), index=df.index)
    if f"sniper_long_weight_{suffix}" in df.columns:
        df["sniper_long_weight"] = df[f"sniper_long_weight_{suffix}"].astype(np.float32)
    if f"sniper_short_label_{suffix}" in df.columns:
        df["sniper_long_label_short"] = df[f"sniper_short_label_{suffix}"].astype(np.uint8)
    if f"sniper_short_ret_pct_{suffix}" in df.columns:
        df["sniper_long_ret_pct_short"] = df[f"sniper_short_ret_pct_{suffix}"].astype(np.float32)
    if f"sniper_short_weight_{suffix}" in df.columns:
        df["sniper_long_weight_short"] = df[f"sniper_short_weight_{suffix}"].astype(np.float32)
    return df
