# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Backtest de portfólio multi-cripto (carteira única) para o Sniper.

Objetivo:
- Operar vários símbolos ao mesmo tempo com capital compartilhado
- Selecionar "melhores" entradas no mesmo timestamp sem olhar o futuro
- Aumentar exposição quando há poucas posições abertas e distribuir quando há muitas

Estratégia (sem vazamento):
- Em cada timestamp de sinal, ranqueia candidatos por um score contemporâneo (ex.: p_entry)
- Aceita até max_positions e respeita orçamento de exposição total (total_exposure)
- Tamanho por trade é definido no momento da entrada usando apenas estado atual:
    desired = total_exposure / (open_positions + 1)
    w = min(max_trade_exposure, remaining_budget, desired)

Observação importante:
- Não há rebalanceamento de posições já abertas (evita suposições complexas).
- Equity é atualizada quando trades fecham (event-driven).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import heapq

import numpy as np
import pandas as pd

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window


@dataclass
class PortfolioConfig:
    max_positions: int = 10
    total_exposure: float = 1.0
    max_trade_exposure: float = 0.25
    min_trade_exposure: float = 0.02
    rank_mode: str = "p_entry"
    exit_min_hold_bars: int = 0
    exit_confirm_bars: int = 1
    # Filtro de diversificacao por correlacao entre candidatos do mesmo timestamp.
    corr_filter_enabled: bool = False
    corr_window_bars: int = 144
    corr_min_obs: int = 96
    corr_max_with_market: float = 0.85
    corr_max_pair: float = 0.92
    corr_keep_top_n: int = 0
    corr_abs: bool = True
    corr_debug: bool = False
    # Diversificacao contra o livro JA aberto.
    corr_open_filter_enabled: bool = False
    corr_open_window_bars: int = 144
    corr_open_min_obs: int = 96
    corr_open_reduce_start: float = 0.60
    corr_open_hard_reject: float = 0.92
    corr_open_min_weight_mult: float = 0.20
    # Exposure multiplier: quando > 0, total_exposure efetivo = equity * multiplier
    # (substitui cap fixo por cap proporcional ao valor atual da carteira).
    exposure_multiplier: float = 0.0


@dataclass
class CandidateTrade:
    symbol: str
    entry_i: int
    entry_ts: pd.Timestamp
    exit_i: int
    exit_ts: pd.Timestamp
    r_net: float
    num_adds: int
    reason: str
    score: float


@dataclass
class ExecutedTrade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    r_net: float
    weight: float
    reason: str
    num_adds: int


@dataclass
class PortfolioBacktestResult:
    trades: List[ExecutedTrade]
    equity_curve: pd.Series  # daily
    max_dd: float
    diagnostics: dict[str, Any] | None = None
    corr_trace: pd.DataFrame | None = None


@dataclass
class SymbolData:
    df: pd.DataFrame  # precisa ter index + close/high/low
    p_entry: np.ndarray
    p_danger: np.ndarray
    # p_exit = span EMA previsto (em barras) por timestamp.
    p_exit: np.ndarray
    # thresholds (podem ser overrides globais na simulação)
    tau_entry: float
    tau_danger: float
    tau_add: float
    tau_danger_add: float
    tau_exit: float
    entry_windows_minutes: tuple[int, ...] | None = None
    # walk-forward: qual período usar em cada timestamp (mesma ordem da lista `periods`)
    period_id: np.ndarray | None = None
    # lista de PeriodModel do wf (para entry/danger on-the-fly)
    periods: list | None = None
    # caches para performance (preenchidos em `simulate_portfolio`)
    idx: pd.DatetimeIndex | None = None
    close: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None


def _safe_corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    n = int(np.sum(m))
    if n < int(min_obs) or n < 3:
        return float("nan")
    aa = a[m].astype(np.float64, copy=False)
    bb = b[m].astype(np.float64, copy=False)
    sa = float(np.std(aa))
    sb = float(np.std(bb))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = float(np.corrcoef(aa, bb)[0, 1])
    return r if np.isfinite(r) else float("nan")


def _window_returns(sd: SymbolData, i: int, bars: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    idx = sd.idx if (sd.idx is not None) else pd.to_datetime(sd.df.index)
    close = sd.close if (sd.close is not None) else sd.df["close"].to_numpy(np.float64, copy=False)
    if i <= 0 or i >= len(close):
        return idx[:0], np.empty((0,), dtype=np.float64)
    start = max(1, int(i) - int(max(2, bars)) + 1)
    sl_idx = idx[start : int(i) + 1]
    sl_px = close[start - 1 : int(i) + 1]
    if sl_px.size < 2:
        return idx[:0], np.empty((0,), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(sl_px.astype(np.float64, copy=False)))
    return pd.DatetimeIndex(sl_idx), rets


def _window_returns_at_ts(sd: SymbolData, ts: pd.Timestamp, bars: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    idx = sd.idx if (sd.idx is not None) else pd.to_datetime(sd.df.index)
    if len(idx) == 0:
        return pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)
    pos = int(np.searchsorted(idx.values, np.datetime64(pd.to_datetime(ts)), side="right") - 1)
    if pos <= 0:
        return pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)
    return _window_returns(sd, pos, bars)


def _corr_stats_against_open(
    *,
    sym: str,
    ts: pd.Timestamp,
    symbols: Dict[str, SymbolData],
    open_weights: dict[str, float],
    cfg: PortfolioConfig,
) -> dict[str, float]:
    if (not bool(cfg.corr_open_filter_enabled)) or (not open_weights):
        return {}
    sd = symbols.get(sym)
    if sd is None:
        return {}
    bars = int(max(4, int(cfg.corr_open_window_bars)))
    min_obs = int(max(3, int(cfg.corr_open_min_obs)))
    corr_abs = bool(cfg.corr_abs)
    idx_w, ret_w = _window_returns_at_ts(sd, ts, bars=bars)
    if ret_w.size == 0:
        return {}
    vals: list[tuple[str, float, float]] = []
    for osym, ow in open_weights.items():
        if not np.isfinite(float(ow)) or float(ow) <= 0.0 or osym == sym:
            continue
        osd = symbols.get(osym)
        if osd is None:
            continue
        oidx, oret = _window_returns_at_ts(osd, ts, bars=bars)
        if oret.size == 0:
            continue
        if len(idx_w) == len(oidx) and len(idx_w) > 0 and bool(np.array_equal(idx_w.values, oidx.values)):
            rc = _safe_corr(ret_w, oret, min_obs=min_obs)
        else:
            s1 = pd.Series(ret_w, index=idx_w)
            s2 = pd.Series(oret, index=oidx)
            j = s1.to_frame("a").join(s2.to_frame("b"), how="inner")
            rc = _safe_corr(j["a"].to_numpy(np.float64, copy=False), j["b"].to_numpy(np.float64, copy=False), min_obs=min_obs)
        if np.isfinite(rc):
            vals.append((osym, abs(float(rc)) if corr_abs else float(rc), float(ow)))
    if not vals:
        return {}
    corr_vals = np.asarray([v[1] for v in vals], dtype=np.float64)
    w_vals = np.asarray([max(0.0, v[2]) for v in vals], dtype=np.float64)
    w_sum = float(np.sum(w_vals))
    weighted_mean = float(np.sum(corr_vals * w_vals) / w_sum) if w_sum > 0.0 else float(np.mean(corr_vals))
    correlated_exposure = float(np.sum(w_vals[corr_vals >= float(max(0.0, cfg.corr_open_reduce_start))]))
    return {
        "open_corr_count": float(len(vals)),
        "open_max_corr": float(np.max(corr_vals)),
        "open_mean_corr": float(np.mean(corr_vals)),
        "open_weighted_mean_corr": float(weighted_mean),
        "open_correlated_exposure": float(correlated_exposure),
    }


def _filter_batch_by_correlation(
    batch: list[tuple[pd.Timestamp, float, str, int, float, float, int]],
    *,
    symbols: Dict[str, SymbolData],
    market_ret: pd.Series | None,
    cfg: PortfolioConfig,
) -> tuple[list[tuple[pd.Timestamp, float, str, int, float, float, int]], dict[str, float | int]]:
    if (not bool(cfg.corr_filter_enabled)) or len(batch) <= 1:
        return batch, {}

    win = int(max(4, int(cfg.corr_window_bars)))
    min_obs = int(max(3, int(cfg.corr_min_obs)))
    corr_abs = bool(cfg.corr_abs)
    max_mkt = float(cfg.corr_max_with_market)
    max_pair = float(cfg.corr_max_pair)
    keep_n = int(max(0, int(cfg.corr_keep_top_n)))

    # batch ja chega ordenado por score (melhor -> pior)
    out: list[tuple[pd.Timestamp, float, str, int, float, float, int]] = []
    wr_by_sym: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    mkt_corr_cache: dict[str, float] = {}

    def _corr_with_market(sym: str) -> float:
        if sym in mkt_corr_cache:
            return mkt_corr_cache[sym]
        if market_ret is None:
            mkt_corr_cache[sym] = float("nan")
            return mkt_corr_cache[sym]
        idx_w, ret_w = wr_by_sym[sym]
        if ret_w.size == 0:
            mkt_corr_cache[sym] = float("nan")
            return mkt_corr_cache[sym]
        mkt_w = market_ret.reindex(idx_w).to_numpy(dtype=np.float64, copy=False)
        r = _safe_corr(ret_w, mkt_w, min_obs=min_obs)
        mkt_corr_cache[sym] = float(r)
        return mkt_corr_cache[sym]

    for cand in batch:
        _ts, _neg_sc, sym, i, _pe0, _te0, _ema = cand
        sd = symbols.get(sym)
        if sd is None:
            continue
        wr_by_sym[sym] = _window_returns(sd, int(i), bars=win)

    for cand in batch:
        _ts, _neg_sc, sym, i, _pe0, _te0, _ema = cand
        idx_w, ret_w = wr_by_sym.get(sym, (pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)))
        if ret_w.size == 0:
            continue

        # Forca manter o topo de score, depois aplica filtro.
        if keep_n > 0 and len(out) < keep_n:
            out.append(cand)
            continue

        rm = _corr_with_market(sym)
        if np.isfinite(rm):
            rm_cmp = abs(rm) if corr_abs else rm
            if rm_cmp > max_mkt:
                continue

        ok = True
        for kept in out:
            ks = kept[2]
            k_idx, k_ret = wr_by_sym.get(ks, (pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)))
            if k_ret.size == 0:
                continue
            # Alinha por timestamp para pares de simbolos.
            if len(idx_w) == len(k_idx) and len(idx_w) > 0 and bool(np.array_equal(idx_w.values, k_idx.values)):
                rp = _safe_corr(ret_w, k_ret, min_obs=min_obs)
            else:
                s1 = pd.Series(ret_w, index=idx_w)
                s2 = pd.Series(k_ret, index=k_idx)
                j = s1.to_frame("a").join(s2.to_frame("b"), how="inner")
                rp = _safe_corr(j["a"].to_numpy(np.float64, copy=False), j["b"].to_numpy(np.float64, copy=False), min_obs=min_obs)
            if np.isfinite(rp):
                rp_cmp = abs(rp) if corr_abs else rp
                if rp_cmp > max_pair:
                    ok = False
                    break
        if ok:
            out.append(cand)

    kept = out if out else batch[:1]

    def _pair_mean_abs(symbols_list: list[str]) -> float:
        if len(symbols_list) < 2:
            return float("nan")
        vals: list[float] = []
        for i in range(len(symbols_list)):
            s1 = symbols_list[i]
            i1, r1 = wr_by_sym.get(s1, (pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)))
            if r1.size == 0:
                continue
            for j in range(i + 1, len(symbols_list)):
                s2 = symbols_list[j]
                i2, r2 = wr_by_sym.get(s2, (pd.DatetimeIndex([]), np.empty((0,), dtype=np.float64)))
                if r2.size == 0:
                    continue
                if len(i1) == len(i2) and len(i1) > 0 and bool(np.array_equal(i1.values, i2.values)):
                    rp = _safe_corr(r1, r2, min_obs=min_obs)
                else:
                    s1s = pd.Series(r1, index=i1)
                    s2s = pd.Series(r2, index=i2)
                    jn = s1s.to_frame("a").join(s2s.to_frame("b"), how="inner")
                    rp = _safe_corr(jn["a"].to_numpy(np.float64, copy=False), jn["b"].to_numpy(np.float64, copy=False), min_obs=min_obs)
                if np.isfinite(rp):
                    vals.append(abs(float(rp)) if corr_abs else float(rp))
        return float(np.mean(vals)) if vals else float("nan")

    batch_syms = [c[2] for c in batch if c[2] in wr_by_sym]
    kept_syms = [c[2] for c in kept if c[2] in wr_by_sym]
    batch_mkt: list[float] = []
    for s in batch_syms:
        rm = _corr_with_market(s)
        if np.isfinite(rm):
            batch_mkt.append(abs(float(rm)) if corr_abs else float(rm))
    kept_mkt: list[float] = []
    for s in kept_syms:
        rm = _corr_with_market(s)
        if np.isfinite(rm):
            kept_mkt.append(abs(float(rm)) if corr_abs else float(rm))

    stats: dict[str, float | int] = {
        "batch_size": int(len(batch)),
        "kept_size": int(len(kept)),
        "rejected_size": int(max(0, len(batch) - len(kept))),
        "accept_ratio": float(float(len(kept)) / float(max(1, len(batch)))),
        "batch_avg_pair_corr": float(_pair_mean_abs(batch_syms)),
        "kept_avg_pair_corr": float(_pair_mean_abs(kept_syms)),
        "batch_avg_market_corr": float(np.mean(batch_mkt)) if batch_mkt else float("nan"),
        "kept_avg_market_corr": float(np.mean(kept_mkt)) if kept_mkt else float("nan"),
    }

    if bool(cfg.corr_debug):
        print(
            f"[portfolio][corr] batch={int(stats['batch_size'])} kept={int(stats['kept_size'])} "
            f"acc={float(stats['accept_ratio']):.2f} "
            f"pair={float(stats['kept_avg_pair_corr']) if np.isfinite(float(stats['kept_avg_pair_corr'])) else float('nan'):.3f} "
            f"mkt={float(stats['kept_avg_market_corr']) if np.isfinite(float(stats['kept_avg_market_corr'])) else float('nan'):.3f}",
            flush=True,
        )
    return kept, stats


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_calibration(p: np.ndarray, calib: dict) -> np.ndarray:
    if not isinstance(calib, dict):
        return p
    if calib.get("type") != "platt":
        return p
    a = float(calib.get("coef", 1.0))
    b = float(calib.get("intercept", 0.0))
    prior_shift = float(calib.get("prior_shift", 0.0) or 0.0)
    space = str(calib.get("space", "prob")).strip().lower()
    pp = np.asarray(p, dtype=np.float64)
    if space == "logit":
        eps = 1e-6
        pp = np.clip(pp, eps, 1.0 - eps)
        z = a * np.log(pp / (1.0 - pp)) + b + prior_shift
    else:
        z = a * pp + b + prior_shift
    return _sigmoid(z)


def _rank_score(p_entry: float, p_danger: float, mode: str) -> float:
    if mode == "p_entry":
        return float(p_entry)
    return float(p_entry)


def _simulate_one_trade(
    sd: SymbolData,
    *,
    start_i: int,
    contract: TradeContract,
    candle_sec: int = 60,
    exit_min_hold_bars: int = 3,
    exit_confirm_bars: int = 1,
    p_entry_override: float | None = None,
    tau_entry_override: float | None = None,
    exit_ema_span_override: int | None = None,
) -> CandidateTrade | None:
    df = sd.df
    idx = sd.idx if (sd.idx is not None) else pd.to_datetime(df.index)
    n = int(len(df))
    if start_i < 0 or start_i >= n:
        return None

    close = sd.close if (sd.close is not None) else df["close"].to_numpy(np.float64, copy=False)
    high = sd.high if (sd.high is not None) else df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = sd.low if (sd.low is not None) else df.get("low", df["close"]).to_numpy(np.float64, copy=False)

    px0 = float(close[start_i])
    if not np.isfinite(px0) or px0 <= 0.0:
        return None

    # valida condição de entrada no candle start_i
    if p_entry_override is None:
        pe0 = float(sd.p_entry[start_i]) if np.isfinite(sd.p_entry[start_i]) else 0.0
    else:
        pe0 = float(p_entry_override)
    tau_entry_use = float(sd.tau_entry if tau_entry_override is None else tau_entry_override)
    if not (pe0 >= float(tau_entry_use)):
        return None

    entry_i = int(start_i)
    entry_price = px0
    avg_price = px0
    last_fill = px0
    total_size = float(contract.add_sizing[0]) if contract.add_sizing else 1.0
    num_adds = 0

    # agenda sizes
    size_sched = tuple(float(x) for x in contract.add_sizing) if contract.add_sizing else (1.0,)
    if len(size_sched) < contract.max_adds + 1:
        size_sched = size_sched + (size_sched[-1],) * (contract.max_adds + 1 - len(size_sched))

    j_limit = int(n - 1)
    min_exit_i = entry_i + int(max(0, int(exit_min_hold_bars)))
    if min_exit_i > j_limit:
        return None

    # percorre até fechar
    ema_span = int(
        exit_ema_span_override
        if exit_ema_span_override is not None
        else exit_ema_span_from_window(contract, int(candle_sec))
    )
    span_state = float(max(1, ema_span)) if ema_span > 0 else 0.0
    ema_alpha = 2.0 / float(span_state + 1.0) if span_state > 0.0 else 0.0
    ema_offset = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)
    ema = float(entry_price) * (1.0 - ema_offset) if span_state > 0.0 else 0.0
    span_smooth = 0.85
    span_min = 2.0
    span_max = float(max(2400, int(max(1, ema_span)) * 6))
    exit_streak = 0
    for j in range(entry_i + 1, j_limit + 1):
        px = float(close[j])
        if not np.isfinite(px) or px <= 0.0:
            continue
        hi = float(high[j]) if np.isfinite(high[j]) else px
        lo = float(low[j]) if np.isfinite(low[j]) else px
        time_in_trade = int(j - entry_i)
        if span_state > 0.0:
            try:
                pred_span = float(sd.p_exit[j]) if (sd.p_exit is not None and j < len(sd.p_exit)) else np.nan
            except Exception:
                pred_span = np.nan
            if np.isfinite(pred_span) and pred_span > 1.0:
                if pred_span < span_min:
                    pred_span = span_min
                if pred_span > span_max:
                    pred_span = span_max
                span_state = float((span_smooth * span_state) + ((1.0 - span_smooth) * pred_span))
                if span_state < span_min:
                    span_state = span_min
                if span_state > span_max:
                    span_state = span_max
                ema_alpha = 2.0 / float(span_state + 1.0)
            ema = ema + (ema_alpha * (px - ema))
            if px < ema:
                exit_streak += 1
            else:
                exit_streak = 0
            if exit_streak >= int(max(1, exit_confirm_bars)):
                exit_px = px
                reason = "EMA"
                exit_i = j
                break

        # adds (se ainda no ciclo)
        if num_adds < int(contract.max_adds):
            trigger = last_fill * (1.0 - float(contract.add_spacing_pct))
            if trigger > 0.0 and lo <= trigger:
                pe = float(sd.p_entry[j]) if np.isfinite(sd.p_entry[j]) else 0.0
                next_size = float(size_sched[num_adds + 1])
                risk_after = 0.0
                if (pe >= float(sd.tau_add)) and (risk_after <= float(contract.risk_max_cycle_pct) + 1e-9):
                    new_total = total_size + next_size
                    avg_price = (avg_price * total_size + trigger * next_size) / new_total
                    total_size = new_total
                    last_fill = trigger
                    num_adds += 1
    else:
        # se não quebrou, fecha no fim do range
        exit_i = int(j_limit)
        exit_px = float(close[exit_i]) if np.isfinite(close[exit_i]) else float(close[entry_i])
        reason = "EOD"

    # custos (mesma ideia do simulador): (entradas + saída) * fee+slippage
    entries = 1 + int(num_adds)
    sides = entries + 1
    costs = sides * float(contract.fee_pct_per_side + contract.slippage_pct)
    r = (float(exit_px) / float(avg_price)) - 1.0
    r_net = float(r - costs)

    score = _rank_score(pe0, 0.0, mode="p_entry")
    return CandidateTrade(
        symbol="",
        entry_i=int(entry_i),
        entry_ts=pd.to_datetime(idx[int(entry_i)]),
        exit_i=int(exit_i),
        exit_ts=pd.to_datetime(idx[int(exit_i)]),
        r_net=float(r_net),
        num_adds=int(num_adds),
        reason=str(reason),
        score=float(score),
    )


def simulate_portfolio(
    symbols: Dict[str, SymbolData],
    *,
    cfg: PortfolioConfig | None = None,
    contract: TradeContract | None = None,
    candle_sec: int = 60,
    # opcional: callback de progresso por timestamp "atual" do loop
    progress_cb: Callable[[pd.Timestamp], None] | None = None,
    progress_every: int = 200,
) -> PortfolioBacktestResult:
    cfg = cfg or PortfolioConfig()
    contract = contract or DEFAULT_TRADE_CONTRACT
    default_exit_span = int(max(1, exit_ema_span_from_window(contract, int(candle_sec))))

    # heap de próximos candidatos: (ts, -score, symbol, idx)
    entry_heap: list[tuple[pd.Timestamp, float, str, int, str, float, float, int]] = []

    # estado por símbolo: ponteiro do scan
    ptr: Dict[str, int] = {s: 0 for s in symbols.keys()}

    # precompute caches por símbolo (evita custo enorme repetido por trade)
    for sym, sd in symbols.items():
        if sd.idx is None:
            sd.idx = pd.to_datetime(sd.df.index)
        if sd.close is None:
            sd.close = sd.df["close"].to_numpy(np.float64, copy=False)
        if sd.high is None:
            sd.high = sd.df.get("high", sd.df["close"]).to_numpy(np.float64, copy=False)
        if sd.low is None:
            sd.low = sd.df.get("low", sd.df["close"]).to_numpy(np.float64, copy=False)

    market_ret: pd.Series | None = None
    if bool(cfg.corr_filter_enabled):
        # Mercado "geral" = media cross-sectional de retornos log por timestamp.
        # Alinha automaticamente por timestamp entre simbolos.
        ret_cols: list[pd.Series] = []
        for sym, sd in symbols.items():
            if sd.idx is None or sd.close is None:
                continue
            px = np.asarray(sd.close, dtype=np.float64)
            if px.size < 2:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.diff(np.log(px))
            if r.size < 2:
                continue
            ret_cols.append(pd.Series(r, index=sd.idx[1:], name=str(sym)))
        if ret_cols:
            market_ret = pd.concat(ret_cols, axis=1).mean(axis=1, skipna=True)

    def _next_entry(sym: str) -> tuple[pd.Timestamp, float, int, str, float, float, int] | None:
        sd = symbols[sym]
        df = sd.df
        idx = sd.idx if (sd.idx is not None) else pd.to_datetime(df.index)
        n = int(len(df))
        i = int(ptr[sym])
        pe_slice = sd.p_entry[i:]
        valid_mask = (pe_slice >= float(sd.tau_entry)) & np.isfinite(pe_slice)
        
        if not np.any(valid_mask):
            ptr[sym] = n
            return None
            
        first_valid_offset = int(np.argmax(valid_mask))
        match_i = i + first_valid_offset
        pe = float(sd.p_entry[match_i])
        
        best_win = None
        if sd.entry_windows_minutes is not None and len(sd.entry_windows_minutes) > 0:
            best_win = sd.entry_windows_minutes[0]
            
        if sd.p_exit is not None and match_i < len(sd.p_exit) and np.isfinite(sd.p_exit[match_i]) and float(sd.p_exit[match_i]) > 1.0:
            ema_span = int(max(2, round(float(sd.p_exit[match_i]))))
        else:
            ema_span = (
                int(max(1, round((float(best_win or 0.0) * 60.0) / float(max(1, candle_sec)))))
                if best_win
                else int(default_exit_span)
            )

        sc = _rank_score(pe, 0.0, cfg.rank_mode)
        ptr[sym] = match_i
        return pd.to_datetime(idx[match_i]), float(sc), int(match_i), float(pe), float(sd.tau_entry), int(ema_span)

    for sym in list(symbols.keys()):
        nxt = _next_entry(sym)
        if nxt is None:
            continue
        ts, sc, i, pe0, te0, ema_span = nxt
        heapq.heappush(entry_heap, (ts, -sc, sym, i, pe0, te0, ema_span))

    # heap de posições abertas: (exit_ts, symbol, entry_ts, weight, r_net, reason, num_adds)
    open_heap: list[tuple[pd.Timestamp, str, pd.Timestamp, float, float, str, int]] = []
    open_set: set[str] = set()
    open_weights: dict[str, float] = {}
    used_exposure = 0.0

    eq = 1.0
    eq_events: list[tuple[pd.Timestamp, float]] = []
    out_trades: list[ExecutedTrade] = []
    corr_rows: list[dict[str, Any]] = []
    corr_batches = 0
    corr_rejected = 0
    corr_total_candidates = 0
    corr_total_kept = 0

    def _close_until(t: pd.Timestamp) -> None:
        nonlocal eq, used_exposure
        while open_heap and open_heap[0][0] <= t:
            exit_ts, sym, entry_ts, w, r_net, reason, num_adds = heapq.heappop(open_heap)
            if sym in open_set:
                open_set.remove(sym)
                open_weights.pop(sym, None)
                used_exposure = max(0.0, float(used_exposure) - float(w))
            eq = float(eq) * (1.0 + float(w) * float(r_net))
            eq_events.append((pd.to_datetime(exit_ts), float(eq)))
            out_trades.append(
                ExecutedTrade(
                    symbol=str(sym),
                    entry_ts=pd.to_datetime(entry_ts),
                    exit_ts=pd.to_datetime(exit_ts),
                    r_net=float(r_net),
                    weight=float(w),
                    reason=str(reason),
                    num_adds=int(num_adds),
                )
            )

    iter_n = 0
    pevery = int(max(1, int(progress_every)))
    while entry_heap:
        ts, neg_sc, sym, i, pe0, te0, ema_span = heapq.heappop(entry_heap)
        t = pd.to_datetime(ts)
        iter_n += 1
        if progress_cb is not None and (iter_n % pevery == 0):
            try:
                progress_cb(pd.to_datetime(t))
            except Exception:
                pass
        _close_until(t)

        # agrupa todos os candidatos no mesmo timestamp
        batch = [(ts, neg_sc, sym, i, pe0, te0, ema_span)]
        while entry_heap and entry_heap[0][0] == ts:
            batch.append(heapq.heappop(entry_heap))

        # ordena por score desc (neg_sc asc)
        batch.sort(key=lambda x: x[1])
        batch, corr_stats = _filter_batch_by_correlation(batch, symbols=symbols, market_ret=market_ret, cfg=cfg)
        if corr_stats:
            corr_batches += 1
            corr_total_candidates += int(corr_stats.get("batch_size", 0) or 0)
            corr_total_kept += int(corr_stats.get("kept_size", 0) or 0)
            corr_rejected += int(corr_stats.get("rejected_size", 0) or 0)
            corr_rows.append({"ts": pd.to_datetime(t), **corr_stats})

        for _ts, _neg_sc, _sym, _i, _pe0, _te0, _ema_span in batch:
            if _sym in open_set:
                # já em posição
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            if int(cfg.max_positions) > 0 and len(open_set) >= int(cfg.max_positions):
                # sem capacidade; tenta a próxima ocorrência deste símbolo
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            if float(cfg.exposure_multiplier) > 0.0:
                effective_total_exposure = float(eq) * float(cfg.exposure_multiplier)
            else:
                effective_total_exposure = float(cfg.total_exposure)
            remaining = float(effective_total_exposure) - float(used_exposure)
            if remaining <= 1e-9:
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            desired = float(effective_total_exposure) / float(max(1, len(open_set) + 1))
            w = float(min(float(cfg.max_trade_exposure), remaining, desired))
            open_corr_stats = _corr_stats_against_open(
                sym=str(_sym),
                ts=pd.to_datetime(t),
                symbols=symbols,
                open_weights=open_weights,
                cfg=cfg,
            )
            if open_corr_stats:
                max_corr = float(open_corr_stats.get("open_max_corr", float("nan")))
                weighted_mean_corr = float(open_corr_stats.get("open_weighted_mean_corr", float("nan")))
                if np.isfinite(max_corr) and max_corr >= float(cfg.corr_open_hard_reject):
                    ptr[_sym] = int(_i) + 1
                    nxt = _next_entry(_sym)
                    if nxt is not None:
                        nts, sc, ni, pe0, te0, ema_span = nxt
                        heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                    corr_rows.append(
                        {
                            "ts": pd.to_datetime(t),
                            "open_filter_reject": 1,
                            "symbol": str(_sym),
                            **open_corr_stats,
                        }
                    )
                    continue
                if np.isfinite(weighted_mean_corr):
                    reduce_start = float(max(0.0, min(0.99, cfg.corr_open_reduce_start)))
                    if weighted_mean_corr > reduce_start:
                        rel = (weighted_mean_corr - reduce_start) / max(1e-9, 1.0 - reduce_start)
                        mult = 1.0 - rel
                        mult = float(max(float(cfg.corr_open_min_weight_mult), min(1.0, mult)))
                        w *= mult
                corr_rows.append(
                    {
                        "ts": pd.to_datetime(t),
                        "open_filter_reject": 0,
                        "symbol": str(_sym),
                        "post_weight": float(w),
                        **open_corr_stats,
                    }
                )
            if w < float(cfg.min_trade_exposure):
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            # aceita: simula trade deste símbolo a partir de _i
            sd = symbols[_sym]
            tr = _simulate_one_trade(
                sd,
                start_i=int(_i),
                contract=contract,
                candle_sec=int(candle_sec),
                exit_min_hold_bars=int(cfg.exit_min_hold_bars),
                exit_confirm_bars=int(cfg.exit_confirm_bars),
                p_entry_override=float(_pe0),
                tau_entry_override=float(_te0),
                exit_ema_span_override=int(_ema_span),
            )
            if tr is None:
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            tr.symbol = str(_sym)
            open_set.add(str(_sym))
            open_weights[str(_sym)] = float(w)
            used_exposure = float(used_exposure) + float(w)
            heapq.heappush(
                open_heap,
                (
                    pd.to_datetime(tr.exit_ts),
                    str(_sym),
                    pd.to_datetime(tr.entry_ts),
                    float(w),
                    float(tr.r_net),
                    str(tr.reason),
                    int(tr.num_adds),
                ),
            )

            # avança ponteiro do símbolo para depois do exit
            ptr[_sym] = int(tr.exit_i) + 1
            nxt = _next_entry(_sym)
            if nxt is not None:
                nts, sc, ni, pe0, te0, ema_span = nxt
                heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))

    # fecha tudo no final do último evento conhecido
    if open_heap:
        last_ts = max([x[0] for x in open_heap])
        _close_until(pd.to_datetime(last_ts))

    start_ts = None
    try:
        start_ts = min(pd.to_datetime(sd.df.index.min()) for sd in symbols.values() if sd.df is not None and len(sd.df))
    except Exception:
        start_ts = None
    if start_ts is None:
        start_ts = pd.Timestamp.utcnow()
    eq_events = [(pd.to_datetime(start_ts), 1.0)] + list(eq_events)

    eq_ser = pd.Series(
        data=[e for _, e in eq_events],
        index=pd.to_datetime([t for t, _ in eq_events]),
        name="equity",
    ).sort_index()
    eq_daily = eq_ser.resample("1D").last().ffill()

    # MDD em base diária (aprox. conservadora)
    eq_arr = eq_daily.to_numpy(np.float64, copy=False)
    if eq_arr.size:
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.nanmax(dd))
    else:
        max_dd = 0.0

    corr_trace_df: pd.DataFrame | None = None
    if corr_rows:
        corr_trace_df = pd.DataFrame(corr_rows)
        try:
            corr_trace_df["ts"] = pd.to_datetime(corr_trace_df["ts"])
            corr_trace_df = corr_trace_df.set_index("ts").sort_index()
        except Exception:
            pass

    diagnostics: dict[str, Any] | None = None
    if bool(cfg.corr_filter_enabled):
        diagnostics = {
            "corr_filter_enabled": True,
            "corr_batches": int(corr_batches),
            "corr_total_candidates": int(corr_total_candidates),
            "corr_total_kept": int(corr_total_kept),
            "corr_total_rejected": int(corr_rejected),
            "corr_accept_ratio": float(float(corr_total_kept) / float(max(1, corr_total_candidates))),
        }

    return PortfolioBacktestResult(
        trades=out_trades,
        equity_curve=eq_daily,
        max_dd=max_dd,
        diagnostics=diagnostics,
        corr_trace=corr_trace_df,
    )


__all__ = [
    "PortfolioConfig",
    "SymbolData",
    "ExecutedTrade",
    "PortfolioBacktestResult",
    "simulate_portfolio",
]

