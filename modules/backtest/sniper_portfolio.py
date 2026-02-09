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
from typing import Dict, List, Tuple
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


@dataclass
class SymbolData:
    df: pd.DataFrame  # precisa ter index + close/high/low
    p_entry: np.ndarray
    p_danger: np.ndarray
    # p_exit unused (mantido por compatibilidade).
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
    use_ema_exit = ema_span > 0
    ema_alpha = 2.0 / float(ema_span + 1) if use_ema_exit else 0.0
    ema_offset = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)
    ema = float(entry_price) * (1.0 - ema_offset) if use_ema_exit else 0.0
    exit_streak = 0
    for j in range(entry_i + 1, j_limit + 1):
        px = float(close[j])
        if not np.isfinite(px) or px <= 0.0:
            continue
        hi = float(high[j]) if np.isfinite(high[j]) else px
        lo = float(low[j]) if np.isfinite(low[j]) else px
        time_in_trade = int(j - entry_i)
        if use_ema_exit:
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

    def _next_entry(sym: str) -> tuple[pd.Timestamp, float, int, str, float, float, int] | None:
        sd = symbols[sym]
        df = sd.df
        idx = sd.idx if (sd.idx is not None) else pd.to_datetime(df.index)
        n = int(len(df))
        i = int(ptr[sym])
        while i < n:
            pe = float(sd.p_entry[i]) if np.isfinite(sd.p_entry[i]) else 0.0
            best_win = None
            if sd.entry_windows_minutes is not None and len(sd.entry_windows_minutes) > 0:
                best_win = sd.entry_windows_minutes[0]
            ema_span = int(max(1, round((float(best_win or 0.0) * 60.0) / float(max(1, candle_sec))))) if best_win else 0

            if pe >= float(sd.tau_entry):
                sc = _rank_score(pe, 0.0, cfg.rank_mode)
                ptr[sym] = i
                return pd.to_datetime(idx[i]), float(sc), int(i), float(pe), float(sd.tau_entry), int(ema_span)
            i += 1
        ptr[sym] = n
        return None

    for sym in list(symbols.keys()):
        nxt = _next_entry(sym)
        if nxt is None:
            continue
        ts, sc, i, pe0, te0, ema_span = nxt
        heapq.heappush(entry_heap, (ts, -sc, sym, i, pe0, te0, ema_span))

    # heap de posições abertas: (exit_ts, symbol, entry_ts, weight, r_net, reason, num_adds)
    open_heap: list[tuple[pd.Timestamp, str, pd.Timestamp, float, float, str, int]] = []
    open_set: set[str] = set()
    used_exposure = 0.0

    eq = 1.0
    eq_events: list[tuple[pd.Timestamp, float]] = []
    out_trades: list[ExecutedTrade] = []

    def _close_until(t: pd.Timestamp) -> None:
        nonlocal eq, used_exposure
        while open_heap and open_heap[0][0] <= t:
            exit_ts, sym, entry_ts, w, r_net, reason, num_adds = heapq.heappop(open_heap)
            if sym in open_set:
                open_set.remove(sym)
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

            remaining = float(cfg.total_exposure) - float(used_exposure)
            if remaining <= 1e-9:
                ptr[_sym] = int(_i) + 1
                nxt = _next_entry(_sym)
                if nxt is not None:
                    nts, sc, ni, pe0, te0, ema_span = nxt
                    heapq.heappush(entry_heap, (nts, -sc, _sym, ni, pe0, te0, ema_span))
                continue

            desired = float(cfg.total_exposure) / float(max(1, len(open_set) + 1))
            w = float(min(float(cfg.max_trade_exposure), remaining, desired))
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

    if not eq_events:
        eq_events = [(pd.Timestamp.utcnow(), float(eq))]

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

    return PortfolioBacktestResult(trades=out_trades, equity_curve=eq_daily, max_dd=max_dd)


__all__ = [
    "PortfolioConfig",
    "SymbolData",
    "ExecutedTrade",
    "PortfolioBacktestResult",
    "simulate_portfolio",
]

