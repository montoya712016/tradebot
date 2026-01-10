# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd

if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = Path(__file__).resolve().parents[2].parent / "cache_sniper" / "numba"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache_dir)

from numba import njit

try:
    from .labels import apply_trade_contract_labels
except Exception:
    from prepare_features.labels import apply_trade_contract_labels  # type: ignore[import]

try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    try:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]
    except Exception:
        from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT


@dataclass
class SniperDataset:
    entry: pd.DataFrame
    add: pd.DataFrame
    danger: pd.DataFrame
    exit: pd.DataFrame
    meta: Dict[str, float | int]


def _infer_candle_sec(idx: pd.DatetimeIndex) -> int:
    if len(idx) < 2:
        return 60
    try:
        dt = float((idx[1] - idx[0]).total_seconds())
        if not np.isfinite(dt) or dt <= 0.0:
            return 60
        return int(round(dt))
    except Exception:
        return 60


def _ensure_contract_labels(
    df: pd.DataFrame,
    *,
    contract: TradeContract,
    candle_sec: int,
) -> None:
    cols_needed = {
        "sniper_entry_label",
        "sniper_mae_pct",
        "sniper_exit_code",
        "sniper_exit_wait_bars",
        "sniper_danger_label",
    }
    if not cols_needed.issubset(df.columns):
        apply_trade_contract_labels(df, contract=contract, candle_sec=candle_sec)


def _normalize_size_schedule(add_sizing: Sequence[float], max_adds: int) -> Tuple[float, ...]:
    if not add_sizing:
        base = (1.0,)
    else:
        base = tuple(float(x) for x in add_sizing if float(x) > 0.0)
    if not base:
        base = (1.0,)
    if len(base) < max_adds + 1:
        last = base[-1]
        extra = tuple(last for _ in range(max_adds + 1 - len(base)))
        base = base + extra
    return base


def _collect_add_snapshots(
    df: pd.DataFrame,
    *,
    contract: TradeContract,
    candle_sec: int,
    max_add_starts: int = 20_000,
    seed: int = 42,
) -> Dict[str, List[float | int]]:
    close = df["close"].to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    entry_label = df["sniper_entry_label"].to_numpy(np.uint8, copy=False)
    exit_code = df["sniper_exit_code"].to_numpy(np.int8, copy=False)

    n = int(len(df))
    timeout_bars = int(contract.timeout_bars(candle_sec))
    spacing = float(contract.add_spacing_pct)
    tp_pct = float(contract.tp_min_pct)
    sl_pct = float(contract.sl_pct)
    max_adds = int(contract.max_adds)
    risk_cap = float(contract.risk_max_cycle_pct)
    size_schedule = np.asarray(_normalize_size_schedule(contract.add_sizing, max_adds), dtype=np.float64)

    # amostra starts para não fazer O(N*timeout) em séries enormes
    rng = np.random.default_rng(int(seed))
    starts_pos = np.flatnonzero(entry_label == 1)
    starts_neg = np.flatnonzero(entry_label == 0)
    max_add_starts = int(max_add_starts)
    if max_add_starts <= 0:
        max_add_starts = 1
    want_pos = min(starts_pos.size, max_add_starts // 2)
    want_neg = max_add_starts - want_pos
    sel_pos = starts_pos if want_pos >= starts_pos.size else rng.choice(starts_pos, size=int(want_pos), replace=False)
    sel_neg = starts_neg if want_neg >= starts_neg.size else rng.choice(starts_neg, size=int(want_neg), replace=False)
    start_idx = np.unique(np.concatenate([sel_pos, sel_neg]).astype(np.int64, copy=False))

    idx_j, num_adds_a, time_in_trade_a, dd_a, dtp_a, dsl_a, avg_a, last_a, r_used_a, r_if_a, lab_a, ex_a, cnt = _collect_add_snapshots_numba(
        close,
        high,
        low,
        entry_label,
        exit_code,
        start_idx,
        timeout_bars,
        tp_pct,
        sl_pct,
        spacing,
        max_adds,
        risk_cap,
        size_schedule,
    )

    if cnt <= 0:
        return dict(idx=[], num_adds=[], time_in_trade=[], dd_pct=[], dist_to_tp=[], dist_to_sl=[],
                    avg_entry_price=[], last_fill_price=[], risk_used_pct=[], risk_if_add_pct=[], label=[], exit_code=[])

    idx_j = idx_j[:cnt]
    snapshots: Dict[str, List[float | int]] = dict(
        idx=idx_j.astype(int).tolist(),
        num_adds=num_adds_a[:cnt].astype(int).tolist(),
        time_in_trade=time_in_trade_a[:cnt].astype(int).tolist(),
        dd_pct=dd_a[:cnt].astype(float).tolist(),
        dist_to_tp=dtp_a[:cnt].astype(float).tolist(),
        dist_to_sl=dsl_a[:cnt].astype(float).tolist(),
        avg_entry_price=avg_a[:cnt].astype(float).tolist(),
        last_fill_price=last_a[:cnt].astype(float).tolist(),
        risk_used_pct=r_used_a[:cnt].astype(float).tolist(),
        risk_if_add_pct=r_if_a[:cnt].astype(float).tolist(),
        label=lab_a[:cnt].astype(int).tolist(),
        exit_code=ex_a[:cnt].astype(int).tolist(),
    )
    return snapshots


@njit(cache=True)
def _hold_return_no_add_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    j: int,
    j_end: int,
    avg_price: float,
    tp_pct: float,
    sl_pct: float,
) -> float:
    """
    Retorno (bruto) de "segurar" a partir de j, assumindo:
    - não fazer mais adds
    - TP/SL fixos em torno de avg_price
    - se não bater TP/SL até j_end, sai no close[j_end]
    """
    if not np.isfinite(avg_price) or avg_price <= 0.0:
        return 0.0
    tp_price = avg_price * (1.0 + tp_pct)
    sl_price = avg_price * (1.0 - sl_pct)
    if j_end <= j:
        cj = close[j]
        return (cj / avg_price) - 1.0 if np.isfinite(cj) else 0.0
    for k in range(j + 1, j_end + 1):
        c = close[k]
        h = high[k]
        l = low[k]
        if not np.isfinite(c):
            break
        if not np.isfinite(h):
            h = c
        if not np.isfinite(l):
            l = c
        # ordem conservadora: SL primeiro
        if l <= sl_price:
            return -float(sl_pct)
        if h >= tp_price:
            return float(tp_pct)
    c_end = close[j_end]
    if not np.isfinite(c_end) or avg_price <= 0.0:
        return 0.0
    return (c_end / avg_price) - 1.0


@njit(cache=True)
def _collect_exit_snapshots_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_label: np.ndarray,
    start_idx: np.ndarray,
    timeout_bars: int,
    tp_pct: float,
    sl_pct: float,
    spacing: float,
    max_adds: int,
    risk_cap: float,
    size_schedule: np.ndarray,
    exit_stride_bars: int,
    exit_lookahead_bars: int,
    exit_margin_pct: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, int
]:
    """
    Snapshots "em posição" para treinar Exit model (com cycle_*).

    Label supervisionado (label_exit):
      Intenção: alinhar ao que você propôs (pico local pós-compra):
      - Para cada "compra" (start), escolhemos uma janela posterior (limitada ao tempo em posição).
      - Identificamos o máximo local (pico) dentro dessa janela.
      - label_exit=1 para snapshots próximos do pico (em preço e em tempo).
      - label_exit=0 para snapshots em que ainda há subida "em sequência" (tendência forte)
        e também para pontos após o pico.
    """
    n = close.size
    stride = int(exit_stride_bars)
    if stride <= 0:
        stride = 1
    look = int(exit_lookahead_bars)
    if look <= 0:
        look = 1

    approx_per_start = (timeout_bars // stride) + 3
    max_snap = max(1, start_idx.size * approx_per_start)

    idx_j = np.empty(max_snap, np.int32)
    num_adds_a = np.empty(max_snap, np.int32)
    time_in_trade_a = np.empty(max_snap, np.int32)
    dd_a = np.empty(max_snap, np.float32)
    dtp_a = np.empty(max_snap, np.float32)
    dsl_a = np.empty(max_snap, np.float32)
    avg_a = np.empty(max_snap, np.float64)
    last_a = np.empty(max_snap, np.float64)
    r_used_a = np.empty(max_snap, np.float32)
    r_if_a = np.empty(max_snap, np.float32)
    lab_exit_a = np.empty(max_snap, np.uint8)
    out_cnt = 0

    for si in range(start_idx.size):
        start = int(start_idx[si])
        if start < 0 or start >= n - 3:
            continue
        px0 = close[start]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue

        # referência (mantemos o acesso para reduzir risco de otimização enganosa do numba)
        _ = entry_label[start]

        avg_price = px0
        last_fill = px0
        total_size = float(size_schedule[0])
        num_adds_now = 0

        j_limit = start + timeout_bars
        if j_limit >= n:
            j_limit = n - 1

        # guardaremos onde começam os snapshots deste trade para rotular depois
        out_start = out_cnt
        trade_end = start  # último candle (índice) ainda "em posição" (não inclui candle de fechamento)

        for j in range(start + 1, j_limit + 1):
            c = close[j]
            h = high[j]
            l = low[j]
            if not np.isfinite(c):
                break
            if not np.isfinite(h):
                h = c
            if not np.isfinite(l):
                l = c

            t_in = j - start
            # fecha por contrato (não gera snapshot no candle de fechamento)
            if (l <= (avg_price * (1.0 - sl_pct))) or (h >= (avg_price * (1.0 + tp_pct))) or (t_in >= timeout_bars):
                trade_end = j - 1
                break

            # aplica add (para o snapshot refletir o estado atual)
            if num_adds_now < max_adds:
                trigger = last_fill * (1.0 - spacing)
                if trigger > 0.0 and l <= trigger:
                    next_size = float(size_schedule[num_adds_now + 1])
                    risk_after = (total_size + next_size) * sl_pct
                    if risk_after <= risk_cap + 1e-9:
                        new_total = total_size + next_size
                        avg_price = (avg_price * total_size + trigger * next_size) / new_total
                        total_size = new_total
                        last_fill = trigger
                        num_adds_now += 1

            # snapshot periódico
            if (t_in % stride) != 0:
                continue

            tp_price = avg_price * (1.0 + tp_pct)
            sl_price = avg_price * (1.0 - sl_pct)
            dd = (c / avg_price) - 1.0
            dist_tp = (tp_price - c) / tp_price if tp_price > 0.0 else 0.0
            dist_sl = (c - sl_price) / sl_price if sl_price > 0.0 else 0.0
            risk_used = total_size * sl_pct
            if num_adds_now < max_adds:
                next_size2 = float(size_schedule[num_adds_now + 1])
                risk_if = (total_size + next_size2) * sl_pct
            else:
                risk_if = risk_used

            idx_j[out_cnt] = j
            num_adds_a[out_cnt] = num_adds_now
            time_in_trade_a[out_cnt] = t_in
            dd_a[out_cnt] = dd
            dtp_a[out_cnt] = dist_tp
            dsl_a[out_cnt] = dist_sl
            avg_a[out_cnt] = avg_price
            last_a[out_cnt] = last_fill
            r_used_a[out_cnt] = risk_used
            r_if_a[out_cnt] = risk_if
            # rotulamos depois, quando soubermos o pico local do trade
            lab_exit_a[out_cnt] = np.uint8(0)
            out_cnt += 1

            if out_cnt >= max_snap:
                return (
                    idx_j,
                    num_adds_a,
                    time_in_trade_a,
                    dd_a,
                    dtp_a,
                    dsl_a,
                    avg_a,
                    last_a,
                    r_used_a,
                    r_if_a,
                    lab_exit_a,
                    out_cnt,
                )

        # Se o trade não "quebrou" pelo if acima (edge), garante trade_end dentro do range.
        if trade_end < start:
            trade_end = start
        if trade_end > j_limit:
            trade_end = j_limit

        # Rotulagem pós-trade: pico local e vizinhança
        if out_cnt > out_start:
            # janela posterior à compra, limitada ao tempo em posição
            win_end = start + look
            if win_end > trade_end:
                win_end = trade_end
            if win_end > start:
                # pico (última ocorrência do máximo) usando CLOSE
                peak_j = start + 1
                peak_px = close[peak_j]
                for k in range(start + 1, win_end + 1):
                    ck = close[k]
                    if not np.isfinite(ck):
                        break
                    if ck >= peak_px:
                        peak_px = ck
                        peak_j = k

                # parâmetros de "proximidade" (em tempo e preço)
                near_bars = int(max(60, 3 * stride))   # janela maior para aumentar positivos
                tol_px = peak_px * (1.0 - exit_margin_pct)
                profit_floor = max(0.003, exit_margin_pct * 0.5)  # lucro mínimo mais permissivo
                up_seq_len = 3  # "subida em sequência": 3 closes seguidos subindo

                for t in range(out_start, out_cnt):
                    js = int(idx_j[t])
                    if js <= 0:
                        continue
                    # após o pico => negativo
                    if js > peak_j:
                        lab_exit_a[t] = np.uint8(0)
                        continue
                    # muito longe do pico => negativo
                    if js < (peak_j - near_bars):
                        lab_exit_a[t] = np.uint8(0)
                        continue
                    # longe do preço do pico => negativo
                    if close[js] < tol_px:
                        lab_exit_a[t] = np.uint8(0)
                        continue
                    # precisa estar com lucro mínimo (usando avg_price do snapshot)
                    ap = avg_a[t]
                    if not np.isfinite(ap) or ap <= 0.0:
                        lab_exit_a[t] = np.uint8(0)
                        continue
                    if ((close[js] / ap) - 1.0) < profit_floor:
                        lab_exit_a[t] = np.uint8(0)
                        continue

                    # se ainda está em subida "em sequência", mantém negativo
                    if (js + up_seq_len) <= peak_j:
                        ok = True
                        prev = close[js]
                        for m in range(1, up_seq_len + 1):
                            cur = close[js + m]
                            if not np.isfinite(cur):
                                ok = False
                                break
                            if cur <= prev:
                                ok = False
                                break
                            prev = cur
                        if ok:
                            lab_exit_a[t] = np.uint8(0)
                            continue

                    # caso contrário: positivo (próximo do pico e sem subida forte contínua)
                    lab_exit_a[t] = np.uint8(1)

    return (
        idx_j,
        num_adds_a,
        time_in_trade_a,
        dd_a,
        dtp_a,
        dsl_a,
        avg_a,
        last_a,
        r_used_a,
        r_if_a,
        lab_exit_a,
        out_cnt,
    )


def _collect_exit_snapshots(
    df: pd.DataFrame,
    *,
    contract: TradeContract,
    candle_sec: int,
    max_exit_starts: int = 8_000,
    exit_stride_bars: int = 15,
    exit_lookahead_bars: int | None = None,
    exit_margin_pct: float = 0.006,
    seed: int = 42,
) -> Dict[str, List[float | int]]:
    close = df["close"].to_numpy(np.float64, copy=False)
    high = df.get("high", df["close"]).to_numpy(np.float64, copy=False)
    low = df.get("low", df["close"]).to_numpy(np.float64, copy=False)
    entry_label = df["sniper_entry_label"].to_numpy(np.uint8, copy=False)

    timeout_bars = int(contract.timeout_bars(candle_sec))
    spacing = float(contract.add_spacing_pct)
    tp_pct = float(contract.tp_min_pct)
    sl_pct = float(contract.sl_pct)
    max_adds = int(contract.max_adds)
    risk_cap = float(contract.risk_max_cycle_pct)
    size_schedule = np.asarray(_normalize_size_schedule(contract.add_sizing, max_adds), dtype=np.float64)

    rng = np.random.default_rng(int(seed) + 17)
    starts_pos = np.flatnonzero(entry_label == 1)
    max_exit_starts = int(max_exit_starts)
    if max_exit_starts <= 0:
        max_exit_starts = 1

    # IMPORTANT: Exit deve ser treinado em estados "realistas" (após entradas plausíveis).
    # Usar starts negativos (entry_label==0) polui o dataset com posições que você não teria aberto.
    if starts_pos.size == 0:
        return dict(
            idx=[],
            num_adds=[],
            time_in_trade=[],
            dd_pct=[],
            dist_to_tp=[],
            dist_to_sl=[],
            avg_entry_price=[],
            last_fill_price=[],
            risk_used_pct=[],
            risk_if_add_pct=[],
            label_exit=[],
        )

    want_pos = min(starts_pos.size, max_exit_starts)
    start_idx = starts_pos if want_pos >= starts_pos.size else rng.choice(starts_pos, size=int(want_pos), replace=False)
    start_idx = np.unique(start_idx.astype(np.int64, copy=False))

    if exit_lookahead_bars is None:
        # horizonte curto por padrão (evita ensinar o modelo a sair "cedo demais" olhando muito longe)
        exit_lookahead_bars = int(min(max(5, timeout_bars), 240))

    idx_j, num_adds_a, time_in_trade_a, dd_a, dtp_a, dsl_a, avg_a, last_a, r_used_a, r_if_a, lab_exit_a, cnt = _collect_exit_snapshots_numba(
        close,
        high,
        low,
        entry_label,
        start_idx,
        timeout_bars,
        tp_pct,
        sl_pct,
        spacing,
        max_adds,
        risk_cap,
        size_schedule,
        int(exit_stride_bars),
        int(exit_lookahead_bars),
        float(exit_margin_pct),
    )

    if cnt <= 0:
        return dict(
            idx=[],
            num_adds=[],
            time_in_trade=[],
            dd_pct=[],
            dist_to_tp=[],
            dist_to_sl=[],
            avg_entry_price=[],
            last_fill_price=[],
            risk_used_pct=[],
            risk_if_add_pct=[],
            label_exit=[],
        )

    idx_j = idx_j[:cnt]
    return dict(
        idx=idx_j.astype(int).tolist(),
        num_adds=num_adds_a[:cnt].astype(int).tolist(),
        time_in_trade=time_in_trade_a[:cnt].astype(int).tolist(),
        dd_pct=dd_a[:cnt].astype(float).tolist(),
        dist_to_tp=dtp_a[:cnt].astype(float).tolist(),
        dist_to_sl=dsl_a[:cnt].astype(float).tolist(),
        avg_entry_price=avg_a[:cnt].astype(float).tolist(),
        last_fill_price=last_a[:cnt].astype(float).tolist(),
        risk_used_pct=r_used_a[:cnt].astype(float).tolist(),
        risk_if_add_pct=r_if_a[:cnt].astype(float).tolist(),
        label_exit=lab_exit_a[:cnt].astype(int).tolist(),
    )


@njit(cache=True)
def _collect_add_snapshots_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_label: np.ndarray,
    exit_code: np.ndarray,
    start_idx: np.ndarray,
    timeout_bars: int,
    tp_pct: float,
    sl_pct: float,
    spacing: float,
    max_adds: int,
    risk_cap: float,
    size_schedule: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, int
]:
    n = close.size
    max_snap = start_idx.size * max_adds
    idx_j = np.empty(max_snap, np.int32)
    num_adds_a = np.empty(max_snap, np.int32)
    time_in_trade_a = np.empty(max_snap, np.int32)
    dd_a = np.empty(max_snap, np.float32)
    dtp_a = np.empty(max_snap, np.float32)
    dsl_a = np.empty(max_snap, np.float32)
    avg_a = np.empty(max_snap, np.float64)
    last_a = np.empty(max_snap, np.float64)
    r_used_a = np.empty(max_snap, np.float32)
    r_if_a = np.empty(max_snap, np.float32)
    lab_a = np.empty(max_snap, np.uint8)
    ex_a = np.empty(max_snap, np.int8)
    out_cnt = 0

    for si in range(start_idx.size):
        start = int(start_idx[si])
        if start < 0 or start >= n - 2:
            continue
        px0 = close[start]
        if not np.isfinite(px0) or px0 <= 0.0:
            continue

        avg_price = px0
        last_fill = px0
        total_size = float(size_schedule[0])
        num_adds_now = 0
        tp_price = avg_price * (1.0 + tp_pct)
        sl_price = avg_price * (1.0 - sl_pct)
        j_limit = start + timeout_bars
        if j_limit >= n:
            j_limit = n - 1

        for j in range(start + 1, j_limit + 1):
            c = close[j]
            h = high[j]
            l = low[j]
            if not np.isfinite(c):
                break
            if not np.isfinite(h):
                h = c
            if not np.isfinite(l):
                l = c

            t_in = j - start
            if (l <= sl_price) or (h >= tp_price) or (t_in >= timeout_bars):
                break

            if num_adds_now >= max_adds:
                continue

            trigger = last_fill * (1.0 - spacing)
            if trigger <= 0.0 or l > trigger:
                continue

            next_size = float(size_schedule[num_adds_now + 1])
            risk_before = total_size * sl_pct
            risk_after = (total_size + next_size) * sl_pct
            if risk_after > risk_cap + 1e-9:
                continue

            dd = (c / avg_price) - 1.0
            dist_tp = (tp_price - c) / tp_price if tp_price > 0.0 else 0.0
            dist_sl = (c - sl_price) / sl_price if sl_price != 0.0 else 0.0

            idx_j[out_cnt] = j
            num_adds_a[out_cnt] = num_adds_now
            time_in_trade_a[out_cnt] = t_in
            dd_a[out_cnt] = dd
            dtp_a[out_cnt] = dist_tp
            dsl_a[out_cnt] = dist_sl
            avg_a[out_cnt] = avg_price
            last_a[out_cnt] = last_fill
            r_used_a[out_cnt] = risk_before
            r_if_a[out_cnt] = risk_after
            lab_a[out_cnt] = entry_label[j]
            ex_a[out_cnt] = exit_code[j]
            out_cnt += 1

            # aplica add (sempre, para evoluir o estado e gerar próximos snapshots)
            new_total = total_size + next_size
            avg_price = (avg_price * total_size + trigger * next_size) / new_total
            total_size = new_total
            last_fill = trigger
            num_adds_now += 1
            tp_price = avg_price * (1.0 + tp_pct)
            sl_price = avg_price * (1.0 - sl_pct)

            if out_cnt >= max_snap:
                return idx_j, num_adds_a, time_in_trade_a, dd_a, dtp_a, dsl_a, avg_a, last_a, r_used_a, r_if_a, lab_a, ex_a, out_cnt

    return idx_j, num_adds_a, time_in_trade_a, dd_a, dtp_a, dsl_a, avg_a, last_a, r_used_a, r_if_a, lab_a, ex_a, out_cnt


def build_sniper_datasets(
    df: pd.DataFrame,
    *,
    contract: TradeContract | None = None,
    candle_sec: int | None = None,
    max_add_starts: int = 20_000,
    max_exit_starts: int = 8_000,
    exit_stride_bars: int = 15,
    exit_lookahead_bars: int | None = None,
    # Margem do exit: em 1m, 0.2% costuma ser pequeno demais e gera label_exit quase morto.
    exit_margin_pct: float = 0.006,
    seed: int = 42,
) -> SniperDataset:
    """
    Constrói quatro dataframes:
        - entry: todas as barras com label sniper_entry_label (fora de posição)
        - add: snapshots capturados quando o ciclo hipotético atinge pontos de add
        - danger: todas as barras com label sniper_danger_label
        - exit: snapshots em posição para treinar Exit model (label_exit)
    Cada dataframe já inclui colunas de estado (cycle_*).
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    candle_sec = int(candle_sec or _infer_candle_sec(df.index))
    _ensure_contract_labels(df, contract=contract, candle_sec=candle_sec)

    entry_mask = df["sniper_entry_label"].notna()
    entry_df = df.loc[entry_mask].copy()
    entry_df["ts"] = entry_df.index
    entry_df["cycle_is_add"] = 0
    entry_df["cycle_num_adds"] = 0
    entry_df["cycle_time_in_trade"] = 0
    entry_df["cycle_dd_pct"] = 0.0
    entry_df["cycle_dist_to_tp"] = 1.0
    entry_df["cycle_dist_to_sl"] = 0.0
    entry_df["cycle_avg_entry_price"] = entry_df["close"]
    entry_df["cycle_last_fill_price"] = entry_df["close"]
    entry_df["cycle_risk_used_pct"] = contract.sl_pct
    entry_df["cycle_risk_if_add_pct"] = contract.sl_pct + contract.add_spacing_pct
    entry_df["label_entry"] = entry_df["sniper_entry_label"].astype(np.uint8)

    add_snap = _collect_add_snapshots(df, contract=contract, candle_sec=candle_sec, max_add_starts=int(max_add_starts), seed=int(seed))
    if add_snap["idx"]:
        add_idx = np.array(add_snap["idx"], dtype=int)
        add_df = df.iloc[add_idx].copy()
        add_df["ts"] = add_df.index
        add_df["cycle_is_add"] = 1
        add_df["cycle_num_adds"] = np.array(add_snap["num_adds"], dtype=np.int32)
        add_df["cycle_time_in_trade"] = np.array(add_snap["time_in_trade"], dtype=np.int32)
        add_df["cycle_dd_pct"] = np.array(add_snap["dd_pct"], dtype=np.float32)
        add_df["cycle_dist_to_tp"] = np.array(add_snap["dist_to_tp"], dtype=np.float32)
        add_df["cycle_dist_to_sl"] = np.array(add_snap["dist_to_sl"], dtype=np.float32)
        add_df["cycle_avg_entry_price"] = np.array(add_snap["avg_entry_price"], dtype=np.float64)
        add_df["cycle_last_fill_price"] = np.array(add_snap["last_fill_price"], dtype=np.float64)
        add_df["cycle_risk_used_pct"] = np.array(add_snap["risk_used_pct"], dtype=np.float32)
        add_df["cycle_risk_if_add_pct"] = np.array(add_snap["risk_if_add_pct"], dtype=np.float32)
        add_df["label_entry"] = np.array(add_snap["label"], dtype=np.uint8)
        add_df["sniper_exit_code"] = np.array(add_snap["exit_code"], dtype=np.int8)
    else:
        add_df = pd.DataFrame(columns=list(df.columns) + [
            "ts",
            "cycle_is_add",
            "cycle_num_adds",
            "cycle_time_in_trade",
            "cycle_dd_pct",
            "cycle_dist_to_tp",
            "cycle_dist_to_sl",
            "cycle_avg_entry_price",
            "cycle_last_fill_price",
            "cycle_risk_used_pct",
            "cycle_risk_if_add_pct",
            "label_entry",
        ])

    danger_df = df.copy()
    danger_df["ts"] = danger_df.index
    danger_df["label_danger"] = danger_df["sniper_danger_label"].fillna(0).astype(np.uint8)

    exit_snap = _collect_exit_snapshots(
        df,
        contract=contract,
        candle_sec=candle_sec,
        max_exit_starts=int(max_exit_starts),
        exit_stride_bars=int(exit_stride_bars),
        exit_lookahead_bars=exit_lookahead_bars,
        exit_margin_pct=float(exit_margin_pct),
        seed=int(seed),
    )
    if exit_snap["idx"]:
        exit_idx = np.array(exit_snap["idx"], dtype=int)
        exit_df = df.iloc[exit_idx].copy()
        exit_df["ts"] = exit_df.index
        exit_df["cycle_is_add"] = 0
        exit_df["cycle_num_adds"] = np.array(exit_snap["num_adds"], dtype=np.int32)
        exit_df["cycle_time_in_trade"] = np.array(exit_snap["time_in_trade"], dtype=np.int32)
        exit_df["cycle_dd_pct"] = np.array(exit_snap["dd_pct"], dtype=np.float32)
        exit_df["cycle_dist_to_tp"] = np.array(exit_snap["dist_to_tp"], dtype=np.float32)
        exit_df["cycle_dist_to_sl"] = np.array(exit_snap["dist_to_sl"], dtype=np.float32)
        exit_df["cycle_avg_entry_price"] = np.array(exit_snap["avg_entry_price"], dtype=np.float64)
        exit_df["cycle_last_fill_price"] = np.array(exit_snap["last_fill_price"], dtype=np.float64)
        exit_df["cycle_risk_used_pct"] = np.array(exit_snap["risk_used_pct"], dtype=np.float32)
        exit_df["cycle_risk_if_add_pct"] = np.array(exit_snap["risk_if_add_pct"], dtype=np.float32)
        exit_df["label_exit"] = np.array(exit_snap["label_exit"], dtype=np.uint8)
    else:
        exit_df = pd.DataFrame(columns=list(df.columns) + [
            "ts",
            "cycle_is_add",
            "cycle_num_adds",
            "cycle_time_in_trade",
            "cycle_dd_pct",
            "cycle_dist_to_tp",
            "cycle_dist_to_sl",
            "cycle_avg_entry_price",
            "cycle_last_fill_price",
            "cycle_risk_used_pct",
            "cycle_risk_if_add_pct",
            "label_exit",
        ])

    meta = dict(
        tp_pct=contract.tp_min_pct,
        sl_pct=contract.sl_pct,
        timeout_bars=contract.timeout_bars(candle_sec),
        add_spacing_pct=contract.add_spacing_pct,
        max_adds=contract.max_adds,
        risk_max_cycle_pct=contract.risk_max_cycle_pct,
        candle_sec=candle_sec,
        exit_stride_bars=int(exit_stride_bars),
        exit_lookahead_bars=int(exit_lookahead_bars) if exit_lookahead_bars is not None else int(min(max(5, contract.timeout_bars(candle_sec)), 240)),
        exit_margin_pct=float(exit_margin_pct),
    )
    return SniperDataset(entry=entry_df, add=add_df, danger=danger_df, exit=exit_df, meta=meta)


def warmup_sniper_dataset_numba() -> None:
    """
    A primeira compilação de Numba pode parecer "travada" em datasets grandes.
    Chamamos um warmup pequeno para compilar o kernel de ADD snapshots antes do treino.
    """
    close = np.array([1.0, 0.99, 1.01, 0.98, 1.02], dtype=np.float64)
    high = close.copy()
    low = close.copy()
    entry_label = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    exit_code = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    start_idx = np.array([1, 3], dtype=np.int64)
    size_sched = np.array([1.0, 0.5, 0.25], dtype=np.float64)
    _ = _collect_add_snapshots_numba(
        close,
        high,
        low,
        entry_label,
        exit_code,
        start_idx,
        10,         # timeout_bars
        0.05,       # tp_pct
        0.05,       # sl_pct
        0.01,       # spacing
        2,          # max_adds
        0.20,       # risk_cap
        size_sched,
    )


__all__ = [
    "SniperDataset",
    "build_sniper_datasets",
    "warmup_sniper_dataset_numba",
]

