# -*- coding: utf-8 -*-
"""
Funções centralizadas para simulação de backtest e carregamento de parâmetros.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import heapq
import os
from collections import deque

# Numba opcional (fallback no-op se indisponível)
try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f
        return _wrap


# Caches globais (por janela) para cruzamentos EMA de confirmação e ATR por símbolo
# Estrutura:
#   _CROSS_CACHE[conf_ema_win][symbol] = (cross_up uint8, cross_dn uint8)
#   _ATR_CACHE[exit_atr_win][symbol] = atr float32
_CROSS_CACHE: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
_ATR_CACHE: dict[int, dict[str, np.ndarray]] = {}


# -----------------------------
# Núcleo numba otimizado
# -----------------------------
@njit(cache=True)
def _ema_1d(alpha: float, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def _entry_signals_core_numba(p_buy: np.ndarray, p_sho: np.ndarray, thresh: float,
                              close: np.ndarray, conf_ema_win: int,
                              thresh_secondary: float, thresh_window: int) -> tuple:
    """
    Calcula sinais de entrada de forma SIMPLES:
    - Compra quando P_buy >= thresh
    - Shorta quando P_short >= thresh
    
    SEM filtro de EMA - confiamos apenas no modelo!
    """
    n = close.shape[0]
    buy_dec = np.zeros(n, dtype=np.uint8)
    sho_dec = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return buy_dec, sho_dec
    
    for i in range(n):
        # SIMPLES: P >= thresh = sinal!
        buy_hit = (not np.isnan(p_buy[i])) and (p_buy[i] >= thresh)
        sho_hit = (not np.isnan(p_sho[i])) and (p_sho[i] >= thresh)
        
        # Se ambos estão acima do thresh, escolhe o maior
        both = buy_hit and sho_hit
        if both:
            if p_buy[i] >= p_sho[i]:
                buy_dec[i] = 1
                sho_dec[i] = 0
            else:
                buy_dec[i] = 0
                sho_dec[i] = 1
        else:
            buy_dec[i] = 1 if buy_hit else 0
            sho_dec[i] = 1 if sho_hit else 0
    
    return buy_dec, sho_dec


@njit(cache=True)
def _atr_ema_numba(close: np.ndarray, high: np.ndarray, low: np.ndarray, exit_atr_win: int) -> np.ndarray:
    n = close.shape[0]
    tr = np.zeros(n, dtype=np.float32)
    if n == 0:
        return tr
    tr[0] = abs(high[0] - low[0])
    for i in range(1, n):
        pc = close[i - 1]
        a = abs(high[i] - low[i])
        b = abs(high[i] - pc)
        c = abs(low[i] - pc)
        # máximo do true range
        tr[i] = a if (a >= b and a >= c) else (b if b >= c else c)
    alpha = 2.0 / (float(exit_atr_win) + 1.0)
    atr = _ema_1d(alpha, tr)
    return atr


@njit(cache=True)
def _backtest_core_numba(
    close: np.ndarray,
    buy_dec: np.ndarray,
    sho_dec: np.ndarray,
    atr: np.ndarray,
    alpha: float,
    exit_k: float,
    fee_per_side: float,
) -> tuple:
    """
    Núcleo Numba otimizado para backtest simples (sem DCA, sem panic mode).
    Retorna: (trades_arr, equity, long_line, short_line)
    """
    n = close.shape[0]
    
    # Arrays de saída
    max_trades = n // 2 + 1
    trades = np.zeros((max_trades, 3), dtype=np.int64)  # (entry_idx, exit_idx, side)
    trade_count = 0
    
    equity = np.ones(n, dtype=np.float32)
    long_line = np.full(n, np.nan, dtype=np.float32)
    short_line = np.full(n, np.nan, dtype=np.float32)
    
    # Estado
    state = 0  # 0=flat, 1=long, -1=short
    entry_idx = 0
    entry_price = 0.0
    trailing = np.nan
    eq = 1.0
    
    for j in range(n):
        px = close[j]
        if not np.isfinite(px):
            equity[j] = eq
            continue
        
        a = atr[j] if np.isfinite(atr[j]) else 0.0
        
        if state == 0:
            # Flat - procura entrada
            if buy_dec[j]:
                state = 1
                entry_idx = j
                entry_price = px
                trailing = px - exit_k * a
            elif sho_dec[j]:
                state = -1
                entry_idx = j
                entry_price = px
                trailing = px + exit_k * a
        
        elif state == 1:
            # Long - atualiza trailing e verifica saída
            seed = px - exit_k * a
            if np.isfinite(trailing):
                trailing = alpha * seed + (1.0 - alpha) * trailing
            else:
                trailing = seed
            long_line[j] = trailing
            
            if np.isfinite(trailing) and px < trailing:
                # Sai do long
                r = (px / entry_price) - 1.0 - 2.0 * fee_per_side
                eq = eq * (1.0 + r)
                if trade_count < max_trades:
                    trades[trade_count, 0] = entry_idx
                    trades[trade_count, 1] = j
                    trades[trade_count, 2] = 1
                    trade_count += 1
                state = 0
                trailing = np.nan
        
        elif state == -1:
            # Short - atualiza trailing e verifica saída
            seed = px + exit_k * a
            if np.isfinite(trailing):
                trailing = alpha * seed + (1.0 - alpha) * trailing
            else:
                trailing = seed
            short_line[j] = trailing
            
            if np.isfinite(trailing) and px > trailing:
                # Sai do short
                r = (entry_price / px) - 1.0 - 2.0 * fee_per_side
                eq = eq * (1.0 + r)
                if trade_count < max_trades:
                    trades[trade_count, 0] = entry_idx
                    trades[trade_count, 1] = j
                    trades[trade_count, 2] = -1
                    trade_count += 1
                state = 0
                trailing = np.nan
        
        equity[j] = eq
    
    # Retorna apenas trades usados
    return (trades[:trade_count], equity, long_line, short_line)


def ema_alpha_for_u(
    uval: float,
    *,
    use_dynamic: bool,
    exit_ema_win: int,
    exit_ema_min: int | None = None,
    exit_ema_max: int | None = None,
    u_to_win_scale: float | None = None,
) -> float:
    """
    Calcula o alpha da EMA de saída baseado no valor de U previsto.
    
    Se use_dynamic=False, retorna alpha fixo baseado em exit_ema_win.
    Se use_dynamic=True, mapeia |U| para uma janela entre exit_ema_min e exit_ema_max.
    """
    if not use_dynamic:
        return float(2.0 / (int(exit_ema_win) + 1.0))
    
    if exit_ema_min is None or exit_ema_max is None or u_to_win_scale is None:
        return float(2.0 / (int(exit_ema_win) + 1.0))
    
    try:
        uabs = float(abs(uval))
        frac = float(np.tanh(uabs / float(u_to_win_scale)))  # 0..~1
        win = int(round(float(exit_ema_min) + (float(exit_ema_max - exit_ema_min) * frac)))
        win = max(int(exit_ema_min), min(int(exit_ema_max), int(win)))
        return float(2.0 / (win + 1.0))
    except Exception:
        return float(2.0 / (int(exit_ema_win) + 1.0))


def calculate_entry_signals(
    p_buy: np.ndarray,
    p_sho: np.ndarray,
    thresh: float,
    close: np.ndarray | pd.Series,
    conf_ema_win: int,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplificação total: sinaliza compra/short sempre que P >= thresh.
    Quando ambos lados passam do limiar, escolhe o maior P para evitar empate.
    """
    thresh = float(thresh)
    if isinstance(p_buy, pd.Series):
        p_buy_arr = p_buy.to_numpy()
        p_sho_arr = p_sho.to_numpy()
    else:
        p_buy_arr = np.asarray(p_buy)
        p_sho_arr = np.asarray(p_sho)
    
    buy_hits = np.isfinite(p_buy_arr) & (p_buy_arr >= thresh)
    sho_hits = np.isfinite(p_sho_arr) & (p_sho_arr >= thresh)
    
    both = buy_hits & sho_hits
    choose_buy = both & (p_buy_arr >= p_sho_arr)
    choose_sho = both & (~choose_buy)
    
    buy_dec = (buy_hits & (~both)) | choose_buy
    sho_dec = (sho_hits & (~both)) | choose_sho
    
    buy_dec = buy_dec.astype(bool, copy=False)
    sho_dec = sho_dec.astype(bool, copy=False)
    
    # cross_up/down são apenas placeholders para manter compatibilidade com o restante do código
    cross_up = np.zeros_like(buy_dec, dtype=bool)
    cross_down = np.zeros_like(sho_dec, dtype=bool)
    return buy_dec, sho_dec, cross_up, cross_down


def calculate_atr(
    close: np.ndarray | pd.Series,
    high: np.ndarray | pd.Series | None = None,
    low: np.ndarray | pd.Series | None = None,
    exit_atr_win: int = 14,
) -> np.ndarray | pd.Series:
    """
    Calcula ATR (Average True Range) usando True Range.
    
    Args:
        close: Preços de fechamento
        high: Preços máximos (opcional, usa close se None)
        low: Preços mínimos (opcional, usa close se None)
        exit_atr_win: Janela para EMA do ATR
    
    Returns:
        Array ou Series com valores de ATR
    """
    if high is None:
        high = close
    if low is None:
        low = close
    
    if isinstance(close, pd.Series):
        prev_close = close.shift(1)
        tr_raw = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_all = tr_raw.ewm(span=int(exit_atr_win), adjust=False).mean()
        return atr_all
    else:
        # NumPy array - calcula manualmente
        n = len(close)
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = abs(high[0] - low[0]) if n > 0 else 0.0
        for i in range(1, n):
            prev_c = close[i-1]
            tr[i] = max(
                abs(high[i] - low[i]),
                abs(high[i] - prev_c),
                abs(low[i] - prev_c)
            )
        # EMA do TR
        alpha = float(2.0 / (int(exit_atr_win) + 1.0))
        atr = np.zeros(n, dtype=np.float64)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1]
        return atr


def simulate_backtest_fast(
    df_sym: pd.DataFrame,
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
) -> dict:
    """
    Backtest RÁPIDO e SIMPLES para otimização GA.
    
    Recebe DataFrame com colunas: close, high, low, p_buy, p_short.
    Retorna métricas: eq_final, win_rate, max_dd, n_trades, pf, ret_total.
    
    Sem DCA, sem panic mode - apenas lógica básica de entrada/saída.
    """
    if df_sym is None or df_sym.empty:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    # Extrai arrays
    close = df_sym['close'].to_numpy(np.float32, copy=False)
    high = df_sym['high'].to_numpy(np.float32, copy=False) if 'high' in df_sym.columns else close
    low = df_sym['low'].to_numpy(np.float32, copy=False) if 'low' in df_sym.columns else close
    p_buy = df_sym['p_buy'].to_numpy(np.float32, copy=False)
    p_sho = df_sym['p_short'].to_numpy(np.float32, copy=False)
    
    n = len(close)
    if n < 10:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    # Calcula sinais usando função Numba otimizada
    if thresh_secondary is None:
        thresh_secondary = float(thresh) * 0.82
    
    buy_dec, sho_dec = _entry_signals_core_numba(
        p_buy, p_sho, float(thresh), close, int(conf_ema_win),
        float(thresh_secondary), int(thresh_window)
    )
    
    # ATR usando função Numba
    atr = _atr_ema_numba(close, high, low, int(exit_atr_win))
    
    # Simula backtest usando função Numba otimizada
    alpha = float(2.0 / (int(exit_ema_win) + 1.0))
    
    (trades_arr, equity, _, _) = _backtest_core_numba(
        close, buy_dec, sho_dec, atr,
        alpha, float(exit_k), float(fee_per_side)
    )
    
    # Calcula métricas
    n_trades = int(trades_arr.shape[0]) if trades_arr is not None and len(trades_arr) > 0 else 0
    
    if n_trades == 0:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    # Calcula retornos dos trades
    returns = []
    for i in range(n_trades):
        ei, xi, side = int(trades_arr[i, 0]), int(trades_arr[i, 1]), int(trades_arr[i, 2])
        px_e, px_x = float(close[ei]), float(close[xi])
        if side == 1:  # long
            r = (px_x / px_e) - 1.0
        else:  # short
            r = (px_e / px_x) - 1.0
        returns.append(r - 2.0 * float(fee_per_side))
    
    returns = np.array(returns, dtype=np.float32)
    wins = returns[returns > 0.0]
    losses = returns[returns <= 0.0]
    
    win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(-losses.sum()) if len(losses) > 0 else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else 0.0
    
    # Equity e drawdown
    eq_final = float(equity[-1]) if len(equity) > 0 and np.isfinite(equity[-1]) else 1.0
    equity_max = np.maximum.accumulate(equity)
    drawdowns = (equity_max - equity) / np.where(equity_max > 0, equity_max, 1.0)
    max_dd = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    return dict(
        eq_final=eq_final,
        win_rate=win_rate,
        max_dd=max_dd,
        n_trades=n_trades,
        pf=pf,
        ret_total=eq_final - 1.0,
    )


def simulate_backtest(
    df: pd.DataFrame,
    Xdf: pd.DataFrame,
    *,
    p_buy: np.ndarray,
    p_sho: np.ndarray,
    u_pred: np.ndarray | None = None,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float,
    use_dynamic_exit_ema: bool = False,
    exit_ema_min: int | None = None,
    exit_ema_max: int | None = None,
    u_to_win_scale: float | None = None,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
    hold_signal_thresh: float | None = None,
    hold_signal_lookback: int = 5,
    panic_dd_thresh: float = 0.15,
    panic_abandon_thresh: float = 0.20,
    panic_abandon_candles: int = 20,
    dca_enabled: bool = True,
    dca_max_entries: int = 3,
    dca_drop_pct: float = 0.10,
    dca_min_p: float = 0.50,
) -> dict:
    """
    Simula backtest completo com trailing stop baseado em EMA e ATR.
    
    Args:
        hold_signal_thresh: NÃO fecha pelo trailing stop enquanto P estiver acima deste valor.
        hold_signal_lookback: Janela para calcular média do sinal.
        panic_dd_thresh: Se drawdown > este valor, entra em "modo pânico".
        panic_abandon_thresh: Em modo pânico, só sai se P < este valor...
        panic_abandon_candles: ...por este número de candles CONSECUTIVOS.
        dca_enabled: Habilita DCA (Dollar Cost Averaging) - recompras durante quedas.
        dca_max_entries: Máximo de entradas totais (ex: 3 = até 3x alavancagem).
        dca_drop_pct: Queda % do preço médio necessária para fazer DCA (ex: 0.10 = 10%).
        dca_min_p: P(buy/short) mínimo para permitir DCA (ex: 0.50 = modelo ainda >50% confiante).
    
    Retorna dict com trades, equity, linhas de trailing stop, índices de entrada/saída.
    """
    # Calcula sinais de entrada usando função compartilhada
    close_series = df['close'].reindex(Xdf.index)
    close_arr = close_series.to_numpy(np.float64)
    ema_conf_vals = None
    if int(conf_ema_win) > 1:
        ema_conf_vals = close_series.ewm(span=int(conf_ema_win), adjust=False).mean().to_numpy(np.float64)
    buy_dec, sho_dec, cross_up, cross_dn = calculate_entry_signals(
        p_buy, p_sho, thresh, close_series, conf_ema_win,
        thresh_secondary=thresh_secondary, thresh_window=thresh_window
    )
    
    # ATR para trailing stop usando função compartilhada
    high_series = df.get('high', df['close']).reindex(Xdf.index) if 'high' in df.columns else None
    low_series = df.get('low', df['close']).reindex(Xdf.index) if 'low' in df.columns else None
    atr = calculate_atr(close_series, high_series, low_series, exit_atr_win)
    
    # Alpha padrão
    alpha_default = float(2.0 / (int(exit_ema_win) + 1.0))
    
    # Simulação
    state = 0  # 0=flat, 1=long, -1=short
    long_line_plot = np.full(len(Xdf), np.nan, np.float64)
    short_line_plot = np.full(len(Xdf), np.nan, np.float64)
    entry_long_idx: list[int] = []
    exit_long_idx: list[int] = []
    entry_short_idx: list[int] = []
    exit_short_idx: list[int] = []
    trailing = np.nan
    trades: list[tuple[int, int, int]] = []  # (entry_idx, exit_idx, side)
    equity_delta = np.zeros(len(Xdf), dtype=np.float64)
    
    # Pilhas de posições (DCA progressivo simplificado)
    open_long_entries: list[int] = []
    open_long_prices: list[float] = []
    open_short_entries: list[int] = []
    open_short_prices: list[float] = []
    
    alpha_cur = alpha_default

    # Memória recente de probabilidades altas (ex: últimos 10 minutos)
    RECENT_PROB_LOOKBACK_MIN = 10
    lookback_ns = int(RECENT_PROB_LOOKBACK_MIN * 60 * 1e9)
    ts_ns = Xdf.index.view(np.int64)
    recent_buy_ready = np.zeros(len(Xdf), dtype=bool)
    recent_sho_ready = np.zeros(len(Xdf), dtype=bool)
    buy_history: deque[int] = deque()
    sho_history: deque[int] = deque()
    for j in range(len(Xdf)):
        tsj = ts_ns[j]
        while buy_history and tsj - ts_ns[buy_history[0]] > lookback_ns:
            buy_history.popleft()
        while sho_history and tsj - ts_ns[sho_history[0]] > lookback_ns:
            sho_history.popleft()
        if buy_history:
            recent_buy_ready[j] = True
        if sho_history:
            recent_sho_ready[j] = True
        if bool(buy_dec[j]):
            buy_history.append(j)
        if bool(sho_dec[j]):
            sho_history.append(j)
    
    for j, ts in enumerate(Xdf.index):
        px = close_arr[j]
        if not np.isfinite(px):
            continue
        allow_long = True
        allow_short = True
        if ema_conf_vals is not None:
            ema_val = ema_conf_vals[j]
            if not np.isfinite(ema_val):
                allow_long = False
                allow_short = False
            else:
                allow_long = px > ema_val
                allow_short = px < ema_val
        buy_signal_now = bool(buy_dec[j])
        sho_signal_now = bool(sho_dec[j])
        buy_signal_effective = buy_signal_now or recent_buy_ready[j]
        sho_signal_effective = sho_signal_now or recent_sho_ready[j]
        
        if state == 0:
            if buy_signal_effective and allow_long:
                atr0 = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else 0.0
                trailing = float(px) - float(exit_k) * atr0
                if u_pred is not None and use_dynamic_exit_ema:
                    alpha_cur = ema_alpha_for_u(
                        float(u_pred[j]),
                        use_dynamic=True,
                        exit_ema_win=exit_ema_win,
                        exit_ema_min=exit_ema_min,
                        exit_ema_max=exit_ema_max,
                        u_to_win_scale=u_to_win_scale,
                    ) if np.isfinite(u_pred[j]) else alpha_default
                else:
                    alpha_cur = alpha_default
                state = 1
                entry_long_idx.append(j)
                open_long_entries.append(j)
                open_long_prices.append(float(px))
                long_line_plot[j] = trailing
            elif sho_signal_effective and allow_short:
                atr0 = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else 0.0
                trailing = float(px) + float(exit_k) * atr0
                if u_pred is not None and use_dynamic_exit_ema:
                    alpha_cur = ema_alpha_for_u(
                        float(u_pred[j]),
                        use_dynamic=True,
                        exit_ema_win=exit_ema_win,
                        exit_ema_min=exit_ema_min,
                        exit_ema_max=exit_ema_max,
                        u_to_win_scale=u_to_win_scale,
                    ) if np.isfinite(u_pred[j]) else alpha_default
                else:
                    alpha_cur = alpha_default
                state = -1
                entry_short_idx.append(j)
                open_short_entries.append(j)
                open_short_prices.append(float(px))
                short_line_plot[j] = trailing
        
        elif state == 1:  # long
            # Flip imediato: sinal de short antes do cruzamento encerra e inverte
            if bool(sho_dec[j]):
                exit_long_idx.append(j)
                total_r = 0.0
                for ei, entry_price in zip(open_long_entries, open_long_prices):
                    trades.append((ei, j, 1))
                    r = (px / float(entry_price)) - 1.0
                    r_net = r - 2.0 * float(fee_per_side)
                    total_r += r_net
                if total_r != 0.0:
                    equity_delta[j] += total_r
                open_long_entries.clear()
                open_long_prices.clear()
                # abre short imediatamente (se confirmação permitir)
                if not allow_short:
                    continue
                atr0 = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else 0.0
                trailing = float(px) + float(exit_k) * atr0
                if u_pred is not None and use_dynamic_exit_ema:
                    alpha_cur = ema_alpha_for_u(
                        float(u_pred[j]),
                        use_dynamic=True,
                        exit_ema_win=exit_ema_win,
                        exit_ema_min=exit_ema_min,
                        exit_ema_max=exit_ema_max,
                        u_to_win_scale=u_to_win_scale,
                    ) if np.isfinite(u_pred[j]) else alpha_default
                else:
                    alpha_cur = alpha_default
                state = -1
                entry_short_idx.append(j)
                open_short_entries.append(j)
                open_short_prices.append(float(px))
                short_line_plot[j] = trailing
                continue
            
            trailing = alpha_cur * float(px) + (1.0 - alpha_cur) * float(trailing)
            long_line_plot[j] = trailing
            
            # DCA long: adiciona entradas sempre que o preço atual estiver abaixo da última compra
            if dca_enabled and len(open_long_entries) < dca_max_entries:
                last_entry_price = open_long_prices[-1] if open_long_prices else float(px)
                if last_entry_price > 0.0 and float(px) < last_entry_price:
                    atr_val = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else None
                    if atr_val is not None and atr_val > 0.0:
                        trailing = float(px) - float(exit_k) * atr_val
                        long_line_plot[j] = trailing
                        entry_long_idx.append(j)
                        open_long_entries.append(j)
                        open_long_prices.append(float(px))
                        continue
            
            # Verifica se deve sair: trailing stop atingido
            if np.isfinite(trailing) and (px < trailing):
                state = 0
                exit_long_idx.append(j)
                total_r = 0.0
                for ei, entry_price in zip(open_long_entries, open_long_prices):
                    trades.append((ei, j, 1))
                    r = (px / float(entry_price)) - 1.0
                    r_net = r - 2.0 * float(fee_per_side)
                    total_r += r_net
                if total_r != 0.0:
                    equity_delta[j] += total_r
                open_long_entries.clear()
                open_long_prices.clear()
        
        else:  # state == -1 (short)
            # Flip imediato: sinal de buy antes do cruzamento encerra e inverte
            if bool(buy_dec[j]):
                exit_short_idx.append(j)
                total_r = 0.0
                for ei, entry_price in zip(open_short_entries, open_short_prices):
                    trades.append((ei, j, -1))
                    r = (float(entry_price) / px) - 1.0
                    r_net = r - 2.0 * float(fee_per_side)
                    total_r += r_net
                if total_r != 0.0:
                    equity_delta[j] += total_r
                open_short_entries.clear()
                open_short_prices.clear()
                # abre long imediatamente (se confirmação permitir)
                if not allow_long:
                    continue
                atr0 = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else 0.0
                trailing = float(px) - float(exit_k) * atr0
                if u_pred is not None and use_dynamic_exit_ema:
                    alpha_cur = ema_alpha_for_u(
                        float(u_pred[j]),
                        use_dynamic=True,
                        exit_ema_win=exit_ema_win,
                        exit_ema_min=exit_ema_min,
                        exit_ema_max=exit_ema_max,
                        u_to_win_scale=u_to_win_scale,
                    ) if np.isfinite(u_pred[j]) else alpha_default
                else:
                    alpha_cur = alpha_default
                state = 1
                entry_long_idx.append(j)
                open_long_entries.append(j)
                open_long_prices.append(float(px))
                long_line_plot[j] = trailing
                continue
            
            trailing = alpha_cur * float(px) + (1.0 - alpha_cur) * float(trailing)
            short_line_plot[j] = trailing
            
            # DCA short: adiciona entradas sempre que o preço atual estiver acima da última venda
            if dca_enabled and len(open_short_entries) < dca_max_entries:
                last_entry_price = open_short_prices[-1] if open_short_prices else float(px)
                if last_entry_price > 0.0 and float(px) > last_entry_price:
                    atr_val = float(atr.iloc[j] if isinstance(atr, pd.Series) else atr[j]) if (pd.notna(atr.iloc[j]) if isinstance(atr, pd.Series) else np.isfinite(atr[j])) else None
                    if atr_val is not None and atr_val > 0.0:
                        trailing = float(px) + float(exit_k) * atr_val
                        short_line_plot[j] = trailing
                        entry_short_idx.append(j)
                        open_short_entries.append(j)
                        open_short_prices.append(float(px))
                        continue
            
            # Verifica se deve sair: trailing stop atingido
            if np.isfinite(trailing) and (px > trailing):
                state = 0
                exit_short_idx.append(j)
                total_r = 0.0
                for ei, entry_price in zip(open_short_entries, open_short_prices):
                    trades.append((ei, j, -1))
                    r = (float(entry_price) / px) - 1.0
                    r_net = r - 2.0 * float(fee_per_side)
                    total_r += r_net
                if total_r != 0.0:
                    equity_delta[j] += total_r
                open_short_entries.clear()
                open_short_prices.clear()
    
    # Fecha forçadamente no fim
    if state != 0:
        last_j = len(Xdf) - 1
        if state == 1:
            exit_long_idx.append(last_j)
            total_r = 0.0
            last_price = float(df.loc[Xdf.index[last_j], 'close'])
            for ei, entry_price in zip(open_long_entries, open_long_prices):
                trades.append((ei, last_j, 1))
                r = (last_price / float(entry_price)) - 1.0
                r_net = r - 2.0 * float(fee_per_side)
                total_r += r_net
            if total_r != 0.0:
                equity_delta[last_j] += total_r
            open_long_entries.clear()
            open_long_prices.clear()
        else:
            exit_short_idx.append(last_j)
            total_r = 0.0
            last_price = float(df.loc[Xdf.index[last_j], 'close'])
            for ei, entry_price in zip(open_short_entries, open_short_prices):
                trades.append((ei, last_j, -1))
                r = (float(entry_price) / last_price) - 1.0
                r_net = r - 2.0 * float(fee_per_side)
                total_r += r_net
            if total_r != 0.0:
                equity_delta[last_j] += total_r
            open_short_entries.clear()
            open_short_prices.clear()
    
    # Calcula equity
    eq = 1.0
    equity = np.ones(len(Xdf), dtype=np.float64)
    for j in range(len(Xdf)):
        delta = equity_delta[j]
        if delta != 0.0:
            eq *= (1.0 + delta)
        equity[j] = eq
    
    return dict(
        trades=trades,
        equity=equity,
        long_line_plot=long_line_plot,
        short_line_plot=short_line_plot,
        entry_long_idx=entry_long_idx,
        exit_long_idx=exit_long_idx,
        entry_short_idx=entry_short_idx,
        exit_short_idx=exit_short_idx,
    )


def load_ga_params(run_dir: Path) -> dict | None:
    """
    Carrega parâmetros otimizados pelo GA de um run_dir.
    
    Args:
        run_dir: Diretório do run (deve conter ga_best_params.json)
    
    Returns:
        Dict com parâmetros ou None se não encontrado
    """
    ga_params_path = run_dir / 'ga_best_params.json'
    if not ga_params_path.exists():
        return None
    
    try:
        return json.loads(ga_params_path.read_text(encoding='utf-8'))
    except Exception:
        return None


def apply_ga_params(
    ga_params: dict | None,
    *,
    default_thresh: float = 0.85,
    default_conf_ema_win: int = 2,
    default_exit_ema_win: int = 50,
    default_exit_atr_win: int = 14,
    default_exit_k: float = 1.0,
    default_fee_per_side: float = 0.001 / 2.0,
    default_exit_ema_min: int = 8,
    default_exit_ema_max: int = 60,
    default_u_to_win_scale: float = 2.0,
    default_use_dynamic_exit_ema: bool = True,
    default_thresh_secondary: float | None = None,  # Se None, calcula como 0.82 * thresh
    default_thresh_window: int = 5,
    default_hold_signal_thresh: float | None = 0.5,  # Segura posição se sinal > 50%
) -> dict:
    """
    Aplica parâmetros GA a valores padrão, retornando dict com parâmetros efetivos.
    
    Returns:
        Dict com chaves: thresh, conf_ema_win, exit_ema_win, exit_atr_win, exit_k,
        fee_per_side, exit_ema_min, exit_ema_max, u_to_win_scale, use_dynamic_exit_ema,
        thresh_secondary, thresh_window, hold_signal_thresh
    """
    if ga_params is None:
        thresh = default_thresh
        thresh_secondary = default_thresh_secondary if default_thresh_secondary is not None else (thresh * 0.82)
        return {
            'thresh': thresh,
            'conf_ema_win': default_conf_ema_win,
            'exit_ema_win': default_exit_ema_win,
            'exit_atr_win': default_exit_atr_win,
            'exit_k': default_exit_k,
            'fee_per_side': default_fee_per_side,
            'exit_ema_min': default_exit_ema_min,
            'exit_ema_max': default_exit_ema_max,
            'u_to_win_scale': default_u_to_win_scale,
            'use_dynamic_exit_ema': default_use_dynamic_exit_ema,
            'thresh_secondary': thresh_secondary,
            'thresh_window': default_thresh_window,
            'hold_signal_thresh': default_hold_signal_thresh,
            'hold_signal_lookback': 5,
            'panic_dd_thresh': 0.15,
            'panic_abandon_thresh': 0.20,
            'panic_abandon_candles': 20,
            'dca_enabled': True,
            'dca_max_entries': 3,
            'dca_drop_pct': 0.10,
            'dca_min_p': 0.50,
        }
    
    thresh = float(ga_params.get('thresh', default_thresh))
    conf_ema_win = int(ga_params.get('conf_ema_win', default_conf_ema_win))
    exit_ema_win = int(ga_params.get('exit_ema_win', default_exit_ema_win))
    exit_atr_win = int(ga_params.get('exit_atr_win', default_exit_atr_win))
    exit_k = float(ga_params.get('exit_k', default_exit_k))
    
    # Thresh secondary e window
    thresh_secondary = ga_params.get('thresh_secondary', None)
    if thresh_secondary is None:
        thresh_secondary = thresh * 0.82  # 82% do thresh principal
    else:
        thresh_secondary = float(thresh_secondary)
    thresh_window = int(ga_params.get('thresh_window', default_thresh_window))
    
    # Fee
    fee_per_side = default_fee_per_side
    frt = ga_params.get('fee_round_trip', None)
    if frt is not None:
        try:
            fee_per_side = float(frt) / 2.0
        except Exception:
            pass
    
    # Dynamic exit EMA
    if 'use_dynamic_exit_ema' in ga_params:
        use_dynamic_exit_ema = bool(ga_params.get('use_dynamic_exit_ema', False))
    else:
        use_dynamic_exit_ema = False
    
    # hold_signal_thresh: carrega do GA ou usa default
    hold_signal_thresh = ga_params.get('hold_signal_thresh', default_hold_signal_thresh)
    if hold_signal_thresh is not None:
        hold_signal_thresh = float(hold_signal_thresh)
    
    return {
        'thresh': thresh,
        'conf_ema_win': conf_ema_win,
        'exit_ema_win': exit_ema_win,
        'exit_atr_win': exit_atr_win,
        'exit_k': exit_k,
        'fee_per_side': fee_per_side,
        'exit_ema_min': default_exit_ema_min,
        'exit_ema_max': default_exit_ema_max,
        'u_to_win_scale': default_u_to_win_scale,
        'use_dynamic_exit_ema': use_dynamic_exit_ema,
        'thresh_secondary': thresh_secondary,
        'thresh_window': thresh_window,
        'hold_signal_thresh': hold_signal_thresh,
        'hold_signal_lookback': int(ga_params.get('hold_signal_lookback', 5)),
        'panic_dd_thresh': float(ga_params.get('panic_dd_thresh', 0.15)),
        'panic_abandon_thresh': float(ga_params.get('panic_abandon_thresh', 0.20)),
        'panic_abandon_candles': int(ga_params.get('panic_abandon_candles', 20)),
        'dca_enabled': bool(ga_params.get('dca_enabled', True)),
        'dca_max_entries': int(ga_params.get('dca_max_entries', 3)),
        'dca_drop_pct': float(ga_params.get('dca_drop_pct', 0.10)),
        'dca_min_p': float(ga_params.get('dca_min_p', 0.50)),
    }


def simulate_portfolio_backtest(
    symbols: list[str],
    preds_by_sym: dict[str, pd.DataFrame],
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float = 0.0005,
    capital_per_trade: float = 0.10,  # 10% do capital por trade
    leverage: float = 3.0,  # alavancagem 3x (permite até 30 trades simultâneos)
    max_concurrent_trades: int = 30,  # máximo de trades simultâneos (10% * 3x = até 30 trades)
) -> dict:
    """
    Simula uma carteira real com alocação de capital para múltiplos símbolos:
    - 10% do capital por trade (sem multiplicar retorno por leverage)
    - Alavancagem 3x permite até 30 trades simultâneos (mas cada trade usa 10% do capital)
    - Prioriza por U-value quando há mais de 30 sinais simultâneos
    
    Args:
        symbols: Lista de símbolos a simular
        preds_by_sym: Dict {symbol: DataFrame} com colunas ['close', 'p_buy', 'p_short', 'u_pred']
        thresh: Limiar de probabilidade para entrada
        conf_ema_win: Janela EMA de confirmação
        exit_ema_win: Janela EMA de saída (trailing stop)
        exit_atr_win: Janela ATR para trailing stop
        exit_k: Multiplicador ATR para trailing stop
        fee_per_side: Taxa por lado (ex: 0.0005 = 0.05%)
        capital_per_trade: Fração de capital por trade (ex: 0.10 = 10%)
        leverage: Alavancagem (ex: 3.0 = 3x)
        max_concurrent_trades: Máximo de trades simultâneos
    
    Returns:
        Dict com métricas agregadas:
        - eq_final: Equity final da carteira
        - win_rate: Taxa de acerto
        - max_dd: Drawdown máximo
        - n_trades: Número total de trades
        - pf: Profit factor
        - ret_total: Retorno total composto
    """
    # Constrói uma linha do tempo global (união) para um único eixo X.
    # Observação: isso pode aumentar a RAM; os símbolos sem dado naquele instante ficam como NaN e são ignorados.
    common_timestamps = None
    for s in symbols:
        df = preds_by_sym.get(s)
        if df is None or df.empty:
            continue
        if common_timestamps is None:
            common_timestamps = df.index
        else:
            common_timestamps = common_timestamps.union(df.index)
    
    if common_timestamps is None or len(common_timestamps) == 0:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    # Prepara dados por símbolo
    data_by_sym: dict[str, dict] = {}
    for s in symbols:
        df = preds_by_sym.get(s)
        if df is None or df.empty:
            continue
        df_aligned = df.reindex(common_timestamps)
        data_by_sym[s] = {
            'close': (df_aligned['close'].to_numpy(copy=False)).astype(np.float32, copy=False),
            'high': ((df_aligned['high'] if 'high' in df_aligned.columns else df_aligned['close']).to_numpy(copy=False)).astype(np.float32, copy=False),
            'low': ((df_aligned['low'] if 'low' in df_aligned.columns else df_aligned['close']).to_numpy(copy=False)).astype(np.float32, copy=False),
            'p_buy': (df_aligned['p_buy'].to_numpy(copy=False)).astype(np.float32, copy=False),
            'p_short': (df_aligned['p_short'].to_numpy(copy=False)).astype(np.float32, copy=False),
            'u_pred': (df_aligned['u_pred'].to_numpy(copy=False)).astype(np.float32, copy=False) if 'u_pred' in df_aligned.columns else np.full(len(df_aligned), np.nan, np.float32),
        }
    
    if not data_by_sym:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    n = len(common_timestamps)
    portfolio_eq = 1.0
    portfolio_eq_max = 1.0
    max_dd = 0.0
    
    # Trades abertos: dict {symbol: (entry_idx, entry_price, side, trailing)}
    # side: 1=long, -1=short
    open_trades: dict[str, tuple[int, float, int, float]] = {}
    
    # Parâmetros de EMA e ATR (calculados por símbolo)
    exit_alpha = float(2.0 / (int(exit_ema_win) + 1.0))
    
    # Calcula sinais e ATR por símbolo usando funções compartilhadas (mantém apenas o essencial na memória)
    atr_by_sym: dict[str, np.ndarray] = {}
    buy_dec_by_sym: dict[str, np.ndarray] = {}
    sho_dec_by_sym: dict[str, np.ndarray] = {}
    
    for s, data in data_by_sym.items():
        close = data['close']
        high = data['high']
        low  = data['low']
        p_buy = data['p_buy']
        p_sho = data['p_short']
        
        # Calcula sinais usando função compartilhada (usa padrão para thresh_secondary e thresh_window)
        buy_dec, sho_dec, cross_up, cross_dn = calculate_entry_signals(
            p_buy, p_sho, thresh, close, conf_ema_win,
            thresh_secondary=None, thresh_window=5
        )
        buy_dec_by_sym[s] = buy_dec
        sho_dec_by_sym[s] = sho_dec
        
        # Calcula ATR usando função compartilhada
        atr = calculate_atr(close, high, low, exit_atr_win)
        atr_np = atr if isinstance(atr, np.ndarray) else atr.to_numpy()
        atr_by_sym[s] = atr_np.astype(np.float32, copy=False)
    
    win_count = 0
    total_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    for i in range(n):
        # Fecha trades que atingiram trailing stop
        trades_to_close: list[str] = []
        for s, (entry_idx, entry_price, side, trailing) in open_trades.items():
            data = data_by_sym[s]
            close_price = data['close'][i]
            if not np.isfinite(close_price):
                # Sem dado neste timestamp para este símbolo; mantém trade aberto
                continue
            atr = atr_by_sym[s]
            
            # Atualiza trailing stop
            if side == 1:  # long
                trailing = exit_alpha * close_price + (1.0 - exit_alpha) * trailing
                if close_price < trailing:
                    trades_to_close.append(s)
            else:  # short
                trailing = exit_alpha * close_price + (1.0 - exit_alpha) * trailing
                if close_price > trailing:
                    trades_to_close.append(s)
            
            # Atualiza trailing no dict
            open_trades[s] = (entry_idx, entry_price, side, trailing)
        
        # Fecha trades
        for s in trades_to_close:
            entry_idx, entry_price, side, _ = open_trades.pop(s)
            data = data_by_sym[s]
            exit_price = data['close'][i]
            if not np.isfinite(exit_price):
                # Se não há preço válido, re-insere o trade e posterga o fechamento
                open_trades[s] = (entry_idx, entry_price, side, _)
                continue
            
            # Calcula retorno do trade
            if side == 1:  # long
                r_net = (exit_price / entry_price) - 1.0 - 2.0 * fee_per_side
            else:  # short
                r_net = (entry_price / exit_price) - 1.0 - 2.0 * fee_per_side
            
            # Aplica capital por trade (10% por trade, sem multiplicar por leverage)
            # A alavancagem só permite ter mais trades simultâneos, não multiplica o retorno
            r_portfolio = r_net * capital_per_trade
            portfolio_eq *= (1.0 + r_portfolio)
            
            total_trades += 1
            if r_net > 0.0:
                win_count += 1
                gross_profit += r_net
            else:
                gross_loss += -r_net
            
            if portfolio_eq > portfolio_eq_max:
                portfolio_eq_max = portfolio_eq
            dd = (portfolio_eq_max - portfolio_eq) / (portfolio_eq_max if portfolio_eq_max > 0.0 else 1.0)
            if dd > max_dd:
                max_dd = dd
        
        # Se já temos 30 trades abertos, não abre novos
        if len(open_trades) >= max_concurrent_trades:
            continue
        
        # Coleta sinais disponíveis
        signals: list[tuple[str, int, float]] = []  # (symbol, side, priority_score)
        for s, data in data_by_sym.items():
            if s in open_trades:
                continue  # já tem trade aberto
            
            u_pred = data['u_pred'][i]
            p_buy_val = data['p_buy'][i]
            p_sho_val = data['p_short'][i]
            
            # Usa sinais já calculados pela função compartilhada
            buy_signal = bool(buy_dec_by_sym[s][i])
            sho_signal = bool(sho_dec_by_sym[s][i])
            
            # Prioridade: usa u_pred se disponível, senão usa max(p_buy, p_sho)
            if buy_signal and sho_signal:
                if p_buy_val >= p_sho_val:
                    priority = float(u_pred) if np.isfinite(u_pred) else float(p_buy_val)
                    signals.append((s, 1, priority))
                else:
                    priority = float(u_pred) if np.isfinite(u_pred) else float(p_sho_val)
                    signals.append((s, -1, priority))
            elif buy_signal:
                priority = float(u_pred) if np.isfinite(u_pred) else float(p_buy_val)
                signals.append((s, 1, priority))
            elif sho_signal:
                priority = float(u_pred) if np.isfinite(u_pred) else float(p_sho_val)
                signals.append((s, -1, priority))
        
        # Prioriza e abre até max_concurrent_trades
        if signals:
            # Ordena por prioridade (maior primeiro)
            signals.sort(key=lambda x: x[2], reverse=True)
            slots_available = max_concurrent_trades - len(open_trades)
            signals_to_open = signals[:slots_available]
            
            for s, side, _ in signals_to_open:
                data = data_by_sym[s]
                entry_price = data['close'][i]
                atr = atr_by_sym[s][i]
                if not (np.isfinite(entry_price) and np.isfinite(atr)):
                    continue
                
                if side == 1:  # long
                    trailing = entry_price - exit_k * atr
                else:  # short
                    trailing = entry_price + exit_k * atr
                
                open_trades[s] = (i, entry_price, side, trailing)
    
    # Fecha trades restantes no final
    for s, (entry_idx, entry_price, side, _) in open_trades.items():
        data = data_by_sym[s]
        exit_price = data['close'][-1]
        
        if side == 1:  # long
            r_net = (exit_price / entry_price) - 1.0 - 2.0 * fee_per_side
        else:  # short
            r_net = (entry_price / exit_price) - 1.0 - 2.0 * fee_per_side
        
        # Aplica capital por trade (10% por trade, sem multiplicar por leverage)
        r_portfolio = r_net * capital_per_trade
        portfolio_eq *= (1.0 + r_portfolio)
        
        total_trades += 1
        if r_net > 0.0:
            win_count += 1
            gross_profit += r_net
        else:
            gross_loss += -r_net
        
        if portfolio_eq > portfolio_eq_max:
            portfolio_eq_max = portfolio_eq
        dd = (portfolio_eq_max - portfolio_eq) / (portfolio_eq_max if portfolio_eq_max > 0.0 else 1.0)
        if dd > max_dd:
            max_dd = dd
    
    wr = (win_count / total_trades) if total_trades > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0
    ret_total = portfolio_eq - 1.0
    
    return dict(
        eq_final=float(portfolio_eq),
        win_rate=float(wr),
        max_dd=float(max_dd),
        n_trades=int(total_trades),
        pf=float(pf),
        ret_total=float(ret_total),
    )


# ----------------------------------
# Pré-cálculo de trades por símbolo
# ----------------------------------
@njit(cache=True)
def _precompute_trades_core(close: np.ndarray, buy_dec: np.ndarray, sho_dec: np.ndarray,
                            atr: np.ndarray, exit_alpha: float, exit_k: float, fee_per_side: float) -> tuple:
    """
    Pré-calcula trades por símbolo permitindo múltiplos trades sequenciais.
    Um novo trade só pode abrir após o anterior fechar (não permite trades simultâneos no mesmo símbolo).
    """
    n = close.shape[0]
    entry_idx = np.empty(n, dtype=np.int64)
    exit_idx = np.empty(n, dtype=np.int64)
    side_arr = np.empty(n, dtype=np.int8)
    r_net_arr = np.empty(n, dtype=np.float64)
    count = 0
    i = 0
    in_trade = False  # flag para rastrear se há trade aberto
    side = 0  # 1=long, -1=short
    entry_price = 0.0
    trailing = 0.0
    entry_i = 0
    
    while i < n:
        c = close[i]
        a = atr[i]
        if not np.isfinite(c) or not np.isfinite(a):
            i += 1
            continue
        
        if not in_trade:
            # Não há trade aberto: pode entrar
            enter_long = (buy_dec[i] == 1) and not (sho_dec[i] == 1)
            enter_short = (sho_dec[i] == 1) and not (buy_dec[i] == 1)
            if enter_long or enter_short:
                side = 1 if enter_long else -1
                entry_price = c
                trailing = entry_price - exit_k * a if side == 1 else entry_price + exit_k * a
                in_trade = True
                entry_i = i
                i += 1
                continue
        
        if in_trade:
            # Há trade aberto: atualiza trailing e verifica saída
            cj = close[i]
            if not np.isfinite(cj):
                i += 1
                continue
            
            # Atualiza trailing stop
            trailing = exit_alpha * cj + (1.0 - exit_alpha) * trailing
            
            # Verifica se deve fechar
            should_exit = False
            if side == 1:  # long
                should_exit = cj < trailing
            else:  # short
                should_exit = cj > trailing
            
            if should_exit:
                # Fecha o trade
                if side == 1:
                    r_net = (cj / entry_price) - 1.0 - 2.0 * fee_per_side
                else:
                    r_net = (entry_price / cj) - 1.0 - 2.0 * fee_per_side
                entry_idx[count] = entry_i
                exit_idx[count] = i
                side_arr[count] = side
                r_net_arr[count] = r_net
                count += 1
                in_trade = False
                # Após fechar, avança para o próximo índice (pode entrar novamente se houver sinal)
                i += 1
            else:
                # Trade continua aberto, avança
                i += 1
    
    # Fecha trade aberto no final se houver
    if in_trade and n > 0:
        last_price = close[n - 1]
        if np.isfinite(last_price):
            if side == 1:
                r_net = (last_price / entry_price) - 1.0 - 2.0 * fee_per_side
            else:
                r_net = (entry_price / last_price) - 1.0 - 2.0 * fee_per_side
            entry_idx[count] = entry_i
            exit_idx[count] = n - 1
            side_arr[count] = side
            r_net_arr[count] = r_net
            count += 1
    
    return entry_idx, exit_idx, side_arr, r_net_arr, count


def precompute_trades_for_symbol(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    p_buy: np.ndarray,
    p_sho: np.ndarray,
    u_pred: np.ndarray,
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
) -> list[tuple[int, int, int, float, float]]:
    """
    Retorna lista de trades: (entry_idx, exit_idx, side, r_net, u_at_entry)
    """
    # Calcula thresh_secondary se não fornecido
    if thresh_secondary is None:
        thresh_secondary = float(thresh) * 0.82
    else:
        thresh_secondary = float(thresh_secondary)
    
    # sinais e ATR (Numba-optimized)
    close = close.astype(np.float32, copy=False)
    high = high.astype(np.float32, copy=False)
    low  = low.astype(np.float32, copy=False)
    p_buy = p_buy.astype(np.float32, copy=False)
    p_sho = p_sho.astype(np.float32, copy=False)
    buy_dec, sho_dec = _entry_signals_core_numba(
        p_buy, p_sho, float(thresh), close, int(conf_ema_win),
        thresh_secondary, int(thresh_window)
    )
    atr = _atr_ema_numba(close, high, low, int(exit_atr_win))
    exit_alpha = float(2.0 / (int(exit_ema_win) + 1.0))
    # numba core
    e_idx, x_idx, sides, r_nets, cnt = _precompute_trades_core(close, buy_dec, sho_dec, atr, exit_alpha, float(exit_k), float(fee_per_side))
    trades: list[tuple[int, int, int, float, float]] = []
    for k in range(int(cnt)):
        ei = int(e_idx[k]); xi = int(x_idx[k]); sd = int(sides[k]); rn = float(r_nets[k])
        # prioridade usa u_pred se disponível; fallback para max(p_buy, p_sho) no ponto de entrada
        u = float(u_pred[ei]) if ei < len(u_pred) and np.isfinite(u_pred[ei]) else float(max(p_buy[ei], p_sho[ei]))
        trades.append((ei, xi, sd, rn, u))
    return trades


def _get_cross_for_symbol(symbol: str, data: dict, conf_ema_win: int) -> tuple[np.ndarray, np.ndarray]:
    cache = _CROSS_CACHE.get(int(conf_ema_win))
    if cache is None:
        cache = {}
        _CROSS_CACHE[int(conf_ema_win)] = cache
    v = cache.get(symbol)
    if v is not None:
        return v
    close = data['close'].astype(np.float32, copy=False)
    alpha = float(2.0 / (int(conf_ema_win) + 1.0))
    ema = _ema_1d(alpha, close)
    n = len(close)
    c_prev = np.empty(n, dtype=np.float32)
    e_prev = np.empty(n, dtype=np.float32)
    if n > 0:
        c_prev[0] = close[0]
        e_prev[0] = ema[0]
    for i in range(1, n):
        c_prev[i] = close[i - 1]
        e_prev[i] = ema[i - 1]
    cross_up = ((close > ema) & (c_prev <= e_prev)).astype(np.uint8, copy=False)
    cross_dn = ((close < ema) & (c_prev >= e_prev)).astype(np.uint8, copy=False)
    cache[symbol] = (cross_up, cross_dn)
    return cross_up, cross_dn


def _get_atr_for_symbol(symbol: str, data: dict, exit_atr_win: int) -> np.ndarray:
    cache = _ATR_CACHE.get(int(exit_atr_win))
    if cache is None:
        cache = {}
        _ATR_CACHE[int(exit_atr_win)] = cache
    v = cache.get(symbol)
    if v is not None:
        return v
    close = data['close'].astype(np.float32, copy=False)
    high = data['high'].astype(np.float32, copy=False)
    low  = data['low'].astype(np.float32, copy=False)
    atr = _atr_ema_numba(close, high, low, int(exit_atr_win)).astype(np.float32, copy=False)
    cache[symbol] = atr
    return atr


def precompute_trades_for_symbol_cached(
    symbol: str,
    data: dict,
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
) -> list[tuple[int, int, int, float, float]]:
    close = data['close'].astype(np.float32, copy=False)
    p_buy = data['p_buy'].astype(np.float32, copy=False)
    p_sho = data['p_short'].astype(np.float32, copy=False)
    u_pred = data['u_pred'].astype(np.float32, copy=False) if 'u_pred' in data else np.full(len(close), np.nan, np.float32)

    # Calcula thresh_secondary se não fornecido
    if thresh_secondary is None:
        thresh_secondary = float(thresh) * 0.82
    else:
        thresh_secondary = float(thresh_secondary)

    # Fast reject: se não há nenhum ponto acima do thresh ou thresh_secondary, pula
    # (usa max de p_buy/p_sho ignorando NaN)
    max_pb = np.nanmax(p_buy) if p_buy.size else np.nan
    max_ps = np.nanmax(p_sho) if p_sho.size else np.nan
    if not (np.isfinite(max_pb) or np.isfinite(max_ps)):
        return []
    if (np.isfinite(max_pb) and max_pb < float(thresh_secondary)) and (np.isfinite(max_ps) and max_ps < float(thresh_secondary)):
        return []

    # Usa caches
    cross_up, cross_dn = _get_cross_for_symbol(symbol, data, int(conf_ema_win))
    atr = _get_atr_for_symbol(symbol, data, int(exit_atr_win))

    # Verifica se já passou do thresh nas últimas thresh_window velas (janela de memória)
    n = len(p_buy)
    buy_had_thresh = np.zeros(n, dtype=bool)
    sho_had_thresh = np.zeros(n, dtype=bool)
    
    for i in range(n):
        start_idx = max(0, i - thresh_window + 1)
        window_buy = p_buy[start_idx:i+1]
        window_sho = p_sho[start_idx:i+1]
        buy_had_thresh[i] = np.any(np.isfinite(window_buy) & (window_buy >= float(thresh)))
        sho_had_thresh[i] = np.any(np.isfinite(window_sho) & (window_sho >= float(thresh)))

    # buy/short hits: pode comprar se está acima do thresh agora OU (já esteve acima E está acima do secundário)
    buy_hits_now = np.isfinite(p_buy) & (p_buy >= float(thresh))
    buy_hits_secondary = np.isfinite(p_buy) & (p_buy >= float(thresh_secondary)) & buy_had_thresh
    buy_hits = buy_hits_now | buy_hits_secondary
    
    sho_hits_now = np.isfinite(p_sho) & (p_sho >= float(thresh))
    sho_hits_secondary = np.isfinite(p_sho) & (p_sho >= float(thresh_secondary)) & sho_had_thresh
    sho_hits = sho_hits_now | sho_hits_secondary
    
    both = buy_hits & sho_hits
    choose_buy = both & (p_buy >= p_sho)
    choose_sho = both & (~choose_buy)
    buy_dec = ((buy_hits & (~both)) | choose_buy) & (cross_up.astype(bool))
    sho_dec = ((sho_hits & (~both)) | choose_sho) & (cross_dn.astype(bool))

    # core
    exit_alpha = float(2.0 / (int(exit_ema_win) + 1.0))
    e_idx, x_idx, sides, r_nets, cnt = _precompute_trades_core(
        close, buy_dec.astype(np.uint8, copy=False), sho_dec.astype(np.uint8, copy=False),
        atr, exit_alpha, float(exit_k), float(fee_per_side)
    )
    trades: list[tuple[int, int, int, float, float]] = []
    for k in range(int(cnt)):
        ei = int(e_idx[k]); xi = int(x_idx[k]); sd = int(sides[k]); rn = float(r_nets[k])
        u = float(u_pred[ei]) if ei < len(u_pred) and np.isfinite(u_pred[ei]) else float(max(p_buy[ei], p_sho[ei]))
        trades.append((ei, xi, sd, rn, u))
    return trades


def simulate_portfolio_from_trades(
    symbols: list[str],
    trades_by_sym: dict[str, list[tuple[int, int, int, float, float]]],
    *,
    capital_per_trade: float = 0.10,
    max_concurrent_trades: int = 30,
    show_progress: bool = False,
    return_curve: bool = False,
    common_timestamps: pd.DatetimeIndex | None = None,
) -> dict:
    """
    Agrega trades pré-calculados por símbolo simulando a carteira com capacidade limitada.
    
    Se return_curve=True, retorna também equity_curve e active_trades (arrays por timestamp).
    Requer common_timestamps quando return_curve=True.
    """
    # Junta entradas
    entries: list[tuple[int, str, int, float, float, int]] = []  # (entry_idx, symbol, side, r_net, u_entry, exit_idx)
    for s, trades in trades_by_sym.items():
        for (ei, xi, sd, rn, u) in trades:
            entries.append((ei, s, sd, rn, u, xi))
    if not entries:
        result = dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
        if return_curve and common_timestamps is not None:
            result['equity_curve'] = np.ones(len(common_timestamps), dtype=np.float32)
            result['active_trades'] = np.zeros(len(common_timestamps), dtype=np.int32)
        return result
    entries.sort(key=lambda t: t[0])

    portfolio_eq = 1.0
    portfolio_eq_max = 1.0
    max_dd = 0.0
    total_trades = 0
    win_count = 0
    gross_profit = 0.0
    gross_loss = 0.0

    # Para equity curve e trades ativos
    equity_curve = None
    active_trades = None
    if return_curve:
        if common_timestamps is None:
            raise ValueError("common_timestamps é obrigatório quando return_curve=True")
        equity_curve = np.ones(len(common_timestamps), dtype=np.float32)
        active_trades = np.zeros(len(common_timestamps), dtype=np.int32)

    # heap de abertos por exit_idx
    open_heap: list[tuple[int, str, int, float]] = []  # (exit_idx, symbol, side, r_net)
    i = 0
    INF = 1 << 60

    # barra de progresso interna (segundo nível)
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(iterable, desc="", total=None, **kwargs):
            return iterable
    pbar = tqdm(total=len(entries), desc="[carteira] entries", leave=False, mininterval=0.8, dynamic_ncols=True) if show_progress else None

    last_updated_idx = -1  # rastreia último índice atualizado na equity curve
    
    while i < len(entries) or open_heap:
        next_entry_idx = entries[i][0] if i < len(entries) else INF
        next_exit_idx = open_heap[0][0] if open_heap else INF
        
        # Determina próximo evento
        if next_exit_idx <= next_entry_idx:
            event_idx = next_exit_idx
        else:
            event_idx = next_entry_idx
        
        # Preenche equity curve até o evento atual (com valores anteriores)
        if return_curve and event_idx < INF:
            start_fill = max(0, last_updated_idx + 1)
            end_fill = min(event_idx, len(equity_curve))
            if start_fill < end_fill:
                equity_curve[start_fill:end_fill] = float(portfolio_eq)
                active_trades[start_fill:end_fill] = len(open_heap)
        
        # processa exits
        if next_exit_idx <= next_entry_idx:
            ex_idx = next_exit_idx
            # fecha todos com este exit_idx
            to_close: list[tuple[int, str, int, float]] = []
            while open_heap and open_heap[0][0] == ex_idx:
                to_close.append(heapq.heappop(open_heap))
            for (_xi, _sym, side, r_net) in to_close:
                r_portfolio = r_net * capital_per_trade
                portfolio_eq *= (1.0 + r_portfolio)
                total_trades += 1
                if r_net > 0.0:
                    win_count += 1
                    gross_profit += r_net
                else:
                    gross_loss += -r_net
                if portfolio_eq > portfolio_eq_max:
                    portfolio_eq_max = portfolio_eq
                dd = (portfolio_eq_max - portfolio_eq) / (portfolio_eq_max if portfolio_eq_max > 0.0 else 1.0)
                if dd > max_dd:
                    max_dd = dd
            # Atualiza equity curve após fechar trades
            if return_curve and ex_idx < len(equity_curve):
                equity_curve[ex_idx] = float(portfolio_eq)
                active_trades[ex_idx] = len(open_heap)
                last_updated_idx = ex_idx
            continue
        
        # processa entries no next_entry_idx
        cur_idx = next_entry_idx
        same_entries: list[tuple[int, str, int, float, float, int]] = []
        while i < len(entries) and entries[i][0] == cur_idx:
            same_entries.append(entries[i])
            i += 1
            if pbar is not None:
                pbar.update(1)
        slots = max_concurrent_trades - len(open_heap)
        if slots <= 0:
            # Atualiza trades ativos mesmo quando não há slots (mantém valor anterior)
            if return_curve and cur_idx < len(active_trades):
                active_trades[cur_idx] = len(open_heap)
                equity_curve[cur_idx] = float(portfolio_eq)
                last_updated_idx = cur_idx
            continue
        # prioriza pelos maiores u_entry
        same_entries.sort(key=lambda t: t[4], reverse=True)
        accept = same_entries[:slots]
        for (ei, sym, side, r_net, _u, xi) in accept:
            heapq.heappush(open_heap, (xi, sym, side, r_net))
        # Atualiza trades ativos após abrir novos
        if return_curve and cur_idx < len(active_trades):
            active_trades[cur_idx] = len(open_heap)
            equity_curve[cur_idx] = float(portfolio_eq)
            last_updated_idx = cur_idx

    if pbar is not None:
        pbar.close()

    # Preenche o restante da equity curve até o final
    if return_curve:
        start_fill = max(0, last_updated_idx + 1)
        if start_fill < len(equity_curve):
            equity_curve[start_fill:] = float(portfolio_eq)
            active_trades[start_fill:] = len(open_heap)

    wr = (win_count / total_trades) if total_trades > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0
    ret_total = portfolio_eq - 1.0
    
    result = dict(
        eq_final=float(portfolio_eq),
        win_rate=float(wr),
        max_dd=float(max_dd),
        n_trades=int(total_trades),
        pf=float(pf),
        ret_total=float(ret_total),
    )
    
    if return_curve:
        result['equity_curve'] = equity_curve
        result['active_trades'] = active_trades
    
    return result


def simulate_portfolio_backtest_fast(
    symbols: list[str],
    preds_by_sym: dict[str, pd.DataFrame],
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float = 0.0005,
    capital_per_trade: float = 0.10,
    max_concurrent_trades: int = 30,
    show_progress: bool = False,
) -> dict:
    """
    Variante rápida: pré-calcula trades por símbolo e agrega na carteira.
    """
    # eixo temporal global (união)
    common_timestamps = None
    for s in symbols:
        df = preds_by_sym.get(s)
        if df is None or df.empty:
            continue
        common_timestamps = df.index if common_timestamps is None else common_timestamps.union(df.index)
    if common_timestamps is None or len(common_timestamps) == 0:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)

    # pré-calcula trades por símbolo (apenas válidos)
    trades_by_sym: dict[str, list[tuple[int, int, int, float, float]]] = {}
    syms_iter = [s for s in symbols if (preds_by_sym.get(s) is not None and not preds_by_sym.get(s).empty)]
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            def tqdm(iterable, desc="", total=None, **kwargs):
                return iterable
        iterator = tqdm(syms_iter, desc="[pre] símbolos", total=len(syms_iter), leave=False, mininterval=0.8, dynamic_ncols=True)
    else:
        iterator = syms_iter
    for s in iterator:
        df = preds_by_sym.get(s)  # df garantido não vazio em syms_iter
        df_aligned = df.reindex(common_timestamps)
        close = df_aligned['close'].to_numpy(np.float32, copy=False)
        high = (df_aligned['high'] if 'high' in df_aligned.columns else df_aligned['close']).to_numpy(np.float32, copy=False)
        low  = (df_aligned['low'] if 'low' in df_aligned.columns else df_aligned['close']).to_numpy(np.float32, copy=False)
        p_buy = df_aligned['p_buy'].to_numpy(np.float32, copy=False)
        p_sho = df_aligned['p_short'].to_numpy(np.float32, copy=False)
        u_pred = (df_aligned['u_pred'].to_numpy(np.float32, copy=False) if 'u_pred' in df_aligned.columns else np.full(len(df_aligned), np.nan, np.float32))
        trades = precompute_trades_for_symbol(
            close, high, low, p_buy, p_sho, u_pred,
            thresh=thresh, conf_ema_win=conf_ema_win,
            exit_ema_win=exit_ema_win, exit_atr_win=exit_atr_win,
            exit_k=exit_k, fee_per_side=fee_per_side
        )
        if trades:
            trades_by_sym[s] = trades

    # agrega
    metrics = simulate_portfolio_from_trades(
        symbols, trades_by_sym,
        capital_per_trade=capital_per_trade,
        max_concurrent_trades=max_concurrent_trades,
        show_progress=show_progress,
    )
    return metrics


# ----------------------------------
# Variante preparada (sem reindex a cada indivíduo)
# ----------------------------------
def prepare_portfolio_data(
    symbols: list[str],
    preds_by_sym: dict[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, dict[str, dict]]:
    """
    Prepara (uma vez) o eixo temporal global (união) e arrays alinhados por símbolo.
    Retorna (common_timestamps, data_by_sym) onde data_by_sym[s] tem arrays numpy float32.
    """
    common_timestamps = None
    for s in symbols:
        df = preds_by_sym.get(s)
        if df is None or df.empty:
            continue
        common_timestamps = df.index if common_timestamps is None else common_timestamps.union(df.index)
    if common_timestamps is None:
        common_timestamps = pd.DatetimeIndex([])
    
    data_by_sym: dict[str, dict] = {}
    for s in symbols:
        df = preds_by_sym.get(s)
        if df is None or df.empty:
            continue
        df_aligned = df.reindex(common_timestamps)
        data_by_sym[s] = {
            'close': df_aligned['close'].to_numpy(np.float32, copy=False),
            'high': (df_aligned['high'] if 'high' in df_aligned.columns else df_aligned['close']).to_numpy(np.float32, copy=False),
            'low':  (df_aligned['low'] if 'low' in df_aligned.columns else df_aligned['close']).to_numpy(np.float32, copy=False),
            'p_buy': df_aligned['p_buy'].to_numpy(np.float32, copy=False),
            'p_short': df_aligned['p_short'].to_numpy(np.float32, copy=False),
            'u_pred': (df_aligned['u_pred'].to_numpy(np.float32, copy=False) if 'u_pred' in df_aligned.columns else np.full(len(df_aligned), np.nan, np.float32)),
        }
    return common_timestamps, data_by_sym


def simulate_portfolio_backtest_timestamp_by_timestamp(
    symbols: list[str],
    common_timestamps: pd.DatetimeIndex,
    data_by_sym: dict[str, dict],
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float = 0.0005,
    capital_per_trade: float = 0.10,
    max_concurrent_trades: int = 30,
    show_progress: bool = False,
    return_curve: bool = False,
    thresh_secondary: float | None = None,
    thresh_window: int = 5,
) -> dict:
    """
    Processa backtest timestamp por timestamp, verificando todos os símbolos simultaneamente.
    Mais realista que pré-calcular trades, pois simula execução em tempo real.
    """
    if common_timestamps is None or len(common_timestamps) == 0:
        result = dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
        if return_curve:
            result['equity_curve'] = np.ones(0, dtype=np.float32)
            result['active_trades'] = np.zeros(0, dtype=np.int32)
        return result
    
    n_ts = len(common_timestamps)
    
    # Prepara arrays de sinais e indicadores para todos os símbolos (uma vez)
    # Estrutura: para cada símbolo, temos arrays alinhados ao common_timestamps
    signals_by_sym: dict[str, dict] = {}
    for s in symbols:
        data = data_by_sym.get(s)
        if data is None:
            continue
        
        close = data['close']
        high = data['high']
        low = data['low']
        p_buy = data['p_buy']
        p_sho = data['p_short']
        u_pred = data.get('u_pred', np.full(n_ts, np.nan, np.float32))
        
        # Calcula thresh_secondary se não fornecido
        if thresh_secondary is None:
            thresh_sec = float(thresh) * 0.82
        else:
            thresh_sec = float(thresh_secondary)
        
        # Calcula sinais de entrada usando função Numba
        buy_dec, sho_dec = _entry_signals_core_numba(
            p_buy.astype(np.float32, copy=False),
            p_sho.astype(np.float32, copy=False),
            float(thresh),
            close.astype(np.float32, copy=False),
            int(conf_ema_win),
            thresh_sec,
            int(thresh_window)
        )
        
        # Calcula ATR
        atr = _atr_ema_numba(
            close.astype(np.float32, copy=False),
            high.astype(np.float32, copy=False),
            low.astype(np.float32, copy=False),
            int(exit_atr_win)
        )
        
        signals_by_sym[s] = {
            'close': close,
            'buy_dec': buy_dec,
            'sho_dec': sho_dec,
            'atr': atr,
            'u_pred': u_pred,
            'p_buy': p_buy,  # Para fallback de prioridade
            'p_sho': p_sho,  # Para fallback de prioridade
        }
    
    # Estado da carteira
    portfolio_eq = 1.0
    portfolio_eq_max = 1.0
    max_dd = 0.0
    total_trades = 0
    win_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    # Trades abertos: dict[symbol] = dict(side=1/-1, entry_idx=int, entry_price=float, trailing=float)
    open_trades: dict[str, dict] = {}
    
    # Estatísticas de diagnóstico
    total_signals_found = 0
    total_signals_used = 0
    
    # Equity curve e trades ativos
    equity_curve = None
    active_trades = None
    if return_curve:
        equity_curve = np.ones(n_ts, dtype=np.float32)
        active_trades = np.zeros(n_ts, dtype=np.int32)
    
    exit_alpha = float(2.0 / (int(exit_ema_win) + 1.0))
    
    # Barra de progresso
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(iterable, desc="", total=None, **kwargs):
            return iterable
    
    iterator = tqdm(range(n_ts), desc="[carteira] timestamps", total=n_ts, leave=False, mininterval=0.5, dynamic_ncols=True) if show_progress else range(n_ts)
    
    for i in iterator:
        # 1) Fecha trades que devem ser fechados (verifica trailing stop)
        to_close: list[str] = []
        for s, trade_info in open_trades.items():
            signals = signals_by_sym.get(s)
            if signals is None:
                continue
            
            close_price = signals['close'][i]
            if not np.isfinite(close_price):
                continue
            
            side = trade_info['side']
            trailing = trade_info['trailing']
            
            # Atualiza trailing stop
            trailing = exit_alpha * close_price + (1.0 - exit_alpha) * trailing
            trade_info['trailing'] = trailing
            
            # Verifica se deve fechar
            should_exit = False
            if side == 1:  # long
                should_exit = close_price < trailing
            else:  # short
                should_exit = close_price > trailing
            
            if should_exit:
                # Fecha o trade
                entry_price = trade_info['entry_price']
                if side == 1:
                    r_net = (close_price / entry_price) - 1.0 - 2.0 * fee_per_side
                else:
                    r_net = (entry_price / close_price) - 1.0 - 2.0 * fee_per_side
                
                r_portfolio = r_net * capital_per_trade
                portfolio_eq *= (1.0 + r_portfolio)
                total_trades += 1
                
                if r_net > 0.0:
                    win_count += 1
                    gross_profit += r_net
                else:
                    gross_loss += -r_net
                
                if portfolio_eq > portfolio_eq_max:
                    portfolio_eq_max = portfolio_eq
                dd = (portfolio_eq_max - portfolio_eq) / (portfolio_eq_max if portfolio_eq_max > 0.0 else 1.0)
                if dd > max_dd:
                    max_dd = dd
                
                to_close.append(s)
        
        # Remove trades fechados
        for s in to_close:
            del open_trades[s]
        
        # 2) Verifica novos sinais de entrada
        slots_available = max_concurrent_trades - len(open_trades)
        if slots_available > 0:
            # Coleta todos os sinais disponíveis neste timestamp
            candidates: list[tuple[str, int, float]] = []  # (symbol, side, u_priority)
            
            for s, signals in signals_by_sym.items():
                # Pula se já tem trade aberto neste símbolo
                if s in open_trades:
                    continue
                
                buy_dec_val = int(signals['buy_dec'][i])  # Converte para int explícito
                sho_dec_val = int(signals['sho_dec'][i])  # Converte para int explícito
                close_price = signals['close'][i]
                atr_val = signals['atr'][i]
                u_pred_val = signals['u_pred'][i]
                
                if not np.isfinite(close_price) or not np.isfinite(atr_val) or atr_val <= 0:
                    continue
                
                # Verifica sinais (buy_dec e sho_dec são uint8: 0 ou 1)
                if buy_dec_val == 1 and sho_dec_val == 0:
                    # Long
                    u_priority = float(u_pred_val) if np.isfinite(u_pred_val) else float(signals['p_buy'][i])
                    candidates.append((s, 1, u_priority))
                elif sho_dec_val == 1 and buy_dec_val == 0:
                    # Short
                    u_priority = float(u_pred_val) if np.isfinite(u_pred_val) else float(signals['p_sho'][i])
                    candidates.append((s, -1, u_priority))
            
            # Prioriza por U-value e seleciona os melhores
            if candidates:
                total_signals_found += len(candidates)
                candidates.sort(key=lambda x: x[2], reverse=True)
                selected = candidates[:slots_available]
                total_signals_used += len(selected)
                
                for s, side, _ in selected:
                    signals = signals_by_sym[s]
                    close_price = float(signals['close'][i])
                    atr_val = float(signals['atr'][i])
                    
                    # Inicializa trailing stop
                    trailing = close_price - exit_k * atr_val if side == 1 else close_price + exit_k * atr_val
                    
                    open_trades[s] = {
                        'side': side,
                        'entry_idx': i,
                        'entry_price': close_price,
                        'trailing': trailing,
                    }
        
        # 3) Atualiza equity curve e trades ativos
        if return_curve:
            equity_curve[i] = float(portfolio_eq)
            active_trades[i] = len(open_trades)
    
    # Fecha trades abertos no final (força fechamento)
    for s, trade_info in open_trades.items():
        signals = signals_by_sym.get(s)
        if signals is None:
            continue
        
        last_price = signals['close'][n_ts - 1]
        if not np.isfinite(last_price):
            continue
        
        side = trade_info['side']
        entry_price = trade_info['entry_price']
        
        if side == 1:
            r_net = (last_price / entry_price) - 1.0 - 2.0 * fee_per_side
        else:
            r_net = (entry_price / last_price) - 1.0 - 2.0 * fee_per_side
        
        r_portfolio = r_net * capital_per_trade
        portfolio_eq *= (1.0 + r_portfolio)
        total_trades += 1
        
        if r_net > 0.0:
            win_count += 1
            gross_profit += r_net
        else:
            gross_loss += -r_net
    
    wr = (win_count / total_trades) if total_trades > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0
    ret_total = portfolio_eq - 1.0
    
    result = dict(
        eq_final=float(portfolio_eq),
        win_rate=float(wr),
        max_dd=float(max_dd),
        n_trades=int(total_trades),
        pf=float(pf),
        ret_total=float(ret_total),
        _debug_total_signals_found=int(total_signals_found),
        _debug_total_signals_used=int(total_signals_used),
    )
    
    if return_curve:
        result['equity_curve'] = equity_curve
        result['active_trades'] = active_trades
    
    return result


def simulate_portfolio_backtest_fast_prepared(
    symbols: list[str],
    common_timestamps: pd.DatetimeIndex,
    data_by_sym: dict[str, dict],
    *,
    thresh: float,
    conf_ema_win: int,
    exit_ema_win: int,
    exit_atr_win: int,
    exit_k: float,
    fee_per_side: float = 0.0005,
    capital_per_trade: float = 0.10,
    max_concurrent_trades: int = 30,
    show_progress: bool = False,
    return_curve: bool = False,
) -> dict:
    """
    Igual ao fast, mas usa dados já alinhados (evita reindex por indivíduo).
    """
    if common_timestamps is None or len(common_timestamps) == 0:
        return dict(eq_final=1.0, win_rate=0.0, max_dd=0.0, n_trades=0, pf=0.0, ret_total=0.0)
    
    # pré-calcula trades por símbolo usando arrays preparados
    trades_by_sym: dict[str, list[tuple[int, int, int, float, float]]] = {}
    syms_iter = [s for s in symbols if s in data_by_sym and data_by_sym.get(s) is not None]
    
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            def tqdm(iterable, desc="", total=None, **kwargs):
                return iterable
        # Usa mininterval menor e update mais frequente
        iterator = tqdm(syms_iter, desc="[pre] símbolos", total=len(syms_iter), leave=False, mininterval=0.3, dynamic_ncols=True, ncols=80)
    else:
        iterator = syms_iter
    
    for s in iterator:
        data = data_by_sym.get(s)
        try:
            trades = precompute_trades_for_symbol_cached(
                s, data,
                thresh=thresh, conf_ema_win=conf_ema_win,
                exit_ema_win=exit_ema_win, exit_atr_win=exit_atr_win,
                exit_k=exit_k, fee_per_side=fee_per_side
            )
            if trades:
                trades_by_sym[s] = trades
        except Exception:
            continue
    
    # agrega
    metrics = simulate_portfolio_from_trades(
        symbols, trades_by_sym,
        capital_per_trade=capital_per_trade,
        max_concurrent_trades=max_concurrent_trades,
        show_progress=show_progress,
        return_curve=return_curve,
        common_timestamps=common_timestamps if return_curve else None,
    )
    return metrics

