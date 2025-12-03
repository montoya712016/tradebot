# -*- coding: utf-8 -*-
r"""
Dataset builder (entrada): gera índices de positivos (regras) e negativos
balanceados (near-rule, pre/post-timing, far) e oferece um teste visual simples.
"""
from typing import Dict, Tuple
import numpy as np, pandas as pd
from numba import njit
import matplotlib.pyplot as plt, matplotlib.dates as mdates

# Suporte a execução como pacote OU como script direto (fallback absoluto)
try:
    from .pf_config import (
        RULE_LOOKBACK_MIN, RULE_DROP_PCT, RULE_RISE_PCT, RULE_U_HI, RULE_U_LO,
        RULE_MAX_DIST_FROM_EXTREME_PCT, WEIGHT_RULE,
    )
except Exception:
    import sys, pathlib
    _PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in sys.path:
        sys.path.insert(0, str(_WORKSPACE))
    from my_project.prepare_features.pf_config import (
        RULE_LOOKBACK_MIN, RULE_DROP_PCT, RULE_RISE_PCT, RULE_U_HI, RULE_U_LO,
        RULE_MAX_DIST_FROM_EXTREME_PCT, WEIGHT_RULE,
    )


# Constante para janela da EMA de confirmação (em minutos)
CONFIRM_EMA_SPAN_MINUTES: int = 2

# Parâmetros de peso (compartilhados com o trainer)
_WEIGHT_CFG = WEIGHT_RULE or {}
_WEIGHT_U_MAX = float(_WEIGHT_CFG.get("u_weight_max", 10.0))
_WEIGHT_ALPHA_POS = float(_WEIGHT_CFG.get("alpha_pos", 2.0))
_WEIGHT_BETA_NEG = float(_WEIGHT_CFG.get("beta_neg", 3.0))
_WEIGHT_GAMMA_EASE = float(_WEIGHT_CFG.get("gamma_ease", 0.6))
_WEIGHT_NEG_FLOOR = float(_WEIGHT_CFG.get("neg_floor", 0.5))

def _minutes_to_bars(idx: pd.DatetimeIndex, minutes: int) -> int:
    if len(idx) < 2:
        return 1
    dt = float((idx[1] - idx[0]).total_seconds() or 60.0)
    return max(1, int(round((minutes * 60.0) / max(1.0, dt))))


@njit(cache=True)
def _rolling_max_nb(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size
    out = np.empty(n, np.float32)
    out[:] = np.nan
    dq = np.empty(n, np.int64)
    head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] <= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = x[dq[head]]
    return out


@njit(cache=True)
def _rolling_min_nb(x: np.ndarray, win: int) -> np.ndarray:
    n = x.size
    out = np.empty(n, np.float32)
    out[:] = np.nan
    dq = np.empty(n, np.int64)
    head = 0; tail = 0
    for i in range(n):
        while tail > head and x[dq[tail-1]] >= x[i]:
            tail -= 1
        dq[tail] = i; tail += 1
        left = i - win + 1
        while tail > head and dq[head] < left:
            head += 1
        if left >= 0:
            out[i] = x[dq[head]]
    return out


@njit(cache=True)
def _find_idx_of_max(close: np.ndarray, start: int, end: int) -> int:
    """Encontra o índice do máximo no intervalo [start, end)"""
    if start >= end:
        return start
    idx = start
    val = close[start]
    for i in range(start + 1, end):
        if close[i] > val:
            val = close[i]
            idx = i
    return idx

@njit(cache=True)
def _find_idx_of_min(close: np.ndarray, start: int, end: int) -> int:
    """Encontra o índice do mínimo no intervalo [start, end)"""
    if start >= end:
        return start
    idx = start
    val = close[start]
    for i in range(start + 1, end):
        if close[i] < val:
            val = close[i]
            idx = i
    return idx

@njit(cache=True)
def _compute_reversal_masks_nb(close: np.ndarray, win: int, drop_pct: float, rise_pct: float,
                                max_distance_from_extreme_pct: float = RULE_MAX_DIST_FROM_EXTREME_PCT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula máscaras de REVERSÃO EM V com verificação de ORDEM TEMPORAL:
    
    Para ser um FUNDO em V:
      1. O MÁXIMO da janela veio ANTES do MÍNIMO (preço CAIU)
      2. Queda significativa: (max - min) / max >= drop_pct
      3. Preço atual PERTO do mínimo: (close - min) / min <= max_distance_pct
      
    Para ser um PICO em V:
      1. O MÍNIMO da janela veio ANTES do MÁXIMO (preço SUBIU)
      2. Alta significativa: (max - min) / min >= rise_pct
      3. Preço atual PERTO do máximo: (max - close) / max <= max_distance_pct
    """
    n = close.size
    is_bottom = np.zeros(n, dtype=np.bool_)
    is_top = np.zeros(n, dtype=np.bool_)
    
    for i in range(win, n):
        start = i - win
        end = i + 1  # inclui o ponto atual
        
        # Encontra índices do máximo e mínimo na janela
        idx_max = _find_idx_of_max(close, start, end)
        idx_min = _find_idx_of_min(close, start, end)
        
        val_max = close[idx_max]
        val_min = close[idx_min]
        
        if val_min <= 0 or val_max <= 0:
            continue
        
        # Queda: do máximo ao mínimo (em % do máximo)
        drop_pct_actual = (val_max - val_min) / val_max
        
        # Alta: do mínimo ao máximo (em % do mínimo)  
        rise_pct_actual = (val_max - val_min) / val_min
        
        # Distância do preço atual até o MÍNIMO (em % do mínimo)
        distance_from_min = (close[i] - val_min) / val_min
        
        # Distância do preço atual até o MÁXIMO (em % do máximo)
        distance_from_max = (val_max - close[i]) / val_max
        
        # FUNDO em V: 
        # 1. Máximo veio ANTES do mínimo (preço caiu)
        # 2. Queda significativa
        # 3. Preço atual muito perto do mínimo
        if idx_max < idx_min and drop_pct_actual >= drop_pct and distance_from_min <= max_distance_from_extreme_pct:
            is_bottom[i] = True
        
        # PICO em V:
        # 1. Mínimo veio ANTES do máximo (preço subiu)
        # 2. Alta significativa
        # 3. Preço atual muito perto do máximo
        if idx_min < idx_max and rise_pct_actual >= rise_pct and distance_from_max <= max_distance_from_extreme_pct:
            is_top[i] = True
    
    return is_bottom, is_top


@njit(cache=True)
def _compute_drop_rise_nb(close: np.ndarray, win: int, drop_pct: float, rise_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    n = close.size
    drop_mask = np.zeros(n, dtype=np.bool_)
    rise_mask = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return drop_mask, rise_mask

    rmax = _rolling_max_nb(close, win)
    rmin = _rolling_min_nb(close, win)
    prev_max = np.empty(n, np.float32)
    prev_min = np.empty(n, np.float32)
    prev_max[0] = close[0]
    prev_min[0] = close[0]
    for i in range(1, n):
        prev_max[i] = rmax[i-1]
        prev_min[i] = rmin[i-1]

    for i in range(1, n):
        mx = prev_max[i]
        mn = prev_min[i]
        if mx > 0.0:
            drop_mask[i] = ((close[i] / (mx + 1e-9)) - 1.0) <= (-float(drop_pct))
        if mn > 0.0:
            rise_mask[i] = ((close[i] / (mn + 1e-9)) - 1.0) >= float(rise_pct)

    return drop_mask, rise_mask


def compute_rule_masks(
    df: pd.DataFrame,
    *,
    lookback_min: int,
    drop_pct: float,
    rise_pct: float,
    u_hi: float,
    u_lo: float,
    max_distance_from_extreme_pct: float = RULE_MAX_DIST_FROM_EXTREME_PCT,
) -> Tuple[pd.Series, pd.Series]:
    """
    Computa máscaras de REVERSÃO EM V:
    - buy_mask: fundo com queda significativa + preço MUITO perto do mínimo + U alto
    - peak_mask: pico com alta significativa + preço MUITO perto do máximo + U baixo
    
    max_distance_from_extreme_pct: máximo 1.5% de distância do extremo
    Se caiu 10%, só marca se subiu no máximo 1.5% desde o fundo!
    """
    close = df["close"].to_numpy(np.float32)
    win = _minutes_to_bars(df.index, int(lookback_min))
    if win < 1:
        win = 1
    
    try:
        is_bottom, is_top = _compute_reversal_masks_nb(
            close, int(win), float(drop_pct), float(rise_pct), float(max_distance_from_extreme_pct)
        )
    except Exception:
        # Fallback simples
        is_bottom = np.zeros(len(close), dtype=bool)
        is_top = np.zeros(len(close), dtype=bool)

    # U_total se existir; senão compõe
    u_tot = df.get("U_total")
    if u_tot is None:
        if ("U_compra" in df.columns) and ("U_venda" in df.columns):
            u_tot = (df["U_compra"].astype(np.float32) + df["U_venda"].astype(np.float32))
        else:
            u_tot = pd.Series(np.zeros(len(df)), index=df.index)
    
    drop_mask, rise_mask = _compute_drop_rise_nb(close, win, float(drop_pct), float(rise_pct))

    # Combina: posição no extremo + U indica lucro + queda/alta mínima na janela
    buy_mask  = (u_tot.to_numpy(np.float32) > float(u_hi)) & is_bottom & drop_mask
    peak_mask = (u_tot.to_numpy(np.float32) < float(u_lo)) & is_top & rise_mask
    
    return pd.Series(buy_mask, index=df.index), pd.Series(peak_mask, index=df.index)


def _compute_side_weights_for_plot(
    u_total: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    """Replica a lógica de pesos usada no trainer para exibir w_buy/w_short."""
    assert side in ("long", "short")
    n = u_total.size
    w = np.full(n, np.nan, dtype=np.float32)
    if not pos_mask.any() and not neg_mask.any():
        return w

    u_clip = np.clip(u_total.astype(np.float32, copy=False), -_WEIGHT_U_MAX, _WEIGHT_U_MAX)

    if pos_mask.any():
        if side == "long":
            score_pos = np.clip(u_clip[pos_mask] / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
        else:
            score_pos = np.clip((-u_clip[pos_mask]) / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
        w_pos = 1.0 + _WEIGHT_ALPHA_POS * score_pos
        w[pos_mask] = w_pos.astype(np.float32, copy=False)

    if neg_mask.any():
        u_neg = u_clip[neg_mask]
        if side == "long":
            score_bad = np.clip((-u_neg) / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
            w_neg = 1.0 + _WEIGHT_BETA_NEG * score_bad
            score_good = np.clip(u_neg / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
        else:
            score_bad = np.clip(u_neg / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
            w_neg = 1.0 + _WEIGHT_BETA_NEG * score_bad
            score_good = np.clip((-u_neg) / max(1e-6, _WEIGHT_U_MAX), 0.0, 1.0)
        w_neg *= (1.0 - _WEIGHT_GAMMA_EASE * score_good)
        w_neg = np.maximum(w_neg, _WEIGHT_NEG_FLOOR)
        w[neg_mask] = w_neg.astype(np.float32, copy=False)

    return w


def get_strict_rule_indices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    buy = df.get("rev_buy_rule", pd.Series(False, index=df.index)).astype(bool)
    peak = df.get("peak_rule", pd.Series(False, index=df.index)).astype(bool)
    return np.flatnonzero(buy.to_numpy()), np.flatnonzero(peak.to_numpy())


def build_entry_dataset_indices(
    df: pd.DataFrame,
    *,
    loose_drop: float = 0.015,
    loose_rise: float = 0.015,
    loose_u_hi: float = 1.5,
    loose_u_lo: float = -1.5,
    lookback_min: int = RULE_LOOKBACK_MIN,
    pre_off_min: Tuple[int, ...] = (5, 15, 30),
    post_off_min: Tuple[int, ...] = (5, 15, 30),
    min_dist_min: int = 3,
    far_per_pos: int = 2,
    far_avoid_min: int = 60,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Retorna dicionário de arrays de índices:
      - pos_long, pos_short
      - neg_near_long, neg_near_short (looser-rule vs strict-rule)
      - neg_pre_long,  neg_pre_short  (offsets negativos temporais)
      - neg_post_long, neg_post_short (offsets positivos temporais)
    """
    n = len(df)
    # strict (já computado pelo pipeline)
    idx_pos_long, idx_pos_short = get_strict_rule_indices(df)

    # looser masks (near-rule) — via Numba
    m_buy_loose, m_peak_loose = compute_rule_masks(
        df,
        lookback_min=int(lookback_min),
        drop_pct=float(loose_drop),
        rise_pct=float(loose_rise),
        u_hi=float(loose_u_hi),
        u_lo=float(loose_u_lo),
    )
    m_buy_strict  = pd.Series(False, index=df.index)
    m_peak_strict = pd.Series(False, index=df.index)
    if idx_pos_long.size:  m_buy_strict.iloc[idx_pos_long] = True
    if idx_pos_short.size: m_peak_strict.iloc[idx_pos_short] = True
    near_buy_mask  = (m_buy_loose.astype(bool)  & (~m_buy_strict))
    near_peak_mask = (m_peak_loose.astype(bool) & (~m_peak_strict))

    idx_near_long  = np.flatnonzero(near_buy_mask.to_numpy())
    idx_near_short = np.flatnonzero(near_peak_mask.to_numpy())

    # pré/pós offsets
    taken = np.zeros(n, dtype=np.uint8)
    if idx_pos_long.size: taken[idx_pos_long] = 1
    if idx_pos_short.size: taken[idx_pos_short] = 1

    bars_pre  = np.array([_minutes_to_bars(df.index, int(m)) for m in pre_off_min], dtype=np.int64)
    bars_post = np.array([_minutes_to_bars(df.index, int(m)) for m in post_off_min], dtype=np.int64)
    md_bars = _minutes_to_bars(df.index, int(min_dist_min)) if min_dist_min > 0 else 0

    @njit(cache=True)
    def _collect_offsets_nb(src_idx: np.ndarray, bars: np.ndarray, sign: int, n_tot: int, taken_mask: np.ndarray, min_dist: int) -> np.ndarray:
        cnt = 0
        # worst-case buffer
        buf = np.empty(src_idx.size * bars.size, np.int64)
        for ii in range(src_idx.size):
            i = src_idx[ii]
            for bb in range(bars.size):
                j = i + sign * int(bars[bb])
                if j <= 0 or j >= n_tot:
                    continue
                if taken_mask[j] != 0:
                    continue
                buf[cnt] = j
                cnt += 1
                taken_mask[j] = 1
        if cnt == 0:
            return np.empty(0, np.int64)
        arr = buf[:cnt]
        arr.sort()
        if min_dist <= 0 or arr.size <= 1:
            return arr
        # thin by min_dist
        keep_count = 1
        last = arr[0]
        for k in range(1, arr.size):
            if arr[k] - last >= min_dist:
                keep_count += 1
                last = arr[k]
        out = np.empty(keep_count, np.int64)
        out[0] = arr[0]
        idx_out = 1
        last2 = arr[0]
        for k in range(1, arr.size):
            if arr[k] - last2 >= min_dist:
                out[idx_out] = arr[k]
                idx_out += 1
                last2 = arr[k]
        return out

    idx_pre_long  = _collect_offsets_nb(idx_pos_long,  bars_pre,  -1, n, taken.copy(), md_bars)
    idx_post_long = _collect_offsets_nb(idx_pos_long,  bars_post, +1, n, taken.copy(), md_bars)
    idx_pre_short = _collect_offsets_nb(idx_pos_short, bars_pre,  -1, n, taken.copy(), md_bars)
    idx_post_short= _collect_offsets_nb(idx_pos_short, bars_post, +1, n, taken.copy(), md_bars)

    # far-random: amostrar pontos longe de qualquer evento (estrito, near e offsets)
    taken_far = np.zeros(n, dtype=np.uint8)
    rad_min = max(far_avoid_min, min_dist_min)
    rad_bars = _minutes_to_bars(df.index, int(rad_min)) if int(rad_min) > 0 else 0
    all_counts = (idx_pos_long.size + idx_pos_short.size + idx_near_long.size + idx_near_short.size +
                  idx_pre_long.size + idx_post_long.size + idx_pre_short.size + idx_post_short.size)
    if all_counts > 0 and rad_bars > 0:
        all_idx = np.concatenate([
            idx_pos_long, idx_pos_short, idx_near_long, idx_near_short,
            idx_pre_long, idx_post_long, idx_pre_short, idx_post_short
        ]).astype(np.int64)

        @njit(cache=True)
        def _mark_windows_nb(mark: np.ndarray, centers: np.ndarray, radius: int, n_tot: int):
            for t in range(centers.size):
                i = centers[t]
                L = 0 if i - radius < 0 else i - radius
                R = n_tot if i + radius + 1 > n_tot else i + radius + 1
                for j in range(L, R):
                    mark[j] = 1
        _mark_windows_nb(taken_far, all_idx, int(rad_bars), n)

    @njit(cache=True)
    def _sample_eligible_reservoir(mask: np.ndarray, need: int, seed_i: int) -> np.ndarray:
        # mask: 0 => elegível
        nloc = mask.size
        if need <= 0:
            return np.empty(0, np.int64)
        out = np.empty(need, np.int64)
        # LCG simples para determinismo
        state = seed_i if seed_i > 0 else 1234567
        cnt = 0
        for i in range(nloc):
            if mask[i] != 0:
                continue
            cnt += 1
            if cnt <= need:
                out[cnt-1] = i
            else:
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                r = state % cnt  # 0..cnt-1
                if r < need:
                    out[r] = i
        if cnt < need:
            return out[:cnt]
        return out

    total_pos = int(idx_pos_long.size + idx_pos_short.size)
    need = max(0, int(far_per_pos) * total_pos)
    idx_far = _sample_eligible_reservoir(taken_far, int(need), int(seed))

    return dict(
        pos_long=idx_pos_long,
        pos_short=idx_pos_short,
        neg_near_long=idx_near_long,
        neg_near_short=idx_near_short,
        neg_pre_long=idx_pre_long,
        neg_post_long=idx_post_long,
        neg_pre_short=idx_pre_short,
        neg_post_short=idx_post_short,
        neg_far=idx_far,
    )


def _build_binary_series(
    df: pd.DataFrame,
    idxs: Dict[str, np.ndarray],
    *,
    side: str = "long",
    confirm_ema_span_min: int | None = CONFIRM_EMA_SPAN_MINUTES,
) -> pd.Series:
    """
    Constrói uma série binária (float) com 1 nos positivos selecionados e 0 nos negativos,
    deixando NaN nos demais timestamps. Opcionalmente aplica confirmação por EMA:
    - long: exige close > EMA(span)
    - short: exige close < EMA(span)
    """
    assert side in ("long", "short")
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)
    if side == "long":
        pos = idxs.get("pos_long", np.empty(0, dtype=np.int64))
        neg = np.unique(np.concatenate([
            idxs.get("neg_near_long", np.empty(0, dtype=np.int64)),
            idxs.get("neg_pre_long",  np.empty(0, dtype=np.int64)),
            idxs.get("neg_post_long", np.empty(0, dtype=np.int64)),
            idxs.get("neg_far",       np.empty(0, dtype=np.int64)),
        ]))
    else:
        pos = idxs.get("pos_short", np.empty(0, dtype=np.int64))
        neg = np.unique(np.concatenate([
            idxs.get("neg_near_short", np.empty(0, dtype=np.int64)),
            idxs.get("neg_pre_short",  np.empty(0, dtype=np.int64)),
            idxs.get("neg_post_short", np.empty(0, dtype=np.int64)),
            idxs.get("neg_far",        np.empty(0, dtype=np.int64)),
        ]))

    pos_mask = np.zeros(n, dtype=bool)
    if pos.size:
        pos_mask[pos] = True
    # Confirmação por EMA (opcional, causal)
    if confirm_ema_span_min is not None and int(confirm_ema_span_min) > 0:
        span = max(1, int(round(confirm_ema_span_min * 60.0 / max(1.0, float((df.index[1]-df.index[0]).total_seconds() or 60.0)))))
        # usa pandas ewm causal
        ema = df["close"].ewm(span=span, adjust=False).mean().to_numpy(np.float64)
        close_arr = df["close"].to_numpy(np.float64)
        if side == "long":
            pos_mask &= (close_arr > ema)
        else:
            pos_mask &= (close_arr < ema)
    y[pos_mask] = 1.0
    if neg.size:
        # negativos ficam 0 apenas onde ainda não marcamos 1
        neg_mask = np.zeros(n, dtype=bool); neg_mask[neg] = True
        neg_mask &= (~pos_mask)
        y[neg_mask] = 0.0
    return pd.Series(y, index=df.index, name=f"y_{side}_bin")


def plot_dataset_labels(
    df: pd.DataFrame,
    idxs: Dict[str, np.ndarray],
    *,
    confirm_ema_span_min: int | None = CONFIRM_EMA_SPAN_MINUTES,
    show: bool = True,
) -> None:
    """
    Visualiza close + rótulos e os MESMOS pesos utilizados no trainer:
      - Topo: close com scatter dos positivos (buy '^', short 'v'), divididos em early/confirmed (EMA opcional)
      - Meio: pesos BUY calculados via `WEIGHT_RULE`
      - Base: pesos SHORT na mesma escala
    Assim conseguimos inspecionar onde o modelo será punido com mais força (fake positive)
    e onde ele pode errar sem tanto custo (fake negative).
    """
    n = len(df)
    if n == 0:
        return
    close_arr = df["close"].to_numpy(np.float64)
    u_total = df.get("U_total", pd.Series(np.zeros(len(df)), index=df.index)).astype(np.float32).to_numpy()
    # Máscaras de positivos/negativos vindos DOS ÍNDICES do dataset
    pos_buy_mask_all = np.zeros(n, dtype=bool)
    pos_sho_mask_all = np.zeros(n, dtype=bool)
    idx_pos_long = idxs.get("pos_long", np.empty(0, dtype=np.int64))
    idx_pos_short = idxs.get("pos_short", np.empty(0, dtype=np.int64))
    if idx_pos_long.size:
        pos_buy_mask_all[idx_pos_long] = True
    if idx_pos_short.size:
        pos_sho_mask_all[idx_pos_short] = True

    neg_buy_mask_all = np.zeros(n, dtype=bool)
    for key in ("neg_near_long", "neg_pre_long", "neg_post_long", "neg_far"):
        arr = idxs.get(key, np.empty(0, dtype=np.int64))
        if arr.size:
            neg_buy_mask_all[arr] = True
    neg_buy_mask_all &= ~pos_buy_mask_all

    neg_sho_mask_all = np.zeros(n, dtype=bool)
    for key in ("neg_near_short", "neg_pre_short", "neg_post_short", "neg_far"):
        arr = idxs.get(key, np.empty(0, dtype=np.int64))
        if arr.size:
            neg_sho_mask_all[arr] = True
    neg_sho_mask_all &= ~pos_sho_mask_all

    # confirmação (apenas para colorir early vs confirmed)
    pos_buy_conf = np.zeros(n, dtype=bool)
    pos_sho_conf = np.zeros(n, dtype=bool)
    if confirm_ema_span_min is not None and int(confirm_ema_span_min) > 0 and len(df.index) > 1:
        span = max(1, int(round(confirm_ema_span_min * 60.0 / max(1.0, float((df.index[1]-df.index[0]).total_seconds() or 60.0)))))
        ema = df["close"].ewm(span=span, adjust=False).mean().to_numpy(np.float64)
        pos_buy_conf = pos_buy_mask_all & (close_arr > ema)
        pos_sho_conf = pos_sho_mask_all & (close_arr < ema)
    pos_buy_early = pos_buy_mask_all & (~pos_buy_conf)
    pos_sho_early = pos_sho_mask_all & (~pos_sho_conf)

    # Pesos idênticos ao trainer (evita discrepâncias na leitura)
    w_buy = _compute_side_weights_for_plot(u_total, pos_buy_mask_all, neg_buy_mask_all, side="long")
    w_sho = _compute_side_weights_for_plot(u_total, pos_sho_mask_all, neg_sho_mask_all, side="short")
    x = mdates.date2num(df.index.to_numpy())
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={'hspace':0.18, 'height_ratios':[2,1,1]})
    # 1) Preço + pontos 1
    axs[0].plot(x, df["close"].to_numpy(), color="k", lw=0.8)
    # scatter: distinguindo early (claro) e confirmados (escuro)
    pos_buy_c_idx   = np.flatnonzero(pos_buy_conf)
    pos_buy_e_idx   = np.flatnonzero(pos_buy_early)
    pos_sho_c_idx   = np.flatnonzero(pos_sho_conf)
    pos_sho_e_idx   = np.flatnonzero(pos_sho_early)
    if pos_buy_e_idx.size:
        axs[0].scatter(x[pos_buy_e_idx], df["close"].to_numpy()[pos_buy_e_idx],
                       s=16, c="#7bed9f", marker="^", linewidths=0.0, alpha=0.80, label="buy (early)")
    if pos_buy_c_idx.size:
        axs[0].scatter(x[pos_buy_c_idx], df["close"].to_numpy()[pos_buy_c_idx],
                       s=22, c="#27ae60", marker="^", linewidths=0.0, alpha=0.95, label="buy (confirmed)")
    if pos_sho_e_idx.size:
        axs[0].scatter(x[pos_sho_e_idx], df["close"].to_numpy()[pos_sho_e_idx],
                       s=16, c="#f5b7b1", marker="v", linewidths=0.0, alpha=0.80, label="short (early)")
    if pos_sho_c_idx.size:
        axs[0].scatter(x[pos_sho_c_idx], df["close"].to_numpy()[pos_sho_c_idx],
                       s=22, c="#c0392b", marker="v", linewidths=0.0, alpha=0.95, label="short (confirmed)")
    axs[0].set_title(f"Close | labels do dataset (EMA conf. p/ cor={confirm_ema_span_min}min)"); axs[0].grid(True, alpha=0.3)
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(loc="upper left")
    # 2) Pesos BUY
    axs[1].plot(x, w_buy, color="#2ecc71", lw=1.1, label="w_buy")
    axs[1].set_title("Pesos BUY (mesma lógica usada no trainer)"); axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper left")
    # 3) Pesos SHORT
    axs[2].plot(x, w_sho, color="#e74c3c", lw=1.1, label="w_short")
    axs[2].set_title("Pesos SHORT (mesma lógica usada no trainer)"); axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper left")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')
    if show:
        plt.show()
    else:
        plt.close(fig)

def _plot_balance_preview(df: pd.DataFrame, idxs: Dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt, matplotlib.dates as mdates
    x = mdates.date2num(df.index.to_numpy())
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.plot(x, df["close"].to_numpy(), color="k", lw=0.7, label="close")
    def _p(points: np.ndarray, marker: str, color: str, label: str, ms: float=5.0):
        if points.size:
            ax.plot(x[points], df["close"].to_numpy()[points], marker=marker, linestyle="None", color=color, markersize=ms, label=label)
    _p(idxs.get("pos_long", np.empty(0, dtype=np.int64)),  "^", "green",  "buy+")
    _p(idxs.get("pos_short", np.empty(0, dtype=np.int64)), "v", "red",    "peak+")
    _p(idxs.get("neg_near_long", np.empty(0, dtype=np.int64)),  "x", "gold",   "near(buy)")
    _p(idxs.get("neg_near_short", np.empty(0, dtype=np.int64)), "x", "orange", "near(peak)")
    _p(idxs.get("neg_pre_long", np.empty(0, dtype=np.int64)),  "o", "lime",  "pre(buy)", 4.0)
    _p(idxs.get("neg_post_long", np.empty(0, dtype=np.int64)), "o", "olive", "post(buy)", 4.0)
    _p(idxs.get("neg_pre_short", np.empty(0, dtype=np.int64)),  "o", "salmon","pre(peak)", 4.0)
    _p(idxs.get("neg_post_short", np.empty(0, dtype=np.int64)), "o", "maroon","post(peak)",4.0)
    _p(idxs.get("neg_far", np.empty(0, dtype=np.int64)),        ".", "purple","far-rand", 3.5)
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    plt.title("Preview balanceamento (entrada)")
    plt.tight_layout(); plt.show()


def _sample_from_bucket(rng: np.random.Generator, arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if k >= arr.size:
        return arr.copy()
    return rng.choice(arr, size=int(k), replace=False)


def balance_side_indices(
    idxs: Dict[str, np.ndarray],
    *,
    side: str,           # 'long' ou 'short'
    ratio_neg_per_pos: int | float = 3,
    bucket_weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05),  # near, pre, post, far
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Seleciona negativos para atingir ~1:ratio por lado, com mistura por buckets.

    Retorna dict com keys: pos, neg, neg_near, neg_pre, neg_post, neg_far (selecionados).
    """
    assert side in ("long", "short")
    rng = np.random.default_rng(int(seed))

    pos = idxs["pos_long"] if side == "long" else idxs["pos_short"]
    near = idxs["neg_near_long"] if side == "long" else idxs["neg_near_short"]
    pre  = idxs["neg_pre_long"] if side == "long" else idxs["neg_pre_short"]
    post = idxs["neg_post_long"] if side == "long" else idxs["neg_post_short"]
    far  = idxs.get("neg_far", np.empty(0, dtype=np.int64))

    target_neg = int(round(float(ratio_neg_per_pos) * pos.size))
    if target_neg <= 0 or pos.size == 0:
        return dict(pos=pos, neg=np.empty(0, dtype=np.int64), neg_near=np.empty(0, dtype=np.int64), neg_pre=np.empty(0, dtype=np.int64), neg_post=np.empty(0, dtype=np.int64), neg_far=np.empty(0, dtype=np.int64))

    w_near, w_pre, w_post, w_far = bucket_weights
    w_sum = max(1e-9, w_near + w_pre + w_post + w_far)
    w_near, w_pre, w_post, w_far = [w / w_sum for w in (w_near, w_pre, w_post, w_far)]

    k_near = int(round(target_neg * w_near))
    k_pre  = int(round(target_neg * w_pre))
    k_post = int(round(target_neg * w_post))
    k_far  = target_neg - (k_near + k_pre + k_post)

    sel_near = _sample_from_bucket(rng, near, k_near)
    sel_pre  = _sample_from_bucket(rng, pre,  k_pre)
    sel_post = _sample_from_bucket(rng, post, k_post)
    # re-alocar sobra para far
    k_far = max(0, target_neg - (sel_near.size + sel_pre.size + sel_post.size))
    sel_far  = _sample_from_bucket(rng, far,  k_far)

    neg = np.unique(np.concatenate([sel_near, sel_pre, sel_post, sel_far]))

    # se ainda faltou por limite de buckets, completa do pool total remanescente
    pool_all = np.unique(np.concatenate([near, pre, post, far]))
    if neg.size < target_neg and pool_all.size > neg.size:
        remaining = np.setdiff1d(pool_all, neg, assume_unique=True)
        need = min(target_neg - neg.size, remaining.size)
        extra = _sample_from_bucket(rng, remaining, int(need))
        neg = np.unique(np.concatenate([neg, extra]))

    return dict(pos=pos, neg=neg, neg_near=sel_near, neg_pre=sel_pre, neg_post=sel_post, neg_far=sel_far)

def balance_both_sides(
    idxs: Dict[str, np.ndarray],
    *,
    ratio_neg_per_pos: int | float = 3,
    bucket_weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05),
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        "long":  balance_side_indices(idxs, side="long",  ratio_neg_per_pos=ratio_neg_per_pos, bucket_weights=bucket_weights, seed=seed),
        "short": balance_side_indices(idxs, side="short", ratio_neg_per_pos=ratio_neg_per_pos, bucket_weights=bucket_weights, seed=seed),
    }

if __name__ == "__main__":
    # Teste isolado: busca dados, roda prepare_features.run, constrói índices e plota
    try:
        from .prepare_features import (
            run,
            FLAGS,
            DEFAULT_SYMBOL,
            DEFAULT_DAYS,
            DEFAULT_REMOVE_TAIL_DAYS,
            DEFAULT_CANDLE_SEC,
        )
        from .data import load_ohlc_1m_series, to_ohlc_from_1m
    except Exception:
        from my_project.prepare_features.prepare_features import (
            run,
            FLAGS,
            DEFAULT_SYMBOL,
            DEFAULT_DAYS,
            DEFAULT_REMOVE_TAIL_DAYS,
            DEFAULT_CANDLE_SEC,
        )
        from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m

    TEST_SYMBOL = DEFAULT_SYMBOL
    TEST_DAYS = 360          # tamanho da janela que queremos analisar
    TEST_REMOVE_TAIL = 360   # quantos dias mais recentes remover (shift de 1 ano)

    total_days = int(TEST_DAYS + TEST_REMOVE_TAIL)
    raw = load_ohlc_1m_series(
        TEST_SYMBOL,
        total_days,
        remove_tail_days=int(TEST_REMOVE_TAIL),
    )
    df_ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
    # Garante que rótulos e regras de pivô sejam gerados para visualização
    try:
        flags_viz = dict(FLAGS)
    except Exception:
        flags_viz = {}
    flags_viz["label"] = True
    flags_viz["pivots"] = True
    df_ohlc = run(df_ohlc, flags=flags_viz, plot=False)

    idxs = build_entry_dataset_indices(
        df_ohlc,
        loose_drop=0.02, loose_rise=0.02, loose_u_hi=2.0, loose_u_lo=-2.0,
        lookback_min=RULE_LOOKBACK_MIN,
        pre_off_min=(5, 15, 30), post_off_min=(5, 15, 30), min_dist_min=3,
    )

    # Contagens simples
    def _c(name):
        arr = idxs.get(name, np.empty(0, dtype=np.int64))
        return int(arr.size)
    print((
        f"[dataset-bal] pos_long={_c('pos_long'):,} | pos_short={_c('pos_short'):,} | "
        f"near_long={_c('neg_near_long'):,} | near_short={_c('neg_near_short'):,} | "
        f"pre_long={_c('neg_pre_long'):,} | post_long={_c('neg_post_long'):,} | "
        f"pre_short={_c('neg_pre_short'):,} | post_short={_c('neg_post_short'):,} | "
        f"far={_c('neg_far'):,}"
    ).replace(',', '.'), flush=True)

    # Visual: alvos binários (sem scatter), com confirmação por EMA curta
    plot_dataset_labels(df_ohlc, idxs, confirm_ema_span_min=CONFIRM_EMA_SPAN_MINUTES, show=True)

    # Balanceamento 1:3 por lado, com mistura de buckets
    balanced = balance_both_sides(idxs, ratio_neg_per_pos=3, bucket_weights=(0.5, 0.3, 0.15, 0.05), seed=42)
    bl, bs = balanced["long"], balanced["short"]
    print((
        f"[balanced] long: pos={bl['pos'].size:,} neg={bl['neg'].size:,} (1:{(bl['neg'].size/max(1,bl['pos'].size)):.2f}) | "
        f"short: pos={bs['pos'].size:,} neg={bs['neg'].size:,} (1:{(bs['neg'].size/max(1,bs['pos'].size)):.2f})"
    ).replace(',', '.'), flush=True)


