# -*- coding: utf-8 -*-
from __future__ import annotations
"""
GA simples para calibrar parâmetros de execução do entry (binário + EMA).

O GA otimiza (por while/ETF de múltiplos símbolos):
  - thresh (limiar de decisão para P)
  - conf_ema_win (EMA curta de confirmação de entrada)
  - exit_ema_win (EMA de saída – trailing)
  - exit_atr_win (ATR do seed do trailing)
  - exit_k (offset do seed: ±K*ATR)

Somente OHLC e previsões (P(buy)/P(short)) são usados para o backtest.
Para evitar recálculos pesados a cada avaliação, este script gera (e
reutiliza) um cache parquet por símbolo com as colunas necessárias:
['open','high','low','close','p_buy','p_short'].

Como fonte das previsões, usamos o run mais recente (ou indicado) e
recalculamos UMA vez por símbolo, salvando em otimizacao/cache/.
"""

import json, time, random, os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback simples se tqdm não estiver disponível
    def tqdm(iterable, desc="", total=None, **kwargs):
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        if total:
            print(f"{desc} [0/{total}]", end="", flush=True)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} [{i+1}/{total}]", end="", flush=True)
            yield item
        if total:
            print()  # nova linha ao final

try:
    from ..prepare_features.prepare_features import run as pf_run, DEFAULT_CANDLE_SEC, build_flags, FEATURE_KEYS
    from ..prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_ROOT = _Path(__file__).resolve().parents[1]  # .../my_project
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from my_project.prepare_features.prepare_features import run as pf_run, DEFAULT_CANDLE_SEC, build_flags, FEATURE_KEYS
    from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m


SAVE_DIR = Path(__file__).resolve().parents[2] / "models_classifier"
# Run padrão = último treinamento concluído. Pode ser sobrescrito via env GA_RUN_HINT.
DEFAULT_MARKET_LABEL = "market_cap_150B_50M"
DEFAULT_RUN_NAME = "wf_002"
DEFAULT_RUN_PATH = (SAVE_DIR / DEFAULT_MARKET_LABEL / DEFAULT_RUN_NAME)

# Imports centralizados - mesmas funções que teste_individual.py usa
try:
    from ..test.load_models import (
        find_latest_run, find_latest_run_any, build_X_from_features,
        get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run, read_anchor_end, get_data_window_end
    )
    from ..test.backtest import simulate_portfolio_backtest, simulate_portfolio_backtest_fast, simulate_portfolio_backtest_fast_prepared
    from ..entry_predict_core import predict_buy_short_proba_for_segments, preload_boosters
except Exception:
    from my_project.test.load_models import (
        find_latest_run, find_latest_run_any, build_X_from_features,
        get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run, read_anchor_end, get_data_window_end
    )
    from my_project.test.backtest import simulate_portfolio_backtest, simulate_portfolio_backtest_fast, simulate_portfolio_backtest_fast_prepared
    from my_project.entry_predict_core import predict_buy_short_proba_for_segments, preload_boosters

try:
    import xgboost as xgb
except Exception:
    xgb = None

CACHE_DIR = Path(__file__).resolve().parents[1] / "otimizacao" / "cache"
OUT_DIR   = Path(__file__).resolve().parents[1] / "otimizacao" / "out"
for _d in (CACHE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Contexto preparado para backtest rápido (eixo global + arrays por símbolo)
_PREPARED_CTX: tuple | None = None

def _predict_symbol_once(symbol: str, run_dir: Path, periods: list[int], days: int, *, skip_days: int = 0, require_cache: bool = False, flex_skip_if_empty: bool = True, bypass_cache_on_recalc: bool = False) -> pd.DataFrame | None:
    """Calcula OHLC + P(buy)/P(short) uma única vez e devolve DataFrame.
       Resultado é salvo em cache parquet para reutilização.
    """
    cache_path = CACHE_DIR / f"{symbol}_{days}d_skip{int(skip_days)}_{run_dir.name}.parquet"
    full_cache_path = CACHE_DIR / f"{symbol}_full_{run_dir.name}.parquet"

    def _slice_window(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return df_in
        # ATENÇÃO: o OHLC já foi carregado com remove_tail_days=skip_days,
        # então NÃO devemos subtrair skip_days novamente aqui.
        right = df_in.index.max()
        left  = right - pd.Timedelta(days=int(days))
        return df_in[(df_in.index > left) & (df_in.index <= right)]

    if (not bool(bypass_cache_on_recalc)) or bool(require_cache):
        if cache_path.exists():
            try:
                dfc = pd.read_parquet(cache_path)
                dfc = _slice_window(dfc)
                if require_cache and (dfc is None or dfc.empty):
                    raise RuntimeError(f"{symbol}: cache vazio para janela solicitada")
                # Se não é obrigatório usar cache e o cache está vazio, seguimos para recomputar
                if dfc is not None and not dfc.empty:
                    return dfc
            except Exception:
                pass
    # tenta cache completo do símbolo e recorta
    if (not bool(bypass_cache_on_recalc)) or bool(require_cache):
        if full_cache_path.exists():
            try:
                dff = pd.read_parquet(full_cache_path)
                dff = _slice_window(dff)
                if require_cache and (dff is None or dff.empty):
                    raise RuntimeError(f"{symbol}: cache_full vazio para janela solicitada")
                if dff is not None and not dff.empty:
                    print(f"[ga] usando cache_full para {symbol}: rows={len(dff)}")
                    return dff
            except Exception:
                pass
    # fallback: qualquer parquet do símbolo (pega o mais recente)
    fallback: pd.DataFrame | None = None
    if (not bool(bypass_cache_on_recalc)) or bool(require_cache):
        try:
            cand = sorted(CACHE_DIR.glob(f"{symbol}_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                fallback = _slice_window(pd.read_parquet(cand[0]))
                if require_cache and (fallback is None or fallback.empty):
                    raise RuntimeError(f"{symbol}: cache fallback vazio para janela solicitada")
                if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                    print(f"[ga] usando cache fallback para {symbol}: {cand[0].name}")
                    return fallback
        except Exception:
            fallback = None
    if require_cache:
        # não recalcula pipeline, apenas sinaliza ausência
        raise RuntimeError(f"{symbol}: cache parquet ausente — defina require_cache=False para recalcular")

    # carrega dados brutos e gera features (sem labels/pivots para acelerar)
    # Com days=365 e skip_days=365: queremos 1 ano de dados há 1 ano atrás
    # Carregamos (days + skip_days) = 730 dias e removemos os últimos skip_days = 365 dias
    # Resultado: dados de [t_end-730, t_end-365] = 365 dias há 365 dias atrás
    total_days_to_load = int(int(days) + int(skip_days))
    raw = load_ohlc_1m_series(symbol, total_days_to_load, remove_tail_days=int(skip_days))
    if raw.empty:
        raise RuntimeError(f"{symbol}: dados OHLC vazios após load_ohlc_1m_series")
    ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
    if ohlc.empty:
        raise RuntimeError(f"{symbol}: OHLC vazio após to_ohlc_from_1m")
    # Log do range de dados carregados
    if len(ohlc) > 0:
        data_start = ohlc.index[0]
        data_end = ohlc.index[-1]
        data_days = (data_end - data_start).total_seconds() / 86400
        print(f"[ga] {symbol}: dados carregados: {data_start.strftime('%Y-%m-%d')} até {data_end.strftime('%Y-%m-%d')} (~{data_days:.0f} dias)")
    # flags: ligar todas as famílias de features usadas no treinamento (seguro) exceto label/pivots
    FLAGS_FOR_PRED = build_flags(enable=FEATURE_KEYS, label=False)
    FLAGS_FOR_PRED["pivots"] = False
    df = pf_run(ohlc, flags=FLAGS_FOR_PRED, plot=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"{symbol}: df vazio após prepare_features")

    # carrega feature_columns
    feat_cols = load_feature_columns(run_dir, periods)
    Xdf = build_X_from_features(df, feat_cols)
    idx = Xdf.index
    if len(idx) == 0:
        raise RuntimeError(f"{symbol}: X vazio após build_X")

    # segmentos de 90d
    t_end = idx[-1]
    segments: list[Tuple[int, pd.Timestamp, pd.Timestamp]] = []
    prev = t_end
    for i in range(1, 9):
        left = t_end - pd.Timedelta(days=90*i)
        segments.append((90*i, left, prev))
        prev = left
    # Carrega cutoffs antes da predição para filtrar períodos inválidos
    # Usa função centralizada de load_models.py
    cutoffs_map, last_train_ts0 = load_training_cutoffs(run_dir, symbol, periods)
    data_start = Xdf.index[0] if len(Xdf.index) > 0 else None
    data_end = Xdf.index[-1] if len(Xdf.index) > 0 else None
    
    # Filtra períodos válidos usando função centralizada
    periods_valid = get_valid_periods(
        run_dir=run_dir,
        symbol=symbol,
        data_start=data_start,
        data_end=data_end,
        periods_available=periods,
    )
    
    if not periods_valid:
        raise RuntimeError(f"{symbol}: nenhum período válido após verificação de cutoffs")
    
    # previsão usando apenas períodos válidos
    p_buy, p_sho, u_pred, used_spans, _ = predict_buy_short_proba_for_segments(Xdf, run_dir, periods_valid, print_timing=False, symbol=symbol)
    
    # Aplica corte OOS por período/símbolo para não usar dados vistos no treino
    # CRÍTICO: Garante que não usamos dados que foram vistos no treinamento
    try:
        idx = Xdf.index
        if len(idx) == 0:
            return None
        for (lft, rgt, psel) in list(used_spans or []):
            cutoff = cutoffs_map.get(int(psel))
            if cutoff is None:
                # Fallback: usa last_train_ts0 - p se disponível
                if last_train_ts0 is not None and pd.notna(last_train_ts0):
                    cutoff = last_train_ts0 - pd.Timedelta(days=int(psel))
                else:
                    continue
            if cutoff is None:
                continue
            lft = pd.to_datetime(lft); rgt = pd.to_datetime(rgt)
            cutoff = pd.to_datetime(cutoff)
            # Se o cutoff está no futuro, não há problema (todos os dados são anteriores)
            if data_end is not None and cutoff > data_end:
                continue
            # Máscara do segmento atual
            mspan = (idx > lft) & (idx <= rgt)
            # Máscara de dados que foram vistos no treino (ts <= cutoff)
            moos = (idx <= cutoff)
            # Dados problemáticos: estão no segmento E foram vistos no treino
            bad = mspan & moos
            if bad.any():
                n_bad = int(bad.sum())
                p_buy[bad] = np.nan
                p_sho[bad] = np.nan
                if n_bad > 0:
                    print(f"[ga] {symbol} periodo {psel}d: zerou {n_bad} preds (ts <= cutoff {cutoff.strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"[ga] {symbol}: erro ao aplicar cutoffs OOS: {e}")
        import traceback
        traceback.print_exc()

    # salva apenas o essencial para backtest leve (inclui u_pred para priorização)
    # inclui high/low para ATR realista quando disponível
    close_series = df['close'].reindex(Xdf.index)
    # tenta obter high/low do OHLC original; caso não exista no df, usa close como fallback
    try:
        high_series = ohlc['high'].reindex(Xdf.index)
        low_series = ohlc['low'].reindex(Xdf.index)
    except Exception:
        high_series = close_series
        low_series = close_series
    base_out = pd.DataFrame(index=Xdf.index, data={
        'close': close_series.astype(np.float32),
        'high': high_series.astype(np.float32),
        'low': low_series.astype(np.float32),
        'p_buy': p_buy.astype(np.float32), 'p_short': p_sho.astype(np.float32),
        'u_pred': u_pred.astype(np.float32) if u_pred is not None else np.full(len(Xdf), np.nan, np.float32),
    })
    base_out = base_out.dropna()
    out = _slice_window(base_out)
    if out.empty and bool(flex_skip_if_empty) and int(skip_days) > 0:
        # fallback reduzindo skip_days gradualmente: 75%, 50%, 25% e 0
        for frac in (75, 50, 25, 0):
            try_skip = int((int(skip_days) * frac) // 100)
            t_end = base_out.index.max()
            right = t_end - pd.Timedelta(days=int(try_skip)) if int(try_skip) > 0 else t_end
            left  = right - pd.Timedelta(days=int(days))
            out_try = base_out[(base_out.index > left) & (base_out.index <= right)]
            if out_try is not None and not out_try.empty:
                print(f"[ga] {symbol}: fallback skip_days {int(skip_days)} -> {try_skip}")
                out = out_try
                break
    if out.empty:
        raise RuntimeError(f"{symbol}: preds vazias (sem modelos aplicáveis para períodos {periods})")
    try:
        # salva janela específica
        out.to_parquet(cache_path, index=True)
        try:
            print(f"[ga] cache janela salvo: {cache_path.name} | rows={len(out)}")
        except Exception:
            pass
        # salva cache completo (para futuras janelas sem recálculo)
        # só salva se não existir ou se base_out for maior (atualiza)
        need_full = True
        if full_cache_path.exists():
            try:
                old = pd.read_parquet(full_cache_path)
                if isinstance(old, pd.DataFrame) and (len(old) >= len(base_out)):
                    need_full = False
            except Exception:
                need_full = True
        if need_full:
            base_out.to_parquet(full_cache_path, index=True)
            try:
                print(f"[ga] cache FULL salvo: {full_cache_path.name} | rows={len(base_out)}")
            except Exception:
                pass
    except Exception:
        pass
    return out


# Funções removidas - agora importadas de load_models.py:
# - _choose_run -> choose_run
# - _read_anchor_end -> read_anchor_end
# - _backtest_params e _bt_numba -> removidas (não são mais usadas, substituídas por simulate_portfolio_backtest)


# _portfolio_backtest removido - agora usa simulate_portfolio_backtest de backtest.py


def _fitness(symbols: list[str], preds_by_sym: dict[str, pd.DataFrame],
             genome: dict, *, fee_per_side: float, executor: ThreadPoolExecutor | None = None,
             weights: dict[str, float] | None = None) -> float:
    """
    Fitness BALANCEADO entre consistência e ATIVIDADE:
    - Maximiza Calmar Ratio (retorno / drawdown) = lucro consistente
    - Favorece win rate alto (mais previsível)
    - Favorece PF alto (ganhos > perdas)
    - INCENTIVA mais trades (não queremos sistema parado!)
    - DD máximo 30%
    
    Objetivo: operar ATIVAMENTE com lucro consistente.
    """
    det = _aggregate_metrics_for_genome(symbols, preds_by_sym, genome, fee_per_side=fee_per_side, executor=executor)
    if det.get("used_symbols", 0) == 0:
        return -1e9
    
    win_rate = float(det.get("wr_mean", 0.0))
    pf = float(det.get("pf_mean", 0.0))
    ret_total = float(det.get("ret_mean", 0.0))
    max_dd = float(det.get("dd_max", 1.0))
    n_trades = int(det.get("trades_sum", 0))
    n_symbols = int(det.get("used_symbols", 1))
    
    # ============================================
    # TRADES POR SÍMBOLO - Métrica crucial!
    # ============================================
    trades_per_sym = n_trades / max(n_symbols, 1)
    
    # Penalizações duras
    if ret_total <= 0:
        return ret_total - 1.0  # Score negativo
    
    if max_dd > 0.30:
        return -max_dd  # Penaliza DD alto
    
    # ============================================
    # MÍNIMO DE TRADES POR SÍMBOLO
    # Meta: pelo menos 10 trades/símbolo/ano = ~1 a cada 5 semanas
    # ============================================
    MIN_TRADES_PER_SYM = 5.0   # Mínimo absoluto
    TARGET_TRADES_PER_SYM = 20.0  # Ideal
    
    if trades_per_sym < MIN_TRADES_PER_SYM:
        # Penaliza severamente: poucos trades = sistema inútil
        return -1.0 + (trades_per_sym / MIN_TRADES_PER_SYM) * 0.5
    
    # ============================================
    # CALMAR RATIO: Retorno / Drawdown
    # ============================================
    calmar = ret_total / max(max_dd, 0.01)
    
    # ============================================
    # BÔNUS POR WIN RATE (WR > 50% = bom)
    # ============================================
    if win_rate >= 0.50:
        wr_bonus = 1.0 + (win_rate - 0.50) * 0.8  # +40% para WR=100%
    else:
        wr_bonus = win_rate / 0.50  # Penaliza WR < 50%
    
    # ============================================
    # BÔNUS POR PROFIT FACTOR (PF > 1.5 = bom)
    # ============================================
    if pf >= 1.0:
        pf_bonus = 1.0 + min(pf - 1.0, 2.0) * 0.15  # Cap em PF=3 (+30% bônus)
    else:
        pf_bonus = pf  # Penaliza PF < 1
    
    # ============================================
    # BÔNUS POR BAIXO DRAWDOWN
    # ============================================
    dd_bonus = 1.0 + max(0, 0.20 - max_dd) * 1.5  # Até +30% para DD < 20%
    
    # ============================================
    # BÔNUS POR ATIVIDADE (trades por símbolo)
    # Escala de 1.0 a 2.0 conforme aproxima do target
    # ============================================
    activity_ratio = min(trades_per_sym / TARGET_TRADES_PER_SYM, 2.0)  # Cap em 2x target
    activity_bonus = 1.0 + activity_ratio * 0.5  # +50% a +100% para sistema ativo
    
    # ============================================
    # SCORE FINAL: Calmar × WR × PF × DD × ATIVIDADE
    # ============================================
    score = calmar * wr_bonus * pf_bonus * dd_bonus * activity_bonus
    
    return float(score)


def _mutate(g: dict, *, rng: random.Random) -> dict:
    """Mutação com ranges flexíveis para cada parâmetro."""
    h = dict(g)
    
    # Parâmetros de entrada - thresh BAIXO para mais trades!
    if rng.random() < 0.4: 
        h['thresh'] = min(0.75, max(0.30, h['thresh'] + rng.uniform(-0.08, 0.08)))  # Cap em 0.75!
    if rng.random() < 0.3: 
        h['conf_ema_win'] = int(min(25, max(2, h['conf_ema_win'] + rng.randint(-3, 3))))  # mínimo 2
    
    # Parâmetros de saída - janelas maiores
    if rng.random() < 0.4: 
        h['exit_ema_win'] = int(min(150, max(10, h['exit_ema_win'] + rng.randint(-10, 10))))
    if rng.random() < 0.3: 
        h['exit_atr_win'] = int(min(40, max(5, h['exit_atr_win'] + rng.randint(-3, 3))))
    if rng.random() < 0.4: 
        h['exit_k'] = min(4.0, max(0.3, h['exit_k'] + rng.uniform(-0.25, 0.25)))
    
    # Parâmetros anti-pânico
    if rng.random() < 0.3:
        h['hold_signal_thresh'] = min(0.75, max(0.20, h['hold_signal_thresh'] + rng.uniform(-0.08, 0.08)))
    if rng.random() < 0.3:
        h['panic_dd_thresh'] = min(0.35, max(0.05, h['panic_dd_thresh'] + rng.uniform(-0.03, 0.03)))
    if rng.random() < 0.3:
        h['panic_abandon_thresh'] = min(0.45, max(0.05, h['panic_abandon_thresh'] + rng.uniform(-0.05, 0.05)))
    if rng.random() < 0.3:
        h['panic_abandon_candles'] = int(min(80, max(3, h['panic_abandon_candles'] + rng.randint(-8, 8))))
    
    # Parâmetros DCA
    if rng.random() < 0.3:
        h['dca_max_entries'] = int(min(8, max(1, h['dca_max_entries'] + rng.randint(-1, 1))))
    if rng.random() < 0.3:
        h['dca_drop_pct'] = min(0.30, max(0.02, h['dca_drop_pct'] + rng.uniform(-0.03, 0.03)))
    if rng.random() < 0.3:
        h['dca_min_p'] = min(0.75, max(0.25, h['dca_min_p'] + rng.uniform(-0.08, 0.08)))
    
    return h


def _crossover(a: dict, b: dict, *, rng: random.Random) -> dict:
    """Crossover uniforme - cada gene vem de um dos pais."""
    c = {}
    for k in a.keys():
        c[k] = (a[k] if rng.random() < 0.5 else b[k])
    return c


def _backtest_single_symbol(args: tuple) -> dict | None:
    """Executa backtest para um único símbolo. Usado em paralelo."""
    symbol, df_sym, genome, fee_per_side = args
    
    try:
        from my_project.test.backtest import simulate_backtest_fast
    except Exception:
        from ..test.backtest import simulate_backtest_fast
    
    try:
        result = simulate_backtest_fast(
            df_sym,
            thresh=float(genome.get('thresh', 0.85)),
            conf_ema_win=int(genome.get('conf_ema_win', 2)),
            exit_ema_win=int(genome.get('exit_ema_win', 50)),
            exit_atr_win=int(genome.get('exit_atr_win', 14)),
            exit_k=float(genome.get('exit_k', 1.0)),
            fee_per_side=float(fee_per_side),
            thresh_secondary=float(genome.get('thresh', 0.85)) * 0.82,
            thresh_window=5,
        )
        return result
    except Exception:
        return None


def _aggregate_metrics_for_genome(symbols: list[str], preds_by_sym: dict[str, pd.DataFrame], genome: dict, *, fee_per_side: float, executor: ThreadPoolExecutor | None = None) -> dict:
    """
    Executa backtests individuais em paralelo para cada símbolo e agrega métricas.
    Usa simulate_backtest_fast (leve, rápido, sem DCA/panic para GA).
    """
    results = []
    valid_syms = [s for s in symbols if s in preds_by_sym and preds_by_sym[s] is not None and not preds_by_sym[s].empty]
    
    if not valid_syms:
        return dict(used_symbols=0)
    
    # Prepara argumentos para execução paralela
    args_list = [(s, preds_by_sym[s], genome, fee_per_side) for s in valid_syms]
    
    # Executa em paralelo usando ProcessPoolExecutor se disponível, senão sequencial
    if executor is not None:
        futures = [executor.submit(_backtest_single_symbol, args) for args in args_list]
        for fut in futures:
            try:
                r = fut.result(timeout=30)  # timeout de 30s por símbolo
                if r is not None:
                    results.append(r)
            except Exception:
                pass
    else:
        for args in args_list:
            r = _backtest_single_symbol(args)
            if r is not None:
                results.append(r)
    
    if not results:
        return dict(used_symbols=0)
    
    # Agrega métricas de todos os símbolos
    eq_finals = [r.get('eq_final', 1.0) for r in results]
    win_rates = [r.get('win_rate', 0.0) for r in results]
    max_dds = [r.get('max_dd', 0.0) for r in results]
    pfs = [r.get('pf', 0.0) for r in results]
    ret_totals = [r.get('ret_total', 0.0) for r in results]
    n_trades_list = [r.get('n_trades', 0) for r in results]
    
    return dict(
        used_symbols=len(results),
        eq_mean=float(np.mean(eq_finals)),
        eq_median=float(np.median(eq_finals)),
        wr_mean=float(np.mean(win_rates)),
        dd_max=float(np.max(max_dds)),  # Pior drawdown entre todos os símbolos
        pf_mean=float(np.mean([p for p in pfs if p > 0])) if any(p > 0 for p in pfs) else 0.0,
        ret_mean=float(np.mean(ret_totals)),
        trades_sum=int(sum(n_trades_list)),
    )


def run_ga(*,
    run_hint: str | None = None,
    symbols: list[str] | None = None,
    days: int = 360,
    skip_days: int = 360,
    pop_size: int = 500,
    generations: int = 15,
    elite: int = 25,
    fee_round_trip: float = 0.001,
    seed: int = 42,
    workers: int = 32,
) -> dict:
    rng = random.Random(int(seed))
    env_run_hint = os.environ.get("GA_RUN_HINT")
    default_run_hint = str(DEFAULT_RUN_PATH) if DEFAULT_RUN_PATH.exists() else None
    selected_run_hint = run_hint or env_run_hint or default_run_hint
    if selected_run_hint and selected_run_hint != run_hint:
        print(f"[ga] usando run_hint='{selected_run_hint}' (env/auto)")
    run_hint = selected_run_hint
    # Alinha o run ao fim da janela de dados (evita cutoffs muito à frente)
    try:
        target_end = get_data_window_end(int(days), int(skip_days))
    except Exception:
        target_end = None
    run_dir, periods = choose_run(SAVE_DIR, run_hint, target_end=target_end)
    print(f"[ga] run_dir={run_dir} | periods={periods}")
    # Seleciona períodos disponíveis
    periods_avail = sorted([int(p) for p in periods if int(p) > 0])
    
    # Ajuste crucial: se a janela termina em 'target_end' e os modelos foram ancorados em 'anchor_end',
    # precisamos de períodos p tais que cutoff(anchor_end - p) < target_end  =>  p > (anchor_end - target_end)
    anchor_end = read_anchor_end(run_dir)
    if (anchor_end is not None) and (target_end is not None):
        delta_days = int(max(0, (anchor_end - target_end).days))
        # Loga delta apenas para diagnóstico; não restringe períodos aqui.
        # A aplicação OOS precisa acontecer por vela/segmento após a predição (mais precisa).
        print(f"[ga] anchor_end={anchor_end.date()} | target_end={target_end.date()} | delta={delta_days}d => periods_candidatos={periods_avail}")
    else:
        print(f"[ga] aviso: anchor_end/target_end indisponíveis para ajuste fino de períodos.")
    
    # Carrega todos os modelos possíveis inicialmente (90..720) para acelerar e simplificar.
    preload_boosters(run_dir, periods_avail, print_timing=True)
    periods_use = periods_avail

    # símbolos
    if not symbols:
        syms: list[str] = []
        try:
            # Usa exatamente o conjunto do treinamento (sym_map.json) para casar com o dataset
            sym_map_path = run_dir / "dataset" / "sym_map.json"
            if sym_map_path.exists():
                syms = json.loads(sym_map_path.read_text(encoding="utf-8"))
        except Exception:
            syms = []
        if not syms:
            # Fallback: usa todos do top_market_cap.txt dentro do range, sem truncar top 50
            top_file = Path(__file__).resolve().parents[1] / "top_market_cap.txt"
            cap_map: dict[str, float] = {}
            try:
                txt = top_file.read_text(encoding="utf-8")
                for line in txt.splitlines():
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    s = k.strip().upper()
                    if not s.endswith("USDT"):
                        s = s + "USDT"
                    try:
                        cap_map[s] = float(int(v.strip().replace(".", "").replace(",", "")))
                    except Exception:
                        continue
            except Exception:
                cap_map = {}
            MIN_CAP = 50_000_000
            MAX_CAP = 150_000_000_000
            syms = [
                s for s, cap in cap_map.items()
                if s not in {"BTCUSDT", "ETHUSDT"}
                and MIN_CAP <= cap <= MAX_CAP
            ]
        # USA TODOS os símbolos do treinamento (sem limite)
        symbols = syms if syms else ["ADAUSDT","XRPUSDT","SOLUSDT","ATOMUSDT","SUIUSDT","XLMUSDT"]
    print(f"[ga] symbols={len(symbols)} — {symbols[:10]}{'...' if len(symbols)>10 else ''}")

    # Usa períodos filtrados (já definido acima)
    periods_to_use = periods_use
    
    # gera (ou carrega) previsões por símbolo — sequencial (sem paralelização)
    preds_by_sym: dict[str, pd.DataFrame] = {}
    for s in symbols:
        t0 = time.time()
        df_sym = None
        try:
            # Tenta carregar do cache primeiro
            df_sym = _predict_symbol_once(s, run_dir, periods_to_use, int(days), skip_days=int(skip_days), require_cache=True)
            dt = time.time() - t0
            rows = len(df_sym) if df_sym is not None else 0
            if rows <= 0:
                raise RuntimeError(f"{s}: cache vazio (0 linhas)")
            print(f"[ga] cached preds {s}: rows={rows} | {dt:.2f}s", flush=True)
        except Exception as e:
            # Recalcula se cache não disponível
            try:
                t1 = time.time()
                df_sym = _predict_symbol_once(s, run_dir, periods_to_use, int(days), skip_days=int(skip_days), require_cache=False, bypass_cache_on_recalc=True)
                dtr = time.time() - t1
                rows = len(df_sym) if df_sym is not None else 0
                print(f"[ga] preds {s} recalculadas: rows={rows} | {dtr:.2f}s", flush=True)
            except Exception as e2:
                print(f"[ga] {s}: falhou gerar preds mesmo após recalcular: {e2}", flush=True)
                df_sym = None
        
        if isinstance(df_sym, pd.DataFrame) and not df_sym.empty:
            preds_by_sym[s] = df_sym

    fee_per_side = float(fee_round_trip) / 2.0
    
    if not preds_by_sym:
        print("[ga] AVISO: nenhum símbolo tem predições válidas. Verifique os logs acima.", flush=True)
        return dict(best_genome={}, best_fitness=-1e9, generations=0)

    # população inicial com todos os parâmetros
    def _rand_gene() -> dict:
        return dict(
            # Parâmetros de entrada - faixa ampla para testar extremos
            thresh=rng.uniform(0.20, 0.90),      # agora permite desde super agressivo até ultra conservador
            conf_ema_win=rng.randint(1, 60),     # inclui 1 para desativar filtro ou EMAs longas
            
            # Parâmetros de saída - janelas maiores
            exit_ema_win=rng.randint(10, 160),
            exit_atr_win=rng.randint(5, 60),
            exit_k=rng.uniform(0.3, 4.0),
            
            # Parâmetros anti-pânico - ranges flexíveis
            hold_signal_thresh=rng.uniform(0.10, 0.90),
            panic_dd_thresh=rng.uniform(0.03, 0.40),
            panic_abandon_thresh=rng.uniform(0.05, 0.60),
            panic_abandon_candles=rng.randint(3, 90),
            
            # Parâmetros DCA - mais flexíveis
            dca_max_entries=rng.randint(1, 8),
            dca_drop_pct=rng.uniform(0.01, 0.35),
            dca_min_p=rng.uniform(0.20, 0.80),
        )

    pop: list[dict] = [_rand_gene() for _ in range(int(pop_size))]
    history: list[dict] = []

    # Relatório de símbolos válidos antes de iniciar
    valid = [s for s, df in preds_by_sym.items() if isinstance(df, pd.DataFrame) and not df.empty]
    print(f"[ga] símbolos válidos para otimização: {len(valid)} / {len(symbols)}")
    if not valid:
        print("[ga] aviso: nenhum símbolo tem dados válidos na janela pedida (days/skip). Considere reduzir skip_days ou days, ou permitir recalcular caches.")
    
    # Thread pool para paralelizar backtest (avaliação de genomas)
    executor: ThreadPoolExecutor | None = None
    try:
        if int(workers) > 1:
            executor = ThreadPoolExecutor(max_workers=int(max(1, workers)))
            print(f"[ga] paralelização de backtest ativada: {int(workers)} workers")
        
        # Timer para primeira avaliação (ajuda a estimar tempo total)
        import time as _time
        first_eval_done = False
        
        # GA com paralelização de backtest
        for gen in range(int(generations)):
            scored: list[tuple[float, dict]] = []
            gen_desc = f"gen={gen:02d}/{generations-1}"
            gen_t0 = _time.time()
            
            if executor is not None:
                # Com paralelização: submete todos e atualiza barra conforme completam
                futures_map = {}
                for g in pop:
                    fut = executor.submit(_fitness, symbols, preds_by_sym, g, fee_per_side=fee_per_side, executor=None)
                    futures_map[fut] = g
                
                # Usa as_completed para atualizar barra em tempo real
                pbar = tqdm(total=len(futures_map), desc=f"[ga] {gen_desc}", leave=False, ncols=100)
                completed = 0
                for fut in as_completed(futures_map):
                    g = futures_map[fut]
                    try:
                        f = fut.result()
                    except Exception as e:
                        print(f"\n[ga] erro avaliando genoma: {e}")
                        f = -1e9
                    scored.append((f, g))
                    completed += 1
                    pbar.update(1)
                    
                    # Log do primeiro resultado para debug
                    if not first_eval_done:
                        first_eval_done = True
                        t_first = _time.time() - gen_t0
                        print(f"\n[ga] 1º genoma avaliado em {t_first:.1f}s", flush=True)
                pbar.close()
            else:
                # Sem paralelização: avalia sequencialmente com barra
                for i, g in enumerate(tqdm(pop, desc=f"[ga] {gen_desc}", total=len(pop), leave=False, ncols=100)):
                    f = _fitness(symbols, preds_by_sym, g, fee_per_side=fee_per_side, executor=None)
                    scored.append((f, g))
                    
                    if not first_eval_done:
                        first_eval_done = True
                        t_first = _time.time() - gen_t0
                        print(f"\n[ga] 1º genoma avaliado em {t_first:.1f}s", flush=True)
            
            # Tempo total da geração
            gen_time = _time.time() - gen_t0
            scored.sort(key=lambda x: x[0], reverse=True)
            best_f, best_g = scored[0]
            # métricas detalhadas do melhor da geração
            det = _aggregate_metrics_for_genome(symbols, preds_by_sym, best_g, fee_per_side=fee_per_side, executor=executor)
            if det.get("used_symbols", 0) > 0:
                # Log compacto das métricas + tempo
                n_syms = det.get('used_symbols', 1)
                tps = det['trades_sum'] / max(n_syms, 1)  # trades per symbol
                metrics = f"ret={det['ret_mean']*100:+.1f}% pf={det['pf_mean']:.2f} wr={det['wr_mean']:.0%} dd={det['dd_max']:.0%} trades={det['trades_sum']} ({tps:.1f}/sym)"
                params = f"th={best_g.get('thresh', 0.85):.2f} dca={best_g.get('dca_max_entries', 3)}x"
                print(f"[ga] gen={gen:02d} ({gen_time:.0f}s) f={best_f:.4f} | {metrics} | {params}")
            else:
                print(f"[ga] gen={gen:02d} ({gen_time:.0f}s) f={best_f:.6f} | sem símbolos válidos")
            history.append(dict(gen=int(gen), fitness=float(best_f), genome=best_g, summary=det))
            
            # Salva JSON intermediário a cada geração
            try:
                partial_res = dict(
                    run_dir=str(run_dir),
                    generation=int(gen),
                    total_generations=int(generations),
                    best_genome=best_g,
                    best_fitness=float(best_f),
                    best_summary=det,
                    history=history,
                    updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                partial_path = OUT_DIR / f"ga_partial_{run_dir.name}.json"
                partial_path.write_text(json.dumps(partial_res, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass  # não interrompe se falhar ao salvar
            
            # próxima geração
            next_pop: list[dict] = [scored[i][1] for i in range(min(int(elite), len(scored)))]
            while len(next_pop) < int(pop_size):
                pa = scored[rng.randint(0, max(1, len(scored)//2))][1]
                pb = scored[rng.randint(0, max(1, len(scored)//2))][1]
                child = _crossover(pa, pb, rng=rng)
                child = _mutate(child, rng=rng)
                next_pop.append(child)
            pop = next_pop
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # final — melhor da última geração
    scored_final: list[tuple[float, dict]] = []
    for g in pop:
        f = _fitness(symbols, preds_by_sym, g, fee_per_side=fee_per_side, executor=None)
        scored_final.append((f, g))
    scored_final.sort(key=lambda x: x[0], reverse=True)
    best_f, best_g = scored_final[0]
    det_final = _aggregate_metrics_for_genome(symbols, preds_by_sym, best_g, fee_per_side=fee_per_side, executor=None)

    res = dict(
        run_dir=str(run_dir),
        periods=periods,
        symbols=symbols,
        fee_round_trip=float(fee_round_trip),
        window=dict(days=int(days), skip_days=int(skip_days)),
        best_genome=best_g,
        best_fitness=float(best_f),
        best_summary=det_final,
        history=history,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    out_path = OUT_DIR / f"ga_result_{run_dir.name}_{int(time.time())}.json"
    try:
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ga] resultado salvo em {out_path}")
    except Exception:
        pass
    # salva um arquivo compacto junto ao run para o run_entry_predict consumir
    compact = dict(
        # Parâmetros de entrada
        thresh=float(best_g.get('thresh', 0.85)),
        conf_ema_win=int(best_g.get('conf_ema_win', 2)),
        
        # Parâmetros de saída
        exit_ema_win=int(best_g.get('exit_ema_win', 50)),
        exit_atr_win=int(best_g.get('exit_atr_win', 14)),
        exit_k=float(best_g.get('exit_k', 1.0)),
        
        # Parâmetros anti-pânico
        hold_signal_thresh=float(best_g.get('hold_signal_thresh', 0.50)),
        panic_dd_thresh=float(best_g.get('panic_dd_thresh', 0.15)),
        panic_abandon_thresh=float(best_g.get('panic_abandon_thresh', 0.20)),
        panic_abandon_candles=int(best_g.get('panic_abandon_candles', 20)),
        
        # Parâmetros DCA
        dca_enabled=True,
        dca_max_entries=int(best_g.get('dca_max_entries', 3)),
        dca_drop_pct=float(best_g.get('dca_drop_pct', 0.10)),
        dca_min_p=float(best_g.get('dca_min_p', 0.50)),
        
        # Meta
        fee_round_trip=float(fee_round_trip),
        window=dict(days=int(days), skip_days=int(skip_days)),
        created_at=res['created_at'],
        symbols=len(symbols),
    )
    try:
        (Path(res['run_dir']) / "ga_best_params.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ga] parâmetros compactos salvos em {Path(res['run_dir']) / 'ga_best_params.json'}")
    except Exception as e:
        print(f"[ga] aviso: não foi possível salvar ga_best_params.json no run_dir: {e}")
    return res


if __name__ == "__main__":
    """
    Configuração otimizada para i9 13th gen (32 threads) + 64GB RAM:
    
    - Todas as criptos do treinamento (~200+)
    - 32 workers (usa todos os threads)
    - pop_size=500, generations=15 → 7500 avaliações
    - Bounds FLEXÍVEIS: thresh a partir de 0.40, EMAs até 120+
    
    Salvamento a cada geração: ga_partial_<run>.json
    
    Fitness focado em CONSISTÊNCIA (Calmar ratio):
    - Maximiza retorno/drawdown
    - Favorece WR e PF altos
    - DD máximo 30%
    """
    _ = run_ga(
        run_hint=None,
        symbols=None,        # Usa TODAS do treinamento
        days=360,            # 1 ano de dados
        skip_days=360,       # Dados de 1 ano atrás (OOS)
        pop_size=500,        # 500 genomas por geração
        generations=15,      # 15 gerações
        elite=25,            # Top 25 sobrevivem
        fee_round_trip=0.001,
        seed=42,
        workers=32,          # TODOS os 32 threads
    )
    print("✓ GA concluído.")
