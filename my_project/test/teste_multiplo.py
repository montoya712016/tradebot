# -*- coding: utf-8 -*-
"""
Backtest Multi-Crypto com Gestão de Portfólio

Simula uma carteira de criptos onde:
1. A cada timestamp, verifica sinais de TODAS as criptos
2. Prioriza pela probabilidade (maior P primeiro)
3. Respeita limite de alavancagem (ex: 5x = 500% do capital)
4. Usa parâmetros otimizados do GA
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import time
import json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# imports com fallback (suporte a execução direta)
try:
    from ..prepare_features.prepare_features import DEFAULT_CANDLE_SEC
    from ..prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_ROOT = _Path(__file__).resolve().parents[1]
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from my_project.prepare_features.prepare_features import DEFAULT_CANDLE_SEC
    from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m

SAVE_DIR = Path(__file__).resolve().parents[2] / "models_classifier"
GA_CACHE_DIR = Path(__file__).resolve().parents[1] / "otimizacao" / "cache"

# Imports centralizados
try:
    from .load_models import (
        find_latest_run, build_X_from_features, get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run
    )
    from .backtest import (
        load_ga_params, apply_ga_params, prepare_portfolio_data, 
        simulate_portfolio_backtest_timestamp_by_timestamp
    )
    from ..entry_predict_core import predict_buy_short_proba_for_segments, preload_boosters
    from ..prepare_features.prepare_features import run as pf_run, build_flags, FEATURE_KEYS
except Exception:
    from my_project.test.load_models import (
        find_latest_run, build_X_from_features, get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run
    )
    from my_project.test.backtest import (
        load_ga_params, apply_ga_params, prepare_portfolio_data,
        simulate_portfolio_backtest_timestamp_by_timestamp
    )
    from my_project.entry_predict_core import predict_buy_short_proba_for_segments, preload_boosters
    from my_project.prepare_features.prepare_features import run as pf_run, build_flags, FEATURE_KEYS


# ============================================
# CONFIGURAÇÕES DE PORTFÓLIO
# ============================================
class PortfolioConfig:
    """Configurações de gestão de portfólio."""
    
    def __init__(
        self,
        leverage_max: float = 5.0,        # Alavancagem máxima (ex: 5x = 500%)
        capital_per_trade_pct: float = 0.20,  # % do capital por trade (ex: 20%)
        min_trades: int = 1,              # Mínimo de trades simultâneos
        priority_by: str = "probability",  # "probability" ou "u_pred"
    ):
        self.leverage_max = leverage_max
        self.capital_per_trade_pct = capital_per_trade_pct
        self.priority_by = priority_by
        
        # Calcula máximo de trades baseado em alavancagem
        # Ex: 5x leverage, 20% por trade → max 25 trades (100%/20% × 5)
        self.max_concurrent_trades = int((1.0 / capital_per_trade_pct) * leverage_max)
        self.min_trades = min(min_trades, self.max_concurrent_trades)
    
    def __repr__(self):
        return (
            f"PortfolioConfig(leverage={self.leverage_max}x, "
            f"capital/trade={self.capital_per_trade_pct*100:.0f}%, "
            f"max_trades={self.max_concurrent_trades})"
        )


def load_predictions_from_ga_cache(
    run_name: str = "wf_001",
    symbols: list[str] | None = None,
    max_symbols: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Carrega predições do cache do GA (muito mais rápido que recalcular).
    
    Args:
        run_name: Nome do run (ex: "wf_001")
        symbols: Lista de símbolos específicos (None = todos do cache)
        max_symbols: Limita número de símbolos (None = todos)
    
    Returns:
        Dict[symbol, DataFrame] com colunas: close, high, low, p_buy, p_short, u_pred
    """
    cache_files = list(GA_CACHE_DIR.glob(f"*_full_{run_name}.parquet"))
    
    if not cache_files:
        # Tenta padrão alternativo
        cache_files = list(GA_CACHE_DIR.glob(f"*_360d_skip360_{run_name}.parquet"))
    
    if not cache_files:
        print(f"[cache] Nenhum cache encontrado para {run_name} em {GA_CACHE_DIR}")
        return {}
    
    # Filtra por símbolos se especificado
    if symbols is not None:
        symbols_set = set(s.upper() for s in symbols)
        cache_files = [f for f in cache_files if f.name.split("_")[0].upper() in symbols_set]
    
    # Limita número de símbolos
    if max_symbols is not None and len(cache_files) > max_symbols:
        cache_files = cache_files[:max_symbols]
    
    print(f"[cache] Carregando {len(cache_files)} arquivos de cache...")
    
    preds: dict[str, pd.DataFrame] = {}
    for path in cache_files:
        try:
            symbol = path.name.split("_")[0]
            df = pd.read_parquet(path)
            if len(df) > 0:
                preds[symbol] = df
        except Exception as e:
            print(f"[cache] Erro ao carregar {path.name}: {e}")
    
    print(f"[cache] {len(preds)} símbolos carregados com sucesso")
    return preds


def compute_predictions_from_models(
    run_dir: Path,
    periods: list[int],
    symbols: list[str],
    *,
    days: int,
    skip_days: int = 0,
) -> dict[str, pd.DataFrame]:
    """Calcula previsões diretamente dos modelos (sem cache)."""
    if not symbols:
        return {}

    # Pré-carrega modelos para acelerar
    preload_boosters(run_dir, periods, print_timing=True)

    from concurrent.futures import ThreadPoolExecutor

    def _predict_one(s: str) -> tuple[str, pd.DataFrame | None]:
        try:
            raw = load_ohlc_1m_series(s, int(days + skip_days), remove_tail_days=int(skip_days))
            if raw.empty:
                return s, None
            ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
            if ohlc.empty:
                return s, None

            FLAGS = build_flags(enable=FEATURE_KEYS, label=False)
            FLAGS["pivots"] = False
            df = pf_run(ohlc, flags=FLAGS, plot=False)
            if df is None or len(df) == 0:
                return s, None

            feat_cols = load_feature_columns(run_dir, periods)
            Xdf = build_X_from_features(df, feat_cols)
            if Xdf.empty:
                return s, None

            periods_valid = get_valid_periods(
                run_dir=run_dir,
                symbol=s,
                data_start=Xdf.index[0],
                data_end=Xdf.index[-1],
                periods_available=periods,
            )
            if not periods_valid:
                return s, None

            p_buy, p_sho, u_pred, _, _ = predict_buy_short_proba_for_segments(
                Xdf, run_dir, periods_valid, print_timing=False, symbol=s
            )

            close_series = df["close"].reindex(Xdf.index)
            high_series = ohlc.get("high", close_series).reindex(Xdf.index)
            low_series = ohlc.get("low", close_series).reindex(Xdf.index)

            out = pd.DataFrame(
                index=Xdf.index,
                data={
                    "close": close_series.astype(np.float32),
                    "high": high_series.astype(np.float32),
                    "low": low_series.astype(np.float32),
                    "p_buy": p_buy.astype(np.float32),
                    "p_short": p_sho.astype(np.float32),
                    "u_pred": u_pred.astype(np.float32) if u_pred is not None else np.nan,
                },
            ).dropna()
            return s, out
        except Exception:
            return s, None

    preds: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, df in tqdm(
            ex.map(_predict_one, symbols),
            total=len(symbols),
            desc="[preds] Recalculando",
        ):
            if df is not None and len(df) > 0:
                preds[s] = df
    return preds


def load_symbols_from_sym_map(run_dir: Path) -> list[str]:
    """Carrega símbolos do sym_map.json do treinamento."""
    sym_map_path = run_dir / "dataset" / "sym_map.json"
    if sym_map_path.exists():
        try:
            return json.loads(sym_map_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def main(
    symbols: list[str] | None = None,
    days: int = 360,
    skip_days: int = 0,
    run_hint: str | None = None,
    *,
    # Configurações de portfólio
    leverage_max: float = 5.0,
    capital_per_trade_pct: float = 0.20,
    priority_by: str = "probability",
    # Opções
    use_ga_cache: bool = True,
    max_symbols: int | None = None,
    plot: bool = True,
):
    """
    Executa backtest multi-crypto com gestão de portfólio inteligente.
    
    Args:
        symbols: Lista de símbolos. Se None, usa todos do treinamento
        days: Número de dias de dados para backtest
        skip_days: Dias a remover do final (para OOS)
        run_hint: Dica para escolher o run (ex: 'wf_001')
        
        leverage_max: Alavancagem máxima (ex: 5.0 = 500%)
        capital_per_trade_pct: % do capital por trade (ex: 0.20 = 20%)
        priority_by: "probability" (P maior primeiro) ou "u_pred" (U maior primeiro)
        
        use_ga_cache: Se True, usa cache do GA (muito mais rápido)
        max_symbols: Limita número de símbolos (None = todos)
        plot: Se True, mostra gráficos
    
    Exemplo:
        leverage=5x, capital=20% → max 25 trades simultâneos (100%/20% × 5)
        leverage=3x, capital=10% → max 30 trades simultâneos (100%/10% × 3)
    """
    t_start = time.time()
    
    # Configuração de portfólio
    config = PortfolioConfig(
        leverage_max=leverage_max,
        capital_per_trade_pct=capital_per_trade_pct,
        priority_by=priority_by,
    )
    print(f"\n{'='*80}")
    print(f"BACKTEST MULTI-CRYPTO - GESTÃO DE PORTFÓLIO")
    print(f"{'='*80}")
    print(f"[config] {config}")
    
    # 1) Escolhe run e carrega parâmetros GA
    if run_hint is None:
        run_hint = "market_cap_150B_50M"
    
    run_dir, periods = choose_run(SAVE_DIR, run_hint)
    want_periods = [i*90 for i in range(1, 9)]  # 90..720
    avail = [p for p in want_periods if p in periods]
    if not avail:
        raise RuntimeError(f"Nenhum período válido encontrado. Disponíveis: {periods}")
    
    run_name = run_dir.name  # ex: "wf_001"
    print(f"[models] run_dir={run_dir}")
    print(f"[models] períodos disponíveis: {avail}")
    
    # Carrega parâmetros GA
    ga_params = load_ga_params(run_dir)
    params = apply_ga_params(ga_params)
    
    # Usa thresh do GA, mas DESATIVA filtro EMA (conf_ema_win=1)
    # Assim, compra sempre que P >= thresh, sem confirmação extra
    thresh = params['thresh']  # GA: 0.724
    conf_ema_win = 1  # DESATIVA filtro EMA! (era params['conf_ema_win'])
    exit_ema_win = params['exit_ema_win']
    exit_atr_win = params['exit_atr_win']
    exit_k = params['exit_k']
    fee_per_side = params['fee_per_side']
    thresh_secondary = params.get('thresh_secondary', None)
    thresh_window = params.get('thresh_window', 5)
    
    print(f"[params] thresh={thresh:.3f} | conf_ema={conf_ema_win} | exit_ema={exit_ema_win} | "
          f"exit_atr={exit_atr_win} | exit_k={exit_k:.3f} | fee={fee_per_side:.4f}")
    
    # 2) Carrega símbolos
    if symbols is None:
        # Usa símbolos do treinamento
        symbols = load_symbols_from_sym_map(run_dir)
        if not symbols:
            # Fallback: usa top_market_cap.txt
            top_file = Path(__file__).resolve().parents[1] / "top_market_cap.txt"
            if top_file.exists():
                cap_map: dict[str, float] = {}
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
                        cap_map[s] = float(int(v.strip().replace(".","").replace(",","")))
                    except Exception:
                        continue
                MIN_CAP, MAX_CAP = 50_000_000, 150_000_000_000
                symbols = sorted([s for s, cap in cap_map.items() 
                                if s not in {"BTCUSDT", "ETHUSDT"} and MIN_CAP <= cap <= MAX_CAP],
                               key=lambda s: -cap_map[s])
    
    if max_symbols is not None and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
    
    print(f"[symbols] {len(symbols)} símbolos: {symbols[:5]}{'...' if len(symbols)>5 else ''}")
    
    # 3) Carrega predições
    preds_by_sym: dict[str, pd.DataFrame] = {}
    if use_ga_cache:
        print("[preds] Usando cache do GA (rápido)...")
        preds_by_sym = load_predictions_from_ga_cache(
            run_name=run_name,
            symbols=symbols,
            max_symbols=max_symbols,
        )
        if not preds_by_sym:
            print("[preds] Cache ausente. Recalculando diretamente dos modelos...")

    if not preds_by_sym:
        preds_by_sym = compute_predictions_from_models(
            run_dir,
            avail,
            symbols,
            days=int(days),
            skip_days=int(skip_days),
        )
    
    if not preds_by_sym:
        print("[erro] Nenhum símbolo com predições válidas!")
        return None
    
    print(f"[preds] {len(preds_by_sym)} símbolos com dados válidos")
    
    # 4) Prepara dados para backtest
    print("[backtest] Preparando dados...")
    common_ts, data_by_sym = prepare_portfolio_data(list(preds_by_sym.keys()), preds_by_sym)
    
    if common_ts is None or len(common_ts) == 0:
        print("[erro] Nenhum timestamp comum entre os símbolos!")
        return None
    
    print(f"[backtest] {len(common_ts)} timestamps | {common_ts[0]} a {common_ts[-1]}")
    
    # 5) Executa backtest
    print(f"[backtest] Executando simulação (max {config.max_concurrent_trades} trades simultâneos)...")
    
    metrics = simulate_portfolio_backtest_timestamp_by_timestamp(
        list(preds_by_sym.keys()),
        common_ts,
        data_by_sym,
        thresh=float(thresh),
        conf_ema_win=int(conf_ema_win),
        exit_ema_win=int(exit_ema_win),
        exit_atr_win=int(exit_atr_win),
        exit_k=float(exit_k),
        fee_per_side=float(fee_per_side),
        capital_per_trade=float(config.capital_per_trade_pct),
        max_concurrent_trades=int(config.max_concurrent_trades),
        show_progress=True,
        return_curve=True,
        thresh_secondary=thresh_secondary,
        thresh_window=thresh_window,
    )
    
    # 6) Mostra resultados
    t_total = time.time() - t_start
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS DO BACKTEST")
    print(f"{'='*80}")
    print(f"Símbolos:                   {len(preds_by_sym)}")
    print(f"Período:                    {days} dias")
    print(f"Alavancagem máxima:         {config.leverage_max}x")
    print(f"Capital por trade:          {config.capital_per_trade_pct*100:.0f}%")
    print(f"Máx. trades simultâneos:    {config.max_concurrent_trades}")
    print(f"-"*80)
    print(f"Retorno Total:              {metrics['ret_total']*100:+.2f}%")
    print(f"Equity Final:               {metrics['eq_final']:.4f}")
    print(f"Win Rate:                   {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor:              {metrics['pf']:.2f}")
    print(f"Max Drawdown:               {metrics['max_dd']*100:.1f}%")
    print(f"Total de Trades:            {metrics['n_trades']}")
    
    if metrics['n_trades'] > 0:
        trades_per_day = metrics['n_trades'] / days
        avg_ret_per_trade = metrics['ret_total'] / metrics['n_trades'] * 100
        print(f"Trades por dia:             {trades_per_day:.2f}")
        print(f"Retorno médio por trade:    {avg_ret_per_trade:+.3f}%")
    
    if 'active_trades' in metrics:
        active = metrics['active_trades']
        print(f"Trades ativos (máx):        {int(np.max(active))}")
        print(f"Trades ativos (média):      {np.mean(active):.1f}")
    
    # Métricas de diagnóstico
    if '_debug_total_signals_found' in metrics:
        signals_found = metrics['_debug_total_signals_found']
        signals_used = metrics['_debug_total_signals_used']
        print(f"-"*80)
        print(f"Sinais encontrados:         {signals_found}")
        print(f"Sinais utilizados:          {signals_used} ({signals_used/max(signals_found,1)*100:.1f}%)")
    
    print(f"-"*80)
    print(f"Tempo total:                {t_total:.1f}s")
    print(f"{'='*80}\n")
    
    # 7) Plots
    if plot and 'equity_curve' in metrics and 'active_trades' in metrics:
        equity_curve = metrics['equity_curve']
        active_trades = metrics['active_trades']
        
        fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Plot 1: Trades ativos
        ax = axs[0]
        ax.fill_between(common_ts, 0, active_trades, alpha=0.4, color='#3498db')
        ax.plot(common_ts, active_trades, color='#2980b9', linewidth=1)
        ax.axhline(config.max_concurrent_trades, color='#e74c3c', linestyle='--', 
                   linewidth=1.5, label=f'Limite ({config.max_concurrent_trades})')
        ax.set_ylabel('Trades Ativos', fontsize=11)
        ax.set_title(f'Trades Ativos | {len(preds_by_sym)} criptos | Leverage {config.leverage_max}x', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Plot 2: Equity curve
        ax = axs[1]
        ax.plot(common_ts, equity_curve, color='#2ecc71', linewidth=1.5)
        ax.axhline(1.0, color='#888888', linestyle='--', linewidth=1)
        ax.fill_between(common_ts, 1.0, equity_curve, 
                        where=(equity_curve >= 1.0), color='#2ecc71', alpha=0.3)
        ax.fill_between(common_ts, 1.0, equity_curve, 
                        where=(equity_curve < 1.0), color='#e74c3c', alpha=0.3)
        ax.set_ylabel('Equity', fontsize=11)
        ax.set_title(f'Equity Curve | Retorno: {metrics["ret_total"]*100:+.2f}% | '
                     f'DD: {metrics["max_dd"]*100:.1f}%', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax = axs[2]
        equity_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_max - equity_curve) / np.maximum(equity_max, 1e-8) * 100
        ax.fill_between(common_ts, 0, drawdown, color='#e74c3c', alpha=0.4)
        ax.plot(common_ts, drawdown, color='#c0392b', linewidth=1)
        ax.axhline(config.leverage_max * 10, color='#f39c12', linestyle='--', 
                   linewidth=1, label=f'Warning ({config.leverage_max*10:.0f}%)')
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.set_xlabel('Data', fontsize=11)
        ax.set_title('Drawdown', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.invert_yaxis()
        
        # Formatação do eixo X
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    return metrics


if __name__ == '__main__':
    """
    Exemplo de uso:
    
    1. Backtest padrão (5x leverage, 20% por trade = max 25 trades):
       python teste_multiplo.py
    
    2. Mais conservador (3x leverage, 10% por trade = max 30 trades):
       Modifique: leverage_max=3.0, capital_per_trade_pct=0.10
    
    3. Mais agressivo (10x leverage, 50% por trade = max 20 trades):
       Modifique: leverage_max=10.0, capital_per_trade_pct=0.50
    """
    
    # ================== CONFIGURAÇÕES ==================
    
    # Símbolos (None = todos do treinamento)
    TEST_SYMBOLS = None
    
    # Período de teste
    TEST_DAYS = 360       # 1 ano de backtest
    TEST_SKIP_DAYS = 0    # 0 = usa dados mais recentes
    
    # Run do modelo
    TEST_RUN_HINT = "market_cap_150B_50M"
    
    # Gestão de portfólio
    TEST_LEVERAGE = 5.0           # 5x de alavancagem máxima
    TEST_CAPITAL_PER_TRADE = 0.20  # 20% do capital por trade
    # → Máximo de 25 trades simultâneos (100%/20% × 5)
    
    # Opções
    TEST_USE_GA_CACHE = True  # Usa cache do GA (MUITO mais rápido)
    TEST_MAX_SYMBOLS = None   # None = todos (ou número específico)
    TEST_PLOT = True
    
    # ==================================================
    
    main(
        symbols=TEST_SYMBOLS,
        days=TEST_DAYS,
        skip_days=TEST_SKIP_DAYS,
        run_hint=TEST_RUN_HINT,
        leverage_max=TEST_LEVERAGE,
        capital_per_trade_pct=TEST_CAPITAL_PER_TRADE,
        use_ga_cache=TEST_USE_GA_CACHE,
        max_symbols=TEST_MAX_SYMBOLS,
        plot=TEST_PLOT,
    )
