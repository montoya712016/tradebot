# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates

# imports com fallback (suporte a execução direta)
try:
    from ..prepare_features.prepare_features import run as pf_run
    from ..prepare_features.prepare_features import DEFAULT_CANDLE_SEC
    from ..prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_ROOT = _Path(__file__).resolve().parents[1]  # .../my_project
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from my_project.prepare_features.prepare_features import run as pf_run
    from my_project.prepare_features.prepare_features import DEFAULT_CANDLE_SEC
    from my_project.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m

try:
    from ..prepare_features.prepare_features import build_flags, FEATURE_KEYS
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_ROOT = _Path(__file__).resolve().parents[1]  # .../my_project
    _WORKSPACE = _PKG_ROOT.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from my_project.prepare_features.prepare_features import build_flags, FEATURE_KEYS

SAVE_DIR = Path(__file__).resolve().parents[2] / "models_classifier"

import json, time

# Imports centralizados
# Imports centralizados - mesmas funções que ga.py usa
try:
    from .load_models import (
        find_latest_run, find_latest_run_any, build_X_from_features, get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run
    )
    from .backtest import load_ga_params, apply_ga_params, simulate_backtest, calculate_entry_signals
    from ..entry_predict_core import predict_buy_short_proba_for_segments
except Exception:
    from my_project.test.load_models import (
        find_latest_run, find_latest_run_any, build_X_from_features, get_valid_periods, load_training_cutoffs,
        load_feature_columns, choose_run
    )
    from my_project.test.backtest import load_ga_params, apply_ga_params, simulate_backtest, calculate_entry_signals
    from my_project.entry_predict_core import predict_buy_short_proba_for_segments


def main(
    symbol: str = 'PSGUSDT',
    days: int = 60,
    run_hint: str | None = None,
    *,
    remove_tail_days: int = 0,
    use_ureg: bool = True,         # usa regressor de U se disponível (apenas logging/inspeção)
):
    # 1) Carrega modelos disponíveis do último run usando função centralizada
    t0 = time.time()
    run_dir, periods = choose_run(SAVE_DIR, run_hint)
    # Apenas janelas OOS (sem 0d)
    want_periods = [i*90 for i in range(1, 9)]  # 90..720
    avail = [p for p in want_periods if p in periods]
    if not avail:
        raise RuntimeError(f"Nenhum dos períodos esperados {want_periods} encontrado. Disponíveis: {periods}")
    print(f"[models] run_dir={run_dir} | periods_disp={periods}", flush=True)
    
    # 1a) Carrega parâmetros otimizados (GA) se existir no run_dir
    ga_params = load_ga_params(run_dir)
    if ga_params is not None:
        print(f"[params] carregado GA de {run_dir / 'ga_best_params.json'}", flush=True)
    
    # Aplica parâmetros GA (ou usa defaults)
    params = apply_ga_params(ga_params)
    thresh = params['thresh']
    conf_ema_win = params['conf_ema_win']
    exit_ema_win = params['exit_ema_win']
    exit_atr_win = params['exit_atr_win']
    exit_k = params['exit_k']
    fee_per_side = params['fee_per_side']
    exit_ema_min = params['exit_ema_min']
    exit_ema_max = params['exit_ema_max']
    u_to_win_scale = params['u_to_win_scale']
    use_dynamic_exit_ema = params['use_dynamic_exit_ema']
    thresh_secondary = params.get('thresh_secondary', None)
    thresh_window = params.get('thresh_window', 5)
    hold_signal_thresh = params.get('hold_signal_thresh', 0.5)
    hold_signal_lookback = params.get('hold_signal_lookback', 5)
    panic_dd_thresh = params.get('panic_dd_thresh', 0.15)
    panic_abandon_thresh = params.get('panic_abandon_thresh', 0.20)
    panic_abandon_candles = params.get('panic_abandon_candles', 20)
    dca_enabled = params.get('dca_enabled', True)
    dca_max_entries = params.get('dca_max_entries', 3)
    dca_drop_pct = params.get('dca_drop_pct', 0.10)
    dca_min_p = params.get('dca_min_p', 0.50)
    
    # Loga parâmetros efetivos
    print(
        "[params] efetivos | "
        f"thresh={thresh:.3f} | thresh_secondary={thresh_secondary:.3f} | thresh_window={thresh_window} | "
        f"conf_ema_win={int(conf_ema_win)} | "
        f"exit_ema_win={int(exit_ema_win)} | exit_atr_win={int(exit_atr_win)} | "
        f"exit_k={float(exit_k):.3f} | fee_per_side={float(fee_per_side):.6f}",
        flush=True
    )
    print(
        f"[params] anti-pânico | hold_thresh={hold_signal_thresh} | lookback={hold_signal_lookback} | "
        f"panic_dd>{panic_dd_thresh:.0%} → só sai se P<{panic_abandon_thresh:.0%} por {panic_abandon_candles} candles",
        flush=True
    )
    print(
        f"[params] DCA | enabled={dca_enabled} | max_entries={dca_max_entries}x (alavancagem) | "
        f"queda p/ DCA: 1ª={dca_drop_pct:.0%}, 2ª={dca_drop_pct*2:.0%} | min_P={dca_min_p:.0%}",
        flush=True
    )
    # Carrega apenas feature_columns (rápido); boosters serão carregados mais tarde, somente os necessários
    feat_cols = load_feature_columns(run_dir, avail)

    # 2) Dados + features — usa exatamente o mesmo acesso que prepare_features
    # Limita dias ao maior período disponível (para acelerar quando faltarem modelos maiores)
    max_model_days = max(avail)
    days_eff = int(min(int(days), int(max_model_days)))
    t_load0 = time.time()
    raw = load_ohlc_1m_series(symbol, int(days_eff + max(0, int(remove_tail_days))), remove_tail_days=int(remove_tail_days))
    t1 = time.time(); print(f"[timing] load_1m={(t1 - t_load0):.2f}s", flush=True)
    ohlc = to_ohlc_from_1m(raw, int(DEFAULT_CANDLE_SEC))
    t2 = time.time(); print(f"[timing] to_ohlc={(t2-t1):.2f}s", flush=True)
    # Predição: precisamos do label/U para avaliar lucro/prejuízo
    # Flags: liga todas as famílias de features para garantir colunas usadas pelos modelos
    FLAGS_TEST = build_flags(enable=FEATURE_KEYS, label=True)
    FLAGS_TEST["pivots"] = False
    df = pf_run(ohlc, flags=FLAGS_TEST, plot=False,)
    t3 = time.time(); print(f"[timing] pf_run(features_only)={(t3-t2):.2f}s", flush=True)
    Xdf = build_X_from_features(df, feat_cols)
    t4 = time.time(); print(f"[timing] build_X={(t4-t3):.2f}s", flush=True)
    if Xdf.empty:
        print('Sem features suficientes para predição.'); return
    # 3) Aplica modelos usando função centralizada (que já faz o mapeamento correto por timestamp)
    idx = Xdf.index
    data_start = idx[0] if len(idx) > 0 else None
    data_end = idx[-1] if len(idx) > 0 else None
    
    # Importa função para determinar períodos válidos
    try:
        from .load_models import get_valid_periods
    except Exception:
        from my_project.test.load_models import get_valid_periods
    
    # Determina quais períodos são válidos baseado nos cutoffs e timestamps dos dados
    periods_valid_for_segments = get_valid_periods(
        run_dir=run_dir,
        symbol=symbol,
        data_start=data_start,
        data_end=data_end,
        periods_available=avail,
    )
    
    print(f"[models] períodos válidos para {symbol}: {periods_valid_for_segments} (de {avail})", flush=True)
    
    if not periods_valid_for_segments:
        print(f"[models] AVISO: nenhum período válido para {symbol}. Não será possível fazer predições.", flush=True)
        return
    
    # Usa função centralizada de predição (que já faz o mapeamento correto por timestamp)
    p_buy, p_sho, u_pred, used_spans, per_times = predict_buy_short_proba_for_segments(
        Xdf=Xdf,
        run_dir=run_dir,
        periods_avail=periods_valid_for_segments,  # Passa apenas períodos válidos
        print_timing=True,
        symbol=symbol,  # Passa símbolo para carregar cutoffs específicos
    )
    
    # Carrega cutoffs para aplicar OOS depois (para verificação e plot)
    per_cut, last_train_ts0 = load_training_cutoffs(run_dir, symbol, periods_valid_for_segments)
    
    # Aplica corte OOS por período/símbolo para não usar dados vistos no treino
    # CRÍTICO: Garante que não usamos dados que foram vistos no treinamento
    try:
        idx = Xdf.index
        if len(idx) > 0:
            for (lft, rgt, psel) in list(used_spans or []):
                cutoff = per_cut.get(int(psel))
                if cutoff is None:
                    continue
                lft = pd.to_datetime(lft)
                rgt = pd.to_datetime(rgt)
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
                    u_pred[bad] = np.nan
                    if n_bad > 0:
                        print(f"[OOS] {symbol} periodo {psel}d: zerou {n_bad} preds (ts <= cutoff {cutoff.strftime('%Y-%m-%d')})", flush=True)
    except Exception as e:
        print(f"[OOS] {symbol}: erro ao aplicar cutoffs OOS: {e}", flush=True)

    # 3b) Decisão única por vela + Marcação por U: verde=lucro, vermelho=prejuízo
    # Usa função compartilhada para calcular sinais (mesma lógica do backtest)
    try:
        u_comp = df.loc[Xdf.index, 'U_compra'].to_numpy(np.float32)
        u_vend = df.loc[Xdf.index, 'U_venda'].to_numpy(np.float32)
    except Exception:
        u_comp = np.zeros(len(Xdf), np.float32)
        u_vend = np.zeros(len(Xdf), np.float32)
    
    # Calcula sinais usando função compartilhada (mesma lógica do backtest)
    close_series = df['close'].reindex(Xdf.index)
    buy_dec, sho_dec, cross_up, cross_dn = calculate_entry_signals(
        p_buy, p_sho, thresh, close_series, conf_ema_win,
        thresh_secondary=thresh_secondary, thresh_window=thresh_window
    )
    
    # Marcação por U: verde=lucro, vermelho=prejuízo
    green_mask = (buy_dec & (u_comp > 0.0)) | (sho_dec & (u_vend < 0.0))
    red_mask   = (buy_dec & (u_comp <= 0.0)) | (sho_dec & (u_vend >= 0.0))
    # evita dupla marcação do mesmo ponto: verde tem precedência
    red_mask = red_mask & (~green_mask)

    # Simulação de backtest usando função centralizada
    bt_result = simulate_backtest(
        df=df,
        Xdf=Xdf,
        p_buy=p_buy,
        p_sho=p_sho,
        u_pred=u_pred if bool(use_ureg) else None,
        thresh=thresh,
        conf_ema_win=conf_ema_win,
        exit_ema_win=exit_ema_win,
        exit_atr_win=exit_atr_win,
        exit_k=exit_k,
        fee_per_side=fee_per_side,
        use_dynamic_exit_ema=use_dynamic_exit_ema,
        exit_ema_min=exit_ema_min if use_dynamic_exit_ema else None,
        exit_ema_max=exit_ema_max if use_dynamic_exit_ema else None,
        u_to_win_scale=u_to_win_scale if use_dynamic_exit_ema else None,
        thresh_secondary=thresh_secondary,
        thresh_window=thresh_window,
        hold_signal_thresh=hold_signal_thresh,
        hold_signal_lookback=hold_signal_lookback,
        panic_dd_thresh=panic_dd_thresh,
        panic_abandon_thresh=panic_abandon_thresh,
        panic_abandon_candles=panic_abandon_candles,
        dca_enabled=dca_enabled,
        dca_max_entries=dca_max_entries,
        dca_drop_pct=dca_drop_pct,
        dca_min_p=dca_min_p,
    )
    
    trades = bt_result['trades']
    equity = bt_result['equity']
    long_line_plot = bt_result['long_line_plot']
    short_line_plot = bt_result['short_line_plot']
    entry_long_idx = bt_result['entry_long_idx']
    exit_long_idx = bt_result['exit_long_idx']
    entry_short_idx = bt_result['entry_short_idx']
    exit_short_idx = bt_result['exit_short_idx']
    
    # Calcula métricas dos trades
    n_trades = len(trades)
    if n_trades > 0:
        returns = []
        for (ei, xi, side) in trades:
            px_e = float(df.loc[Xdf.index[ei], 'close'])
            px_x = float(df.loc[Xdf.index[xi], 'close'])
            if side == 1:  # long
                r = (px_x / px_e) - 1.0
            else:  # short
                r = (px_e / px_x) - 1.0
            r_net = r - 2.0 * float(fee_per_side)
            returns.append(r_net)
        
        returns = np.array(returns)
        wins = returns[returns > 0.0]
        losses = returns[returns <= 0.0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / n_trades) if n_trades > 0 else 0.0
        
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(-losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0
        
        # Calcula drawdown máximo
        equity_finite = equity[np.isfinite(equity)]
        if len(equity_finite) > 0:
            equity_max = np.maximum.accumulate(equity_finite)
            drawdowns = (equity_max - equity_finite) / equity_max
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_dd = 0.0
        
        eq_final = float(equity[-1]) if len(equity) > 0 and np.isfinite(equity[-1]) else 1.0
        ret_total = eq_final - 1.0
        
        avg_return = float(returns.mean()) if len(returns) > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        
        # Print métricas
        print("\n" + "="*70, flush=True)
        print(f"[MÉTRICAS DO BACKTEST - {symbol}]", flush=True)
        print("="*70, flush=True)
        print(f"Equity Final:        {eq_final:.4f} ({ret_total*100:+.2f}%)", flush=True)
        print(f"Total de Trades:     {n_trades}", flush=True)
        print(f"Win Rate:            {win_rate*100:.2f}% ({win_count}W / {loss_count}L)", flush=True)
        print(f"Profit Factor:       {profit_factor:.3f}", flush=True)
        print(f"Max Drawdown:        {max_dd*100:.2f}%", flush=True)
        print(f"Retorno Médio:       {avg_return*100:+.2f}%", flush=True)
        if len(wins) > 0:
            print(f"Retorno Médio Win:   {avg_win*100:+.2f}%", flush=True)
        if len(losses) > 0:
            print(f"Retorno Médio Loss:  {avg_loss*100:+.2f}%", flush=True)
        print("="*70 + "\n", flush=True)
    else:
        print(f"\n[MÉTRICAS] Nenhum trade executado para {symbol}\n", flush=True)
    
    # Calcula EMA de confirmação para plot (se conf_ema_win > 1)
    if int(conf_ema_win) > 1:
        ema_conf_all = df['close'].ewm(span=int(conf_ema_win), adjust=False).mean()
    else:
        # conf_ema_win <= 1 significa filtro desativado, não plota EMA de confirmação
        ema_conf_all = None

    # 3) Plot
    has_ureg = bool(use_ureg) and np.isfinite(u_pred).any()
    # layout: price + P(buy) + P(short) + (ureg opcional) + (p_buy*u_pred opcional) + equity
    nrows = 4 + (1 if has_ureg else 0) + (1 if has_ureg else 0)  # +1 para ureg, +1 para p_buy*u_pred
    height = [3] + [1]*(nrows-1)
    # Remove constrained_layout para evitar erro quando há muitos subplots ou dados vazios
    # Ajusta altura da figura baseado no número de subplots
    fig_height = 9.0 if nrows <= 4 else (9.0 + (nrows - 4) * 1.5)
    fig, axs = plt.subplots(nrows, 1, figsize=(14, fig_height), sharex=True, gridspec_kw={'hspace':0.15, 'height_ratios':height})
    x = mdates.date2num(df.index)
    axs[0].plot(x, df['close'], color='k', lw=0.8); axs[0].set_title(f'{symbol} Close'); axs[0].grid()
    # EMA de confirmação (só plota se conf_ema_win > 1)
    if ema_conf_all is not None:
        try:
            axs[0].plot(x, ema_conf_all, color='#2980b9', lw=1.0, alpha=0.9, label=f'EMA{int(conf_ema_win)} confirm')
        except Exception:
            pass
    # marcações de lucro/prejuízo
    xt_ix = mdates.date2num(Xdf.index)
    try:
        axs[0].scatter(xt_ix[green_mask], df.loc[Xdf.index[green_mask], 'close'], s=14, c='#2ecc71', marker='o', linewidths=0.0, zorder=3, alpha=0.9, label='profit')
    except Exception:
        pass
    try:
        axs[0].scatter(xt_ix[red_mask],   df.loc[Xdf.index[red_mask],   'close'], s=14, c='#e74c3c', marker='o', linewidths=0.0, zorder=3, alpha=0.9, label='loss')
    except Exception:
        pass
    # linhas de saída e marcações de entrada/saída
    try:
        axs[0].plot(xt_ix, long_line_plot,  color='#2ecc71', lw=0.9, alpha=0.7, linestyle='--', label='exit long (ema±atr)')
        axs[0].plot(xt_ix, short_line_plot, color='#e74c3c', lw=0.9, alpha=0.7, linestyle='--', label='exit short (ema±atr)')
        if entry_long_idx:
            axs[0].scatter(xt_ix[np.array(entry_long_idx, dtype=int)], df.loc[Xdf.index[np.array(entry_long_idx, dtype=int)], 'close'], c='#2ecc71', s=22, marker='^', linewidths=0.0, zorder=4, label='long in')
        if exit_long_idx:
            axs[0].scatter(xt_ix[np.array(exit_long_idx, dtype=int)],  df.loc[Xdf.index[np.array(exit_long_idx, dtype=int)],  'close'], c='#27ae60', s=26, marker='x', linewidths=1.2, zorder=4, label='long out')
        if entry_short_idx:
            axs[0].scatter(xt_ix[np.array(entry_short_idx, dtype=int)], df.loc[Xdf.index[np.array(entry_short_idx, dtype=int)], 'close'], c='#e74c3c', s=22, marker='v', linewidths=0.0, zorder=4, label='short in')
        if exit_short_idx:
            axs[0].scatter(xt_ix[np.array(exit_short_idx, dtype=int)],  df.loc[Xdf.index[np.array(exit_short_idx, dtype=int)],  'close'], c='#c0392b', s=26, marker='x', linewidths=1.2, zorder=4, label='short out')
    except Exception:
        pass
    xt = mdates.date2num(Xdf.index)
    axs[1].plot(xt, p_buy,  color='tab:green'); axs[1].set_ylim(-0.05,1.05); axs[1].set_title('P(buy)');  axs[1].grid()
    try:
        axs[1].axhline(float(thresh), color='#888888', lw=0.8, linestyle='--', alpha=0.9)
    except Exception:
        pass
    axs[2].plot(xt, p_sho,  color='tab:red');   axs[2].set_ylim(-0.05,1.05); axs[2].set_title('P(short)'); axs[2].grid()
    try:
        axs[2].axhline(float(thresh), color='#888888', lw=0.8, linestyle='--', alpha=0.9)
    except Exception:
        pass
    # Subplot: U previsto (regressor), se disponível
    if has_ureg:
        ax_u = axs[3]
        ax_u.axhline(0.0, color='#888888', lw=0.8, linestyle='--')
        try:
            ax_u.plot(xt, u_pred, color='#8e44ad', lw=1.0)
        except Exception:
            pass
        ax_u.set_title('U previsto (regressor)')
        ax_u.grid(True, alpha=0.3)
        
        # Subplot: P(buy) * U previsto (combinação de probabilidade e utilidade)
        ax_pbuy_u = axs[4]
        ax_pbuy_u.axhline(0.0, color='#888888', lw=0.8, linestyle='--')
        try:
            p_buy_u = p_buy * u_pred
            ax_pbuy_u.plot(xt, p_buy_u, color='#16a085', lw=1.0)
        except Exception:
            pass
        ax_pbuy_u.set_title('P(buy) × U previsto')
        ax_pbuy_u.grid(True, alpha=0.3)
    
    # Subplot: Equity (backtest realizado por saídas EMA-seed, capital composto)
    eq_ax = axs[-1]
    eq_ax.axhline(1.0, color='#888888', lw=0.8, linestyle='--')
    eq_ax.plot(xt, equity, color='#333333', lw=1.2)
    eq_ax.set_title('Equity (realized compounded, EMA-seed exits)')
    eq_ax.grid(True, alpha=0.3)
    # faixas por período usado (estilo semelhante ao test_plot_month), limitadas ao início dos preços
    xmin = pd.to_datetime(df.index.min())
    xmax = pd.to_datetime(df.index.max())
    # estilo: buy (verde) e short (vermelho), alpha em gradiente por período (mais recente mais opaco)
    used_periods_sorted = sorted(set(int(p) for (_, _, p) in used_spans))
    period_to_alpha = {}
    for i, p in enumerate(used_periods_sorted):
        n = max(1, len(used_periods_sorted)-1)
        frac = (i / n) if n > 0 else 1.0
        period_to_alpha[p] = 0.06 + 0.06 * (1.0 - frac)  # 0.12 mais recente, 0.06 mais antigo
    green = '#2ecc71'; red = '#e74c3c'
    # desenha spans, linhas e rótulos
    for left, right, p_sel in used_spans:
        l = pd.to_datetime(left); r = pd.to_datetime(right)
        # limita ao range do preço
        l = max(l, xmin); r = min(r, xmax)
        if r <= l:
            continue
        a = period_to_alpha.get(int(p_sel), 0.08)
        axs[1].axvspan(l, r, facecolor=green, alpha=a, linewidth=0, zorder=0)
        axs[2].axvspan(l, r, facecolor=red,   alpha=a, linewidth=0, zorder=0)
        for ax in (axs[1], axs[2]):
            ax.axvline(l, color='#888888', alpha=0.35, linewidth=0.8, zorder=1)
        try:
            mid = l + (r - l) / 2
            y0b, y1b = axs[1].get_ylim()
            axs[1].text(mid, y1b - 0.04*(y1b - y0b), f"{int(p_sel)}d",
                        color=green, fontsize=8, ha='center', va='top',
                        alpha=0.85, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.0))
        except Exception:
            pass
    # Verificação OOS: compara spans com cutoffs do treinamento por símbolo
    try:
        dmeta_path = run_dir / 'dataset' / 'meta_long.parquet'
        smap_path  = run_dir / 'dataset' / 'sym_map.json'
        if dmeta_path.exists() and smap_path.exists():
            dmeta = pd.read_parquet(dmeta_path)
            sym_map = json.loads(Path(smap_path).read_text(encoding='utf-8'))
            try:
                sym_id = int(sym_map.index(symbol))
            except ValueError:
                sym_id = None
            if sym_id is not None:
                last_train_ts0 = pd.to_datetime(dmeta.loc[dmeta['sym_id'] == sym_id, 'ts']).max()
                if pd.notna(last_train_ts0):
                    print(f"[OOS] {symbol}: last_train_ts(0d) = {last_train_ts0}", flush=True)
                    # tenta cutoffs precisos por período/símbolo (se salvos)
                    per_cut: dict[int, pd.Timestamp] = {}
                    for p in sorted(set(int(p) for (_, _, p) in used_spans)):
                        jpath = run_dir / f'period_{int(p)}d' / 'dataset_cutoffs_long.json'
                        if jpath.exists():
                            try:
                                j = json.loads(jpath.read_text(encoding='utf-8'))
                                cut_s = j.get(symbol)
                                if cut_s:
                                    per_cut[int(p)] = pd.to_datetime(cut_s)
                            except Exception:
                                pass
                    # desenha linhas de cutoff por período
                    for p in sorted(set(int(p) for (_, _, p) in used_spans)):
                        cutoff = per_cut.get(int(p), last_train_ts0 - pd.Timedelta(days=int(p)))
                        for ax in (axs[1], axs[2]):
                            ax.axvline(cutoff, color='#666666', linestyle='--', linewidth=0.8, alpha=0.9)
                    # checa cada span
                    for (l, r, p_sel) in used_spans:
                        cutoff = per_cut.get(int(p_sel), last_train_ts0 - pd.Timedelta(days=int(p_sel)))
                        ok = (pd.to_datetime(l) > cutoff)
                        status = "OK" if ok else "VIOL"
                        print(f"[OOS] {int(p_sel):3d}d | cutoff={cutoff} | span=({pd.to_datetime(l)}, {pd.to_datetime(r)}) -> {status}", flush=True)
                else:
                    print(f"[OOS] {symbol}: last_train_ts não encontrado no meta_long.", flush=True)
            else:
                print(f"[OOS] {symbol}: não encontrado em sym_map.", flush=True)
    except Exception as _e:
        print(f"[OOS] verificação falhou: {_e}", flush=True)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')
    # Ajusta layout após todos os plots (evita erro do constrained_layout)
    try:
        plt.tight_layout(pad=2.0)
    except Exception:
        # Se tight_layout falhar, tenta ajuste manual
        plt.subplots_adjust(hspace=0.15, top=0.95, bottom=0.08)
    # resumo de tempos de carregamento de modelos
    if per_times:
        tot = sum(t for _, t in per_times)
        detail = ", ".join(f"{p}d={t:.2f}s" for p, t in per_times)
        print(f"[timing] load_models total={tot:.2f}s | {detail}", flush=True)
    else:
        print("[timing] load_models total=0.00s | none", flush=True)
    plt.show()


if __name__ == '__main__':
    # Parâmetros definidos no código (sem argumentos de linha de comando)
    TEST_SYMBOL = 'LAZIOUSDT'
    TEST_DAYS = 360
    TEST_RUN_HINT = 'models_classifier/market_cap_150B_50M/wf_002'
    TEST_REMOVE_TAIL_DAYS = 0
    # Execução
    main(TEST_SYMBOL, days=TEST_DAYS, run_hint=TEST_RUN_HINT, remove_tail_days=int(TEST_REMOVE_TAIL_DAYS))