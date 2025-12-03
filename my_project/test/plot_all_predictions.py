"""
Visualização de Previsões de TODAS as Criptos

Carrega os parquets do cache do GA e plota:
- Subplot 1: P_buy de todas as criptos (sinais de compra)
- Subplot 2: P_short de todas as criptos (sinais de short)

Permite identificar momentos onde o modelo tem sinais fortes em várias criptos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================
CACHE_DIR = Path(__file__).parent.parent / "otimizacao" / "cache"
RUN_NAME = "wf_001"  # Nome do run (ajuste se necessário)

# Filtros de visualização
LAST_N_DAYS = None  # None = TODOS os dados disponíveis (~360 dias)
RESAMPLE_PERIOD = "1h"  # Resample para 1h (reduz ruído visual)
MIN_SIGNAL_TO_PLOT = 0.3  # Só plota linhas de criptos que tiveram P >= este valor

# Visual
ALPHA_LINE = 0.4  # Transparência das linhas individuais
SHOW_MEAN = True  # Mostra média de todas as criptos
SHOW_HEATMAP = True  # Mostra heatmap ao invés de linhas (melhor para muitas criptos)

# Indicador de Tendência de Mercado
MARKET_TREND_THRESH = 0.5  # Threshold para considerar sinal "ativo"
MARKET_TREND_MIN_CRYPTOS = 30  # Mínimo de criptos para considerar "tendência de mercado"


def load_all_predictions() -> dict[str, pd.DataFrame]:
    """Carrega todos os parquets do cache."""
    cache_files = list(CACHE_DIR.glob(f"*_full_{RUN_NAME}.parquet"))
    
    if not cache_files:
        # Tenta padrão alternativo
        cache_files = list(CACHE_DIR.glob(f"*_360d_skip360_{RUN_NAME}.parquet"))
    
    if not cache_files:
        raise FileNotFoundError(f"Nenhum cache encontrado em {CACHE_DIR} para {RUN_NAME}")
    
    print(f"[load] Encontrados {len(cache_files)} arquivos de cache")
    
    def _load_one(path: Path) -> tuple[str, pd.DataFrame | None]:
        try:
            symbol = path.name.split("_")[0]
            df = pd.read_parquet(path)
            if len(df) == 0:
                return symbol, None
            return symbol, df
        except Exception as e:
            print(f"[load] Erro ao carregar {path.name}: {e}")
            return path.name.split("_")[0], None
    
    # Carrega em paralelo
    results = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for symbol, df in ex.map(_load_one, cache_files):
            if df is not None and len(df) > 0:
                results[symbol] = df
    
    print(f"[load] Carregados {len(results)} símbolos com dados válidos")
    return results


def prepare_data(preds_by_sym: dict[str, pd.DataFrame], last_n_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara DataFrames consolidados de P_buy e P_short.
    
    Returns:
        df_buy: DataFrame com colunas = símbolos, index = timestamp, valores = P_buy
        df_short: DataFrame com colunas = símbolos, index = timestamp, valores = P_short
    """
    buy_series = {}
    short_series = {}
    
    for sym, df in preds_by_sym.items():
        if 'p_buy' not in df.columns or 'p_short' not in df.columns:
            continue
        
        # Filtra últimos N dias se especificado
        if last_n_days is not None and len(df) > 0:
            cutoff = df.index.max() - pd.Timedelta(days=last_n_days)
            df = df[df.index >= cutoff]
        
        if len(df) == 0:
            continue
        
        # Resample para reduzir ruído
        if RESAMPLE_PERIOD:
            df_rs = df[['p_buy', 'p_short']].resample(RESAMPLE_PERIOD).mean()
        else:
            df_rs = df[['p_buy', 'p_short']]
        
        buy_series[sym] = df_rs['p_buy']
        short_series[sym] = df_rs['p_short']
    
    # Combina em DataFrames (alinha índices automaticamente)
    df_buy = pd.DataFrame(buy_series)
    df_short = pd.DataFrame(short_series)
    
    # Preenche NaN com 0 (criptos que não existiam naquele timestamp)
    df_buy = df_buy.fillna(0)
    df_short = df_short.fillna(0)
    
    return df_buy, df_short


def filter_symbols_by_signal(df: pd.DataFrame, min_signal: float) -> list[str]:
    """Retorna símbolos que tiveram pelo menos um sinal >= min_signal."""
    return [col for col in df.columns if df[col].max() >= min_signal]


def plot_predictions_lines(df_buy: pd.DataFrame, df_short: pd.DataFrame, title_suffix: str = ""):
    """Plota previsões como linhas sobrepostas."""
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Filtra símbolos com sinais relevantes
    buy_symbols = filter_symbols_by_signal(df_buy, MIN_SIGNAL_TO_PLOT)
    short_symbols = filter_symbols_by_signal(df_short, MIN_SIGNAL_TO_PLOT)
    
    print(f"[plot] {len(buy_symbols)} criptos com P_buy >= {MIN_SIGNAL_TO_PLOT}")
    print(f"[plot] {len(short_symbols)} criptos com P_short >= {MIN_SIGNAL_TO_PLOT}")
    
    # Subplot 1: P_buy
    ax = axs[0]
    for sym in buy_symbols:
        ax.plot(df_buy.index, df_buy[sym], alpha=ALPHA_LINE, linewidth=0.5, label=sym)
    
    if SHOW_MEAN:
        mean_buy = df_buy[buy_symbols].mean(axis=1) if buy_symbols else df_buy.mean(axis=1)
        ax.plot(df_buy.index, mean_buy, color='blue', linewidth=2, alpha=0.9, label='MÉDIA')
    
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='thresh=0.5')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='thresh=0.7')
    ax.set_ylabel('P_buy', fontsize=12)
    ax.set_title(f'Sinais de COMPRA - {len(buy_symbols)} criptos {title_suffix}', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: P_short
    ax = axs[1]
    for sym in short_symbols:
        ax.plot(df_short.index, df_short[sym], alpha=ALPHA_LINE, linewidth=0.5, label=sym)
    
    if SHOW_MEAN:
        mean_short = df_short[short_symbols].mean(axis=1) if short_symbols else df_short.mean(axis=1)
        ax.plot(df_short.index, mean_short, color='red', linewidth=2, alpha=0.9, label='MÉDIA')
    
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='thresh=0.5')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='thresh=0.7')
    ax.set_ylabel('P_short', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_title(f'Sinais de SHORT - {len(short_symbols)} criptos {title_suffix}', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_predictions_heatmap(df_buy: pd.DataFrame, df_short: pd.DataFrame, title_suffix: str = ""):
    """Plota previsões como heatmap (melhor para muitas criptos)."""
    fig, axs = plt.subplots(2, 1, figsize=(18, 12))
    
    # Filtra símbolos com sinais relevantes
    buy_symbols = filter_symbols_by_signal(df_buy, MIN_SIGNAL_TO_PLOT)
    short_symbols = filter_symbols_by_signal(df_short, MIN_SIGNAL_TO_PLOT)
    
    # Resample adicional para heatmap (reduz número de colunas de tempo)
    df_buy_hm = df_buy[buy_symbols].T if buy_symbols else df_buy.T
    df_short_hm = df_short[short_symbols].T if short_symbols else df_short.T
    
    # Reduz para ~200 timestamps para visualização
    n_target = 200
    if df_buy_hm.shape[1] > n_target:
        step = df_buy_hm.shape[1] // n_target
        df_buy_hm = df_buy_hm.iloc[:, ::step]
        df_short_hm = df_short_hm.iloc[:, ::step]
    
    # Subplot 1: P_buy heatmap
    ax = axs[0]
    im = ax.imshow(df_buy_hm.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_ylabel('Cripto', fontsize=12)
    ax.set_title(f'Heatmap P_buy - {len(buy_symbols)} criptos {title_suffix}', fontsize=14)
    
    # Ticks do eixo X (datas)
    n_ticks = min(10, len(df_buy_hm.columns))
    tick_positions = np.linspace(0, len(df_buy_hm.columns)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([df_buy_hm.columns[i].strftime('%m/%d') for i in tick_positions], rotation=45)
    
    # Ticks do eixo Y (top 30 criptos por sinal máximo)
    if len(df_buy_hm) > 30:
        top_idx = df_buy_hm.max(axis=1).nlargest(30).index
        ytick_positions = [list(df_buy_hm.index).index(s) for s in top_idx if s in df_buy_hm.index][:10]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([df_buy_hm.index[i] for i in ytick_positions], fontsize=8)
    
    plt.colorbar(im, ax=ax, label='P_buy')
    
    # Subplot 2: P_short heatmap
    ax = axs[1]
    im = ax.imshow(df_short_hm.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_ylabel('Cripto', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_title(f'Heatmap P_short - {len(short_symbols)} criptos {title_suffix}', fontsize=14)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([df_short_hm.columns[i].strftime('%m/%d') for i in tick_positions], rotation=45)
    
    if len(df_short_hm) > 30:
        top_idx = df_short_hm.max(axis=1).nlargest(30).index
        ytick_positions = [list(df_short_hm.index).index(s) for s in top_idx if s in df_short_hm.index][:10]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([df_short_hm.index[i] for i in ytick_positions], fontsize=8)
    
    plt.colorbar(im, ax=ax, label='P_short')
    
    plt.tight_layout()
    plt.show()


def calculate_market_trend(df_buy: pd.DataFrame, df_short: pd.DataFrame, 
                           thresh: float = 0.5, min_cryptos: int = 30) -> pd.DataFrame:
    """
    Calcula indicador de TENDÊNCIA DE MERCADO.
    
    Retorna DataFrame com:
    - buy_count: número de criptos com P_buy >= thresh
    - short_count: número de criptos com P_short >= thresh
    - sentiment: buy_count - short_count
    - market_bullish: True se buy_count >= min_cryptos (fundo de mercado)
    - market_bearish: True se short_count >= min_cryptos (topo de mercado)
    - market_neutral: nem bullish nem bearish
    
    USO PRÁTICO:
    - Só compra individual se market_bullish = True (confirma com mercado)
    - Só shorta individual se market_bearish = True
    - Aumenta posição quando sentiment é extremo
    """
    buy_count = (df_buy >= thresh).sum(axis=1)
    short_count = (df_short >= thresh).sum(axis=1)
    sentiment = buy_count - short_count
    
    # Normaliza sentiment para % do total de criptos
    n_cryptos = len(df_buy.columns)
    sentiment_pct = sentiment / n_cryptos
    
    result = pd.DataFrame({
        'buy_count': buy_count,
        'short_count': short_count,
        'sentiment': sentiment,
        'sentiment_pct': sentiment_pct,
        'market_bullish': buy_count >= min_cryptos,
        'market_bearish': short_count >= min_cryptos,
    })
    result['market_neutral'] = ~result['market_bullish'] & ~result['market_bearish']
    
    return result


def plot_aggregate_signal(df_buy: pd.DataFrame, df_short: pd.DataFrame, thresh: float = 0.5):
    """
    Plota o número de criptos com sinal >= thresh ao longo do tempo.
    
    Útil para identificar momentos de "consenso de mercado" onde muitas criptos
    estão dando sinal de compra ou short simultaneamente.
    """
    # Calcula tendência de mercado
    trend = calculate_market_trend(df_buy, df_short, thresh, MARKET_TREND_MIN_CRYPTOS)
    
    fig, axs = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    
    # Subplot 1: Número de criptos com sinal de compra
    ax = axs[0]
    ax.fill_between(trend.index, 0, trend['buy_count'], alpha=0.5, color='green', label=f'P_buy >= {thresh}')
    ax.plot(trend.index, trend['buy_count'], color='darkgreen', linewidth=1)
    ax.axhline(y=MARKET_TREND_MIN_CRYPTOS, color='blue', linestyle='--', alpha=0.7, label=f'Min={MARKET_TREND_MIN_CRYPTOS} (tendência)')
    ax.set_ylabel(f'# Criptos com P_buy >= {thresh}', fontsize=11)
    ax.set_title('Sinais de COMPRA Simultâneos (Fundo de Mercado)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Subplot 2: Número de criptos com sinal de short
    ax = axs[1]
    ax.fill_between(trend.index, 0, trend['short_count'], alpha=0.5, color='red', label=f'P_short >= {thresh}')
    ax.plot(trend.index, trend['short_count'], color='darkred', linewidth=1)
    ax.axhline(y=MARKET_TREND_MIN_CRYPTOS, color='blue', linestyle='--', alpha=0.7, label=f'Min={MARKET_TREND_MIN_CRYPTOS} (tendência)')
    ax.set_ylabel(f'# Criptos com P_short >= {thresh}', fontsize=11)
    ax.set_title('Sinais de SHORT Simultâneos (Topo de Mercado)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Subplot 3: Sentimento geral (% do mercado)
    ax = axs[2]
    colors = np.where(trend['sentiment'] >= 0, 'green', 'red')
    ax.bar(trend.index, trend['sentiment'], color=colors, alpha=0.5, width=0.04)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=MARKET_TREND_MIN_CRYPTOS, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-MARKET_TREND_MIN_CRYPTOS, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Sentimento (Buy - Short)', fontsize=11)
    ax.set_title('Sentimento Geral do Mercado', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Estado do mercado (Bullish/Bearish/Neutral)
    ax = axs[3]
    state = np.where(trend['market_bullish'], 1, np.where(trend['market_bearish'], -1, 0))
    colors_state = np.where(state == 1, 'green', np.where(state == -1, 'red', 'gray'))
    ax.fill_between(trend.index, 0, state, alpha=0.6, 
                    color='green', where=(state == 1), label='BULLISH (comprar)')
    ax.fill_between(trend.index, 0, state, alpha=0.6, 
                    color='red', where=(state == -1), label='BEARISH (shortar)')
    ax.fill_between(trend.index, 0, state, alpha=0.3, 
                    color='gray', where=(state == 0), label='NEUTRAL (cuidado)')
    ax.set_ylabel('Estado', fontsize=11)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_title(f'ESTADO DO MERCADO (>={MARKET_TREND_MIN_CRYPTOS} criptos = tendência confirmada)', fontsize=14)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['BEARISH', 'NEUTRAL', 'BULLISH'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas
    bullish_pct = trend['market_bullish'].mean() * 100
    bearish_pct = trend['market_bearish'].mean() * 100
    neutral_pct = trend['market_neutral'].mean() * 100
    
    print(f"\n[market] Estatísticas de Tendência de Mercado:")
    print(f"  BULLISH (comprar): {bullish_pct:.1f}% do tempo")
    print(f"  BEARISH (shortar): {bearish_pct:.1f}% do tempo")
    print(f"  NEUTRAL (cuidado): {neutral_pct:.1f}% do tempo")
    
    # Picos
    max_buy = trend['buy_count'].max()
    max_short = trend['short_count'].max()
    print(f"\n[market] Picos:")
    print(f"  Máximo de criptos com sinal BUY: {max_buy} em {trend['buy_count'].idxmax()}")
    print(f"  Máximo de criptos com sinal SHORT: {max_short} em {trend['short_count'].idxmax()}")
    
    return trend


def main():
    print("=" * 60)
    print("VISUALIZAÇÃO DE PREVISÕES - TODAS AS CRIPTOS")
    print("=" * 60)
    
    # Carrega dados
    preds = load_all_predictions()
    
    if not preds:
        print("[ERRO] Nenhum dado carregado!")
        return
    
    # Prepara dados
    df_buy, df_short = prepare_data(preds, last_n_days=LAST_N_DAYS)
    
    print(f"\n[data] Shape P_buy: {df_buy.shape}")
    print(f"[data] Shape P_short: {df_short.shape}")
    print(f"[data] Período: {df_buy.index.min()} a {df_buy.index.max()}")
    
    # Estatísticas rápidas
    print(f"\n[stats] P_buy máximo: {df_buy.max().max():.3f}")
    print(f"[stats] P_short máximo: {df_short.max().max():.3f}")
    print(f"[stats] P_buy médio: {df_buy.mean().mean():.3f}")
    print(f"[stats] P_short médio: {df_short.mean().mean():.3f}")
    
    # Top criptos com maior sinal
    top_buy = df_buy.max().nlargest(10)
    top_short = df_short.max().nlargest(10)
    
    print(f"\n[top] Maiores P_buy máximos:")
    for sym, val in top_buy.items():
        print(f"  {sym}: {val:.3f}")
    
    print(f"\n[top] Maiores P_short máximos:")
    for sym, val in top_short.items():
        print(f"  {sym}: {val:.3f}")
    
    # Plota
    title_suffix = f"(últimos {LAST_N_DAYS}d)" if LAST_N_DAYS else "(período completo)"
    
    if SHOW_HEATMAP:
        print("\n[plot] Gerando heatmaps...")
        plot_predictions_heatmap(df_buy, df_short, title_suffix)
    else:
        print("\n[plot] Gerando gráficos de linhas...")
        plot_predictions_lines(df_buy, df_short, title_suffix)
    
    # Plota sinais agregados e calcula tendência de mercado
    print("\n[plot] Gerando gráfico de tendência de mercado...")
    trend = plot_aggregate_signal(df_buy, df_short, thresh=MARKET_TREND_THRESH)
    
    # Mostra como usar a tendência no backtest
    print("\n" + "=" * 60)
    print("COMO USAR A TENDÊNCIA DE MERCADO NO BACKTEST")
    print("=" * 60)
    print("""
1. FILTRO DE ENTRADA (mais conservador):
   - Só compra se market_bullish = True
   - Só shorta se market_bearish = True
   - Reduz falsos positivos em mercados laterais

2. SIZING DINÂMICO:
   - sentiment > 50: posição 150% (alta confiança)
   - sentiment 30-50: posição 100% (normal)
   - sentiment < 30: posição 50% (baixa confiança)

3. CONFIRMAÇÃO DE SINAL:
   - Cripto com P_buy = 0.8 + market_bullish → COMPRA FORTE
   - Cripto com P_buy = 0.8 + market_neutral → compra com cautela
   - Cripto com P_buy = 0.8 + market_bearish → NÃO COMPRA (contra tendência)

4. EXIT STRATEGY:
   - Se mercado vira BEARISH enquanto comprado → considera sair
   - Se mercado vira BULLISH enquanto shortado → considera fechar

EXEMPLO DE CÓDIGO PARA BACKTEST:
---------------------------------
# Carrega tendência
trend = calculate_market_trend(df_buy, df_short, thresh=0.5, min_cryptos=30)

# No loop de backtest:
if p_buy >= thresh and trend.loc[timestamp, 'market_bullish']:
    # Compra confirmada pelo mercado - posição maior
    position_size = 1.5
elif p_buy >= thresh and trend.loc[timestamp, 'market_neutral']:
    # Compra sem confirmação - posição normal
    position_size = 1.0
else:
    # Não compra
    position_size = 0
""")
    
    print("\n✓ Visualização concluída!")


if __name__ == "__main__":
    main()

