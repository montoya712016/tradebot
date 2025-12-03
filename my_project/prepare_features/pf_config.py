# -*- coding: utf-8 -*-
# Configuração (somente PF) — janelas e parâmetros usados pelas features/plot

EMA_WINDOWS = [1440, 2880, 4320, 10080]
EMA_PAIRS = [(1440, 4320), (2880, 10080)]
ATR_MIN = [30, 60, 5760]
VOL_MIN = [240, 720, 1440, 10080]
CI_MIN = []
LOGRET_MIN = [1440]
KELTNER_WIDTH_MIN  = [30, 60]
KELTNER_CENTER_MIN = [60, 240, 720]
KELTNER_POS_MIN    = [360, 2880]
KELTNER_Z_MIN      = []
RSI_PRICE_MIN = []
RSI_EMA_PAIRS = [(2880, 1440)]
SLOPE_MIN   = [15, 30]
CCI_MIN     = [240, 1440]
ADX_MIN     = [15, 30, 120]
ZLOG_MIN    = [1440]
MINMAX_MIN  = [240, 720]
SLOPE_RESERR_MIN = [720, 1440, 4320]
REV_WINDOWS = [30, 60, 120]
VOL_RATIO_PAIRS  = [(720, 1440)]
RUN_WINDOWS_MIN = [30, 60]
HHHL_WINDOWS_MIN = [30]
EMA_CONFIRM_SPANS_MIN = [10]
BREAK_LOOKBACK_MIN = [30, 60]
SLOPE_DIFF_PAIRS_MIN = [(3, 10), (5, 15)]
WICK_MEAN_WINDOWS_MIN = [5, 15]

# Pivots / Reversões (parâmetros padrão)
PIVOT_MIN_MOVE_PCT   = 0.010   # 1.0% amplitude mínima entre pivots
PIVOT_MIN_BARS_MIN   = 12      # distância mínima (min) entre pivots
REV_LOOKAHEAD_MIN    = 4*60    # minutos para confirmar reversão
REV_MIN_UP_PCT       = 0.008   # 0.8% de recuperação mínima
IMPULSE_MAX_1BAR_PCT = 0.020   # 2.0% máx por barra

# Extrema locais (modo mais permissivo)
PIVOT_LOCAL_WIN_MIN       = 30  # janela (min) para checar min/max local
PIVOT_LOCAL_MERGE_TOL_MIN = 30  # tolerância (min) para fundir marcas muito próximas
PIVOT_INCLUDE_LOCAL       = True

# ---- Regras de reversão usadas nos labels -----------------------------------
# Em vez de espalhar números mágicos, usamos um "pacote" único e derivamos o que
# for necessário a partir dele. Ajuste aqui para alterar toda a regra.
REVERSAL_RULE = dict(
    lookback_min=6 * 60,   # janelas em minutos para olhar o topo/fundo anterior
    move_pct=0.03,         # queda/alta mínima (% relativa ao topo/fundo anterior)
    u_min_pct=5.0,         # ganho mínimo esperado (U_total) após a reversão
    max_distance_pct=0.015 # tolerância do preço atual em relação ao extremo
)

RULE_LOOKBACK_MIN = int(REVERSAL_RULE["lookback_min"])
RULE_DROP_PCT     = float(REVERSAL_RULE["move_pct"])
RULE_RISE_PCT     = float(REVERSAL_RULE["move_pct"])
RULE_U_HI         = float(REVERSAL_RULE["u_min_pct"])
RULE_U_LO         = -RULE_U_HI
RULE_MAX_DIST_FROM_EXTREME_PCT = float(REVERSAL_RULE["max_distance_pct"])

# Esquema de pesos (reaproveitado no dataflow/training e em visualizações)
WEIGHT_RULE = dict(
    u_weight_max=10.0,  # |U| considerado "muito bom/muito ruim"
    alpha_pos=2.0,      # bônus para positivos fortes (acerto obrigatório)
    beta_neg=3.0,       # penalização para negativos perigosos (fake pos)
    gamma_ease=0.6,     # alívio para negativos "ok" (fake neg)
    neg_floor=0.5,      # limite inferior para peso negativo
)

# Parâmetros de label compartilhados (usados em prepare_features.run)
DEFAULT_P_TGT: float = RULE_U_HI   # alvo de lucro em %
DEFAULT_DMAX: float = 2.0          # drawdown máximo tolerado em %
DEFAULT_DROP_CONFIRM: float = 2.0
DEFAULT_LABEL_CLIP: float = 50.0