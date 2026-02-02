# -*- coding: utf-8 -*-
# Configuração (somente PF) — janelas e parâmetros usados pelas features/plot

EMA_WINDOWS = [120, 360, 720, 1440, 2880, 4320, 10080]
EMA_PAIRS = [(120, 360), (360, 720), (1440, 4320), (2880, 10080)]
ATR_MIN = [15, 30, 60, 5760]
VOL_MIN = [60, 240, 720, 1440, 10080]
CI_MIN = []
LOGRET_MIN = [240, 1440]
KELTNER_WIDTH_MIN  = [30]
KELTNER_CENTER_MIN = [60, 240, 720]
KELTNER_POS_MIN    = [2880]
KELTNER_Z_MIN      = []
RSI_PRICE_MIN = [7, 14]
RSI_EMA_PAIRS = [(5, 9), (9, 14), (2880, 1440)]
SLOPE_MIN   = [5, 10, 15]
CCI_MIN     = [240]
ADX_MIN     = [7, 15, 30, 120]
ZLOG_MIN    = [240, 1440]
MINMAX_MIN  = [15, 30, 60, 240, 720]
SLOPE_RESERR_MIN = [240, 720, 1440, 4320]
REV_WINDOWS = [15, 30, 60, 120]
VOL_RATIO_PAIRS  = [(120, 240), (240, 720), (720, 1440)]
RUN_WINDOWS_MIN = []
HHHL_WINDOWS_MIN = [30]
EMA_CONFIRM_SPANS_MIN = [10]
BREAK_LOOKBACK_MIN = [30, 60]
SLOPE_DIFF_PAIRS_MIN = [(3, 10), (5, 15)]
WICK_MEAN_WINDOWS_MIN = [15]
LIQUIDITY_MIN = []
VOL_Z_SHORT_MIN = [60]
VOL_Z_LONG_MIN = [720]
RANGE_RATIO_PAIRS = []
ROLLING_MINP_MODE = "full"  # full|half|progressive
EMA_MINP_MODE = "full"      # full|half|progressive

# Lista explicita de features a manter (baseado em feature_importance_entry.csv).
# Use [] para nao filtrar.
FEATURE_ALLOWLIST = [
    "cum_ret_pct_240",
    "cum_ret_pct_1440",
    "atr_pct_15",
    "atr_pct_30",
    "vol_ratio_pct_720_1440",
    "vol_ratio_pct_120_240",
    "vol_ratio_pct_240_720",
    "keltner_pos_2880",
    "slope_reserr_pct_1440",
    "bars_since_blow_60",
    "shitidx_pct_2880_10080",
    "adx_120",
    "vol_z",
    "rsi_ema2880_1440",
    "rsi_ema9_14",
    "rsi_ema5_9",
    "rsi_price_7",
    "rsi_price_14",
    "shitidx_pct_1440_4320",
    "vol_pct_60",
    "vol_pct_240",
    "vol_pct_720",
    "vol_pct_1440",
    "keltner_center_pct_60",
    "keltner_center_pct_240",
    "pct_from_max_240",
    "slope_reserr_pct_240",
    "slope_reserr_pct_720",
    "pct_from_min_15",
    "pct_from_min_60",
    "atr_pct_60",
    "cci_240",
    "vol_pct_10080",
    "adx_30",
    "adx_7",
    "pct_from_max_720",
    "pct_from_max_60",
    "rev_speed_up_30",
    "rev_speed_up_15",
    "rev_speed_up_60",
    "keltner_center_pct_720",
    "slope_pct_5",
    "slope_pct_10",
    "slope_pct_15",
    "zlog_240m",
    "zlog_1440m",
    "hl_cnt_30",
    "time_since_min_720",
    "time_since_min_60",
    "adx_15",
    "slope_diff_5_15",
    "rev_speed_up_120",
    "liquidity_ratio",
    "rev_speed_down_30",
    "rev_speed_down_15",
    "rev_speed_down_60",
    "time_since_min_30",
    "pct_from_min_30",
    "time_since_max_30",
    "time_since_max_60",
    "pct_from_min_720",
    "shadow_balance",
    "hh_cnt_30",
    "run_up_len",
    "rev_speed_down_120",
    "pct_from_max_15",
    "pct_from_min_240",
    "wick_lower_mean_15",
    "time_since_max_720",
    "log_volume_ema",
    "slope_diff_3_10",
    "bars_below_ema_10",
    "slope_reserr_pct_4320",
    "bars_above_ema_10",
    "keltner_halfwidth_pct_30",
    "atr_pct_5760",
    "break_high_30",
    "shadow_balance_raw",
]

