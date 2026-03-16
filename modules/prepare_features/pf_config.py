# -*- coding: utf-8 -*-
# Configuração (somente PF) — janelas em MINUTOS.
# Essas janelas são convertidas para número de barras usando o timeframe do dataframe.
# Objetivo atual: funcionar bem no pipeline 5m sem colapsar indicadores para 1-2 barras.

EMA_WINDOWS = [60, 240, 720, 1440, 4320]
EMA_PAIRS = [(60, 240), (240, 1440), (720, 4320)]
ATR_MIN = [30, 60, 240, 1440]
VOL_MIN = [60, 240, 720, 1440, 10080]
CI_MIN = [60, 240]
LOGRET_MIN = [60, 240, 1440]
KELTNER_WIDTH_MIN  = [30, 60, 240]
KELTNER_CENTER_MIN = [60, 240, 720, 1440]
KELTNER_POS_MIN    = [240, 1440, 4320]
KELTNER_Z_MIN      = [240, 1440]
RSI_PRICE_MIN = [30, 60, 240]
RSI_EMA_PAIRS = [(30, 60), (60, 240)]
SLOPE_MIN   = [30, 60, 240, 720]
CCI_MIN     = [60, 240, 1440]
ADX_MIN     = [30, 60, 240]
ZLOG_MIN    = [240, 1440]
MINMAX_MIN  = [30, 60, 240, 720, 1440]
SLOPE_RESERR_MIN = [240, 720, 1440, 4320]
REV_WINDOWS = [30, 60, 240]
VOL_RATIO_PAIRS  = [(60, 240), (240, 1440)]
RUN_WINDOWS_MIN = [30, 60, 240]
HHHL_WINDOWS_MIN = [60, 240]
EMA_CONFIRM_SPANS_MIN = [30, 60, 240]
BREAK_LOOKBACK_MIN = [60, 240, 1440]
SLOPE_DIFF_PAIRS_MIN = [(30, 60), (60, 240)]
WICK_MEAN_WINDOWS_MIN = [30, 60]
LIQUIDITY_MIN = [60, 240, 1440]
VOL_Z_SHORT_MIN = [30]
VOL_Z_LONG_MIN = [240]
RANGE_RATIO_PAIRS = [(30, 240), (240, 1440)]
EFF_MIN = [30, 60, 240, 720, 1440]
ROLLING_MINP_MODE = "full"  # full|half|progressive
EMA_MINP_MODE = "full"      # full|half|progressive

