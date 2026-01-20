# -*- coding: utf-8 -*-
"""
Defaults for stocks prepare_features.
Keep crypto settings untouched.
"""

# Data selection
DEFAULT_SYMBOL: str = "AAPL"
DEFAULT_DAYS: int = 180
DEFAULT_REMOVE_TAIL_DAYS: int = 0
DEFAULT_CANDLE_SEC: int = 60

# Plot
DEFAULT_U_THRESHOLD: float = 0.0
DEFAULT_GREY_ZONE = None

# Feature flags (tune as needed)
FLAGS_STOCKS = {
    "shitidx":      True,
    "atr":          True,
    "rsi":          True,
    "slope":        True,
    "vol":          True,
    "ci":           True,
    "cum_logret":   True,
    "keltner":      True,
    "cci":          True,
    "adx":          True,
    "time_since":   True,
    "zlog":         True,
    "slope_reserr": True,
    "vol_ratio":    True,
    "regime":       True,
    "liquidity":    True,
    "rev_speed":    True,
    "vol_z":        True,
    "shadow":       True,
    "range_ratio":  True,
    "runs":         True,
    "hh_hl":        True,
    "ema_cross":    True,
    "breakout":     True,
    "mom_short":    True,
    "wick_stats":   True,
    "label":        True,
    "plot_candles": True,
}

