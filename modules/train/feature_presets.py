# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from typing import Iterable
import os

try:
    from prepare_features.prepare_features import FEATURE_KEYS, build_flags  # type: ignore
except Exception:
    from prepare_features.prepare_features import FEATURE_KEYS, build_flags  # type: ignore[import]


CORE80_FEATURES = (
    "shitidx_pct_60_240",
    "shitidx_pct_240_1440",
    "shitidx_pct_720_4320",
    "atr_pct_30",
    "atr_pct_60",
    "atr_pct_240",
    "rsi_price_30",
    "rsi_price_60",
    "rsi_price_240",
    "rsi_ema30_60",
    "slope_pct_30",
    "slope_pct_60",
    "slope_pct_240",
    "slope_pct_720",
    "vol_pct_60",
    "vol_pct_240",
    "vol_pct_720",
    "vol_pct_1440",
    "ci_60",
    "cum_ret_pct_60",
    "cum_ret_pct_240",
    "cum_ret_pct_1440",
    "keltner_center_pct_60",
    "keltner_center_pct_240",
    "keltner_center_pct_1440",
    "keltner_pos_240",
    "keltner_halfwidth_pct_30",
    "keltner_halfwidth_pct_60",
    "keltner_halfwidth_pct_240",
    "cci_60",
    "adx_30",
    "adx_60",
    "adx_240",
    "pct_from_min_30",
    "pct_from_min_60",
    "pct_from_min_240",
    "pct_from_min_720",
    "pct_from_max_30",
    "pct_from_max_60",
    "pct_from_max_240",
    "pct_from_max_720",
    "time_since_min_30",
    "time_since_min_60",
    "time_since_min_240",
    "time_since_min_720",
    "zlog_240m",
    "zlog_1440m",
    "slope_reserr_pct_240",
    "slope_reserr_pct_720",
    "slope_reserr_pct_1440",
    "vol_ratio_pct_60_240",
    "vol_ratio_pct_240_1440",
    "log_volume_ema",
    "liquidity_ratio",
    "volume_to_range_ema60",
    "volume_to_range_ema240",
    "rev_speed_up_30",
    "rev_speed_down_30",
    "rev_speed_up_240",
    "rev_speed_down_240",
    "vol_z",
    "signed_vol_z",
    "shadow_balance",
    "shadow_balance_raw",
    "range_ratio_30_240",
    "eff_30",
    "eff_60",
    "eff_240",
    "eff_720",
    "run_up_cnt_30",
    "run_dn_cnt_30",
    "run_up_cnt_240",
    "run_dn_cnt_60",
    "run_up_len",
    "hh_cnt_60",
    "hl_cnt_60",
    "hh_cnt_240",
    "bars_above_ema_30",
    "bars_above_ema_60",
    "bars_since_cross_60",
    "bars_above_ema_240",
    "break_high_60",
    "break_low_60",
    "bars_since_bhigh_240",
    "bars_since_blow_240",
    "slope_diff_30_60",
    "wick_lower_mean_60",
)


CORE65_FEATURES = (
    "shitidx_pct_60_240",
    "shitidx_pct_240_1440",
    "atr_pct_30",
    "atr_pct_60",
    "atr_pct_240",
    "rsi_price_30",
    "rsi_price_60",
    "rsi_price_240",
    "rsi_ema30_60",
    "slope_pct_30",
    "slope_pct_60",
    "slope_pct_240",
    "vol_pct_60",
    "vol_pct_240",
    "vol_pct_720",
    "vol_pct_1440",
    "ci_60",
    "cum_ret_pct_60",
    "cum_ret_pct_240",
    "keltner_center_pct_60",
    "keltner_center_pct_240",
    "keltner_center_pct_1440",
    "keltner_pos_240",
    "keltner_halfwidth_pct_60",
    "cci_60",
    "adx_30",
    "adx_60",
    "pct_from_min_30",
    "pct_from_min_60",
    "pct_from_max_30",
    "pct_from_max_60",
    "time_since_min_30",
    "time_since_min_60",
    "time_since_max_30",
    "time_since_max_60",
    "zlog_240m",
    "slope_reserr_pct_240",
    "slope_reserr_pct_720",
    "slope_reserr_pct_1440",
    "vol_ratio_pct_60_240",
    "log_volume_ema",
    "liquidity_ratio",
    "volume_to_range_ema60",
    "volume_to_range_ema240",
    "rev_speed_up_30",
    "rev_speed_down_30",
    "rev_speed_up_240",
    "rev_speed_down_240",
    "vol_z",
    "signed_vol_z",
    "shadow_balance_raw",
    "range_ratio_30_240",
    "eff_30",
    "eff_60",
    "eff_240",
    "eff_720",
    "run_up_cnt_30",
    "run_dn_cnt_30",
    "run_up_cnt_240",
    "run_up_len",
    "hh_cnt_60",
    "hl_cnt_60",
    "bars_above_ema_60",
    "bars_since_cross_60",
    "bars_above_ema_240",
    "break_high_60",
    "break_low_60",
    "bars_since_bhigh_240",
    "slope_diff_30_60",
    "wick_lower_mean_60",
)


CORE50_FEATURES = (
    "shitidx_pct_60_240",
    "shitidx_pct_240_1440",
    "atr_pct_30",
    "atr_pct_60",
    "rsi_price_30",
    "rsi_price_60",
    "rsi_ema30_60",
    "slope_pct_30",
    "slope_pct_60",
    "slope_pct_240",
    "vol_pct_60",
    "vol_pct_240",
    "vol_pct_720",
    "cum_ret_pct_60",
    "cum_ret_pct_240",
    "keltner_center_pct_60",
    "keltner_center_pct_240",
    "keltner_center_pct_1440",
    "keltner_pos_240",
    "adx_30",
    "adx_60",
    "pct_from_min_30",
    "pct_from_max_30",
    "time_since_min_30",
    "time_since_max_30",
    "zlog_240m",
    "slope_reserr_pct_240",
    "slope_reserr_pct_720",
    "vol_ratio_pct_60_240",
    "log_volume_ema",
    "liquidity_ratio",
    "volume_to_range_ema60",
    "rev_speed_up_30",
    "rev_speed_down_30",
    "vol_z",
    "signed_vol_z",
    "shadow_balance_raw",
    "eff_30",
    "eff_60",
    "eff_240",
    "run_up_cnt_30",
    "run_dn_cnt_30",
    "run_up_cnt_240",
    "hh_cnt_60",
    "hl_cnt_60",
    "bars_above_ema_60",
    "bars_since_cross_60",
    "break_high_60",
    "break_low_60",
    "wick_lower_mean_60",
)


FEATURE_PRESETS = {
    "core80": CORE80_FEATURES,
    "core65": CORE65_FEATURES,
    "core50": CORE50_FEATURES,
}


def _block_from_feature(name: str) -> str:
    if name.startswith("shitidx_pct_"):
        return "shitidx"
    if name.startswith("atr_pct_"):
        return "atr"
    if name.startswith("rsi_price_") or name.startswith("rsi_ema"):
        return "rsi"
    if name.startswith("slope_pct_"):
        return "slope"
    if name.startswith("vol_pct_"):
        return "vol"
    if name.startswith("ci_"):
        return "ci"
    if name.startswith("cum_ret_pct_"):
        return "cum_logret"
    if name.startswith("keltner_"):
        return "keltner"
    if name.startswith("cci_"):
        return "cci"
    if name.startswith("adx_"):
        return "adx"
    if name.startswith("pct_from_") or name.startswith("time_since_"):
        return "time_since"
    if name.startswith("zlog_"):
        return "zlog"
    if name.startswith("slope_reserr_pct_"):
        return "slope_reserr"
    if name.startswith("vol_ratio_pct_"):
        return "vol_ratio"
    if name in {"log_volume_ema", "liquidity_ratio"}:
        return "regime"
    if name.startswith("volume_to_range_ema"):
        return "liquidity"
    if name.startswith("rev_speed_"):
        return "rev_speed"
    if name in {"vol_z", "signed_vol_z"}:
        return "vol_z"
    if name.startswith("shadow_balance"):
        return "shadow"
    if name.startswith("range_ratio_"):
        return "range_ratio"
    if name.startswith("eff_"):
        return "eff"
    if name.startswith("run_"):
        return "runs"
    if name.startswith("hh_cnt_") or name.startswith("hl_cnt_"):
        return "hh_hl"
    if name.startswith("bars_above_ema_") or name.startswith("bars_below_ema_") or name.startswith("bars_since_cross_"):
        return "ema_cross"
    if name.startswith("break_high_") or name.startswith("break_low_") or name.startswith("bars_since_bhigh_") or name.startswith("bars_since_blow_"):
        return "breakout"
    if name.startswith("slope_diff_"):
        return "mom_short"
    if name.startswith("wick_"):
        return "wick_stats"
    return "unknown"


def resolve_feature_preset_name(name: str | None) -> str:
    raw = str(name or "").strip().lower()
    if raw in {"", "default", "full", "all", "none"}:
        return "full"
    if raw not in FEATURE_PRESETS:
        raise ValueError(f"unknown feature preset: {name}")
    return raw


def get_feature_preset(name: str | None) -> tuple[str, ...] | None:
    preset = resolve_feature_preset_name(name)
    if preset == "full":
        return None
    return tuple(str(x) for x in FEATURE_PRESETS[preset])


def feature_flags_for_preset(name: str | None, *, label: bool = True) -> dict[str, bool]:
    cols = get_feature_preset(name)
    if cols is None:
        return build_flags(enable=FEATURE_KEYS, label=label)
    enabled_blocks = sorted({_block_from_feature(col) for col in cols if _block_from_feature(col) != "unknown"})
    return build_flags(enable=enabled_blocks, label=label)


@lru_cache(maxsize=8)
def feature_allowlist_for_preset(name: str | None) -> frozenset[str] | None:
    cols = get_feature_preset(name)
    if cols is None:
        return None
    return frozenset(str(x) for x in cols)


def active_feature_preset_name(default: str = "full") -> str:
    return resolve_feature_preset_name(os.getenv("SNIPER_FEATURE_PRESET", default))


def active_feature_allowlist(default: str = "full") -> frozenset[str] | None:
    return feature_allowlist_for_preset(active_feature_preset_name(default))


def summarize_preset(name: str | None) -> dict[str, object]:
    preset = resolve_feature_preset_name(name)
    cols = get_feature_preset(preset)
    if cols is None:
        return {
            "preset": "full",
            "count": None,
            "blocks": list(FEATURE_KEYS),
        }
    blocks = sorted({_block_from_feature(col) for col in cols if _block_from_feature(col) != "unknown"})
    return {
        "preset": preset,
        "count": len(cols),
        "blocks": blocks,
    }
