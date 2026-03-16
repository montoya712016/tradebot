# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Sequence, Tuple


@dataclass(frozen=True)
class TradeContract:
    """
    Define um contrato fixo de trade usado tanto na geração de labels quanto nas simulações.
    """

    timeframe_sec: int = 300
    entry_label_windows_minutes: Tuple[int, ...] = (240,)
    entry_label_min_profit_pcts: Tuple[float, ...] = (0.03,)
    entry_label_weight_alpha: float = 0.01
    # 24 barras em 5m = 120 minutos efetivos
    exit_ema_span: int = 24
    exit_ema_init_offset_pct: float = 0.005
    fee_pct_per_side: float = 0.0005
    slippage_pct: float = 0.0
    max_adds: int = 0
    add_spacing_pct: float = 0.0
    add_sizing: Tuple[float, ...] = field(default_factory=lambda: (1.0,))
    risk_max_cycle_pct: float = 0.0
    dd_intermediate_limit_pct: float = 0.0
    danger_drop_pct: float = 0.0
    danger_recovery_pct: float = 0.0
    danger_timeout_hours: float = 0.0
    danger_fast_minutes: float = 0.0
    danger_drop_pct_critical: float = 0.0
    danger_stabilize_recovery_pct: float = 0.0
    danger_stabilize_bars: int = 0
    # Stocks-only rule (ignored in crypto defaults)
    forbid_exit_on_gap: bool = False
    gap_hours_forbidden: float = 0.0

    def danger_horizon_bars(self, candle_sec: int) -> int:
        hours = self.danger_timeout_hours
        if hours <= 0:
            return 0
        bars = int(round((hours * 3600.0) / float(max(1, candle_sec))))
        return max(1, bars)

def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        return int(raw) if raw else int(default)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def timeframe_tag(candle_sec: int) -> str:
    candle_sec = int(max(1, candle_sec))
    if candle_sec % 60 == 0:
        return f"{int(candle_sec // 60)}m"
    return f"{int(candle_sec)}s"


def bars_from_minutes(minutes: float, candle_sec: int, *, min_bars: int = 1) -> int:
    return int(max(int(min_bars), round((float(minutes) * 60.0) / float(max(1, int(candle_sec))))))


# Variável única do pipeline crypto.
# Exemplos:
# - 300  -> 5m
# - 600  -> 10m
# - 900  -> 15m
CRYPTO_PIPELINE_CANDLE_SEC = _int_env("CRYPTO_PIPELINE_CANDLE_SEC", 300)
CRYPTO_ENTRY_LABEL_WINDOWS_MINUTES: Tuple[int, ...] = (240,)
CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCTS: Tuple[float, ...] = (_float_env("CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT", 0.03),)
CRYPTO_EXIT_EMA_SPAN_MINUTES: int = _int_env("CRYPTO_EXIT_EMA_SPAN_MINUTES", 120)
CRYPTO_EXIT_EMA_INIT_OFFSET_PCT: float = _float_env("CRYPTO_EXIT_EMA_INIT_OFFSET_PCT", 0.005)


def build_default_crypto_contract(candle_sec: int | None = None) -> TradeContract:
    candle_sec_i = int(max(1, candle_sec or CRYPTO_PIPELINE_CANDLE_SEC))
    return TradeContract(
        timeframe_sec=candle_sec_i,
        entry_label_windows_minutes=CRYPTO_ENTRY_LABEL_WINDOWS_MINUTES,
        entry_label_min_profit_pcts=CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCTS,
        entry_label_weight_alpha=0.01,
        exit_ema_span=bars_from_minutes(CRYPTO_EXIT_EMA_SPAN_MINUTES, candle_sec_i, min_bars=2),
        exit_ema_init_offset_pct=CRYPTO_EXIT_EMA_INIT_OFFSET_PCT,
        fee_pct_per_side=0.0005,
        slippage_pct=0.0,
        max_adds=0,
        add_spacing_pct=0.0,
        add_sizing=(1.0,),
        risk_max_cycle_pct=0.0,
        dd_intermediate_limit_pct=0.0,
        danger_drop_pct=0.0,
        danger_recovery_pct=0.0,
        danger_timeout_hours=0.0,
        danger_fast_minutes=0.0,
        danger_drop_pct_critical=0.0,
        danger_stabilize_recovery_pct=0.0,
        danger_stabilize_bars=0,
        forbid_exit_on_gap=False,
        gap_hours_forbidden=0.0,
    )


def apply_crypto_pipeline_env(candle_sec: int | None = None) -> int:
    candle_sec_i = int(max(1, candle_sec or CRYPTO_PIPELINE_CANDLE_SEC))
    os.environ["CRYPTO_PIPELINE_CANDLE_SEC"] = str(candle_sec_i)
    os.environ["SNIPER_CANDLE_SEC"] = str(candle_sec_i)
    os.environ["PF_CRYPTO_CANDLE_SEC"] = str(candle_sec_i)
    return candle_sec_i


DEFAULT_TRADE_CONTRACT = build_default_crypto_contract()

__all__ = [
    "TradeContract",
    "DEFAULT_TRADE_CONTRACT",
    "CRYPTO_PIPELINE_CANDLE_SEC",
    "CRYPTO_ENTRY_LABEL_WINDOWS_MINUTES",
    "CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCTS",
    "CRYPTO_EXIT_EMA_SPAN_MINUTES",
    "CRYPTO_EXIT_EMA_INIT_OFFSET_PCT",
    "build_default_crypto_contract",
    "apply_crypto_pipeline_env",
    "timeframe_tag",
    "bars_from_minutes",
    "exit_ema_span_from_window",
]


def exit_ema_span_from_window(contract: TradeContract, candle_sec: int = 60) -> int:
    explicit_span = int(getattr(contract, "exit_ema_span", 0) or 0)
    if explicit_span > 0:
        return explicit_span
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    if not windows:
        return 0
    candle_sec = int(max(1, candle_sec))
    w_min = float(windows[0])
    return int(max(1, round((w_min * 60.0) / float(candle_sec))))
