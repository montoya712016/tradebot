# -*- coding: utf-8 -*-
"""
Wrapper to run prepare_features for crypto (debug/visual).
"""
from __future__ import annotations

from pathlib import Path
import sys
import os


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if p.name.lower() == "tradebot":
            root = p
            break
    if root:
        for cand in (root / "modules", root):
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)


_add_repo_paths()

from modules.prepare_features.prepare_features import run_from_flags_dict
from modules.prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
from modules.prepare_features import pf_config as cfg
from modules.prepare_features import features as featmod
from crypto.trade_contract import DEFAULT_TRADE_CONTRACT as CRYPTO_CONTRACT
from modules.prepare_features.prepare_features import (
    DEFAULT_SYMBOL,
    DEFAULT_DAYS,
    DEFAULT_REMOVE_TAIL_DAYS,
    DEFAULT_CANDLE_SEC,
    DEFAULT_U_THRESHOLD,
    DEFAULT_GREY_ZONE,
)


# Exemplo pronto (compatível com o estilo antigo) — pode importar direto como FLAGS
FLAGS_CRYPTO: dict[str, bool] = {
    "shitidx": True,
    "atr": False,
    "rsi": True,
    "slope": False,
    "vol": False,
    "ci": False,
    "cum_logret": False,
    "keltner": False,
    "cci": False,
    "adx": True,
    "time_since": False,
    "zlog": False,
    "slope_reserr": False,
    "vol_ratio": False,
    "regime": False,
    "liquidity": False,
    "rev_speed": False,
    "vol_z": False,
    "shadow": False,
    "range_ratio": False,
    "runs": False,
    "hh_hl": False,
    "ema_cross": False,
    "breakout": False,
    "mom_short": False,
    "wick_stats": False,
    "label": False,
    "plot_candles": True,
}


CFG_CRYPTO_WINDOWS = {
    "ATR_MIN": (30, 60, 5760),
    "RSI_PRICE_MIN": (14,),
    "RSI_EMA_PAIRS": ((9, 14),),
    "SLOPE_MIN": (15, 30),
    "VOL_MIN": (240, 720, 1440, 10080),
    "KELTNER_WIDTH_MIN": (30, 60),
    "KELTNER_CENTER_MIN": (60, 240, 720),
    "KELTNER_POS_MIN": (360, 2880),
    "KELTNER_Z_MIN": (),
    "ADX_MIN": (15, 30, 120),
    "CUM_LOGRET_MIN": (1440,),
}


def _parse_tuple_env(name: str) -> tuple[int, ...] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    out = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return tuple(out) if out else None


def _apply_crypto_windows() -> None:
    for key, default_val in CFG_CRYPTO_WINDOWS.items():
        env_val = _parse_tuple_env(f"PF_CRYPTO_{key}")
        val = env_val if env_val is not None else default_val
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(val))
        if hasattr(featmod, key):
            setattr(featmod, key, tuple(val))


def main() -> None:
    _apply_crypto_windows()
    sym = os.getenv("PF_CRYPTO_SYMBOL", DEFAULT_SYMBOL).strip().upper()
    days = int(os.getenv("PF_CRYPTO_DAYS", DEFAULT_DAYS) or DEFAULT_DAYS)
    tail = int(os.getenv("PF_CRYPTO_TAIL_DAYS", DEFAULT_REMOVE_TAIL_DAYS) or DEFAULT_REMOVE_TAIL_DAYS)
    candle_sec = int(os.getenv("PF_CRYPTO_CANDLE_SEC", DEFAULT_CANDLE_SEC) or DEFAULT_CANDLE_SEC)

    feat_list_raw = os.getenv("PF_CRYPTO_FEATURES", os.getenv("PF_CRYPTO_FEATURE", "")).strip()
    feats = [f.strip().lower() for f in feat_list_raw.replace(";", ",").split(",") if f.strip()]

    raw_1m = load_ohlc_1m_series(sym, int(days), remove_tail_days=int(tail))
    if raw_1m.empty:
        print("Sem dados retornados do MySQL.", flush=True)
        return
    df_ohlc = to_ohlc_from_1m(raw_1m, int(candle_sec))

    if feats:
        flags = {k: False for k in FLAGS_CRYPTO}
        flags.update({"label": False, "plot_candles": True})
        feats_on = []
        for ft in feats:
            if ft in flags:
                flags[ft] = True
                feats_on.append(ft)
        if not feats_on:
            flags["atr"] = True
            feats_on = ["atr"]
    else:
        flags = dict(FLAGS_CRYPTO)
        flags.update({"label": False, "plot_candles": True})
        feats_on = [k for k, v in flags.items() if v and k not in {"label", "plot_candles"}]
        if not feats_on:
            flags["atr"] = True
            feats_on = ["atr"]

    print(f"[crypto] símbolo={sym} dias={days} features_on={feats_on}", flush=True)

    df = run_from_flags_dict(
        df_ohlc,
        flags,
        plot=True,
        u_threshold=float(DEFAULT_U_THRESHOLD),
        grey_zone=DEFAULT_GREY_ZONE,
        show=True,
        trade_contract=CRYPTO_CONTRACT,
    )
    try:
        print(f"OK: rows={len(df):,} | cols={len(df.columns):,}".replace(",", "."), flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
