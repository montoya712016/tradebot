# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Exporta sinais padronizados long/short para a camada RL.

Formato de saída (parquet):
- index: timestamp
- symbol
- close
- mu_long, mu_short, edge, strength, uncertainty
- mu_long_norm, mu_short_norm, edge_norm, strength_norm, uncertainty_norm
- vol_short, vol_long, trend_strength, shock_flag
- vol_short_norm, vol_long_norm, trend_strength_norm
- fwd_ret_1
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            repo_root = p.parent
            for cand in (repo_root, p):
                sp = str(cand)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward  # type: ignore
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore
from config.symbols import load_top_market_cap_symbols, default_top_market_cap_path  # type: ignore
from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract  # type: ignore


@dataclass
class SignalExportConfig:
    asset_class: str = "crypto"
    run_dir: str | None = None
    symbols: tuple[str, ...] = ()
    symbols_file: str | None = None
    symbols_limit: int = 0
    days: int = 365
    total_days_cache: int = 0
    out_path: str = "data/generated/hybrid/supervised_signals.parquet"
    norm_window: int = 2_880
    vol_short_bars: int = 60
    vol_long_bars: int = 720
    trend_bars: int = 360
    shock_k: float = 3.0


def _set_default_supervised_inference_env() -> None:
    # Mantem consistencia da distribuicao dos scores para camada RL.
    # Nao sobrescreve se o usuario ja definiu explicitamente no ambiente.
    os.environ.setdefault("SNIPER_APPLY_PRED_BIAS", "1")
    os.environ.setdefault("SNIPER_CLIP_ENTRY_PRED", "1")


def _find_latest_wf_dir(asset_class: str, run_dir: str | None = None) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"run_dir invalido: {p}")
        return p
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        models_root = _models_root_for_asset(asset_class).resolve()
    except Exception:
        models_root = (Path(__file__).resolve().parents[2].parent / "models_sniper" / asset_class).resolve()
    runs = [p for p in models_root.glob("wf_*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"Nenhum wf_* encontrado em {models_root}")
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]


def _default_flags_for_asset(asset_class: str) -> dict:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        try:
            from stocks import prepare_features_stocks as pfs  # type: ignore

            flags = dict(getattr(pfs, "FLAGS_STOCKS", {}))
            if flags:
                return flags
        except Exception:
            pass
    return dict(GLOBAL_FLAGS_FULL)


def _default_contract_for_asset(asset_class: str) -> TradeContract:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        try:
            from stocks.trade_contract import DEFAULT_TRADE_CONTRACT as STOCKS_CONTRACT  # type: ignore

            return STOCKS_CONTRACT
        except Exception:
            return DEFAULT_TRADE_CONTRACT
    return DEFAULT_TRADE_CONTRACT


def _parse_symbols_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"symbols_file nao encontrado: {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            s = s.split(":", 1)[0].strip()
        out.append(s.upper())
    return out


def _resolve_symbols(cfg: SignalExportConfig) -> list[str]:
    if cfg.symbols:
        syms = [str(s).strip().upper() for s in cfg.symbols if str(s).strip()]
    elif cfg.symbols_file:
        syms = _parse_symbols_file(cfg.symbols_file)
    else:
        fallback = str(default_top_market_cap_path())
        syms = load_top_market_cap_symbols(path=fallback, limit=None, ensure_usdt=(cfg.asset_class == "crypto"))
    if cfg.asset_class == "crypto":
        syms = [s if s.endswith("USDT") else f"{s}USDT" for s in syms]
    if cfg.symbols_limit and cfg.symbols_limit > 0:
        syms = syms[: int(cfg.symbols_limit)]
    if not syms:
        raise RuntimeError("Lista de simbolos vazia")
    return sorted(set(syms))


def _best_from_map(pred_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not pred_map:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    names = sorted(pred_map.keys())
    mats = []
    wins = []
    for name in names:
        arr = np.asarray(pred_map[name], dtype=np.float32)
        mats.append(arr.reshape(-1, 1))
        try:
            w = int(re.sub(r"\D", "", str(name)) or 0)
        except Exception:
            w = 0
        wins.append(np.full_like(arr, float(w), dtype=np.float32).reshape(-1, 1))
    m = np.concatenate(mats, axis=1)
    w = np.concatenate(wins, axis=1)
    safe_m = np.where(np.isfinite(m), m, -np.inf)
    best_idx = np.argmax(safe_m, axis=1)
    rows = np.arange(m.shape[0], dtype=np.int64)
    best = safe_m[rows, best_idx].astype(np.float32, copy=False)
    best[~np.isfinite(best)] = 0.0
    best_w = w[rows, best_idx].astype(np.float32, copy=False)
    return best, best_w


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    w = int(max(32, window))
    m = x.rolling(w, min_periods=max(16, w // 8)).mean()
    s = x.rolling(w, min_periods=max(16, w // 8)).std()
    z = (x - m) / s.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _regime_features(df: pd.DataFrame, cfg: SignalExportConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(np.float64)
    ret_1 = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_short = ret_1.rolling(int(max(5, cfg.vol_short_bars)), min_periods=5).std().fillna(0.0)
    vol_long = ret_1.rolling(int(max(20, cfg.vol_long_bars)), min_periods=20).std().fillna(0.0)
    trend_raw = close.pct_change(int(max(5, cfg.trend_bars))).fillna(0.0)
    trend_strength = trend_raw / (vol_long * np.sqrt(float(max(1, cfg.trend_bars))) + 1e-9)
    shock_flag = (ret_1.abs() > (float(cfg.shock_k) * vol_short.replace(0.0, np.nan))).astype(np.float32).fillna(0.0)

    out["vol_short"] = vol_short.astype(np.float32)
    out["vol_long"] = vol_long.astype(np.float32)
    out["trend_strength"] = trend_strength.astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["shock_flag"] = shock_flag.astype(np.float32)
    return out


def _normalize_signal_features(df: pd.DataFrame, cfg: SignalExportConfig) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "mu_long",
        "mu_short",
        "edge",
        "strength",
        "uncertainty",
        "vol_short",
        "vol_long",
        "trend_strength",
    ):
        out[f"{col}_norm"] = _rolling_z(out[col].astype(np.float64), cfg.norm_window).astype(np.float32)
    return out


def _build_symbol_signals(
    df: pd.DataFrame,
    *,
    periods,
    cfg: SignalExportConfig,
) -> pd.DataFrame:
    p_entry_long_map, p_entry_short_map, _p_danger, _p_exit, _used, period_id = predict_scores_walkforward(
        df,
        periods=periods,
        return_period_id=True,
    )
    mu_long, best_win_long = _best_from_map(dict(p_entry_long_map))
    mu_short, best_win_short = _best_from_map(dict(p_entry_short_map if p_entry_short_map else p_entry_long_map))

    df_out = df[["close"]].copy()
    df_out["mu_long"] = np.asarray(mu_long, dtype=np.float32)
    df_out["mu_short"] = np.asarray(mu_short, dtype=np.float32)
    df_out["edge"] = (df_out["mu_long"] - df_out["mu_short"]).astype(np.float32)
    df_out["strength"] = np.maximum(df_out["mu_long"].to_numpy(), df_out["mu_short"].to_numpy()).astype(np.float32)
    df_out["best_win_long"] = np.asarray(best_win_long, dtype=np.float32)
    df_out["best_win_short"] = np.asarray(best_win_short, dtype=np.float32)
    df_out["period_id"] = np.asarray(period_id, dtype=np.int16)

    mats_long = np.column_stack([np.asarray(v, dtype=np.float32) for v in dict(p_entry_long_map).values()]) if p_entry_long_map else None
    mats_short = np.column_stack([np.asarray(v, dtype=np.float32) for v in dict(p_entry_short_map).values()]) if p_entry_short_map else None
    std_long = np.nanstd(mats_long, axis=1).astype(np.float32) if mats_long is not None and mats_long.shape[1] > 1 else np.zeros(len(df_out), dtype=np.float32)
    std_short = np.nanstd(mats_short, axis=1).astype(np.float32) if mats_short is not None and mats_short.shape[1] > 1 else np.zeros(len(df_out), dtype=np.float32)
    df_out["uncertainty"] = (0.5 * (std_long + std_short)).astype(np.float32)

    df_out = _regime_features(df_out, cfg)
    df_out = _normalize_signal_features(df_out, cfg)

    close = df_out["close"].astype(np.float64)
    df_out["fwd_ret_1"] = ((close.shift(-1) / close) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return df_out


def export_long_short_signals(cfg: SignalExportConfig) -> Path:
    _set_default_supervised_inference_env()
    asset = str(cfg.asset_class or "crypto").lower()
    run_dir = _find_latest_wf_dir(asset, cfg.run_dir)
    periods = load_period_models(run_dir)
    symbols = _resolve_symbols(cfg)
    contract = _default_contract_for_asset(asset)
    flags = _default_flags_for_asset(asset)
    flags["_quiet"] = True

    cache_map = ensure_feature_cache(
        symbols,
        total_days=int(cfg.total_days_cache),
        contract=contract,
        flags=flags,
        asset_class=asset,
    )
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        p = cache_map.get(sym)
        if p is None:
            continue
        if str(p).lower().endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_pickle(p)
        if df.empty:
            continue
        if int(cfg.days) > 0:
            end_ts = pd.to_datetime(df.index.max())
            start_ts = end_ts - pd.Timedelta(days=int(cfg.days))
            df = df.loc[pd.to_datetime(df.index) >= start_ts].copy()
        if len(df) < 1_000:
            continue
        sig = _build_symbol_signals(df, periods=periods, cfg=cfg)
        sig["symbol"] = sym
        frames.append(sig)

    if not frames:
        raise RuntimeError("Nenhum sinal gerado (frames vazio)")
    out = pd.concat(frames, axis=0).sort_index()
    out_path = Path(cfg.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=True)
    return out_path


def _parse_args(argv: Iterable[str] | None = None) -> SignalExportConfig:
    ap = argparse.ArgumentParser(description="Exporta sinais supervisionados long/short para RL")
    ap.add_argument("--asset-class", default="crypto", choices=["crypto", "stocks"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--symbols-limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--total-days-cache", type=int, default=0)
    ap.add_argument("--out", default="data/generated/hybrid/supervised_signals.parquet")
    ap.add_argument("--norm-window", type=int, default=2880)
    ap.add_argument("--vol-short-bars", type=int, default=60)
    ap.add_argument("--vol-long-bars", type=int, default=720)
    ap.add_argument("--trend-bars", type=int, default=360)
    ap.add_argument("--shock-k", type=float, default=3.0)
    ns = ap.parse_args(list(argv) if argv is not None else None)
    return SignalExportConfig(
        asset_class=str(ns.asset_class),
        run_dir=ns.run_dir,
        symbols=tuple(ns.symbols or ()),
        symbols_file=ns.symbols_file,
        symbols_limit=int(ns.symbols_limit),
        days=int(ns.days),
        total_days_cache=int(ns.total_days_cache),
        out_path=str(ns.out),
        norm_window=int(ns.norm_window),
        vol_short_bars=int(ns.vol_short_bars),
        vol_long_bars=int(ns.vol_long_bars),
        trend_bars=int(ns.trend_bars),
        shock_k=float(ns.shock_k),
    )


def main(argv: Iterable[str] | None = None) -> None:
    cfg = _parse_args(argv)
    out = export_long_short_signals(cfg)
    print(f"[supervised] sinais exportados: {out}")


if __name__ == "__main__":
    main()
