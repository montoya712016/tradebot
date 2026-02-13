# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_add_repo_paths()

from backtest.single_symbol import SingleSymbolDemoSettings, run  # type: ignore
from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward, select_entry_mid  # type: ignore
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore
from backtest.single_symbol import _default_contract_for_asset  # type: ignore
from plotting.plotting import plot_backtest_single  # type: ignore


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else str(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    try:
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _latest_wf_run_dir() -> str | None:
    env_root = os.getenv("MODELS_SNIPER_ROOT", "").strip()
    root = Path(env_root) if env_root else Path("D:/astra/models_sniper")
    base_dir = root / "crypto"
    if not base_dir.exists():
        return None
    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("wf_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(runs[0])


def _simulate_threshold_heuristic(
    close: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    tau: float,
    fee_side: float,
    slippage: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(close))
    if n <= 1:
        return np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.int8)
    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = (close[1:] / np.maximum(close[:-1], 1e-12)) - 1.0
    # Unifica os dois regressores em um score direcional:
    # U = long - short_eff
    # Se short vier negativo (escala assinada), usa módulo para não inflar U por subtração de negativo.
    short_eff = np.where(p_short < 0.0, np.abs(p_short), p_short)
    u_score = p_long - short_eff
    desired = np.zeros(n, dtype=np.int8)
    desired[u_score >= float(tau)] = 1
    desired[u_score <= -float(tau)] = -1
    exposure = desired.astype(np.int8, copy=False)
    costs = np.zeros(n, dtype=np.float64)
    turn = np.abs(exposure[1:].astype(np.int16) - exposure[:-1].astype(np.int16)).astype(np.float64, copy=False)
    costs[1:] = turn * float(fee_side + slippage)
    strat_ret = np.zeros(n, dtype=np.float64)
    strat_ret[1:] = exposure[:-1].astype(np.float64, copy=False) * ret[1:] - costs[1:]
    equity = np.cumprod(1.0 + strat_ret)
    return equity.astype(np.float64, copy=False), exposure


def _trades_from_exposure(index: pd.Index, exposure: np.ndarray):
    trades = []
    exp = np.asarray(exposure, dtype=np.int8)
    if exp.size == 0:
        return trades
    cur = int(exp[0])
    start = 0
    for i in range(1, int(exp.size)):
        v = int(exp[i])
        if v == cur:
            continue
        if cur != 0:
            side = "long" if cur > 0 else "short"
            trades.append(SimpleNamespace(entry_ts=index[start], exit_ts=index[i], side=side))
        cur = v
        start = i
    if cur != 0 and exp.size > 1:
        side = "long" if cur > 0 else "short"
        trades.append(SimpleNamespace(entry_ts=index[start], exit_ts=index[-1], side=side))
    return trades


def _auto_tau(u_score: np.ndarray) -> float:
    arr = np.asarray(u_score, dtype=np.float64)
    if arr.size == 0:
        return 5.0
    try:
        p99 = float(np.nanquantile(np.abs(arr), 0.99))
    except Exception:
        p99 = 0.0
    if not np.isfinite(p99):
        p99 = 0.0
    # regressor antigo costuma operar em escala mais alta; classificador puro tende a 0..1.
    return 5.0 if p99 > 5.0 else 0.35


def _run_heuristic_mode(symbol: str, days: int, total_days_cache: int, run_dir: str | None, plot_out: str, plot_candles: bool) -> None:
    run_path = Path(run_dir) if run_dir else None
    if run_path is None or (not run_path.exists()):
        raise RuntimeError("run_dir WF nao encontrado para modo heuristic")
    periods = load_period_models(run_path)
    contract = _default_contract_for_asset("crypto")
    flags = dict(GLOBAL_FLAGS_FULL)
    flags["_quiet"] = True
    cache = ensure_feature_cache(
        [symbol],
        total_days=int(total_days_cache),
        contract=contract,
        flags=flags,
        asset_class="crypto",
    )
    df = pd.read_parquet(cache[symbol])
    end_ts = df.index.max()
    start_ts = end_ts - pd.Timedelta(days=int(days))
    df = df[df.index >= start_ts].copy()
    pred_out = predict_scores_walkforward(df, periods=periods, return_period_id=True, return_cls_maps=True)
    if not isinstance(pred_out, tuple) or len(pred_out) < 2:
        raise RuntimeError("assinatura inesperada de predict_scores_walkforward")
    p_entry_long_map = pred_out[0]
    p_entry_short_map = pred_out[1]
    p_entry_cls_long_map = pred_out[6] if len(pred_out) >= 8 else {}
    p_entry_cls_short_map = pred_out[7] if len(pred_out) >= 8 else {}
    p_long = np.asarray(select_entry_mid(dict(p_entry_long_map)), dtype=np.float64)
    p_short = np.asarray(select_entry_mid(dict(p_entry_short_map)) if p_entry_short_map else np.zeros_like(p_long), dtype=np.float64)
    p_gate_long = np.asarray(select_entry_mid(dict(p_entry_cls_long_map)) if p_entry_cls_long_map else np.ones_like(p_long), dtype=np.float64)
    p_gate_short = np.asarray(select_entry_mid(dict(p_entry_cls_short_map)) if p_entry_cls_short_map else np.ones_like(p_short), dtype=np.float64)
    p_gate_long = np.clip(np.nan_to_num(p_gate_long, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    p_gate_short = np.clip(np.nan_to_num(p_gate_short, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    p_long_eff = p_long * p_gate_long
    p_short_eff = p_short * p_gate_short
    short_eff = np.where(p_short_eff < 0.0, np.abs(p_short_eff), p_short_eff)
    u_score = p_long_eff - short_eff
    tau = _auto_tau(u_score)
    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    equity, exposure = _simulate_threshold_heuristic(
        close,
        p_long_eff,
        p_short_eff,
        tau=tau,
        fee_side=float(getattr(contract, "fee_pct_per_side", 0.0005)),
        slippage=float(getattr(contract, "slippage_pct", 0.0001)),
    )
    eq_end = float(equity[-1]) if equity.size else 1.0
    ret_total = eq_end - 1.0
    dd = 1.0 - (equity / np.maximum.accumulate(equity))
    max_dd = float(np.nanmax(dd)) if dd.size else 0.0
    flips = int(np.sum(np.abs(np.diff(exposure.astype(np.int16, copy=False))) > 0))
    print(
        f"SINGLE-HEUR sym={symbol} days={days} tau={tau:.3f} "
        f"eq={eq_end:.4f} ret={ret_total:+.2%} max_dd={max_dd:.2%} flips={flips}",
        flush=True,
    )
    entry_sig_long = u_score >= tau
    entry_sig_short = u_score <= -tau
    entry_ok = entry_sig_long | entry_sig_short
    trades = _trades_from_exposure(df.index, exposure)
    # Plot principal passa a ser o score combinado U (decisão real do heurístico).
    # Mantemos long/short em mapas auxiliares para inspeção, sem usá-los como gatilho.
    nan_short = np.full_like(u_score, np.nan, dtype=np.float64)
    plot_backtest_single(
        df,
        trades=trades,
        equity=equity,
        p_entry=u_score,
        p_entry_short=nan_short,
        p_entry_map={
            "u_score": np.asarray(u_score, dtype=np.float64),
            "reg_long": np.asarray(p_long, dtype=np.float64),
            "reg_short": np.asarray(p_short, dtype=np.float64),
            "gate_long": np.asarray(p_gate_long, dtype=np.float64),
            "gate_short": np.asarray(p_gate_short, dtype=np.float64),
            "score_long": np.asarray(p_long_eff, dtype=np.float64),
            "score_short": np.asarray(p_short_eff, dtype=np.float64),
        },
        p_entry_long_map=None,
        p_entry_short_map=None,
        p_danger=np.zeros(len(df), dtype=np.float64),
        entry_sig=entry_ok,
        entry_sig_long=entry_sig_long,
        entry_sig_short=entry_sig_short,
        danger_sig=np.zeros(len(df), dtype=bool),
        tau_entry=tau,
        tau_entry_long=tau,
        tau_entry_short=tau,
        tau_danger=1.0,
        title=f"{symbol} | heuristic U=long-short tau={tau:.3f} | ret={ret_total:+.2%} | flips={flips}",
        save_path=plot_out,
        show=True,
        ema_exit=None,
        plot_probs=True,
        plot_signals=False,
        plot_candles=bool(plot_candles),
        probs_simple=False,
    )

def _run_prediction_only_mode(symbol: str, days: int, total_days_cache: int, run_dir: str | None, plot_out: str, plot_candles: bool) -> None:
    run_path = Path(run_dir) if run_dir else None
    if run_path is None or (not run_path.exists()):
        raise RuntimeError("run_dir WF nao encontrado para modo prediction_only")
    periods = load_period_models(run_path)
    contract = _default_contract_for_asset("crypto")
    flags = dict(GLOBAL_FLAGS_FULL)
    flags["_quiet"] = True
    cache = ensure_feature_cache(
        [symbol],
        total_days=int(total_days_cache),
        contract=contract,
        flags=flags,
        asset_class="crypto",
    )
    df = pd.read_parquet(cache[symbol])
    end_ts = df.index.max()
    start_ts = end_ts - pd.Timedelta(days=int(days))
    df = df[df.index >= start_ts].copy()

    pred_out = predict_scores_walkforward(df, periods=periods, return_period_id=True, return_cls_maps=True)
    if not isinstance(pred_out, tuple) or len(pred_out) < 2:
        raise RuntimeError("assinatura inesperada de predict_scores_walkforward")
    p_entry_long_map = pred_out[0]
    p_entry_short_map = pred_out[1]
    p_entry_cls_long_map = pred_out[6] if len(pred_out) >= 8 else {}
    p_entry_cls_short_map = pred_out[7] if len(pred_out) >= 8 else {}

    p_long = np.asarray(select_entry_mid(dict(p_entry_long_map)), dtype=np.float64)
    p_short = np.asarray(select_entry_mid(dict(p_entry_short_map)) if p_entry_short_map else np.zeros_like(p_long), dtype=np.float64)
    p_gate_long = np.asarray(select_entry_mid(dict(p_entry_cls_long_map)) if p_entry_cls_long_map else np.ones_like(p_long), dtype=np.float64)
    p_gate_short = np.asarray(select_entry_mid(dict(p_entry_cls_short_map)) if p_entry_cls_short_map else np.ones_like(p_short), dtype=np.float64)
    p_gate_long = np.clip(np.nan_to_num(p_gate_long, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    p_gate_short = np.clip(np.nan_to_num(p_gate_short, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    p_long_eff = p_long * p_gate_long
    p_short_eff = p_short * p_gate_short
    short_eff = np.where(p_short_eff < 0.0, np.abs(p_short_eff), p_short_eff)
    u_score = p_long_eff - short_eff

    n = len(df)
    equity = np.ones(n, dtype=np.float64)
    entry_sig_long = np.zeros(n, dtype=bool)
    entry_sig_short = np.zeros(n, dtype=bool)
    entry_ok = np.zeros(n, dtype=bool)
    tau = _auto_tau(u_score)
    nan_short = np.full_like(u_score, np.nan, dtype=np.float64)
    print(f"SINGLE-PRED sym={symbol} days={days} rows={n}", flush=True)
    plot_backtest_single(
        df,
        trades=[],
        equity=equity,
        p_entry=u_score,
        p_entry_short=nan_short,
        p_entry_map={
            "u_score": np.asarray(u_score, dtype=np.float64),
            "reg_long": np.asarray(p_long, dtype=np.float64),
            "reg_short": np.asarray(p_short, dtype=np.float64),
            "gate_long": np.asarray(p_gate_long, dtype=np.float64),
            "gate_short": np.asarray(p_gate_short, dtype=np.float64),
            "score_long": np.asarray(p_long_eff, dtype=np.float64),
            "score_short": np.asarray(p_short_eff, dtype=np.float64),
        },
        p_entry_long_map=None,
        p_entry_short_map=None,
        p_danger=np.zeros(n, dtype=np.float64),
        entry_sig=entry_ok,
        entry_sig_long=entry_sig_long,
        entry_sig_short=entry_sig_short,
        danger_sig=np.zeros(n, dtype=bool),
        tau_entry=tau,
        tau_entry_long=tau,
        tau_entry_short=tau,
        tau_danger=1.0,
        title=f"{symbol} | prediction_only | rows={n}",
        save_path=plot_out,
        show=True,
        ema_exit=None,
        plot_probs=True,
        plot_signals=False,
        plot_candles=bool(plot_candles),
        probs_simple=False,
    )


def main() -> None:
    os.environ.setdefault("SNIPER_APPLY_PRED_BIAS", "1")
    symbol = _env_str("BT_SYMBOL", "DOGEUSDT")
    days = _env_int("BT_DAYS", 180)
    total_days_cache = 0
    run_dir = _env_str("BT_RUN_DIR", "") or _latest_wf_run_dir()
    plot_out = _env_str("BT_PLOT_OUT", "data/generated/plots/crypto_single_symbol.html")
    # True = velas, False = linha de close
    plot_candles = _env_bool("BT_PLOT_CANDLES", default=False)
    # BT_MODE=prediction_only (padrao) | heuristic | backtest | pred_only
    bt_mode = _env_str("BT_MODE", "backtest").strip().lower()
    tau_entry = _env_float("BT_TAU_ENTRY", 0.70)
    run_backtest = bt_mode not in {"pred_only", "pred", "scores_only", "scores"}
    print(f"[bt] mode={bt_mode} symbol={symbol} days={days}", flush=True)
    if bt_mode in {"prediction_only", "prediction", "pred_only", "pred", "scores_only", "scores"}:
        _run_prediction_only_mode(symbol, days, max(total_days_cache, days + 30), run_dir, plot_out, plot_candles)
        return
    if bt_mode in {"heuristic", "heur", "simple"}:
        _run_heuristic_mode(symbol, days, max(total_days_cache, days + 30), run_dir, plot_out, plot_candles)
        return

    settings = SingleSymbolDemoSettings(
        asset_class="crypto",
        symbol=symbol,
        days=days,
        total_days_cache=total_days_cache,
        run_dir=run_dir,
        plot_out=plot_out,
        plot_candles=plot_candles,
        run_backtest=run_backtest,
        override_tau_entry=float(tau_entry),
    )
    # Diagnóstico rápido de distribuição das previsões (percentis 5%..95%)
    if _env_bool("BT_PRINT_PCT", default=True):
        try:
            run_path = Path(run_dir) if run_dir else None
            if run_path is not None and run_path.exists():
                periods = load_period_models(run_path)
                flags = dict(GLOBAL_FLAGS_FULL)
                flags["_quiet"] = True
                cache = ensure_feature_cache(
                    [symbol],
                    total_days=0,
                    contract=_default_contract_for_asset("crypto"),
                    flags=flags,
                    asset_class="crypto",
                )
                df = pd.read_parquet(cache[symbol])
                end_ts = df.index.max()
                start_ts = end_ts - pd.Timedelta(days=days)
                df = df[df.index >= start_ts].copy()
                pred_out = predict_scores_walkforward(df, periods=periods, return_period_id=True)
                if not isinstance(pred_out, tuple) or len(pred_out) < 2:
                    raise RuntimeError("assinatura inesperada de predict_scores_walkforward")
                p_entry_long_map = pred_out[0]
                p_entry_short_map = pred_out[1]
                arr = np.asarray(next(iter(p_entry_long_map.values())), dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    pct = np.arange(0, 101, 5)
                    q = np.percentile(arr, pct)
                    print("[bt-diag] pred pct:", " ".join([f"p{int(p):02d}={v:+.5f}" for p, v in zip(pct, q)]), flush=True)
                    print(
                        f"[bt-diag] mean={arr.mean():+.5f} std={arr.std():.5f} min={arr.min():+.5f} max={arr.max():+.5f} "
                        f"pos={float(np.mean(arr > 0)):.2%} neg={float(np.mean(arr < 0)):.2%}",
                        flush=True,
                    )
        except Exception as e:
            print(f"[bt-diag] falhou: {type(e).__name__}: {e}", flush=True)
    run(settings)


if __name__ == "__main__":
    main()
