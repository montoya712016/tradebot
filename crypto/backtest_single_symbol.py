# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys
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
from backtest.sniper_walkforward import load_period_models, predict_scores_walkforward  # type: ignore
from train.sniper_dataflow import ensure_feature_cache, GLOBAL_FLAGS_FULL  # type: ignore
from backtest.single_symbol import _default_contract_for_asset  # type: ignore


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


def main() -> None:
    os.environ.setdefault("SNIPER_APPLY_PRED_BIAS", "1")
    symbol = "STXUSDT"
    days = 720
    total_days_cache = 0
    run_dir = _latest_wf_run_dir()
    plot_out = "data/generated/plots/crypto_single_symbol.html"
    # True = velas, False = linha de close
    plot_candles = _env_bool("BT_PLOT_CANDLES", default=False)
    # BT_MODE=backtest (padrao) | BT_MODE=pred_only
    bt_mode = _env_str("BT_MODE", "pred_only").strip().lower()
    run_backtest = bt_mode not in {"pred_only", "pred", "scores_only", "scores"}

    settings = SingleSymbolDemoSettings(
        asset_class="crypto",
        symbol=symbol,
        days=days,
        total_days_cache=total_days_cache,
        run_dir=run_dir,
        plot_out=plot_out,
        plot_candles=plot_candles,
        run_backtest=run_backtest,
        override_tau_entry=0.05,
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
                p_entry_long_map, p_entry_short_map, _, _, _ = predict_scores_walkforward(df, periods=periods, return_period_id=True)
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
