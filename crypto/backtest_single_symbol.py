# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys


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


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v.strip() or default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in {"0", "false", "no", "off"}


def _latest_wf_run_dir() -> str | None:
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        base_dir = _models_root_for_asset("crypto").resolve()
    except Exception:
        env_root = os.getenv("MODELS_SNIPER_ROOT", "").strip()
        root = Path(env_root) if env_root else Path("D:/astra/models_sniper")
        base_dir = root / "crypto"
    if not base_dir.exists():
        return None
    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("wf_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    min_periods = _env_int("BT_REQUIRE_MIN_PERIODS", 0)
    if min_periods > 0:
        for r in runs:
            try:
                n_periods = sum(1 for d in r.iterdir() if d.is_dir() and d.name.startswith("period_") and d.name.endswith("d"))
                if n_periods >= int(min_periods):
                    return str(r)
            except Exception:
                continue
    return str(runs[0])


def main() -> None:
    symbol = _env_str("BT_SYMBOL", "trumpusdt").upper()
    days = _env_int("BT_DAYS", 2*360)
    total_days_cache = _env_int("BT_TOTAL_DAYS_CACHE", 0)
    run_dir = _env_str("BT_RUN_DIR", "") or _latest_wf_run_dir()
    plot_out = _env_str("BT_PLOT_OUT", "data/generated/plots/crypto_single_symbol.html")
    plot_candles = _env_bool("BT_PLOT_CANDLES", True)
    disable_calib = _env_bool("BT_DISABLE_CALIB", False)
    long_only = _env_bool("BT_LONG_ONLY", True)
    use_exit_model = _env_bool("BT_USE_EXIT_MODEL", False)
    tau_entry = _env_float("BT_TAU_ENTRY", 0.70)
    exit_min_hold_bars = _env_int("BT_EXIT_MIN_HOLD_BARS", 0)
    exit_confirm_bars = _env_int("BT_EXIT_CONFIRM_BARS", 2)
    exit_span_center_smooth = _env_float("BT_EXIT_SPAN_CENTER_SMOOTH", 0.90)
    exit_span_window_pct = _env_float("BT_EXIT_SPAN_WINDOW_PCT", 0.20)
    exit_span_window_steps = _env_float("BT_EXIT_SPAN_WINDOW_STEPS", 2.0)
    exit_span_rate_limit_pct = _env_float("BT_EXIT_SPAN_RATE_LIMIT_PCT", 0.10)

    settings = SingleSymbolDemoSettings(
        asset_class="crypto",
        symbol=symbol,
        days=days,
        total_days_cache=total_days_cache,
        run_dir=run_dir,
        plot_out=plot_out,
        plot_candles=plot_candles,
        disable_entry_calibration=disable_calib,
        long_only=long_only,
        use_exit_model=use_exit_model,
        override_tau_entry=tau_entry,
        exit_min_hold_bars=exit_min_hold_bars,
        exit_confirm_bars=exit_confirm_bars,
        exit_span_center_smooth=exit_span_center_smooth,
        exit_span_window_pct=exit_span_window_pct,
        exit_span_window_steps=exit_span_window_steps,
        exit_span_rate_limit_pct=exit_span_rate_limit_pct,
    )
    try:
        rr = Path(str(run_dir)).resolve() if run_dir else None
        if rr is not None and rr.is_dir():
            n_periods = sum(1 for d in rr.iterdir() if d.is_dir() and d.name.startswith("period_") and d.name.endswith("d"))
            print(f"[backtest-single] run_dir={rr} periods={n_periods}", flush=True)
    except Exception:
        pass
    run(settings)


if __name__ == "__main__":
    main()
