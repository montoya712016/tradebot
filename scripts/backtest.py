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

from backtest.portfolio import PortfolioDemoSettings, run, _default_portfolio_cfg  # type: ignore
from config.trade_contract import (  # type: ignore
    CRYPTO_PIPELINE_CANDLE_SEC,
    apply_crypto_pipeline_env,
    build_default_crypto_contract,
)
from backtest.portfolio_symbol_filters import DEFAULT_PORTFOLIO_BAD_SYMBOLS  # type: ignore


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


def _env_symbols(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        sym = part.strip().upper()
        if not sym:
            continue
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        out.append(sym)
    return out


def _env_optional_int(name: str) -> int | None:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _env_optional_float(name: str) -> float | None:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _latest_wf_run_dir() -> str | None:
    try:
        from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

        base_dir = _models_root_for_asset("crypto").resolve()
    except Exception:
        base_dir = Path("D:/astra/models_sniper/crypto").resolve()
    if not base_dir.exists():
        return None
    runs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("wf_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(runs[0])


def main() -> None:
    candle_sec_default = apply_crypto_pipeline_env(CRYPTO_PIPELINE_CANDLE_SEC)
    days = _env_int("BT_DAYS", 4 * 360)
    total_days_cache = _env_int("BT_TOTAL_DAYS_CACHE", 0)
    run_dir = _env_str("BT_RUN_DIR", "") or _latest_wf_run_dir()
    plot_out = _env_str("BT_PLOT_OUT", "data/generated/plots/crypto_portfolio_equity.html")
    save_plot = _env_bool("BT_SAVE_PLOT", True)
    candle_sec = _env_int("BT_CANDLE_SEC", candle_sec_default)
    tau_entry = _env_float("BT_TAU_ENTRY", 0.61)
    max_symbols = _env_int("BT_MAX_SYMBOLS", 200)
    align_global_window = _env_bool("BT_ALIGN_GLOBAL_WINDOW", True)
    require_feature_cache = _env_bool("BT_REQUIRE_FEATURE_CACHE", False)
    rebuild_on_score_error = _env_bool("BT_REBUILD_ON_SCORE_ERROR", True)
    symbols = _env_symbols("BT_SYMBOLS")
    exclude_symbols = _env_symbols("BT_EXCLUDE_SYMBOLS")
    if _env_bool("BT_USE_DEFAULT_SYMBOL_FILTER", False):
        exclude_symbols = sorted(set(exclude_symbols).union(DEFAULT_PORTFOLIO_BAD_SYMBOLS))
    contract = build_default_crypto_contract(candle_sec)
    cfg = _default_portfolio_cfg()
    # Alinhamento exato com label_017 do Explore:
    # Vencedores costumam ter seletividade via tau, mas filtros de corr desabilitados
    # ou configurados manualmente de forma permissiva.
    cfg.max_positions = 18
    cfg.total_exposure = 0.53
    cfg.max_trade_exposure = 0.15
    cfg.min_trade_exposure = 0.05
    
    if _env_bool("BT_DISABLE_CORRELATION", True):
        cfg.corr_filter_enabled = False
        cfg.corr_open_filter_enabled = False
    v = _env_optional_float("BT_CORR_MAX_WITH_MARKET")
    if v is not None:
        cfg.corr_max_with_market = float(v)
    v = _env_optional_float("BT_CORR_MAX_PAIR")
    if v is not None:
        cfg.corr_max_pair = float(v)
    vi = _env_optional_int("BT_CORR_KEEP_TOP_N")
    if vi is not None:
        cfg.corr_keep_top_n = int(vi)
    v = _env_optional_float("BT_CORR_OPEN_REDUCE_START")
    if v is not None:
        cfg.corr_open_reduce_start = float(v)
    v = _env_optional_float("BT_CORR_OPEN_HARD_REJECT")
    if v is not None:
        cfg.corr_open_hard_reject = float(v)
    v = _env_optional_float("BT_CORR_OPEN_MIN_WEIGHT_MULT")
    if v is not None:
        cfg.corr_open_min_weight_mult = float(v)
    v = _env_optional_float("BT_EXPOSURE_MULTIPLIER")
    if v is not None:
        cfg.exposure_multiplier = float(v)

    settings = PortfolioDemoSettings(
        asset_class="crypto",
        run_dir=run_dir,
        days=days,
        max_symbols=max_symbols,
        total_days_cache=total_days_cache,
        symbols=symbols,
        exclude_symbols=exclude_symbols,
        cfg=cfg,
        save_plot=save_plot,
        plot_out=plot_out,
        override_tau_entry=tau_entry,
        candle_sec=candle_sec,
        contract=contract,
        long_only=True,
        require_feature_cache=require_feature_cache,
        rebuild_on_score_error=rebuild_on_score_error,
        align_global_window=align_global_window,
    )
    try:
        rr = Path(str(run_dir)).resolve() if run_dir else None
        if rr is not None and rr.is_dir():
            n_periods = sum(1 for d in rr.iterdir() if d.is_dir() and d.name.startswith("period_") and d.name.endswith("d"))
            print(f"[backtest-portfolio] run_dir={rr} periods={n_periods}", flush=True)
            print(
                f"[backtest-portfolio] tau={tau_entry:.2f} max_symbols={max_symbols} "
                f"corr_batch={cfg.corr_filter_enabled} corr_open={cfg.corr_open_filter_enabled} "
                f"strict_cache={int(require_feature_cache)} rebuild_on_score_error={int(rebuild_on_score_error)} "
                f"exclude_symbols={len(exclude_symbols)}",
                flush=True,
            )
    except Exception:
        pass
    run(settings)


if __name__ == "__main__":
    main()
