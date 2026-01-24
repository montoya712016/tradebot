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


def main() -> None:
    symbol = os.getenv("STK_SINGLE_SYMBOL", "AAPL").strip().upper()
    days = _env_int("STK_SINGLE_DAYS", 180)
    total_days_cache = _env_int("STK_SINGLE_CACHE_DAYS", 0)
    run_dir = os.getenv("STK_SINGLE_RUN_DIR", "").strip() or None
    plot_out = os.getenv("STK_SINGLE_PLOT_OUT", "data/generated/plots/stocks_single_symbol.html")

    settings = SingleSymbolDemoSettings(
        asset_class="stocks",
        symbol=symbol,
        days=days,
        total_days_cache=total_days_cache,
        run_dir=run_dir,
        plot_out=plot_out,
        override_tau_entry=0.85,
    )
    run(settings)


if __name__ == "__main__":
    main()
