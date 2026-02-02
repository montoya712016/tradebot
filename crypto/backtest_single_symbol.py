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


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in {"0", "false", "no", "off"}


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
    symbol = "ADAUSDT"
    days = 360
    total_days_cache = 0
    run_dir = _latest_wf_run_dir() or r"D:\astra\models_sniper\crypto\wf_024"
    plot_out = "data/generated/plots/crypto_single_symbol.html"
    # True = velas, False = linha de close
    plot_candles = _env_bool("BT_PLOT_CANDLES", default=False)

    settings = SingleSymbolDemoSettings(
        asset_class="crypto",
        symbol=symbol,
        days=days,
        total_days_cache=total_days_cache,
        run_dir=run_dir,
        plot_out=plot_out,
        plot_candles=plot_candles,
        override_tau_entry=0.85,
    )
    run(settings)


if __name__ == "__main__":
    main()
