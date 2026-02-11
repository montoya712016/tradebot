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

from backtest.single_symbol_rl import SingleSymbolRLDemoSettings, run  # type: ignore


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


def main() -> None:
    settings = SingleSymbolRLDemoSettings(
        symbol=_env_str("BT_SYMBOL", "STXUSDT"),
        days=_env_int("BT_DAYS", 365),
        signals_path=(_env_str("BT_RL_SIGNALS", "").strip() or None),
        run_dir=(_env_str("BT_RL_RUN_DIR", "").strip() or None),
        checkpoint=(_env_str("BT_RL_CHECKPOINT", "").strip() or None),
        fold_id=_env_int("BT_RL_FOLD_ID", -1),
        device=_env_str("BT_RL_DEVICE", "cuda"),
        plot_out=(_env_str("BT_RL_PLOT_OUT", "").strip() or None),
        timeline_out=(_env_str("BT_RL_TIMELINE_OUT", "").strip() or None),
        show_plot=_env_bool("BT_SHOW_PLOT", True),
        save_plot=_env_bool("BT_SAVE_PLOT", True),
        save_timeline=_env_bool("BT_SAVE_TIMELINE", True),
        plot_candles=_env_bool("BT_PLOT_CANDLES", True),
    )
    run(settings)


if __name__ == "__main__":
    main()
