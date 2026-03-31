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

from train.wf_portfolio_explorer import ExploreSettings, run  # type: ignore


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v or default


def main() -> None:
    settings = ExploreSettings(
        out_root=_env_str("WF_EXPLORE_OUT_ROOT", "wf_portfolio_explore"),
        results_csv=_env_str("WF_EXPLORE_RESULTS_CSV", "explore_runs.csv"),
        seed=_env_int("WF_EXPLORE_SEED", 42),
        max_label_trials=_env_int("WF_EXPLORE_LABEL_TRIALS", 56),
        retrains_per_label=_env_int("WF_EXPLORE_RETRAINS_PER_LABEL", 2),
        backtests_per_retrain=_env_int("WF_EXPLORE_BACKTESTS_PER_RETRAIN", 21),
        days=_env_int("WF_EXPLORE_DAYS", 4 * 360),
        max_symbols=_env_int("WF_EXPLORE_MAX_SYMBOLS", 0),
        candle_sec=_env_int("WF_EXPLORE_CANDLE_SEC", 300),
        safe_threads=_env_int("WF_EXPLORE_SAFE_THREADS", 8),
        feature_preset=_env_str("WF_EXPLORE_FEATURE_PRESET", "core80"),
    )
    run(settings)


if __name__ == "__main__":
    main()
