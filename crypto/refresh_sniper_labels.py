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


def main() -> None:
    _add_repo_paths()
    os.environ.setdefault("SNIPER_ASSET_CLASS", "crypto")
    from prepare_features.refresh_sniper_labels_in_cache import (  # type: ignore
        RefreshLabelsSettings,
        run as refresh_run,
    )
    settings = RefreshLabelsSettings(
        candle_sec=60,
        mcap_min_usd=50_000_000.0,
        label_clip=0.20,
        label_center=True,
        use_dominant=True,
        dominant_mix=0.50,
        side_mae_penalty=1.25,
        side_time_penalty=0.55,
        side_giveback_penalty=0.65,
        side_cross_penalty=0.35,
        side_rev_lookback_min=90,
        side_chase_penalty=0.90,
        side_reversal_bonus=0.55,
        side_confirm_min=20,
        side_confirm_move=0.0025,
        side_preconfirm_suppress=0.15,
    )
    refresh_run(settings)


if __name__ == "__main__":
    main()
