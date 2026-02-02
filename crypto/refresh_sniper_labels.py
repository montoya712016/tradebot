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
    from trade_contract import TradeContract  # type: ignore
    from prepare_features.refresh_sniper_labels_in_cache import (  # type: ignore
        RefreshLabelsSettings,
        run as refresh_run,
    )

    contract = TradeContract(
        entry_label_windows_minutes=(60,),
        entry_label_min_profit_pcts=(0.03,),
        entry_label_max_dd_pcts=(0.03,),
        entry_label_weight_alpha=0.5,
        exit_ema_span=120,
        exit_ema_init_offset_pct=0.002,
    )
    settings = RefreshLabelsSettings(contract=contract, candle_sec=60)
    refresh_run(settings)


if __name__ == "__main__":
    main()
