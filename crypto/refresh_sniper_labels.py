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
    os.environ["PF_ENTRY_LABEL_NET_PROFIT_THR"] = "0.0"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_POS"] = "0.04"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_NEG"] = "0.03"
    os.environ["PF_ENTRY_WEIGHT_RET_DEADZONE"] = "0.002"
    os.environ["PF_ENTRY_WEIGHT_POS_GAIN"] = "6.0"
    os.environ["PF_ENTRY_WEIGHT_NEG_GAIN"] = "5.0"
    os.environ["PF_ENTRY_WEIGHT_POS_POWER"] = "2.6"
    os.environ["PF_ENTRY_WEIGHT_NEG_POWER"] = "2.2"
    os.environ["PF_ENTRY_WEIGHT_MODE"] = "ret_pct_abs"
    os.environ["PF_ENTRY_WEIGHT_PCT_POWER"] = "1.0"
    os.environ["SNIPER_ENTRY_WEIGHT_BIN_STEP_X10"] = "2"
    from trade_contract import TradeContract  # type: ignore
    from prepare_features.refresh_sniper_labels_in_cache import (  # type: ignore
        RefreshLabelsSettings,
        run as refresh_run,
    )

    contract = TradeContract(
        entry_label_windows_minutes=(240,),
        entry_label_min_profit_pcts=(0.02,),
        entry_label_weight_alpha=0.5,
        exit_ema_span=60,
        exit_ema_init_offset_pct=0.005,
    )
    settings = RefreshLabelsSettings(contract=contract, candle_sec=60)
    refresh_run(settings)


if __name__ == "__main__":
    main()
