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

from train.train_sniper_wf import TrainSniperWFSettings, run  # type: ignore
from crypto.trade_contract import TradeContract  # type: ignore


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v.strip() or default


def main() -> None:
    metric_mode = _env_str("CRYPTO_ENTRY_METRIC_MODE", "aucpr")
    contract = TradeContract(
        entry_label_windows_minutes=(120,),
        entry_label_min_profit_pcts=(0.032,),
        entry_label_weight_alpha=0.5,
        exit_ema_span=120,
        exit_ema_init_offset_pct=0.002,
    )
    settings = TrainSniperWFSettings(
        asset_class="crypto",
        entry_metric_mode=metric_mode,
        contract=contract,
        offsets_step_days=180,
        offsets_days=(0, 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160),
        entry_ratio_neg_per_pos=6.0,
        max_rows_entry=2_000_000,
        min_symbols_used_per_period=30,
    )
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
