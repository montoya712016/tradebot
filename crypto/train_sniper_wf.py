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
from crypto.build_ohlc_cache import run_build as run_ohlc_build  # type: ignore
from prepare_features.refresh_sniper_labels_in_cache import (  # type: ignore
    RefreshLabelsSettings,
    run as run_labels_refresh,
)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v.strip() or default


def main() -> None:
    max_symbols = 0
    total_days = 0
    mcap_min_usd = 100_000_000.0
    mcap_max_usd = 150_000_000_000.0
    refresh_labels_before_train = True

    # Sem flag: sempre prewarm de OHLCV antes do treino.
    pre = run_ohlc_build(
        max_symbols=int(max_symbols),
        days=int(total_days),
        mcap_min_usd=float(mcap_min_usd),
        mcap_max_usd=float(mcap_max_usd),
        refresh=False,
    )
    print(
        f"[train-wf-crypto] ohlc prewarm: ok={pre.get('ok')} skip={pre.get('skip')} fail={pre.get('fail')} sec={pre.get('seconds')}",
        flush=True,
    )

    metric_mode = _env_str("CRYPTO_ENTRY_METRIC_MODE", "aucpr")
    contract = TradeContract(
        exit_ema_span=120,
        exit_ema_init_offset_pct=0.002,
    )
    if bool(refresh_labels_before_train):
        os.environ["SNIPER_ASSET_CLASS"] = "crypto"
        ref = run_labels_refresh(
            RefreshLabelsSettings(
                contract=contract,
                candle_sec=60,
                mcap_min_usd=float(mcap_min_usd),
                mcap_max_usd=float(mcap_max_usd),
                max_symbols=int(max_symbols),
                verbose=True,
            )
        )
        print(
            f"[train-wf-crypto] labels refresh: ok={ref.get('ok')} fail={ref.get('fail')} total={ref.get('total')} sec={ref.get('seconds')}",
            flush=True,
        )

    settings = TrainSniperWFSettings(
        asset_class="crypto",
        entry_metric_mode=metric_mode,
        contract=contract,
        total_days=int(total_days),
        max_symbols=int(max_symbols),
        offsets_step_days=180,
        offsets_days=(0, 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160),
        entry_ratio_neg_per_pos=6.0,
        max_rows_entry=3_000_000,
        use_full_entry_pool=True,
        # SemÃ¢ntica atual: valor por lado (long e short). 5M por lado ~= 10M total.
        full_pool_max_rows_entry=5_000_000,
        min_symbols_used_per_period=30,
    )
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
