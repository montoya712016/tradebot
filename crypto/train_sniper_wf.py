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


def _set_default_env() -> None:
    # Logs de timing por padrão para facilitar diagnóstico
    os.environ.setdefault("SNIPER_TIMINGS", "1")
    os.environ.setdefault("SNIPER_FEATURE_TIMINGS", "1")
    # barra de progresso mais frequente
    os.environ.setdefault("SNIPER_CACHE_PROGRESS_EVERY_S", "3")
    # mais paralelismo para cache
    os.environ.setdefault("SNIPER_CACHE_WORKERS", "16")
    # paralelismo na montagem do dataset (ajuste se a RAM subir demais)
    os.environ.setdefault("SNIPER_DATASET_WORKERS", "8")
    # paralelismo na leitura do pool (parquet)
    os.environ.setdefault("SNIPER_POOL_READERS", "16")
    # debug do dataset (mostra pos/neg e motivos de skip)
    os.environ.setdefault("SNIPER_DS_DEBUG", "0")

def main() -> None:
    _set_default_env()
    # Defina aqui se quiser retomar um wf_XXX existente. Deixe vazio para criar novo.
    resume_run_dir = ""
    # Pool existente (opcional). Se vazio, gera pool do zero no run atual.
    use_existing_pool = False
    existing_pool_root = ""
    metric_mode = _env_str("CRYPTO_ENTRY_METRIC_MODE", "aucpr")
    contract = TradeContract(
        entry_label_windows_minutes=(60,),
        entry_label_min_profit_pcts=(0.03,),
        entry_label_max_dd_pcts=(0.03,),
        entry_label_weight_alpha=0.5,
        exit_ema_init_offset_pct=0.002,
    )
    settings = TrainSniperWFSettings(
        asset_class="crypto",
        entry_metric_mode=metric_mode,
        contract=contract,
        mcap_min_usd=50_000_000.0,
        offsets_step_days=180,
        offsets_days=(0, 180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160),
        entry_ratio_neg_per_pos=4.0,
        entry_pool_ratio_neg_per_pos=12.0,
        entry_label_sides=("long", "short"),
        # limite global por modelo (subsample ponderado por peso)
        max_rows_entry=10_000_000,
        entry_pool_full=True,
        entry_pool_dir=(Path(existing_pool_root) / "entry_pool_full") if use_existing_pool and existing_pool_root else None,
        entry_pool_prefiltered=True,
        use_feature_cache=False,
        min_symbols_used_per_period=30,
        run_dir=resume_run_dir.strip() or None,
    )
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
