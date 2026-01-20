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
from train.sniper_trainer import DEFAULT_STOCKS_SYMBOLS_FILE  # type: ignore
from stocks.trade_contract import DEFAULT_TRADE_CONTRACT as STOCKS_CONTRACT  # type: ignore


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def main() -> None:
    # Defaults mais leves para ações (histórico muito maior que crypto)
    os.environ.setdefault("SNIPER_CACHE_WORKERS", "8")
    os.environ.setdefault("SNIPER_CACHE_PROGRESS_EVERY_S", "5")
    # Habilita logs de tempo de features automaticamente para diagnosticar gargalos
    os.environ.setdefault("SNIPER_FEATURE_TIMINGS", "1")
    # Força driver C do MySQL (mais rápido) a menos que o usuário sobrescreva
    os.environ.setdefault("PF_FORCE_DRIVER", "cext")

    total_days = _env_int("STK_TOTAL_DAYS", 0)  # 0 => todo o histórico disponível
    offsets_step = _env_int("STK_OFFSETS_STEP_DAYS", 180)
    offsets_years = _env_int("STK_OFFSETS_YEARS", 6)
    max_symbols = _env_int("STK_MAX_SYMBOLS", 0)  # 0 => todos os símbolos
    min_used = _env_int("STK_MIN_SYMBOLS_USED_PER_PERIOD", 30)
    max_rows_entry = _env_int("STK_MAX_ROWS_ENTRY", 6_000_000)
    symbols_file = os.getenv("STK_SYMBOLS_FILE", "").strip() or str(DEFAULT_STOCKS_SYMBOLS_FILE)

    settings = TrainSniperWFSettings(
        asset_class="stocks",
        symbols_file=symbols_file,
        contract=STOCKS_CONTRACT,
        total_days=total_days,
        offsets_step_days=offsets_step,
        offsets_years=offsets_years,
        max_symbols=max_symbols,
        min_symbols_used_per_period=min_used,
        max_rows_entry=max_rows_entry,
    )
    run_dir = run(settings)
    print(f"[train-wf-stocks] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
