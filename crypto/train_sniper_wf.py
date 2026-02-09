# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys
import time


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

from train.train_sniper_wf import TrainSniperWFSettings, run as run_wf  # type: ignore
from prepare_features.refresh_sniper_labels_in_cache import RefreshLabelsSettings, run as refresh_labels  # type: ignore
from crypto.trade_contract import TradeContract  # type: ignore


# Final run config (single source of truth)
RUN_DIR_RESUME = ""
REFRESH_LABELS_BEFORE_TRAIN = True
ENTRY_MODEL_TYPE = "catboost"
XGB_DEVICE = "cuda:0"
MCAP_MIN_USD = 100_000_000.0
MCAP_MAX_USD = 150_000_000_000.0
MAX_SYMBOLS = 150
TOTAL_DAYS = 0  # 0 = full history
OFFSET_STEP_DAYS = 60
MAX_OFFSET_DAYS = 2160
INCLUDE_T0 = True
OFFSETS_OVERRIDE: tuple[int, ...] = ()  # ex: (0, 90, 180, ..., 2160)
MAX_ROWS_ENTRY = 10_000_000
MIN_SYMBOLS_USED_PER_PERIOD = 60

# Best params from wf_009
SIDE_MAE_PENALTY = 1.1745451810284482
SIDE_TIME_PENALTY = 0.09718069303742632
SIDE_CROSS_PENALTY = 0.7455489524741807
ENTRY_REG_WEIGHT_ALPHA = 0.5656639957501756
ENTRY_REG_WEIGHT_POWER = 0.9069364698623656
ENTRY_REG_BALANCE_DISTANCE_POWER = 0.10547987991936614
ENTRY_REG_BALANCE_MIN_FRAC = 0.261931245322158


def _set_default_env() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("SNIPER_NUMBA_IMPORT_TIMEOUT", "120")
    os.environ.setdefault("SNIPER_ASSET_CLASS", "crypto")
    os.environ.setdefault("SNIPER_TIMINGS", "1")
    os.environ.setdefault("SNIPER_FEATURE_TIMINGS", "1")
    os.environ.setdefault("SNIPER_CACHE_PROGRESS_EVERY_S", "5")
    # tuned for ~40GB RAM
    os.environ.setdefault("SNIPER_CACHE_WORKERS", "8")
    os.environ.setdefault("SNIPER_DATASET_WORKERS", "4")
    os.environ.setdefault("SNIPER_POOL_READERS", "4")
    os.environ.setdefault("SNIPER_POOL_CHUNK_ROWS", "150000")
    os.environ.setdefault("SNIPER_DS_DEBUG", "0")
    os.environ.setdefault("SNIPER_LABEL_CENTER", "1")
    os.environ.setdefault("SNIPER_TIMING_USE_DOMINANT", "1")
    os.environ.setdefault("SNIPER_TIMING_DOMINANT_MIX", "0.50")
    os.environ.setdefault("SNIPER_USE_TIMING_WEIGHT", "1")
    os.environ.setdefault("SNIPER_BALANCE_SIGN", "0")
    os.environ.setdefault("SNIPER_BALANCE_TAIL_SIGN", "0")
    os.environ.setdefault("SNIPER_BALANCE_BINS_SIGN", "0")
    os.environ.setdefault("SNIPER_TARGET_CENTER_WEIGHTED", "0")
    os.environ.setdefault("SNIPER_TARGET_CENTER_METHOD", "median")
    os.environ.setdefault("SNIPER_ENABLE_CALIBRATION", "1")
    os.environ.setdefault("SNIPER_FORCE_PIECEWISE_CALIB", "1")
    os.environ.setdefault("SNIPER_TAIL_ABS", "4")
    os.environ.setdefault("SNIPER_TAIL_WEIGHT_MULT", "2")
    os.environ.setdefault("SNIPER_WEIGHT_MAX_MULT", "4")
    os.environ.setdefault("SNIPER_ENABLE_REG_SHAPE_WEIGHT", "1")


def _build_offsets() -> tuple[int, ...]:
    if OFFSETS_OVERRIDE:
        offs = sorted({int(x) for x in OFFSETS_OVERRIDE if int(x) >= 0})
        return tuple(offs)
    offs = list(range(int(OFFSET_STEP_DAYS), int(MAX_OFFSET_DAYS) + 1, int(OFFSET_STEP_DAYS)))
    if bool(INCLUDE_T0):
        offs = [0] + offs
    return tuple(int(x) for x in offs)


def _validate_config() -> None:
    if str(ENTRY_MODEL_TYPE).strip().lower() not in {"catboost", "xgb"}:
        raise ValueError("ENTRY_MODEL_TYPE deve ser 'catboost' ou 'xgb'")
    if float(MCAP_MIN_USD) <= 0 or float(MCAP_MAX_USD) <= 0:
        raise ValueError("MCAP_MIN_USD/MCAP_MAX_USD devem ser > 0")
    if float(MCAP_MIN_USD) > float(MCAP_MAX_USD):
        raise ValueError("MCAP_MIN_USD nao pode ser maior que MCAP_MAX_USD")
    if int(MAX_SYMBOLS) <= 0:
        raise ValueError("MAX_SYMBOLS deve ser > 0")
    if int(MAX_ROWS_ENTRY) <= 0:
        raise ValueError("MAX_ROWS_ENTRY deve ser > 0")
    if not OFFSETS_OVERRIDE:
        if int(OFFSET_STEP_DAYS) <= 0:
            raise ValueError("OFFSET_STEP_DAYS deve ser > 0")
        if int(MAX_OFFSET_DAYS) <= 0:
            raise ValueError("MAX_OFFSET_DAYS deve ser > 0")


def _refresh_labels() -> None:
    settings = RefreshLabelsSettings(
        candle_sec=60,
        mcap_min_usd=float(MCAP_MIN_USD),
        mcap_max_usd=float(MCAP_MAX_USD),
        max_symbols=int(MAX_SYMBOLS),
        label_clip=0.20,
        label_center=True,
        use_dominant=True,
        dominant_mix=0.50,
        side_mae_penalty=float(SIDE_MAE_PENALTY),
        side_time_penalty=float(SIDE_TIME_PENALTY),
        side_cross_penalty=float(SIDE_CROSS_PENALTY),
        verbose=True,
    )
    out = refresh_labels(settings)
    print(
        f"[train-wf] refresh done ok={out.get('ok')} fail={out.get('fail')} sec={out.get('seconds')}",
        flush=True,
    )


def main() -> None:
    _set_default_env()
    _validate_config()
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("[train-wf] start: " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    offsets = _build_offsets()
    if not offsets:
        raise RuntimeError("offsets vazio; ajuste OFFSET_STEP_DAYS/MAX_OFFSET_DAYS ou OFFSETS_OVERRIDE")
    min_symbols_per_period = int(min(MAX_SYMBOLS, max(1, MIN_SYMBOLS_USED_PER_PERIOD)))
    print(
        f"[train-wf] config: model={ENTRY_MODEL_TYPE} device={XGB_DEVICE} "
        f"mcap=[{MCAP_MIN_USD:.0f}..{MCAP_MAX_USD:.0f}] max_symbols={MAX_SYMBOLS} "
        f"offsets={len(offsets)} [{offsets[0]}..{offsets[-1]}] "
        f"max_rows={MAX_ROWS_ENTRY} total_days={TOTAL_DAYS} "
        f"min_symbols_period={min_symbols_per_period} refresh={REFRESH_LABELS_BEFORE_TRAIN}",
        flush=True,
    )

    if bool(REFRESH_LABELS_BEFORE_TRAIN):
        _refresh_labels()
    else:
        print("[train-wf] refresh skip (REFRESH_LABELS_BEFORE_TRAIN=False)", flush=True)

    contract = TradeContract(
        entry_label_windows_minutes=(60,),
        exit_ema_init_offset_pct=0.002,
    )
    settings = TrainSniperWFSettings(
        asset_class="crypto",
        contract=contract,
        mcap_min_usd=float(MCAP_MIN_USD),
        mcap_max_usd=float(MCAP_MAX_USD),
        max_symbols=int(MAX_SYMBOLS),
        total_days=int(TOTAL_DAYS),
        offsets_step_days=int(OFFSET_STEP_DAYS),
        offsets_days=offsets,
        entry_model_type=str(ENTRY_MODEL_TYPE),
        xgb_device=str(XGB_DEVICE),
        entry_reg_window_min=60,
        entry_reg_weight_alpha=float(ENTRY_REG_WEIGHT_ALPHA),
        entry_reg_weight_power=float(ENTRY_REG_WEIGHT_POWER),
        entry_reg_balance_bins=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 80.0),
        entry_reg_balance_distance_power=float(ENTRY_REG_BALANCE_DISTANCE_POWER),
        entry_reg_balance_min_frac=float(ENTRY_REG_BALANCE_MIN_FRAC),
        entry_label_mode="split_0_100",
        entry_label_scale=100.0,
        entry_sides=("long", "short"),
        max_rows_entry=int(MAX_ROWS_ENTRY),
        abort_ram_pct=85.0,
        entry_pool_full=True,
        entry_pool_prefiltered=True,
        use_feature_cache=True,
        min_symbols_used_per_period=int(min_symbols_per_period),
        run_dir=RUN_DIR_RESUME.strip() or None,
    )
    run_dir = run_wf(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
