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
REFRESH_LABELS_BEFORE_TRAIN = False
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
SIDE_MAE_PENALTY = 1.10
SIDE_TIME_PENALTY = 0.55
SIDE_GIVEBACK_PENALTY = 0.65
SIDE_CROSS_PENALTY = 0.35
SIDE_REV_LOOKBACK_MIN = 90
SIDE_CHASE_PENALTY = 0.90
SIDE_REVERSAL_BONUS = 0.55
SIDE_CONFIRM_MIN = 20
SIDE_CONFIRM_MOVE = 0.0025
SIDE_PRECONFIRM_SUPPRESS = 0.15
ENTRY_REG_WEIGHT_ALPHA = 0.60
ENTRY_REG_WEIGHT_POWER = 0.90
ENTRY_REG_BALANCE_DISTANCE_POWER = 0.10
ENTRY_REG_BALANCE_MIN_FRAC = 0.25
ENTRY_REG_LABEL_COL_TEMPLATE = "edge_label_{side}"
ENTRY_REG_WEIGHT_COL_TEMPLATE = "edge_weight_{side}"
ENTRY_CLS_ENABLED = True
ENTRY_CLS_MODEL_TYPE = "catboost"
ENTRY_CLS_LABEL_COL_TEMPLATE = "entry_gate_{side}"
ENTRY_CLS_WEIGHT_COL_TEMPLATE = "entry_gate_weight_{side}"
ENTRY_CLS_POSITIVE_THRESHOLD = 75.0
ENTRY_CLS_BALANCE_BINS = ()
ENTRY_CLS_TARGET_POS_RATIO = 0.20
# Gate: aumentar peso da classe negativa reduz fake positives.
# Se ficar "duro" demais, aproxime NEG de POS (ex.: 1.8 -> 1.4).
ENTRY_CLS_POS_WEIGHT = 1.0
ENTRY_CLS_NEG_WEIGHT = 1.8
ENTRY_CLS_PARAMS = {
    "iterations": 1200,
    "learning_rate": 0.04,
    "depth": 6,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "od_type": "Iter",
    "od_wait": 200,
    "verbose": 50,
}
ENTRY_CLS_PARAMS_BY_SIDE = {
    "short": {
        "iterations": 1400,
        "learning_rate": 0.035,
        "depth": 6,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.75,
        "od_wait": 220,
    },
    "long": {
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 6,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "od_wait": 200,
    },
}

# Hiperparâmetros base (catboost)
ENTRY_PARAMS = {
    "iterations": 1800,
    "learning_rate": 0.04,
    "depth": 7,
    "subsample": 0.8,
    "l2_leaf_reg": 6.0,
    "eval_metric": "RMSE",
    "od_type": "Iter",
    "od_wait": 250,
    "verbose": 50,
}

# Short tende a overfitar mais: configuração mais conservadora
ENTRY_PARAMS_BY_SIDE = {
    "short": {
        "iterations": 2400,
        "learning_rate": 0.025,
        "depth": 6,
        "subsample": 0.7,
        "l2_leaf_reg": 12.0,
        "bagging_temperature": 1.0,
        "od_type": "Iter",
        "od_wait": 300,
    },
    "long": {
        "iterations": 1800,
        "learning_rate": 0.04,
        "depth": 7,
        "subsample": 0.8,
        "l2_leaf_reg": 6.0,
        "od_type": "Iter",
        "od_wait": 250,
    },
}


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
    os.environ.setdefault("SNIPER_LABELS_REFRESH_WORKERS", "16")
    os.environ.setdefault("SNIPER_POOL_READERS", "4")
    os.environ.setdefault("SNIPER_POOL_CHUNK_ROWS", "150000")
    os.environ.setdefault("SNIPER_POOL_BUILD_WORKERS", "4")
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
    # Classificador de gate: evitar forcar probs para ~0.5
    os.environ.setdefault("SNIPER_ENTRY_CLS_AUTO_CLASS_WEIGHTS", "")
    os.environ.setdefault("SNIPER_ENTRY_CLS_AUTO_POS_WEIGHT", "0")
    os.environ.setdefault("SNIPER_ENABLE_CLS_CALIB", "1")
    # nao rebalancear por peso para 50/50 (queremos classe 0 dominante)
    os.environ.setdefault("SNIPER_CLS_BALANCE_CLASS_WEIGHT", "0")
    # Labels de gate mais seletivos (maior raridade/precisao):
    # - horizonte menor
    # - TP mais alto
    # - SL mais apertado
    os.environ.setdefault("SNIPER_ENTRY_GATE_TMAX_MIN", "240")
    os.environ.setdefault("SNIPER_ENTRY_GATE_TP_PCT", "0.03")
    os.environ.setdefault("SNIPER_ENTRY_GATE_SL_PCT", "0.006")
    # Thermal guard (pausa treino se CPU/GPU aquecer demais)
    os.environ.setdefault("SNIPER_THERMAL_GUARD", "1")
    os.environ.setdefault("SNIPER_THERMAL_MAX_TEMP_C", "80")
    os.environ.setdefault("SNIPER_THERMAL_RESUME_BELOW_C", "70")
    os.environ.setdefault("SNIPER_THERMAL_CHECK_EVERY_S", "10")
    os.environ.setdefault("SNIPER_THERMAL_COOLDOWN_S", "15")


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
        side_giveback_penalty=float(SIDE_GIVEBACK_PENALTY),
        side_cross_penalty=float(SIDE_CROSS_PENALTY),
        side_rev_lookback_min=int(SIDE_REV_LOOKBACK_MIN),
        side_chase_penalty=float(SIDE_CHASE_PENALTY),
        side_reversal_bonus=float(SIDE_REVERSAL_BONUS),
        side_confirm_min=int(SIDE_CONFIRM_MIN),
        side_confirm_move=float(SIDE_CONFIRM_MOVE),
        side_preconfirm_suppress=float(SIDE_PRECONFIRM_SUPPRESS),
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
        entry_params=dict(ENTRY_PARAMS),
        entry_params_by_side=dict(ENTRY_PARAMS_BY_SIDE),
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
        entry_reg_label_col_template=str(ENTRY_REG_LABEL_COL_TEMPLATE),
        entry_reg_weight_col_template=str(ENTRY_REG_WEIGHT_COL_TEMPLATE),
        entry_cls_enabled=bool(ENTRY_CLS_ENABLED),
        entry_cls_model_type=str(ENTRY_CLS_MODEL_TYPE),
        entry_cls_label_col_template=str(ENTRY_CLS_LABEL_COL_TEMPLATE),
        entry_cls_weight_col_template=str(ENTRY_CLS_WEIGHT_COL_TEMPLATE),
        entry_cls_positive_threshold=float(ENTRY_CLS_POSITIVE_THRESHOLD),
        entry_cls_balance_bins=tuple(float(x) for x in ENTRY_CLS_BALANCE_BINS),
        entry_cls_target_pos_ratio=float(ENTRY_CLS_TARGET_POS_RATIO),
        entry_cls_pos_weight=float(ENTRY_CLS_POS_WEIGHT),
        entry_cls_neg_weight=float(ENTRY_CLS_NEG_WEIGHT),
        entry_cls_params=dict(ENTRY_CLS_PARAMS),
        entry_cls_params_by_side=dict(ENTRY_CLS_PARAMS_BY_SIDE),
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
