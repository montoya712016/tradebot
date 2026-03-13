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


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "y", "on"}


def main() -> None:
    # Config fixa (sem flags/ENV de controle de fluxo)
    max_symbols = 0
    total_days = 0
    mcap_min_usd = 100_000_000.0
    mcap_max_usd = 150_000_000_000.0
    # Apos mudar o label/weight, o proximo treino precisa regravar labels no cache.
    refresh_labels_before_train = True
    prewarm_ohlc_before_train = False
    use_full_entry_pool = True
    # Safe mode (default): forca limites conservadores de recursos para evitar BSOD
    # em cargas longas de cache/dataset. Pode desligar com CRYPTO_FORCE_SAFE_RESOURCES=0.
    force_safe = True
    if bool(force_safe):
        os.environ["SNIPER_CACHE_WORKERS"] = "1"
        os.environ["SNIPER_DATASET_WORKERS"] = "1"
        os.environ["SNIPER_CACHE_RAM_PCT"] = "72"
        os.environ["SNIPER_DATASET_RAM_PCT"] = "75"
        os.environ["SNIPER_CACHE_CRITICAL_RAM_PCT"] = "92"
        os.environ["SNIPER_DATASET_CRITICAL_RAM_PCT"] = "92"
        os.environ["SNIPER_CACHE_MIN_FREE_MB"] = "4096"
        os.environ["SNIPER_DATASET_MIN_FREE_MB"] = "4096"
        os.environ["SNIPER_CACHE_PER_WORKER_MB"] = "1536"
        os.environ["SNIPER_DATASET_PER_WORKER_MB"] = "1536"

    # Calibracao de entry:
    # - platt(logit) costuma gerar probabilidades mais suaves e estaveis OOS
    # - isotonic pode "achatar" em patamares quando o sinal e ruidoso
    os.environ["SNIPER_ENTRY_CALIB_METHOD"] = "platt"
    os.environ["SNIPER_ENTRY_CALIB_USE_WEIGHTS"] = "0"
    os.environ["SNIPER_ENTRY_CALIB_PLATT_C"] = "0.80"
    os.environ["SNIPER_ENTRY_CALIB_COEF_MIN"] = "0.10"
    os.environ["SNIPER_ENTRY_CALIB_COEF_MAX"] = "8.00"
    os.environ["SNIPER_ENTRY_CALIB_INTERCEPT_MIN"] = "-8.00"
    os.environ["SNIPER_ENTRY_CALIB_INTERCEPT_MAX"] = "3.00"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_BLEND"] = "1.0"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_START"] = "0.95"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_POWER"] = "1.00"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_BOOST"] = "1.35"
    os.environ["SNIPER_ENTRY_CALIB_PRIOR_STRENGTH"] = "1.0"
    os.environ["SNIPER_ENTRY_CALIB_PRIOR_SHIFT_CLIP"] = "6.0"
    # Usa weight tambem na loss, com normalizacao controlada no trainer.
    os.environ["SNIPER_ENTRY_USE_LABEL_WEIGHTS"] = "1"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_CLASS_NORM"] = "1"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MIN"] = "0.0"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX"] = "20.0"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_POS"] = "6.0"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_CLIP_MAX_NEG"] = "10.0"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_POWER"] = "1.00"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_POWER_POS"] = "0.65"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_POWER_NEG"] = "0.90"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_MEAN_NORM"] = "0"
    # Mantem probabilidades ancoradas na prevalencia real do recorte.
    os.environ["SNIPER_ENTRY_CLASS_BALANCE_WEIGHTS"] = "0"
    os.environ["SNIPER_ENTRY_CLASS_BALANCE_STRENGTH"] = "0.0"
    # Label/weight: premia trajetoria consistente, nao apenas spike isolado.
    os.environ["PF_ENTRY_LABEL_NET_PROFIT_THR"] = "0.0045"
    os.environ["PF_ENTRY_LABEL_MIN_FUTURE_AVG_NET"] = "0.0012"
    os.environ["PF_ENTRY_LABEL_EARLY_WINDOW_MIN"] = "15"
    os.environ["PF_ENTRY_LABEL_MIN_FUTURE_EARLY_AVG_NET"] = "0.0005"
    os.environ["PF_ENTRY_LABEL_MIN_FUTURE_EARLY_WORST_NET"] = "-0.0035"
    os.environ["PF_ENTRY_LABEL_MIN_RISK_SCORE"] = "0.0000"
    os.environ["PF_ENTRY_LABEL_MIN_FUTURE_EFF"] = "0.09"
    os.environ["PF_ENTRY_LABEL_MIN_FUTURE_POS_FRAC"] = "0.52"
    os.environ["PF_ENTRY_LABEL_MIN_CONSISTENCY_SCORE"] = "0.0015"
    os.environ["PF_ENTRY_LABEL_MIN_CONTRACT_MAE"] = "-0.010"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_POS"] = "0.04"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_NEG"] = "0.03"
    os.environ["PF_ENTRY_WEIGHT_RET_DEADZONE"] = "0.002"
    os.environ["PF_ENTRY_WEIGHT_POS_GAIN"] = "6.0"
    os.environ["PF_ENTRY_WEIGHT_NEG_GAIN"] = "5.0"
    os.environ["PF_ENTRY_WEIGHT_POS_POWER"] = "2.6"
    os.environ["PF_ENTRY_WEIGHT_NEG_POWER"] = "2.2"
    os.environ["PF_ENTRY_WEIGHT_MODE"] = "future_consistency_dir"
    os.environ["PF_ENTRY_WEIGHT_PCT_POWER"] = "1.35"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_WINDOW_MIN"] = "60"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_DD_PENALTY"] = "1.20"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_AVG_SOFTCAP"] = "0.015"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_END_SOFTCAP"] = "0.020"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_EARLY_AVG_SOFTCAP"] = "0.008"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_END_GAIN"] = "0.35"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_EARLY_GAIN"] = "1.00"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_EARLY_DD_PENALTY"] = "1.10"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_EFF_GAIN"] = "0.0200"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_POS_FRAC_GAIN"] = "0.0150"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_POS_SCALE"] = "1.70"
    os.environ["PF_ENTRY_WEIGHT_FUTURE_NEG_SCALE"] = "0.70"
    os.environ["PF_ENTRY_WEIGHT_MAX"] = "20.0"
    # Bins mais finos para weight no sampling.
    os.environ["SNIPER_ENTRY_WEIGHT_BIN_STEP_X10"] = "5"
    os.environ["SNIPER_ENTRY_WEIGHT_BIN_MAX"] = "20.0"
    # Reduz ruido de positivos marginais (0~0.5%) sem remover completamente.
    os.environ["SNIPER_ENTRY_POS_KEEP_FRACTION"] = "0.85"
    os.environ["SNIPER_ENTRY_POS_MIN_WEIGHT"] = "0.75"
    os.environ["SNIPER_ENTRY_NEG_FAVOR_HIGH"] = "1"
    # Pool-full persistido/reusado automaticamente entre execuções.
    os.environ["SNIPER_FULL_POOL_REUSE"] = "1"
    os.environ["SNIPER_FULL_POOL_REBUILD"] = "1"
    # Usa mtime no fingerprint para invalidar pool quando labels forem regravados.
    os.environ["SNIPER_FULL_POOL_FINGERPRINT_MODE"] = "mtime"
    # Evita fallback para pool antigo incompativel quando o hash atual mudar.
    os.environ["SNIPER_FULL_POOL_USE_LAST"] = "0"
    os.environ["SNIPER_FULL_POOL_CACHE_DIR"] = "D:/astra/cache_sniper/full_pool"
    print(f"[train-wf-crypto] full_pool_cache_dir={os.environ['SNIPER_FULL_POOL_CACHE_DIR']}", flush=True)

    # Modo default: reaproveita cache de features e só faz refresh de labels.
    if bool(prewarm_ohlc_before_train):
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
    else:
        print("[train-wf-crypto] ohlc prewarm: skipped (labels-only refresh mode)", flush=True)

    metric_mode = "aucpr"
    contract = TradeContract(
        timeframe_sec=60,
        entry_label_windows_minutes=(240,),
        entry_label_min_profit_pcts=(0.02,),
        exit_ema_span=60,
        exit_ema_init_offset_pct=0.005,
    )
    # Modo simplificado pedido: treino somente de entry (sem regressão de exit).
    os.environ["SNIPER_TRAIN_EXIT_MODEL"] = "0"
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
        # Menos capacidade para reduzir gap train->val (overfit).
        entry_params={
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "eta": 0.04,
            "max_depth": 6,
            "min_child_weight": 8,
            "gamma": 0.20,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "max_bin": 256,
            "lambda": 3.0,
            "alpha": 0.40,
            "tree_method": "hist",
            "device": "cuda:0",
        },
        contract=contract,
        total_days=int(total_days),
        max_symbols=int(max_symbols),
        offsets_step_days=180,
        # Corta tails mais antigos (T-1620+), onde o train-aucpr começou a inflar demais.
        offsets_days=(0, 180, 360, 540, 720, 900, 1080, 1260, 1440),
        entry_ratio_neg_per_pos=3.0,
        max_rows_entry=2_000_000,
        max_rows_exit=0,
        use_full_entry_pool=bool(use_full_entry_pool),
        # Pool full para slicing por periodo (evita rebuild completo por step).
        full_pool_max_rows_entry=2_000_000,
        min_symbols_used_per_period=30,
    )
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
