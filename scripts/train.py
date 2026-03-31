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
from train.feature_presets import feature_flags_for_preset, summarize_preset  # type: ignore
from utils.resource_sizing import apply_env_worker_default  # type: ignore
from config.trade_contract import (  # type: ignore
    CRYPTO_PIPELINE_CANDLE_SEC,
    apply_crypto_pipeline_env,
    build_default_crypto_contract,
    timeframe_tag,
)
from prepare_features.refresh_sniper_labels_in_cache import (  # type: ignore
    RefreshLabelsSettings,
    run as run_labels_refresh,
)
from train.sniper_dataflow import _cache_dir, _cache_format  # type: ignore


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v.strip() or default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    v = os.getenv(name, "").strip()
    if not v:
        return tuple(int(x) for x in default)
    out: list[int] = []
    for part in v.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return tuple(out) if out else tuple(int(x) for x in default)


def _env_symbols(name: str = "TRAIN_SYMBOLS") -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return tuple()
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        sym = str(part or "").strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return tuple(out)


def _load_symbols_from_file(path_str: str) -> tuple[str, ...]:
    path = Path(path_str).expanduser()
    if not path.exists():
        return tuple()
    out: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            sym = str(line or "").strip().upper()
            if sym and (not sym.startswith("#")) and sym not in out:
                out.append(sym)
    except Exception:
        return tuple()
    return tuple(out)


def _apply_train_overrides(prefix: str = "TRAIN_OVR_") -> None:
    applied: list[str] = []
    for name, value in list(os.environ.items()):
        if not name.startswith(prefix):
            continue
        target = name[len(prefix):].strip()
        if not target:
            continue
        os.environ[target] = str(value)
        applied.append(target)
    if applied:
        applied.sort()
        print(f"[train-wf-crypto] env overrides: {', '.join(applied)}", flush=True)


def _feature_cache_bootstrap_needed(asset_class: str, candle_sec: int) -> tuple[bool, Path, int]:
    cache_dir = _cache_dir(asset_class, candle_sec)
    fmt = _cache_format()
    suf = "*.parquet" if fmt == "parquet" else "*.pkl"
    try:
        count = sum(1 for _ in cache_dir.glob(suf))
    except Exception:
        count = 0
    try:
        min_ready_files = int(os.getenv("SNIPER_FEATURE_CACHE_BOOTSTRAP_MIN_FILES", "64") or "64")
    except Exception:
        min_ready_files = 64
    if min_ready_files < 1:
        min_ready_files = 1
    return (count < min_ready_files), cache_dir, int(count)


def main() -> None:
    # --- 1. SETUP PARAMETERS & ENVIRONMENT ---
    # Apply robust "mid-interval" settings early so all constants reflect them
    os.environ["PF_ENTRY_LABEL_NET_PROFIT_THR"] = _env_str("PF_ENTRY_LABEL_NET_PROFIT_THR", "0.025")
    os.environ["CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT"] = _env_str("CRYPTO_ENTRY_LABEL_MIN_PROFIT_PCT", "0.025")
    os.environ["CRYPTO_EXIT_EMA_SPAN_MINUTES"] = _env_str("CRYPTO_EXIT_EMA_SPAN_MINUTES", "80")
    os.environ["SNIPER_ENTRY_TOP_METRIC_MIN_COUNT"] = _env_str("SNIPER_ENTRY_TOP_METRIC_MIN_COUNT", "48")
    
    # --- 1.2 RESOURCE OPTIMIZATION ---
    # Configuração agressiva para o PC de 64GB: maximiza velocidade de cache e dataset
    cache_workers = apply_env_worker_default("SNIPER_CACHE_WORKERS", "feature_cache")
    dataset_workers = apply_env_worker_default("SNIPER_DATASET_WORKERS", "dataset")
    os.environ.setdefault("SNIPER_CACHE_PER_WORKER_MB", "2048")
    os.environ.setdefault("SNIPER_DATASET_PER_WORKER_MB", "2048")
    os.environ.setdefault("SNIPER_CACHE_RAM_PCT", "90.0")
    os.environ.setdefault("SNIPER_DATASET_RAM_PCT", "90.0")
    
    # Safe mode (opcional): se o PC começar a travar, mude force_safe para True
    force_safe = False
    if bool(force_safe):
        os.environ["SNIPER_CACHE_WORKERS"] = "1"
        os.environ["SNIPER_DATASET_WORKERS"] = "1"
        os.environ["SNIPER_CACHE_PER_WORKER_MB"] = "1024"
        os.environ["SNIPER_DATASET_PER_WORKER_MB"] = "1024"
        cache_workers = 1
        dataset_workers = 1

    _apply_train_overrides()
    
    candle_sec = apply_crypto_pipeline_env(CRYPTO_PIPELINE_CANDLE_SEC)
    contract = build_default_crypto_contract(candle_sec)
    feature_preset_name = _env_str("TRAIN_FEATURE_PRESET", _env_str("SNIPER_FEATURE_PRESET", "full")).strip().lower() or "full"
    os.environ["SNIPER_FEATURE_PRESET"] = feature_preset_name
    
    # --- 2. CONFIGURATION ---
    max_symbols = _env_int("TRAIN_MAX_SYMBOLS", 0)  # 0 means all available symbols
    total_days = _env_int("TRAIN_TOTAL_DAYS", 0)
    mcap_min_usd = 100_000_000.0
    mcap_max_usd = 150_000_000_000.0
    
    bootstrap_cache, feature_cache_dir, feature_cache_files = _feature_cache_bootstrap_needed("crypto", int(candle_sec))
    refresh_labels_before_train = (not bool(bootstrap_cache)) and _env_bool("TRAIN_REFRESH_LABELS", True)
    prewarm_ohlc_before_train = False
    use_full_entry_pool = True
    entry_ratio_neg_per_pos = _env_float("TRAIN_ENTRY_RATIO_NEG_PER_POS", 4.0)
    max_rows_entry = _env_int("TRAIN_MAX_ROWS_ENTRY", 2_000_000)
    full_pool_max_rows_entry = _env_int("TRAIN_FULL_POOL_MAX_ROWS_ENTRY", max_rows_entry)
    min_symbols_used_per_period = _env_int("TRAIN_MIN_SYMBOLS_USED_PER_PERIOD", 30)
    offsets_days = _env_int_tuple("TRAIN_OFFSETS_DAYS", (0, 180, 360, 540, 720, 900, 1080, 1260, 1440))
    explicit_symbols = _env_symbols("TRAIN_SYMBOLS")
    if not explicit_symbols:
        explicit_symbols = _load_symbols_from_file(_env_str("TRAIN_SYMBOLS_FILE", ""))
    # Alinhamento com Explorer: usa fixed contract exit, sem treinar modelo de exit
    os.environ["SNIPER_TRAIN_EXIT_MODEL"] = "0"

    # Calibracao de entry:
    # - platt(logit) costuma gerar probabilidades mais suaves e estaveis OOS
    # - isotonic pode "achatar" em patamares quando o sinal e ruidoso
    os.environ["SNIPER_ENTRY_CALIB_METHOD"] = "platt"
    os.environ["SNIPER_ENTRY_CALIB_USE_WEIGHTS"] = "0"
    os.environ["SNIPER_ENTRY_CALIB_PLATT_C"] = "0.20"
    os.environ["SNIPER_ENTRY_CALIB_COEF_MIN"] = "0.35"
    os.environ["SNIPER_ENTRY_CALIB_COEF_MAX"] = "2.60"
    os.environ["SNIPER_ENTRY_CALIB_INTERCEPT_MIN"] = "-2.40"
    os.environ["SNIPER_ENTRY_CALIB_INTERCEPT_MAX"] = "1.20"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_BLEND"] = "0.75"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_START"] = "0.50"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_POWER"] = "1.15"
    os.environ["SNIPER_ENTRY_CALIB_TAIL_BOOST"] = "1.25"
    os.environ["SNIPER_ENTRY_CALIB_PRIOR_STRENGTH"] = "0.45"
    os.environ["SNIPER_ENTRY_CALIB_PRIOR_SHIFT_CLIP"] = "2.0"
    os.environ["SNIPER_ENTRY_CALIB_STABILITY_ENABLE"] = "1"
    os.environ["SNIPER_ENTRY_CALIB_STABILITY_GLOBAL_BLEND"] = "0.18"
    os.environ["SNIPER_ENTRY_CALIB_STABILITY_NEIGHBOR_BLEND"] = "0.28"
    os.environ["SNIPER_ENTRY_CALIB_STABILITY_ROBUST_CLIP_SIGMA"] = "2.0"
    # Seleção do entry focada no topo do ranking do validation.
    os.environ["SNIPER_ENTRY_MODEL_SELECTION"] = "top_precision_combo"
    os.environ["SNIPER_ENTRY_TOP_METRIC_QS"] = "0.0005,0.001,0.0025"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_POWER_NEG"] = "0.90"
    os.environ["SNIPER_ENTRY_LOSS_WEIGHT_MEAN_NORM"] = "0"
    # Mantem probabilidades ancoradas na prevalencia real do recorte.
    os.environ["SNIPER_ENTRY_CLASS_BALANCE_WEIGHTS"] = "0"
    os.environ["SNIPER_ENTRY_CLASS_BALANCE_STRENGTH"] = "0.0"
    # Label: profit-only pelo contrato. Se o trade finaliza acima do threshold,
    # ja e positivo; sem filtros de caminho, eficiencia ou zona neutra.
    os.environ["PF_ENTRY_LABEL_PROFIT_ONLY"] = "1"
    os.environ["PF_ENTRY_LABEL_REQUIRE_NO_DIP"] = "0"
    os.environ["PF_ENTRY_LABEL_ENABLE_NEUTRAL"] = "0"
    os.environ["PF_ENTRY_LABEL_ANY_CANONICAL"] = "0"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_POS"] = "0.04"
    os.environ["PF_ENTRY_WEIGHT_RET_SCALE_NEG"] = "0.03"
    os.environ["PF_ENTRY_WEIGHT_RET_DEADZONE"] = "0.002"
    os.environ["PF_ENTRY_WEIGHT_POS_GAIN"] = "6.0"
    os.environ["PF_ENTRY_WEIGHT_NEG_GAIN"] = "5.0"
    os.environ["PF_ENTRY_WEIGHT_POS_POWER"] = "2.6"
    os.environ["PF_ENTRY_WEIGHT_NEG_POWER"] = "2.2"
    os.environ["PF_ENTRY_WEIGHT_MODE"] = "ret_curve"
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
    # Teste controlado: weight fora do balanceamento. O dataset fica guiado
    # principalmente por contagem de classes, nao por score de weight.
    os.environ["SNIPER_ENTRY_WEIGHT_BIN_STEP_X10"] = "5"
    os.environ["SNIPER_ENTRY_WEIGHT_BIN_MAX"] = "20.0"
    os.environ["SNIPER_ENTRY_POS_KEEP_FRACTION"] = "1.00"
    os.environ["SNIPER_ENTRY_POS_MIN_WEIGHT"] = "0.00"
    os.environ["SNIPER_ENTRY_NEG_MIN_WEIGHT"] = "0.00"
    os.environ["SNIPER_ENTRY_POS_FAVOR_HIGH"] = "0"
    os.environ["SNIPER_ENTRY_NEG_FAVOR_HIGH"] = "0"
    os.environ["SNIPER_ENTRY_USE_WEIGHT_BINS"] = "0"
    os.environ["SNIPER_ENTRY_BIN_PLOT"] = "0"
    # Pool-full persistido/reusado automaticamente entre execuções.
    os.environ["SNIPER_FULL_POOL_REUSE"] = "1"
    os.environ["SNIPER_FULL_POOL_REBUILD"] = "0"
    os.environ["SNIPER_FULL_POOL_FINGERPRINT_MODE"] = "mtime"
    os.environ["SNIPER_FULL_POOL_USE_LAST"] = "0"
    os.environ["SNIPER_FULL_POOL_KEEP_LAST"] = "3"
    os.environ["SNIPER_FULL_POOL_CACHE_DIR"] = "D:/astra/cache_sniper/full_pool"
    full_pool_base_neg_per_pos = _env_float("TRAIN_FULL_POOL_BASE_NEG_PER_POS", float(entry_ratio_neg_per_pos))
    os.environ["SNIPER_FULL_POOL_BASE_RATIO_NEG_PER_POS"] = str(max(float(entry_ratio_neg_per_pos), float(full_pool_base_neg_per_pos)))
    _apply_train_overrides()
    print(
        f"[train-wf-crypto] full_pool_cache_dir={os.environ['SNIPER_FULL_POOL_CACHE_DIR']} "
        f"keep_last={os.environ['SNIPER_FULL_POOL_KEEP_LAST']} "
        f"base_neg_per_pos={os.environ['SNIPER_FULL_POOL_BASE_RATIO_NEG_PER_POS']}",
        flush=True,
    )
    print(
        f"[train-wf-crypto] feature_cache_dir={feature_cache_dir} timeframe={timeframe_tag(int(candle_sec))} "
        f"bootstrap_cache={int(bool(bootstrap_cache))} files={int(feature_cache_files)} "
        f"cache_workers={int(cache_workers)} dataset_workers={int(dataset_workers)}",
        flush=True,
    )
    print(
        f"[train-wf-crypto] sampling_cfg ratio_neg_per_pos={entry_ratio_neg_per_pos:.2f} "
        f"max_rows_entry={int(max_rows_entry)} full_pool_rows={int(full_pool_max_rows_entry)}",
        flush=True,
    )
    print(
        f"[train-wf-crypto] offsets_days={offsets_days} max_symbols={int(max_symbols)} "
        f"min_symbols_used_per_period={int(min_symbols_used_per_period)} "
        f"explicit_symbols={len(explicit_symbols)}",
        flush=True,
    )
    print(
        f"[train-wf-crypto] contract_cfg label_thr={contract.entry_label_min_profit_pcts} "
        f"exit_span={int(contract.exit_ema_span)} offset={float(contract.exit_ema_init_offset_pct):.4f}",
        flush=True,
    )
    print(
        f"[train-wf-crypto] entry_objective={os.environ.get('SNIPER_ENTRY_OBJECTIVE_MODE','binary')} "
        f"selection_metric={os.environ.get('SNIPER_ENTRY_MODEL_SELECTION','aucpr')} "
        f"rank_group_minutes={os.environ.get('SNIPER_ENTRY_RANK_GROUP_MINUTES','4320')}",
        flush=True,
    )

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
    feature_flags = feature_flags_for_preset(feature_preset_name, label=True)
    preset_info = summarize_preset(feature_preset_name)
    print(
        f"[train-wf-crypto] feature_preset={preset_info.get('preset')} "
        f"feature_count={preset_info.get('count')} blocks={len(preset_info.get('blocks') or [])}",
        flush=True,
    )
    # Modo simplificado pedido: treino somente de entry (sem regressão de exit).
    os.environ["SNIPER_TRAIN_EXIT_MODEL"] = "0"
    if bool(refresh_labels_before_train):
        os.environ["SNIPER_ASSET_CLASS"] = "crypto"
        ref = run_labels_refresh(
            RefreshLabelsSettings(
                contract=contract,
                candle_sec=int(candle_sec),
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
    else:
        print(
            "[train-wf-crypto] labels refresh: skipped (feature cache bootstrap mode; trainer will build 5m cache+labels)",
            flush=True,
        )

    settings = TrainSniperWFSettings(
        asset_class="crypto",
        symbols=tuple(explicit_symbols),
        entry_metric_mode=metric_mode,
        # Menos capacidade para reduzir gap train->val (overfit).
        entry_params={
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "eta": 0.05,
            "max_depth": 8,
            "min_child_weight": 4,
            "gamma": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "max_bin": 256,
            "lambda": 2.0,
            "alpha": 0.20,
            "tree_method": "hist",
            "device": "cuda:0",
        },
        contract=contract,
        total_days=int(total_days),
        max_symbols=int(max_symbols),
        offsets_step_days=180,
        offsets_days=tuple(int(x) for x in offsets_days),
        entry_ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
        max_rows_entry=int(max_rows_entry),
        max_rows_exit=0,
        use_full_entry_pool=bool(use_full_entry_pool),
        # Pool full para slicing por periodo (evita rebuild completo por step).
        full_pool_max_rows_entry=int(full_pool_max_rows_entry),
        min_symbols_used_per_period=int(min_symbols_used_per_period),
        feature_flags=feature_flags,
        feature_preset_name=str(preset_info.get("preset") or ""),
    )
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
