# -*- coding: utf-8 -*-
"""
Treinamento dos modelos Sniper (EntryScore + DangerScore) usando walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import json
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:
    raise

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None  # type: ignore

from .sniper_dataflow import (
    prepare_sniper_dataset,
    prepare_sniper_dataset_from_cache,
    ensure_feature_cache,
    GLOBAL_FLAGS_FULL,
    SniperBatch,
)
try:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]


try:
    from utils.paths import models_root as _models_root  # type: ignore

    SAVE_ROOT = _models_root()
except Exception:
    # fallback antigo
    SAVE_ROOT = Path(__file__).resolve().parents[2].parent / "models_sniper"
DEFAULT_SYMBOLS_FILE = Path(__file__).resolve().parents[1] / "top_market_cap.txt"

try:
    from config.symbols import load_market_caps
except Exception:
    # fallback para execução fora de pacote
    from config.symbols import load_market_caps  # type: ignore[import]


@dataclass
class TrainConfig:
    total_days: int = 365 * 5
    offsets_days: Sequence[int] = (90, 180, 270, 360, 450, 540, 630, 720)
    mcap_min_usd: float = 50_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    symbols_file: Path = DEFAULT_SYMBOLS_FILE
    # 0 = sem limite (usa todas elegíveis por market cap)
    max_symbols: int = 0
    # Se o período ficar com poucos símbolos com dados válidos, pula o treino (evita modelo escasso).
    min_symbols_used_per_period: int = 30
    entry_params: dict = None
    danger_params: dict = None
    exit_params: dict = None
    contract: TradeContract = DEFAULT_TRADE_CONTRACT
    # dataset sizing (VRAM/RAM)
    max_rows_entry: int = 4_000_000
    max_rows_danger: int = 2_000_000
    max_rows_exit: int = 2_000_000
    entry_ratio_neg_per_pos: float = 6.0
    danger_ratio_neg_per_pos: float = 4.0
    exit_ratio_neg_per_pos: float = 4.0
    use_feature_cache: bool = True
    # Thresholding (estável):
    # Em datasets muito desbalanceados, otimizar tau por F1/precision pode empurrar tau para ~0.95,
    # matando a taxa de entradas. Para Entry/Exit, usamos uma taxa-alvo de positivos PREDITOS no VAL.
    # Defaults mais agressivos para não “morrer” em tau_entry muito alto.
    # Com 1m, 30 dias ≈ 43k barras; 1% no VAL costuma produzir candidatos suficientes
    # para ao menos 1 entrada/mês (ainda depende do tempo em posição).
    entry_target_pred_pos_frac: float = 0.010  # 1.0% dos pontos do val viram "entry"
    exit_target_pred_pos_frac: float = 0.020   # 2.0% dos snapshots do val viram "exit"
    # piso/teto do tau_entry (evita comprar demais em regimes ruins)
    entry_tau_min: float = 0.60
    entry_tau_max: float = 0.90


DEFAULT_ENTRY_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 9,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    # Observação: QuantileDMatrix costuma usar 256 bins por padrão; manter consistente evita
    # erro "Inconsistent max_bin (512 vs 256)" e permite treinar na GPU.
    "max_bin": 256,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "device": "cuda:0",
}

DEFAULT_DANGER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "max_bin": 256,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "device": "cuda:0",
}

DEFAULT_EXIT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "max_bin": 256,
    "lambda": 1.0,
    "alpha": 0.0,
    "tree_method": "hist",
    "device": "cuda:0",
}



def _select_symbols(cfg: TrainConfig) -> List[str]:
    cap_map = load_market_caps(cfg.symbols_file)
    lo = min(cfg.mcap_min_usd, cfg.mcap_max_usd)
    hi = max(cfg.mcap_min_usd, cfg.mcap_max_usd)
    # ordena por market cap desc
    ranked = sorted(cap_map.items(), key=lambda kv: kv[1], reverse=True)
    symbols: List[str] = []
    max_symbols = int(getattr(cfg, "max_symbols", 0) or 0)
    for sym, cap in ranked:
        if cap < lo or cap > hi:
            continue
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        symbols.append(sym)
        if max_symbols > 0 and len(symbols) >= max_symbols:
            break
    if not symbols:
        raise RuntimeError("Nenhum símbolo elegível para o intervalo de market cap")
    return symbols


def _split_train_val(batch: SniperBatch, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n = batch.X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(batch.ts)
    cut = max(1, int(round(n * (1 - val_frac))))
    tr_idx = order[:cut]
    va_idx = order[cut:]
    if va_idx.size == 0:
        va_idx = tr_idx[-max(1, min(1000, tr_idx.size // 5)) :]
        tr_idx = tr_idx[: tr_idx.size - va_idx.size]
    return tr_idx, va_idx


def _train_xgb_classifier(batch: SniperBatch, params: dict) -> tuple[xgb.Booster, dict]:
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    device = str(params.get("device", "cpu")).lower()
    use_quantile = device.startswith("cuda")
    Xtr = batch.X[tr_idx]
    ytr = batch.y[tr_idx]
    Xva = batch.X[va_idx]
    yva = batch.y[va_idx]

    # Blindagem: evita erro de base_score quando o dataset ficar com 1-classe por sampling.
    y_mean = float(np.mean(ytr)) if ytr.size else 0.5
    if not (0.0 < y_mean < 1.0):
        # Se acontecer, o treino não é informativo (sem negativos/positivos).
        # Não quebra: define base_score neutro e segue (ou você pode optar por "pular período").
        print(f"[xgb] AVISO: y_mean={y_mean:.6f} (1-classe). Ajustando base_score=0.5", flush=True)
        params = dict(params)
        params["base_score"] = 0.5
    elif "base_score" not in params:
        # ajuda convergência / estabilidade
        eps = 1e-6
        params = dict(params)
        params["base_score"] = float(min(1.0 - eps, max(eps, y_mean)))

    # Para GPU + hist, QuantileDMatrix reduz VRAM.
    # IMPORTANTE: o dvalid precisa referenciar o dtrain via `ref=...`.
    if use_quantile:
        mb = int(params.get("max_bin", 256))
        # garante consistência entre Booster params e QuantileDMatrix bins
        params = dict(params)
        params["max_bin"] = mb
        try:
            # Nem todas as versões expõem max_bin no construtor; por isso try/except.
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=mb)
        except TypeError:
            # fallback: usa default do QuantileDMatrix (geralmente 256) e ajusta params
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xva, label=yva)
    watch = [(dtrain, "train"), (dvalid, "val")]
    print(f"[xgb] train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,}".replace(",", "."), flush=True)
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=watch,
            early_stopping_rounds=200,
            verbose_eval=50,
        )
    except Exception as e:
        # fallback adicional: se GPU falhar em runtime (driver/VRAM), tenta CPU
        if device.startswith("cuda"):
            print(f"[xgb] treino em GPU falhou ({type(e).__name__}: {e}) -> tentando CPU", flush=True)
            params_cpu = dict(params)
            params_cpu["device"] = "cpu"
            params_cpu["tree_method"] = "hist"
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
            watch = [(dtrain, "train"), (dvalid, "val")]
            booster = xgb.train(
                params=params_cpu,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=watch,
                early_stopping_rounds=200,
                verbose_eval=50,
            )
        else:
            raise
    val_pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    val_true = batch.y[va_idx]
    calib = _fit_platt(val_pred, val_true)
    preds_cal = calib["transform"](val_pred)
    target = params.get("_target_pred_pos_frac", None)
    if target is not None:
        tau = _choose_threshold_by_target_rate(preds_cal, float(target))
    else:
        tau = _find_best_threshold(preds_cal, val_true)
    meta = {
        "calibrator": calib["params"],
        "threshold": tau,
        "best_iteration": booster.best_iteration,
    }
    return booster, meta


def _train_xgb_regressor(batch: SniperBatch, params: dict, *, y_transform: str = "log1p") -> tuple[xgb.Booster, dict]:
    tr_idx, va_idx = _split_train_val(batch)
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Dataset muito pequeno para treinar")
    device = str(params.get("device", "cpu")).lower()
    use_quantile = device.startswith("cuda")
    Xtr = batch.X[tr_idx]
    ytr0 = batch.y[tr_idx].astype(np.float32, copy=False)
    Xva = batch.X[va_idx]
    yva0 = batch.y[va_idx].astype(np.float32, copy=False)

    def _apply_transform(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float32)
        yy = np.maximum(0.0, yy)
        if y_transform == "none":
            return yy
        if y_transform == "log1p":
            return np.log1p(yy)
        raise ValueError(f"y_transform inválido: {y_transform}")

    ytr = _apply_transform(ytr0)
    yva = _apply_transform(yva0)

    if use_quantile:
        mb = int(params.get("max_bin", 256))
        params = dict(params)
        params["max_bin"] = mb
        try:
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, max_bin=mb)
        except TypeError:
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xva, label=yva)

    watch = [(dtrain, "train"), (dvalid, "val")]
    print(f"[xgb] (reg) train rows={len(tr_idx):,} val rows={len(va_idx):,} feats={batch.X.shape[1]:,}".replace(",", "."), flush=True)
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2500,
            evals=watch,
            early_stopping_rounds=250,
            verbose_eval=50,
        )
    except Exception as e:
        if device.startswith("cuda"):
            print(f"[xgb] treino reg em GPU falhou ({type(e).__name__}: {e}) -> tentando CPU", flush=True)
            params_cpu = dict(params)
            params_cpu["device"] = "cpu"
            params_cpu["tree_method"] = "hist"
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dvalid = xgb.DMatrix(Xva, label=yva)
            watch = [(dtrain, "train"), (dvalid, "val")]
            booster = xgb.train(
                params=params_cpu,
                dtrain=dtrain,
                num_boost_round=2500,
                evals=watch,
                early_stopping_rounds=250,
                verbose_eval=50,
            )
        else:
            raise

    pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    err = pred.astype(np.float64) - yva.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    meta = {
        "transform": y_transform,
        "best_iteration": booster.best_iteration,
        "metrics": {"rmse": rmse, "mae": mae},
    }
    return booster, meta


def _fit_platt(preds: np.ndarray, labels: np.ndarray) -> dict:
    preds = preds.reshape(-1, 1)
    labels = labels.astype(np.int32)
    if LogisticRegression is None:
        return {"params": {"type": "identity"}, "transform": lambda p: p}
    try:
        clf = LogisticRegression(max_iter=200)
        clf.fit(preds, labels)
        def _transform(x: np.ndarray) -> np.ndarray:
            return clf.predict_proba(x.reshape(-1, 1))[:, 1]
        params = {
            "type": "platt",
            "coef": float(clf.coef_[0][0]),
            "intercept": float(clf.intercept_[0]),
        }
        return {
            "params": params,
            "transform": _transform,
        }
    except Exception:
        return {"params": {"type": "identity"}, "transform": lambda p: p}


def _find_best_threshold(preds: np.ndarray, labels: np.ndarray) -> float:
    best_tau = 0.5
    best_score = -1.0
    labels = labels.astype(np.int32)
    for tau in np.linspace(0.3, 0.95, 50):
        mask = preds >= tau
        tp = int(np.sum(labels[mask] == 1))
        fp = int(np.sum(labels[mask] == 0))
        fn = int(np.sum(labels[~mask] == 1))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        score = f1 + precision
        if score > best_score:
            best_score = score
            best_tau = tau
    return float(best_tau)


def _choose_threshold_by_target_rate(preds: np.ndarray, target_pred_pos_frac: float) -> float:
    """
    Tau por quantil para obter aproximadamente target_pred_pos_frac de positivos previstos.
    Usa apenas o conjunto de validação (sem vazamento).
    """
    p = np.asarray(preds, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.5
    frac = float(target_pred_pos_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        return 0.5
    frac = float(min(0.20, max(1e-4, frac)))  # 0.01% .. 20%
    tau = float(np.quantile(p, 1.0 - frac))
    return float(min(0.95, max(0.05, tau)))


def _next_run_dir(base: Path, prefix: str = "wf_") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.glob(f"{prefix}*") if p.is_dir()]
    max_idx = 0
    for p in existing:
        try:
            idx = int(p.name.replace(prefix, ""))
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    run_dir = base / f"{prefix}{max_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_model(booster: xgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    try:
        ubj = path.with_suffix(".ubj")
        booster.save_model(str(ubj))
    except Exception:
        pass


def train_sniper_models(cfg: TrainConfig | None = None) -> Path:
    cfg = cfg or TrainConfig()
    symbols = _select_symbols(cfg)
    print(f"[sniper-train] símbolos={len(symbols)} (max_symbols={cfg.max_symbols})", flush=True)
    run_dir = _next_run_dir(SAVE_ROOT)
    entry_params = dict(DEFAULT_ENTRY_PARAMS if cfg.entry_params is None else cfg.entry_params)
    danger_params = dict(DEFAULT_DANGER_PARAMS if cfg.danger_params is None else cfg.danger_params)
    exit_params = dict(DEFAULT_EXIT_PARAMS if getattr(cfg, "exit_params", None) is None else cfg.exit_params)
    # threshold estável (sem env vars): target rate no VAL
    entry_params["_target_pred_pos_frac"] = float(getattr(cfg, "entry_target_pred_pos_frac", 0.0) or 0.0)
    exit_params["_target_pred_pos_frac"] = float(getattr(cfg, "exit_target_pred_pos_frac", 0.0) or 0.0)

    cache_map = None
    if bool(getattr(cfg, "use_feature_cache", True)):
        print(
            f"[sniper-train] cache: garantindo features+labels 1x (total_days={int(cfg.total_days)})",
            flush=True,
        )
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(cfg.total_days),
            contract=cfg.contract,
            flags=GLOBAL_FLAGS_FULL,
            # Se você pediu histórico completo (total_days<=0), garanta que o cache também foi feito assim.
            strict_total_days=(int(cfg.total_days) <= 0),
        )
        # se algum símbolo falhar no cache, removemos aqui para evitar erro nos períodos
        symbols = [s for s in symbols if s in cache_map]
        print(f"[sniper-train] cache pronto: símbolos_ok={len(symbols)}", flush=True)
        if not symbols:
            raise RuntimeError("Nenhum símbolo restou após gerar cache (ver logs [cache])")

    for tail in cfg.offsets_days:
        print(f"[sniper-train] período T-{tail}d", flush=True)
        if bool(getattr(cfg, "use_feature_cache", True)):
            pack = prepare_sniper_dataset_from_cache(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=cfg.contract,
                cache_map=cache_map,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                danger_ratio_neg_per_pos=float(getattr(cfg, "danger_ratio_neg_per_pos", 4.0)),
                exit_ratio_neg_per_pos=float(getattr(cfg, "exit_ratio_neg_per_pos", 4.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                max_rows_danger=int(getattr(cfg, "max_rows_danger", 1_200_000)),
                max_rows_exit=int(getattr(cfg, "max_rows_exit", 1_200_000)),
                seed=1337,
            )
        else:
            pack = prepare_sniper_dataset(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=cfg.contract,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                danger_ratio_neg_per_pos=float(getattr(cfg, "danger_ratio_neg_per_pos", 4.0)),
                exit_ratio_neg_per_pos=float(getattr(cfg, "exit_ratio_neg_per_pos", 4.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                max_rows_danger=int(getattr(cfg, "max_rows_danger", 1_200_000)),
                max_rows_exit=int(getattr(cfg, "max_rows_exit", 1_200_000)),
                seed=1337,
            )
        try:
            n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            n_skip = len(pack.symbols_skipped) if getattr(pack, "symbols_skipped", None) else 0
            te = str(pd.to_datetime(pack.train_end_utc)) if getattr(pack, "train_end_utc", None) is not None else "None"
            print(f"[sniper-train] {tail}d: train_end_utc={te} | symbols_used={n_used} skipped={n_skip}", flush=True)
        except Exception:
            pass
        # proteção anti-escassez: se poucos símbolos sobreviveram nesse tail, não vale treinar.
        try:
            min_used = int(getattr(cfg, "min_symbols_used_per_period", 0) or 0)
        except Exception:
            min_used = 0
        if min_used > 0:
            try:
                n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            except Exception:
                n_used = 0
            if int(n_used) < int(min_used):
                print(f"[sniper-train] {tail}d: symbols_used={n_used} < min_symbols_used_per_period={min_used} -> pulando", flush=True)
                continue
        if pack.entry.X.size == 0 or pack.danger.X.size == 0 or pack.exit.X.size == 0:
            print(f"[sniper-train] {tail}d: dataset vazio, pulando")
            continue
        try:
            pos_e = int((pack.entry.y >= 0.5).sum())
            neg_e = int((pack.entry.y < 0.5).sum())
            pos_d = int((pack.danger.y >= 0.5).sum())
            neg_d = int((pack.danger.y < 0.5).sum())
            pos_x = int((pack.exit.y >= 0.5).sum())
            neg_x = int((pack.exit.y < 0.5).sum())
            print(
                (f"[sniper-train] dataset entry: pos={pos_e:,} neg={neg_e:,} | danger: pos={pos_d:,} neg={neg_d:,} | exit: pos={pos_x:,} neg={neg_x:,}").replace(",", "."),
                flush=True,
            )
        except Exception:
            pass

        print("[sniper-train] treinando EntryScore...", flush=True)
        entry_model, entry_meta = _train_xgb_classifier(pack.entry, entry_params)
        try:
            tmin = float(getattr(cfg, "entry_tau_min", 0.0) or 0.0)
            tmax = float(getattr(cfg, "entry_tau_max", 1.0) or 1.0)
            entry_meta["threshold"] = float(min(tmax, max(tmin, float(entry_meta["threshold"]))))
        except Exception:
            pass
        print(f"[sniper-train] EntryScore: tau={entry_meta['threshold']:.3f} best_iter={entry_meta['best_iteration']}", flush=True)

        print("[sniper-train] treinando DangerScore...", flush=True)
        danger_model, danger_meta = _train_xgb_classifier(pack.danger, danger_params)
        print(f"[sniper-train] DangerScore: tau={danger_meta['threshold']:.3f} best_iter={danger_meta['best_iteration']}", flush=True)

        print("[sniper-train] treinando ExitScore...", flush=True)
        exit_model, exit_meta = _train_xgb_classifier(pack.exit, exit_params)
        print(f"[sniper-train] ExitScore: tau={exit_meta['threshold']:.3f} best_iter={exit_meta['best_iteration']}", flush=True)

        period_dir = run_dir / f"period_{int(tail)}d"
        (period_dir / "entry_model").mkdir(parents=True, exist_ok=True)
        _save_model(entry_model, period_dir / "entry_model" / "model_entry.json")
        _save_model(danger_model, period_dir / "danger_model" / "model_danger.json")
        _save_model(exit_model, period_dir / "exit_model" / "model_exit.json")

        meta = {
            "entry": {
                "feature_cols": pack.entry.feature_cols,
                "calibration": entry_meta["calibrator"],
                "threshold": entry_meta["threshold"],
            },
            "danger": {
                "feature_cols": pack.danger.feature_cols,
                "calibration": danger_meta["calibrator"],
                "threshold": danger_meta["threshold"],
            },
            "exit": {
                "feature_cols": pack.exit.feature_cols,
                "calibration": exit_meta["calibrator"],
                "threshold": exit_meta["threshold"],
            },
            # Ponto final do treino (auditável): preferimos o valor determinístico vindo do dataflow
            # (cutoff - lookahead). Se não existir, cai no max(ts) do dataset amostrado.
            "train_end_utc": (
                str(pd.to_datetime(pack.train_end_utc)) if getattr(pack, "train_end_utc", None) is not None
                else (
                    np.datetime_as_string(
                        np.minimum(
                            np.max(pack.entry.ts) if pack.entry.ts.size else np.datetime64("NaT"),
                            np.minimum(
                                np.max(pack.danger.ts) if pack.danger.ts.size else np.datetime64("NaT"),
                                np.max(pack.exit.ts) if pack.exit.ts.size else np.datetime64("NaT"),
                            ),
                        ),
                        unit="s",
                    )
                    if (pack.entry.ts.size and pack.danger.ts.size and pack.exit.ts.size)
                    else None
                )
            ),
            "contract": {
                "tp_pct": cfg.contract.tp_min_pct,
                "sl_pct": cfg.contract.sl_pct,
                "timeout_hours": cfg.contract.timeout_hours,
                "add_spacing_pct": cfg.contract.add_spacing_pct,
                "max_adds": cfg.contract.max_adds,
                "risk_max_cycle_pct": cfg.contract.risk_max_cycle_pct,
                "danger_drop_pct": cfg.contract.danger_drop_pct,
                "danger_recovery_pct": cfg.contract.danger_recovery_pct,
                "danger_timeout_hours": cfg.contract.danger_timeout_hours,
                "danger_stabilize_recovery_pct": getattr(cfg.contract, "danger_stabilize_recovery_pct", 0.01),
                "danger_stabilize_bars": getattr(cfg.contract, "danger_stabilize_bars", 30),
            },
            "symbols": (pack.symbols_used if getattr(pack, "symbols_used", None) else pack.symbols),
            "symbols_total": int(len(pack.symbols) if getattr(pack, "symbols", None) else 0),
            "symbols_used": (pack.symbols_used or None),
            "symbols_skipped": (pack.symbols_skipped or None),
        }
        (period_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sniper-train] {tail}d salvo em {period_dir}")

    return run_dir


if __name__ == "__main__":
    run = train_sniper_models()
    print(f"✓ modelos salvos em: {run}")

