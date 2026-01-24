# -*- coding: utf-8 -*-
"""
Treinamento dos modelos Sniper (EntryScore) usando walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

try:
    from .sniper_dataflow import (
        prepare_sniper_dataset,
        prepare_sniper_dataset_from_cache,
        ensure_feature_cache,
        GLOBAL_FLAGS_FULL,
        SniperBatch,
    )
except Exception:
    import sys
    from pathlib import Path

    _HERE = Path(__file__).resolve()
    for _p in _HERE.parents:
        if _p.name.lower() == "modules":
            _sp = str(_p)
            if _sp not in sys.path:
                sys.path.insert(0, _sp)
            break
    from train.sniper_dataflow import (
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
    from utils.paths import models_root_for_asset as _models_root_for_asset  # type: ignore

    SAVE_ROOT = _models_root_for_asset
except Exception:
    # fallback antigo
    SAVE_ROOT = None
DEFAULT_SYMBOLS_FILE = Path(__file__).resolve().parents[1] / "top_market_cap.txt"
DEFAULT_STOCKS_SYMBOLS_FILE = Path(__file__).resolve().parents[2] / "data" / "generated" / "tiingo_universe_seed.txt"

try:
    from config.symbols import load_market_caps
except Exception:
    # fallback para execução fora de pacote
    from config.symbols import load_market_caps  # type: ignore[import]


@dataclass
class TrainConfig:
    total_days: int = 365 * 5
    offsets_days: Sequence[int] = (180, 360, 540, 720, 900, 1_080, 1_260, 1_440, 1_620, 1_800)
    mcap_min_usd: float = 100_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    symbols_file: Path | None = None
    # 0 = sem limite (usa todas elegíveis por market cap)
    max_symbols: int = 0
    # Se o período ficar com poucos símbolos com dados válidos, pula o treino (evita modelo escasso).
    min_symbols_used_per_period: int = 30
    entry_params: dict = None
    contract: TradeContract = DEFAULT_TRADE_CONTRACT
    # dataset sizing (VRAM/RAM)
    max_rows_entry: int = 2_000_000
    entry_ratio_neg_per_pos: float = 6.0
    use_feature_cache: bool = True
    # modo/asset
    asset_class: str = "crypto"
    # lista explícita de símbolos (prioritária a symbols_file/top_market_cap)
    symbols: Sequence[str] | None = None
    # flags de features custom (senão escolhe default por asset)
    feature_flags: dict | None = None
    # onde salvar/ler cache de features (senão usa padrão por asset)
    feature_cache_dir: Path | None = None
    # Thresholds são definidos manualmente em config/thresholds.py (sem calibrar no treino).


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


def _default_symbols_file(asset_class: str) -> Path:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        return DEFAULT_STOCKS_SYMBOLS_FILE
    return DEFAULT_SYMBOLS_FILE


def _select_symbols(cfg: TrainConfig) -> List[str]:
    # prioridade: lista explícita
    if getattr(cfg, "symbols", None):
        raw = [str(s).strip().upper() for s in cfg.symbols if str(s).strip()]
        uniq: List[str] = []
        for s in raw:
            if s and s not in uniq:
                uniq.append(s)
        max_symbols = int(getattr(cfg, "max_symbols", 0) or 0)
        if max_symbols > 0:
            uniq = uniq[:max_symbols]
        return uniq

    asset = str(getattr(cfg, "asset_class", "crypto") or "crypto").lower()
    symbols_file = cfg.symbols_file or _default_symbols_file(asset)
    max_symbols = int(getattr(cfg, "max_symbols", 0) or 0)

    if asset == "stocks":
        if not symbols_file.exists():
            raise RuntimeError(f"Lista de símbolos para stocks não encontrada: {symbols_file}")
        lines = [line.strip().upper() for line in symbols_file.read_text(encoding="utf-8").splitlines()]
        syms: List[str] = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            tok = line.split(",", 1)[0].split(":", 1)[0].strip()
            if tok and tok not in syms:
                syms.append(tok)
            if max_symbols > 0 and len(syms) >= max_symbols:
                break
        if not syms:
            raise RuntimeError("Nenhum símbolo encontrado no arquivo de stocks")
        return syms

    cap_map = load_market_caps(symbols_file)
    lo = min(cfg.mcap_min_usd, cfg.mcap_max_usd)
    hi = max(cfg.mcap_min_usd, cfg.mcap_max_usd)
    # ordena por market cap desc
    ranked = sorted(cap_map.items(), key=lambda kv: kv[1], reverse=True)
    symbols: List[str] = []
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
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else None
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else None

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
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, max_bin=mb)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva, ref=dtrain, max_bin=mb)
        except TypeError:
            # fallback: usa default do QuantileDMatrix (geralmente 256) e ajusta params
            params["max_bin"] = 256
            dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.QuantileDMatrix(Xva, label=yva, weight=wva, ref=dtrain)
        except Exception as e:
            print(f"[xgb] QuantileDMatrix falhou ({type(e).__name__}: {e}) -> fallback DMatrix", flush=True)
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
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
    meta = {
        "calibrator": calib["params"],
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
    wtr = batch.w[tr_idx] if getattr(batch, "w", None) is not None else None
    wva = batch.w[va_idx] if getattr(batch, "w", None) is not None else None

    def _apply_transform(y: np.ndarray) -> np.ndarray:
        yy = np.asarray(y, dtype=np.float32)
        if y_transform == "none":
            return yy
        if y_transform == "log1p":
            yy = np.maximum(0.0, yy)
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
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
    else:
        dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
        dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)

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
            dtrain = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
            dvalid = xgb.DMatrix(Xva, label=yva, weight=wva)
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
    asset_class = str(getattr(cfg, "asset_class", "crypto") or "crypto").lower()

    symbols = _select_symbols(cfg)
    print(
        f"[sniper-train] símbolos={len(symbols)} asset={asset_class} (max_symbols={cfg.max_symbols})",
        flush=True,
    )
    if SAVE_ROOT is None:
        base_root = Path(__file__).resolve().parents[2].parent / "models_sniper"
        save_root = base_root / asset_class
    else:
        save_root = SAVE_ROOT(asset_class)
    run_dir = _next_run_dir(save_root)
    entry_params = dict(DEFAULT_ENTRY_PARAMS if cfg.entry_params is None else cfg.entry_params)

    def _default_feature_flags() -> dict:
        if asset_class == "stocks":
            try:
                from stocks.prepare_features_stocks import FLAGS_STOCKS  # type: ignore

                return dict(FLAGS_STOCKS)
            except Exception:
                pass
        return dict(GLOBAL_FLAGS_FULL)

    feature_flags = dict(cfg.feature_flags or _default_feature_flags())
    contract = cfg.contract

    cache_map = None
    if bool(getattr(cfg, "use_feature_cache", True)):
        print(
            f"[sniper-train] cache: garantindo features+labels 1x (total_days={int(cfg.total_days)})",
            flush=True,
        )
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(cfg.total_days),
            contract=contract,
            flags=feature_flags,
            cache_dir=getattr(cfg, "feature_cache_dir", None),
            asset_class=asset_class,
            # Se você pediu histórico completo (total_days<=0), garanta que o cache também foi feito assim.
            strict_total_days=(int(cfg.total_days) <= 0),
        )
        # se algum símbolo falhar no cache, removemos aqui para evitar erro nos períodos
        symbols = [s for s in symbols if s in cache_map]
        print(f"[sniper-train] cache pronto: símbolos_ok={len(symbols)}", flush=True)
        if not symbols:
            raise RuntimeError("Nenhum símbolo restou após gerar cache (ver logs [cache])")

    for tail in cfg.offsets_days:
        print(f"[sniper-train] periodo T-{tail}d", flush=True)
        if bool(getattr(cfg, "use_feature_cache", True)):
            pack = prepare_sniper_dataset_from_cache(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                cache_map=cache_map,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
            )
        else:
            pack = prepare_sniper_dataset(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
            )
        try:
            n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            n_skip = len(pack.symbols_skipped) if getattr(pack, "symbols_skipped", None) else 0
            te = str(pd.to_datetime(pack.train_end_utc)) if getattr(pack, "train_end_utc", None) is not None else "None"
            print(f"[sniper-train] {tail}d: train_end_utc={te} | symbols_used={n_used} skipped={n_skip}", flush=True)
        except Exception:
            pass
        # protecao anti-escassez: se poucos simbolos sobreviveram nesse tail, nao vale treinar.
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
        if pack.entry_mid.X.size == 0:
            print(f"[sniper-train] {tail}d: dataset vazio, pulando")
            continue
        try:
            pos_e = int((pack.entry_mid.y >= 0.5).sum())
            neg_e = int((pack.entry_mid.y < 0.5).sum())
            print((f"[sniper-train] dataset entry: pos={pos_e:,} neg={neg_e:,}").replace(",", "."), flush=True)
        except Exception:
            pass

        windows = list(getattr(cfg.contract, "entry_label_windows_minutes", []) or [])
        if not windows:
            raise RuntimeError("entry_label_windows_minutes vazio")
        entry_specs = [("mid", int(windows[0]))]
        entry_batches = {"mid": pack.entry_mid}
        entry_models: dict[str, tuple[xgb.Booster, dict]] = {}
        for name, w in entry_specs:
            b = entry_batches.get(name, pack.entry_mid)
            if b is None or b.X.size == 0:
                print(f"[sniper-train] EntryScore {name} ({w}m): dataset vazio, pulando", flush=True)
                continue
            print(f"[sniper-train] treinando EntryScore {name} ({w}m)...", flush=True)
            m, m_meta = _train_xgb_classifier(b, entry_params)
            entry_models[name] = (m, m_meta)
            print(f"[sniper-train] EntryScore {name}: best_iter={m_meta['best_iteration']}", flush=True)

        if not entry_models:
            print(f"[sniper-train] {tail}d: nenhum modelo treinado, pulando", flush=True)
            continue

        period_train_end = None
        try:
            if getattr(pack, "train_end_utc", None) is not None:
                period_train_end = pd.to_datetime(pack.train_end_utc)
            elif pack.entry_mid.ts.size:
                period_train_end = pd.to_datetime(pack.entry_mid.ts.max())
        except Exception:
            period_train_end = None

        period_dir = run_dir / f"period_{int(tail)}d"
        # salva modelo entry (mantem entry_model como alias do "mid")
        for name, w in entry_specs:
            if name not in entry_models:
                continue
            model, _m_meta = entry_models[name]
            d = period_dir / f"entry_model_{int(w)}m"
            d.mkdir(parents=True, exist_ok=True)
            _save_model(model, d / "model_entry.json")
        if "mid" in entry_models:
            (period_dir / "entry_model").mkdir(parents=True, exist_ok=True)
            _save_model(entry_models["mid"][0], period_dir / "entry_model" / "model_entry.json")
        base_name = "mid"
        base_batch = entry_batches[base_name]
        base_meta = entry_models[base_name][1]
        meta = {
            "entry": {
                "feature_cols": base_batch.feature_cols,
                "calibration": base_meta.get("calibrator", {"type": "identity"}),
            },
            # Ponto final do treino (auditavel): preferimos o valor deterministico vindo do dataflow
            # (cutoff - lookahead). Se nao existir, cai no max(ts) do dataset amostrado.
            "train_end_utc": (
                str(pd.to_datetime(period_train_end))
                if period_train_end is not None
                else (np.datetime_as_string(np.max(base_batch.ts), unit="s") if base_batch.ts.size else None)
            ),
            "symbols": (pack.symbols_used if getattr(pack, "symbols_used", None) else pack.symbols),
            "symbols_total": int(len(pack.symbols) if getattr(pack, "symbols", None) else 0),
            "symbols_used": (pack.symbols_used or None),
            "symbols_skipped": (pack.symbols_skipped or None),
        }
        # meta extra por janela
        for name, w in entry_specs:
            if name not in entry_models:
                continue
            _m, _m_meta = entry_models[name]
            meta[f"entry_{int(w)}m"] = {
                "feature_cols": entry_batches[name].feature_cols,
                "calibration": _m_meta["calibrator"],
            }
        (period_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sniper-train] {tail}d salvo em {period_dir}")

    return run_dir


if __name__ == "__main__":
    run = train_sniper_models()
    print(f"[sniper-train] modelos salvos em: {run}")

