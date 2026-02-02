# -*- coding: utf-8 -*-
"""
Treinamento dos modelos Sniper (EntryScore) usando walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return bool(default)
    return v.strip().lower() not in {"0", "false", "no", "off"}


def _enable_vt_mode(stream) -> bool:
    """
    Try to enable ANSI/VT processing on Windows so in-place progress works.
    Returns True if VT is enabled (or not needed), False otherwise.
    """
    if os.name != "nt":
        return True
    try:
        import msvcrt
        import ctypes

        handle = msvcrt.get_osfhandle(stream.fileno())
        kernel32 = ctypes.windll.kernel32
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VT = 0x0004
        new_mode = mode.value | ENABLE_VT
        if not kernel32.SetConsoleMode(handle, new_mode):
            return False
        return True
    except Exception:
        return False

try:
    from .sniper_dataflow import (
        prepare_sniper_dataset,
        prepare_sniper_dataset_from_cache,
        ensure_feature_cache,
        GLOBAL_FLAGS_FULL,
        SniperBatch,
        entry_label_specs,
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
        entry_label_specs,
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
    entry_pool_full: bool = False
    entry_pool_dir: Path | None = None
    entry_pool_prefiltered: bool = True
    entry_pool_ratio_neg_per_pos: float = 10.0
    # quais labels de entry usar ("long", "short" ou ambos)
    entry_label_sides: Sequence[str] = ("long",)
    use_feature_cache: bool = True
    # modo/asset
    asset_class: str = "crypto"
    # lista explícita de símbolos (prioritária a symbols_file/top_market_cap)
    symbols: Sequence[str] | None = None
    # flags de features custom (senão escolhe default por asset)
    feature_flags: dict | None = None
    # onde salvar/ler cache de features (senão usa padrão por asset)
    feature_cache_dir: Path | None = None
    # força reutilizar diretório de run (para retomar wf_XXX)
    run_dir: Path | None = None
    # Thresholds são definidos manualmente em config/thresholds.py (sem calibrar no treino).


DEFAULT_ENTRY_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "eta": 0.05,
    "max_depth": 7,
    "min_child_weight": 5.0,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Observação: QuantileDMatrix costuma usar 256 bins por padrão; manter consistente evita
    # erro "Inconsistent max_bin (512 vs 256)" e permite treinar na GPU.
    "max_bin": 256,
    "lambda": 2.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "device": "cuda:0",
}

# Pool completo (pré-treino): negativos já são filtrados no dataflow (pos x alpha).
POOL_NEG_PER_POS = 6
POOL_NEG_MAX = 0  # 0 = sem limite absoluto; controla apenas por neg_per_pos
POOL_OTHER_POS_WEIGHT_BONUS = 1.0


def _filter_pool_batch(b: SniperBatch, *, neg_per_pos: int, neg_max: int, rng: np.random.Generator) -> SniperBatch:
    y = b.y.astype(np.float32, copy=False)
    pos_idx = np.flatnonzero(y >= 0.5)
    neg_idx = np.flatnonzero(y < 0.5)
    if neg_idx.size == 0:
        keep = pos_idx
    else:
        if pos_idx.size == 0:
            # sem positivos: mantém os negativos mais "pesados" (capado se neg_max>0)
            n_keep = int(min(neg_idx.size, max(1, int(neg_max)))) if neg_max > 0 else int(neg_idx.size)
            w = b.w[neg_idx].astype(np.float64, copy=False)
            if np.all(~np.isfinite(w)) or float(np.nanmax(w)) <= 0.0:
                keep_neg = neg_idx[:n_keep]
            else:
                order = np.argsort(w, kind="mergesort")[::-1]
                keep_neg = neg_idx[order[:n_keep]]
            keep = keep_neg
        else:
            target_neg = int(min(neg_idx.size, max(1, pos_idx.size * int(neg_per_pos))))
            if neg_max > 0:
                target_neg = min(target_neg, int(neg_max))
            w = b.w[neg_idx].astype(np.float64, copy=False)
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            if float(w.sum()) <= 0.0:
                keep_neg = rng.choice(neg_idx, size=target_neg, replace=False)
            else:
                p = w / float(w.sum())
                keep_neg = rng.choice(neg_idx, size=target_neg, replace=False, p=p)
            keep = np.concatenate([pos_idx, keep_neg])
    keep.sort()
    return SniperBatch(
        X=b.X[keep],
        y=b.y[keep],
        w=b.w[keep],
        ts=b.ts[keep],
        sym_id=b.sym_id[keep],
        feature_cols=b.feature_cols,
    )


def _save_entry_pool_symbol(pool_dir: Path, symbol: str, entry_batches: dict[str, SniperBatch]) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    base_batch = None
    for b in entry_batches.values():
        if b is not None and b.X.size > 0:
            base_batch = b
            break
    if base_batch is None:
        return
    df = pd.DataFrame(base_batch.X, columns=list(base_batch.feature_cols))
    df.index = pd.to_datetime(base_batch.ts)
    for name, b in entry_batches.items():
        if b is None or b.X.size == 0:
            continue
        df[f"label_{name}"] = b.y.astype(np.uint8, copy=False)
        df[f"weight_{name}"] = b.w.astype(np.float32, copy=False)
    out_dir = pool_dir / "entry_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{str(symbol).upper()}.parquet"
    df.to_parquet(out_path, index=True)



def _write_pool_meta(pool_dir: Path, windows: Sequence[int], max_ts: pd.Timestamp | None) -> None:
    meta = {
        "windows": {f"w{int(w)}": int(w) for w in windows},
        "layout": "per_symbol_combined",
        "max_ts": None if max_ts is None else str(pd.to_datetime(max_ts)),
    }
    (pool_dir / "pool_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")



def _load_entry_pool(pool_dir: Path) -> tuple[dict[str, list[Path]], pd.Timestamp | None, str]:
    meta_path = pool_dir / "pool_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"pool_meta.json n??o encontrado em {pool_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layout = str(meta.get("layout") or "per_symbol_combined")
    out: dict[str, list[Path]] = {}
    if layout == "per_symbol_combined":
        d = pool_dir / "entry_pool"
        out["_combined"] = sorted(d.glob("*.parquet")) if d.exists() else []
    else:
        for name in (meta.get("windows") or {}).keys():
            d = pool_dir / f"entry_pool_{name}"
            if not d.exists():
                continue
            out[name] = sorted(d.glob("*.parquet"))
    max_ts = meta.get("max_ts")
    return out, (pd.to_datetime(max_ts) if max_ts else None), layout


def _default_symbols_file(asset_class: str) -> Path:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        return DEFAULT_STOCKS_SYMBOLS_FILE
    return DEFAULT_SYMBOLS_FILE


def _sample_indices(
    labels: np.ndarray,
    ratio_neg_per_pos: float,
    max_rows: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    if labels.size == 0:
        return np.empty((0,), dtype=np.int64)
    y = labels
    pos_idx = np.flatnonzero(y >= 0.5)
    neg_idx = np.flatnonzero(y < 0.5)
    if pos_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    ratio = float(max(0.0, ratio_neg_per_pos))
    max_rows = int(max_rows)
    pos_keep_n = int(pos_idx.size)
    if max_rows > 0:
        max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
        pos_keep_n = min(pos_keep_n, max_pos)
    if pos_keep_n < pos_idx.size:
        pos_keep = rng.choice(pos_idx, size=pos_keep_n, replace=False)
    else:
        pos_keep = pos_idx
    neg_target = int(round(pos_keep_n * ratio))
    if max_rows > 0:
        max_neg_allowed = max_rows - pos_keep_n
        if max_neg_allowed < neg_target:
            neg_target = max(0, max_neg_allowed)
    if neg_target <= 0 or neg_idx.size == 0:
        keep_local = pos_keep
    else:
        if neg_target >= neg_idx.size:
            neg_keep = neg_idx
        else:
            neg_keep = rng.choice(neg_idx, size=neg_target, replace=False)
        keep_local = np.concatenate([pos_keep, neg_keep])
    return np.sort(keep_local)


def _sample_indices_weighted(
    labels: np.ndarray,
    weights: np.ndarray | None,
    ratio_neg_per_pos: float,
    max_rows: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    if labels.size == 0:
        return np.empty((0,), dtype=np.int64)
    y = labels
    pos_idx = np.flatnonzero(y >= 0.5)
    neg_idx = np.flatnonzero(y < 0.5)
    if pos_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    ratio = float(max(0.0, ratio_neg_per_pos))
    max_rows = int(max_rows)
    max_pos = pos_idx.size
    if max_rows > 0:
        max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
    if pos_idx.size <= max_pos:
        pos_keep = pos_idx
    else:
        if weights is not None and weights.size == labels.size:
            wpos = weights[pos_idx].astype(np.float64, copy=False)
            wpos = np.nan_to_num(wpos, nan=0.0, posinf=0.0, neginf=0.0)
            if float(wpos.sum()) > 0.0:
                p = wpos / float(wpos.sum())
                pos_keep = rng.choice(pos_idx, size=max_pos, replace=False, p=p)
            else:
                pos_keep = rng.choice(pos_idx, size=max_pos, replace=False)
        else:
            pos_keep = rng.choice(pos_idx, size=max_pos, replace=False)
    target_neg = int(pos_keep.size * ratio)
    if max_rows > 0:
        target_neg = min(target_neg, int(max_rows - pos_keep.size))
    target_neg = max(0, min(target_neg, neg_idx.size))
    if target_neg <= 0 or neg_idx.size == 0:
        keep_local = pos_keep
    else:
        if weights is not None and weights.size == labels.size:
            wneg = weights[neg_idx].astype(np.float64, copy=False)
            wneg = np.nan_to_num(wneg, nan=0.0, posinf=0.0, neginf=0.0)
            if float(wneg.sum()) > 0.0:
                p = wneg / float(wneg.sum())
                neg_keep = rng.choice(neg_idx, size=target_neg, replace=False, p=p)
            else:
                neg_keep = rng.choice(neg_idx, size=target_neg, replace=False)
        else:
            neg_keep = rng.choice(neg_idx, size=target_neg, replace=False)
        keep_local = np.concatenate([pos_keep, neg_keep])
    return np.sort(keep_local)


def _reservoir_merge(
    keys_old: np.ndarray,
    idx_old: np.ndarray,
    keys_new: np.ndarray,
    idx_new: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if keys_old.size == 0:
        if keys_new.size <= k:
            return keys_new, idx_new
        sel = np.argpartition(keys_new, k - 1)[:k]
        return keys_new[sel], idx_new[sel]
    # concat and keep best k (smallest keys)
    keys_all = np.concatenate([keys_old, keys_new])
    idx_all = np.concatenate([idx_old, idx_new])
    if keys_all.size <= k:
        return keys_all, idx_all
    sel = np.argpartition(keys_all, k - 1)[:k]
    return keys_all[sel], idx_all[sel]


def _reservoir_update(
    keys_res: np.ndarray,
    X_res: np.ndarray,
    y_res: np.ndarray,
    w_res: np.ndarray,
    ts_res: np.ndarray,
    keys_new: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    w_new: np.ndarray,
    ts_new: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0 or keys_new.size == 0:
        return keys_res, X_res, y_res, w_res, ts_res
    # if reservoir not full, just take up to k
    if keys_res.size == 0:
        if keys_new.size <= k:
            return keys_new, X_new, y_new, w_new, ts_new
        sel = np.argpartition(keys_new, k - 1)[:k]
        return keys_new[sel], X_new[sel], y_new[sel], w_new[sel], ts_new[sel]
    if keys_res.size >= k:
        # keep only candidates better than current threshold
        thr = np.partition(keys_res, k - 1)[k - 1]
        better = keys_new < thr
        if not np.any(better):
            return keys_res, X_res, y_res, w_res, ts_res
        keys_new = keys_new[better]
        X_new = X_new[better]
        y_new = y_new[better]
        w_new = w_new[better]
        ts_new = ts_new[better]
    if keys_new.size == 0:
        return keys_res, X_res, y_res, w_res, ts_res
    keys_all = np.concatenate([keys_res, keys_new])
    X_all = np.concatenate([X_res, X_new])
    y_all = np.concatenate([y_res, y_new])
    w_all = np.concatenate([w_res, w_new])
    ts_all = np.concatenate([ts_res, ts_new])
    if keys_all.size <= k:
        return keys_all, X_all, y_all, w_all, ts_all
    sel = np.argpartition(keys_all, k - 1)[:k]
    return keys_all[sel], X_all[sel], y_all[sel], w_all[sel], ts_all[sel]


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

    # scale_pos_weight desativado (mantÃ©m o comportamento original)

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
    # aucpr no melhor round (se disponível)
    try:
        best_score = float(getattr(booster, "best_score", float("nan")))
        if np.isfinite(best_score):
            print(f"[xgb] best_val_aucpr={best_score:.5f}", flush=True)
    except Exception:
        pass
    val_pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    val_true = batch.y[va_idx]
    # mÃ©tricas resumidas no fim
    try:
        tau_eval = 0.75
        pred_pos = val_pred >= float(tau_eval)
        tp = int(np.sum(pred_pos & (val_true >= 0.5)))
        fp = int(np.sum(pred_pos & (val_true < 0.5)))
        fn = int(np.sum((~pred_pos) & (val_true >= 0.5)))
        tn = int(np.sum((~pred_pos) & (val_true < 0.5)))
        precision = float(tp / max(1, (tp + fp)))
        recall = float(tp / max(1, (tp + fn)))
        f1 = float((2 * precision * recall) / max(1e-12, (precision + recall)))
        pred_rate = float(np.mean(pred_pos)) if val_true.size else 0.0
        pos_rate = float(np.mean(val_true >= 0.5)) if val_true.size else 0.0
        print(
            f"[xgb] val@tau={tau_eval:.2f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            f"pred_rate={pred_rate:.4f} pos_rate={pos_rate:.4f} (tp={tp} fp={fp} fn={fn} tn={tn})",
            flush=True,
        )
    except Exception:
        pass
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
    timing_on = _env_bool("SNIPER_TIMINGS", default=False)

    t0 = time.perf_counter()
    symbols = _select_symbols(cfg)
    if timing_on:
        print(f"[sniper-train][timing] select_symbols_s={time.perf_counter() - t0:.2f}", flush=True)
    print(
        f"[sniper-train] símbolos={len(symbols)} asset={asset_class} (max_symbols={cfg.max_symbols})",
        flush=True,
    )
    if SAVE_ROOT is None:
        base_root = Path(__file__).resolve().parents[2].parent / "models_sniper"
        save_root = base_root / asset_class
    else:
        save_root = SAVE_ROOT(asset_class)
    if getattr(cfg, "run_dir", None):
        run_dir = Path(cfg.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sniper-train] usando run_dir existente: {run_dir}", flush=True)
    else:
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
        t_cache = time.perf_counter()
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
        if timing_on:
            print(f"[sniper-train][timing] cache_build_s={time.perf_counter() - t_cache:.2f}", flush=True)
        # se algum símbolo falhar no cache, removemos aqui para evitar erro nos períodos
        symbols = [s for s in symbols if s in cache_map]
        print(f"[sniper-train] cache pronto: símbolos_ok={len(symbols)}", flush=True)
        if not symbols:
            raise RuntimeError("Nenhum símbolo restou após gerar cache (ver logs [cache])")

    windows = list(getattr(cfg.contract, "entry_label_windows_minutes", []) or [])
    if not windows:
        raise RuntimeError("entry_label_windows_minutes vazio")
    entry_specs_full = entry_label_specs(cfg.contract, entry_label_sides=getattr(cfg, "entry_label_sides", None))
    if not entry_specs_full:
        raise RuntimeError("entry_label_specs vazio")
    base_spec = entry_specs_full[0]
    base_name = str(base_spec.get("name") or "")
    base_long_name = ""
    base_short_name = ""
    for s in entry_specs_full:
        side = str(s.get("side") or "").strip().lower()
        if side == "long" and not base_long_name:
            base_long_name = str(s.get("name") or "")
        if side == "short" and not base_short_name:
            base_short_name = str(s.get("name") or "")
    if not base_long_name:
        base_long_name = base_name

    pool_dir = None
    pool_df_map: dict[str, dict[str, list[Path]]] | None = None
    pool_max_ts: dict[str, pd.Timestamp | None] | None = None
    pool_layout: dict[str, str] | None = None
    if bool(getattr(cfg, "entry_pool_full", False)):
        pool_dir = Path(cfg.entry_pool_dir) if getattr(cfg, "entry_pool_dir", None) else (run_dir / "entry_pool_full")
        sides = list(getattr(cfg, "entry_label_sides", None) or ("long",))
        pool_df_map = {}
        pool_max_ts = {}
        pool_layout = {}
        pool_ratio = float(getattr(cfg, "entry_pool_ratio_neg_per_pos", 0.0) or 0.0)
        if pool_ratio <= 0:
            pool_ratio = float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0))
        from utils.progress import ProgressPrinter
        # Tenta habilitar VT no Windows para permitir atualizaÃ§Ã£o in-place.
        vt_ok = _enable_vt_mode(sys.stderr)
        for side in sides:
            side_key = str(side or "").strip().lower() or "long"
            side_dir = pool_dir / side_key
            if (side_dir / "pool_meta.json").exists():
                print(f"[sniper-train] pool_full: usando cache {side_key} em {side_dir}", flush=True)
                df_map, max_ts, layout = _load_entry_pool(side_dir)
                pool_df_map[side_key] = df_map
                pool_max_ts[side_key] = max_ts
                pool_layout[side_key] = layout
                continue
            print(f"[sniper-train] pool_full: gerando dataset {side_key} (sem limites)...", flush=True)
            max_ts = None
            progress = ProgressPrinter(
                prefix=f"[sniper-train] pool_full {side_key}:",
                total=len(symbols),
                stream=sys.stderr,
                print_every_s=5.0,
                force_inplace=vt_ok,
            )
            for i, symbol in enumerate(symbols):
                if bool(getattr(cfg, "use_feature_cache", True)):
                    pack_sym = prepare_sniper_dataset_from_cache(
                        [symbol],
                        total_days=int(cfg.total_days),
                        remove_tail_days=0,
                        contract=contract,
                        cache_map=cache_map,
                        entry_ratio_neg_per_pos=float(pool_ratio),
                        max_rows_entry=0,
                        full_entry_pool=True,
                        seed=1337,
                        asset_class=asset_class,
                        feature_flags=feature_flags,
                        entry_label_sides=(side_key,),
                    )
                else:
                    pack_sym = prepare_sniper_dataset(
                        [symbol],
                        total_days=int(cfg.total_days),
                        remove_tail_days=0,
                        contract=contract,
                        entry_ratio_neg_per_pos=float(pool_ratio),
                        max_rows_entry=0,
                        full_entry_pool=True,
                        seed=1337,
                        asset_class=asset_class,
                        feature_flags=feature_flags,
                        entry_label_sides=(side_key,),
                    )
                entry_batches_sym = pack_sym.entry_map or {}
                if not entry_batches_sym:
                    fallback_name = base_long_name if side_key != "short" else (base_short_name or base_long_name)
                    entry_batches_sym = {fallback_name: pack_sym.entry_mid}
                if not bool(getattr(cfg, "entry_pool_prefiltered", True)):
                    rng_pool = np.random.default_rng(1337 + i)
                    entry_batches_sym = {
                        name: _filter_pool_batch(
                            b,
                            neg_per_pos=int(max(1, round(float(pool_ratio)))),
                            neg_max=POOL_NEG_MAX,
                            rng=rng_pool,
                        )
                        for name, b in entry_batches_sym.items()
                        if b is not None and b.X.size > 0
                    }
                _save_entry_pool_symbol(side_dir, symbol, entry_batches_sym)
                for b in entry_batches_sym.values():
                    if b is None or b.ts.size == 0:
                        continue
                    ts_max = pd.to_datetime(b.ts).max()
                    if max_ts is None or ts_max > max_ts:
                        max_ts = ts_max
                if (i + 1) % 2 == 0 or (i + 1) == len(symbols):
                    progress.update(i + 1)
                del pack_sym, entry_batches_sym
            progress.close()
            _write_pool_meta(side_dir, windows, max_ts)
            df_map, max_ts_loaded, layout = _load_entry_pool(side_dir)
            pool_df_map[side_key] = df_map
            pool_max_ts[side_key] = max_ts_loaded
            pool_layout[side_key] = layout
    for tail in cfg.offsets_days:
        period_dir = run_dir / f"period_{int(tail)}d"
        if (period_dir / "entry_model" / "model_entry.json").exists():
            print(f"[sniper-train] {tail}d: jÃ¡ existe ({period_dir}), pulando", flush=True)
            continue
        print(f"[sniper-train] periodo T-{tail}d", flush=True)
        if pool_df_map is None and bool(getattr(cfg, "use_feature_cache", True)):
            t_pack = time.perf_counter()
            pack = prepare_sniper_dataset_from_cache(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                cache_map=cache_map,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                full_entry_pool=bool(getattr(cfg, "entry_pool_full", False)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
                entry_label_sides=getattr(cfg, "entry_label_sides", None),
            )
            if timing_on:
                print(f"[sniper-train][timing] pack_from_cache_s={time.perf_counter() - t_pack:.2f}", flush=True)
        elif pool_df_map is None:
            t_pack = time.perf_counter()
            pack = prepare_sniper_dataset(
                symbols,
                total_days=int(cfg.total_days),
                remove_tail_days=int(tail),
                contract=contract,
                entry_ratio_neg_per_pos=float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                max_rows_entry=int(getattr(cfg, "max_rows_entry", 2_000_000)),
                full_entry_pool=bool(getattr(cfg, "entry_pool_full", False)),
                seed=1337,
                asset_class=asset_class,
                feature_flags=feature_flags,
                entry_label_sides=getattr(cfg, "entry_label_sides", None),
            )
            if timing_on:
                print(f"[sniper-train][timing] pack_build_s={time.perf_counter() - t_pack:.2f}", flush=True)
        else:
            # usa pool completo salvo em disco e filtra por data
            pack = None
        if pack is not None:
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
        if pack is not None and min_used > 0:
            try:
                n_used = len(pack.symbols_used) if getattr(pack, "symbols_used", None) else len(pack.symbols)
            except Exception:
                n_used = 0
            if int(n_used) < int(min_used):
                print(f"[sniper-train] {tail}d: symbols_used={n_used} < min_symbols_used_per_period={min_used} -> pulando", flush=True)
                continue
        entry_specs = [(str(s.get("name")), int(s.get("window") or 0)) for s in entry_specs_full]
        spec_by_name = {str(s.get("name")): s for s in entry_specs_full}
        if pool_df_map is None:
            entry_batches = pack.entry_map or {}
            if not entry_batches:
                entry_batches = {base_name: pack.entry_mid}
            base_batch = entry_batches.get(base_long_name) or entry_batches.get(base_name) or pack.entry_mid
        else:
            entry_batches = {}
            def _pool_for_name(name: str) -> tuple[dict[str, list[Path]] | None, pd.Timestamp | None, str | None]:
                spec = spec_by_name.get(name, {})
                side = str(spec.get("side") or "long").strip().lower() or "long"
                df_map = pool_df_map.get(side) if pool_df_map else None
                max_ts = pool_max_ts.get(side) if pool_max_ts else None
                layout = pool_layout.get(side) if pool_layout else None
                return df_map, max_ts, layout

            def _load_pool_batch_combined(name: str, max_rows: int, ratio: float, rng: np.random.Generator) -> SniperBatch | None:
                df_map, max_ts, layout = _pool_for_name(name)
                files = df_map.get("_combined", []) if df_map else []
                if not files:
                    return None
                cutoff = None
                if max_ts is not None:
                    cutoff = max_ts - pd.Timedelta(days=int(tail))
                ambig_neg_weight_mul = 0.35
                # cache por período + janela (evita reler todos os parquets)
                cache_dir = period_dir / "entry_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_key = f"{int(tail)}d_{name}_{int(max_rows)}_{int(ratio*100)}.npz"
                cache_path = cache_dir / cache_key
                if cache_path.exists():
                    try:
                        loaded = np.load(cache_path, allow_pickle=False)
                        X = loaded["X"]
                        y = loaded["y"]
                        w = loaded["w"]
                        ts = loaded["ts"]
                        feat_cols = loaded["feat_cols"].tolist()
                        sym_id = np.zeros((X.shape[0],), dtype=np.int32)
                        return SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))
                    except Exception:
                        pass
                label_col = f"label_{name}"
                weight_col = f"weight_{name}"
                k = int(max_rows)
                ratio = float(max(0.0, ratio))
                if k <= 0:
                    # fallback: carrega tudo (pode estourar RAM)
                    parts = []
                    for fp in files:
                        df = pd.read_parquet(fp)
                        if cutoff is not None:
                            df = df.loc[pd.to_datetime(df.index) < cutoff]
                        if df.empty:
                            continue
                        parts.append(df)
                    if not parts:
                        return None
                    df_all = pd.concat(parts, axis=0, ignore_index=False)
                    base_feat_cols = [c for c in df_all.columns if not c.startswith("label_") and not c.startswith("weight_")]
                    if label_col not in df_all.columns:
                        return None
                    X = df_all[base_feat_cols].to_numpy(np.float32, copy=True)
                    y = df_all[label_col].to_numpy(dtype=np.float32, copy=False)
                    if weight_col in df_all.columns:
                        w = df_all[weight_col].to_numpy(dtype=np.float32, copy=False)
                    else:
                        w = np.ones((df_all.shape[0],), dtype=np.float32)
                    ts = pd.to_datetime(df_all.index).to_numpy(dtype="datetime64[ns]")
                    sym_id = np.zeros((df_all.shape[0],), dtype=np.int32)
                    return SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(base_feat_cols))

                max_pos = int(max(1, round(k / (1.0 + ratio))))
                max_neg = int(max(0, k - max_pos))
                keys_pos = np.empty((0,), dtype=np.float64)
                keys_neg = np.empty((0,), dtype=np.float64)
                X_pos = np.empty((0, 0), dtype=np.float32)
                X_neg = np.empty((0, 0), dtype=np.float32)
                y_pos = np.empty((0,), dtype=np.float32)
                y_neg = np.empty((0,), dtype=np.float32)
                w_pos = np.empty((0,), dtype=np.float32)
                w_neg = np.empty((0,), dtype=np.float32)
                ts_pos = np.empty((0,), dtype="datetime64[ns]")
                ts_neg = np.empty((0,), dtype="datetime64[ns]")
                base_feat_cols: list[str] | None = None

                total_files = len(files)
                cols_needed: list[str] | None = None
                # descobre colunas uma vez (evita ler tudo em cada parquet)
                try:
                    df0 = pd.read_parquet(files[0])
                    base_feat_cols = [c for c in df0.columns if not c.startswith("label_") and not c.startswith("weight_")]
                    label_cols_all = [c for c in df0.columns if c.startswith("label_")]
                    cols_needed = list(base_feat_cols) + [label_col]
                    if weight_col in df0.columns and weight_col not in cols_needed:
                        cols_needed.append(weight_col)
                    for c in label_cols_all:
                        if c not in cols_needed:
                            cols_needed.append(c)
                    del df0
                except Exception:
                    cols_needed = None
                last_report = -1

                def _load_one(fp: Path) -> pd.DataFrame:
                    df = pd.read_parquet(fp, columns=cols_needed) if cols_needed else pd.read_parquet(fp)
                    if cutoff is not None:
                        df = df.loc[pd.to_datetime(df.index) < cutoff]
                    return df

                max_workers = int(os.getenv("SNIPER_POOL_READERS", "8") or "8")
                max_workers = max(1, min(32, max_workers))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_load_one, fp): fp for fp in files}
                    for idx, fut in enumerate(as_completed(futures)):
                        df = fut.result()
                        if df.empty or label_col not in df.columns:
                            continue
                        if base_feat_cols is None:
                            base_feat_cols = [c for c in df.columns if not c.startswith("label_") and not c.startswith("weight_")]
                        X = df[base_feat_cols].to_numpy(np.float32, copy=True)
                        y = df[label_col].to_numpy(dtype=np.float32, copy=False)
                        if weight_col in df.columns:
                            w = df[weight_col].to_numpy(dtype=np.float32, copy=False)
                        else:
                            w = np.ones((df.shape[0],), dtype=np.float32)
                        ts = pd.to_datetime(df.index).to_numpy(dtype="datetime64[ns]")
                        # negativos ambiguos: mantem, mas reduz peso
                        try:
                            other_pos = np.zeros((df.shape[0],), dtype=bool)
                            for c in df.columns:
                                if c.startswith("label_") and c != label_col:
                                    other_pos |= (df[c].to_numpy(dtype=np.float32, copy=False) >= 0.5)
                            if np.any(other_pos):
                                ambig_mask = (y < 0.5) & other_pos
                                if np.any(ambig_mask):
                                    w = w.copy()
                                    w[ambig_mask] = w[ambig_mask] * ambig_neg_weight_mul
                        except Exception:
                            pass

                        if y.size == 0:
                            continue
                        # keys for reservoir (lower is better)
                        u = rng.random(size=y.shape[0], dtype=np.float64)
                        w_safe = np.maximum(w.astype(np.float64, copy=False), 1e-12)
                        keys = -np.log(u) / w_safe
                        pos_mask = y >= 0.5
                        neg_mask = ~pos_mask
                        if np.any(pos_mask):
                            keys_pos, X_pos, y_pos, w_pos, ts_pos = _reservoir_update(
                                keys_pos,
                                X_pos,
                                y_pos,
                                w_pos,
                                ts_pos,
                                keys[pos_mask],
                                X[pos_mask],
                                y[pos_mask],
                                w[pos_mask],
                                ts[pos_mask],
                                max_pos,
                            )
                        if np.any(neg_mask):
                            keys_neg, X_neg, y_neg, w_neg, ts_neg = _reservoir_update(
                                keys_neg,
                                X_neg,
                                y_neg,
                                w_neg,
                                ts_neg,
                                keys[neg_mask],
                                X[neg_mask],
                                y[neg_mask],
                                w[neg_mask],
                                ts[neg_mask],
                                max_neg,
                            )
                        if total_files:
                            pct = int((idx + 1) * 100 / total_files)
                            if pct != last_report and (pct % 5 == 0 or pct == 100):
                                print(
                                    f"[sniper-train] pool_load {name}: {pct}% ({idx + 1}/{total_files})",
                                    end="\r",
                                    flush=True,
                                )

                if base_feat_cols is None:
                    return None
                print("", flush=True)
                X_all = np.concatenate([X_pos, X_neg]) if X_pos.size or X_neg.size else np.empty((0, len(base_feat_cols)), dtype=np.float32)
                y_all = np.concatenate([y_pos, y_neg]) if y_pos.size or y_neg.size else np.empty((0,), dtype=np.float32)
                w_all = np.concatenate([w_pos, w_neg]) if w_pos.size or w_neg.size else np.empty((0,), dtype=np.float32)
                ts_all = np.concatenate([ts_pos, ts_neg]) if ts_pos.size or ts_neg.size else np.empty((0,), dtype="datetime64[ns]")
                sym_id = np.zeros((X_all.shape[0],), dtype=np.int32)
                try:
                    np.savez_compressed(
                        cache_path,
                        X=X_all,
                        y=y_all,
                        w=w_all,
                        ts=ts_all,
                        feat_cols=np.array(list(base_feat_cols), dtype=object),
                    )
                except Exception:
                    pass
                return SniperBatch(X=X_all, y=y_all, w=w_all, ts=ts_all, sym_id=sym_id, feature_cols=list(base_feat_cols))

            def _load_pool_batch_legacy(name: str) -> SniperBatch | None:
                df_map, max_ts, _layout = _pool_for_name(name)
                files = df_map.get(name, []) if df_map else []
                if not files:
                    return None
                cutoff = None
                if max_ts is not None:
                    cutoff = max_ts - pd.Timedelta(days=int(tail))
                parts = []
                for fp in files:
                    df = pd.read_parquet(fp)
                    if cutoff is not None:
                        df = df.loc[pd.to_datetime(df.index) < cutoff]
                    if df.empty:
                        continue
                    parts.append(df)
                if not parts:
                    return None
                df_f = pd.concat(parts, axis=0, ignore_index=False)
                feat_cols = [c for c in df_f.columns if c not in {"label_entry", "weight", "sym_id"}]
                X = df_f[feat_cols].to_numpy(np.float32, copy=True)
                y = df_f["label_entry"].to_numpy(dtype=np.float32, copy=False)
                w = df_f["weight"].to_numpy(dtype=np.float32, copy=False)
                ts = pd.to_datetime(df_f.index).to_numpy(dtype="datetime64[ns]")
                if "sym_id" in df_f.columns:
                    sym_id = df_f["sym_id"].to_numpy(dtype=np.int32, copy=False)
                else:
                    sym_id = np.zeros((df_f.shape[0],), dtype=np.int32)
                return SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))

            base_name = base_long_name
            _df_map_base, _max_ts_base, base_layout = _pool_for_name(base_name)
            if base_layout == "per_symbol_combined":
                base_batch = _load_pool_batch_combined(
                    base_name,
                    int(getattr(cfg, "max_rows_entry", 2_000_000)),
                    float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                    np.random.default_rng(1337),
                )
            else:
                base_batch = _load_pool_batch_legacy(base_name)
        if base_batch is None or base_batch.X.size == 0:
            print(f"[sniper-train] {tail}d: dataset vazio, pulando")
            continue
        try:
            pos_e = int((base_batch.y >= 0.5).sum())
            neg_e = int((base_batch.y < 0.5).sum())
            print((f"[sniper-train] dataset entry: pos={pos_e:,} neg={neg_e:,}").replace(",", "."), flush=True)
        except Exception:
            pass
        entry_models: dict[str, tuple[xgb.Booster, dict]] = {}
        for name, w in entry_specs:
            if pool_df_map is not None:
                _df_map_cur, _max_ts_cur, cur_layout = _pool_for_name(name)
                if cur_layout == "per_symbol_combined":
                    if name == base_name and base_batch is not None:
                        b = base_batch
                    else:
                        b = _load_pool_batch_combined(
                            name,
                            int(getattr(cfg, "max_rows_entry", 2_000_000)),
                            float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0)),
                            np.random.default_rng(1337 + int(w)),
                        )
                else:
                    b = _load_pool_batch_legacy(name)
                if b is None:
                    b = base_batch
            else:
                b = entry_batches.get(name, base_batch)
            if b is None or b.X.size == 0:
                print(f"[sniper-train] EntryScore {name} ({w}m): dataset vazio, pulando", flush=True)
                continue
            try:
                pos_m = int((b.y >= 0.5).sum())
                neg_m = int((b.y < 0.5).sum())
                print((f"[sniper-train] {name}: pos={pos_m:,} neg={neg_m:,}").replace(",", "."), flush=True)
            except Exception:
                pass
            # se pool full, amostra por modelo aqui
            if bool(getattr(cfg, "entry_pool_full", False)):
                max_rows = int(getattr(cfg, "max_rows_entry", 2_000_000))
                ratio = float(getattr(cfg, "entry_ratio_neg_per_pos", 6.0))
                if max_rows > 0 and b.y.size > max_rows:
                    rng = np.random.default_rng(1337 + int(w))
                    keep = _sample_indices_weighted(
                        b.y.astype(np.float32, copy=False),
                        b.w.astype(np.float32, copy=False),
                        ratio,
                        max_rows,
                        rng=rng,
                    )
                    if keep.size > 0:
                        b = SniperBatch(
                            X=b.X[keep],
                            y=b.y[keep],
                            w=b.w[keep],
                            ts=b.ts[keep],
                            sym_id=b.sym_id[keep],
                            feature_cols=b.feature_cols,
                        )
            print(f"[sniper-train] treinando EntryScore {name} ({w}m)...", flush=True)
            t_train = time.perf_counter()
            m, m_meta = _train_xgb_classifier(b, entry_params)
            if timing_on:
                print(
                    f"[sniper-train][timing] train_entry_{name}_{int(w)}m_s={time.perf_counter() - t_train:.2f}",
                    flush=True,
                )
            entry_models[name] = (m, m_meta)
            print(f"[sniper-train] EntryScore {name}: best_iter={m_meta['best_iteration']}", flush=True)

        if not entry_models:
            print(f"[sniper-train] {tail}d: nenhum modelo treinado, pulando", flush=True)
            continue

        period_train_end = None
        try:
            if pack is not None and getattr(pack, "train_end_utc", None) is not None:
                period_train_end = pd.to_datetime(pack.train_end_utc)
            elif base_batch.ts.size:
                period_train_end = pd.to_datetime(base_batch.ts.max())
        except Exception:
            period_train_end = None

        # salva modelo entry (mantem entry_model como alias do base_long_name)
        for name, w in entry_specs:
            if name not in entry_models:
                continue
            model, _m_meta = entry_models[name]
            spec = spec_by_name.get(name, {})
            side = str(spec.get("side") or "long").strip().lower()
            w_int = int(spec.get("window") or w)
            if side == "short":
                d = period_dir / f"entry_model_short_{w_int}m"
            else:
                d = period_dir / f"entry_model_{w_int}m"
            d.mkdir(parents=True, exist_ok=True)
            _save_model(model, d / "model_entry.json")
        if base_long_name in entry_models:
            (period_dir / "entry_model").mkdir(parents=True, exist_ok=True)
            _save_model(entry_models[base_long_name][0], period_dir / "entry_model" / "model_entry.json")
        if base_short_name and base_short_name in entry_models:
            (period_dir / "entry_model_short").mkdir(parents=True, exist_ok=True)
            _save_model(entry_models[base_short_name][0], period_dir / "entry_model_short" / "model_entry.json")
        if base_long_name in entry_batches:
            base_batch = entry_batches[base_long_name]
        base_meta = entry_models[base_long_name][1] if base_long_name in entry_models else next(iter(entry_models.values()))[1]
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
            "symbols": (
                pack.symbols_used if (pack is not None and getattr(pack, "symbols_used", None)) else (pack.symbols if pack is not None else list(symbols))
            ),
            "symbols_total": int(len(pack.symbols) if pack is not None and getattr(pack, "symbols", None) else len(symbols)),
            "symbols_used": (pack.symbols_used or None) if pack is not None else None,
            "symbols_skipped": (pack.symbols_skipped or None) if pack is not None else None,
        }
        if base_short_name and base_short_name in entry_models:
            short_meta = entry_models[base_short_name][1]
            short_feat_cols = entry_batches.get(base_short_name, base_batch).feature_cols if entry_batches else base_batch.feature_cols
            meta["entry_short"] = {
                "feature_cols": short_feat_cols,
                "calibration": short_meta.get("calibrator", {"type": "identity"}),
            }
        # meta extra por janela
        for name, w in entry_specs:
            if name not in entry_models:
                continue
            _m, _m_meta = entry_models[name]
            if name in entry_batches:
                feat_cols = entry_batches[name].feature_cols
            else:
                # fallback para o batch base carregado on-demand
                feat_cols = base_batch.feature_cols
            spec = spec_by_name.get(name, {})
            side = str(spec.get("side") or "long").strip().lower()
            w_int = int(spec.get("window") or w)
            meta_key = f"entry_short_{w_int}m" if side == "short" else f"entry_{w_int}m"
            meta[meta_key] = {"feature_cols": feat_cols, "calibration": _m_meta["calibrator"]}
        (period_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sniper-train] {tail}d salvo em {period_dir}")

    return run_dir


if __name__ == "__main__":
    run = train_sniper_models()
    print(f"[sniper-train] modelos salvos em: {run}")

