# -*- coding: utf-8 -*-
"""
Calcula importancia de features do modelo de entry (XGBoost).
 - Mostra importancia nativa do modelo (gain/weight).
 - Calcula permutacao (queda em AP) em uma amostra do cache.

Uso (exemplo):
  python scripts/feature_importance_entry.py --run-dir D:/astra/models_sniper/crypto/wf_002 --period period_0d --max-rows 20000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tradebot":
            return p
    return here.parents[2]


def _default_cache_dir() -> Path:
    repo = _repo_root()
    cand = repo.parent / "cache_sniper" / "features_pf_1m"
    if cand.exists():
        return cand
    return repo / "cache_sniper" / "features_pf_1m"


def _load_meta(run_dir: Path, period: str) -> dict:
    meta_path = run_dir / period / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json nao encontrado: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_feature_cols(meta: dict) -> List[str]:
    entry = meta.get("entry") or {}
    cols = entry.get("feature_cols") or []
    if not cols:
        raise RuntimeError("feature_cols vazio em meta.json")
    return list(cols)


def _load_model(run_dir: Path, period: str, model_subdir: str) -> xgb.Booster:
    model_path = run_dir / period / model_subdir / "model_entry.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model_entry.json nao encontrado: {model_path}")
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    return bst


def _list_cache_files(cache_dir: Path) -> List[Path]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir nao encontrado: {cache_dir}")
    files = sorted(cache_dir.glob("*.parquet")) + sorted(cache_dir.glob("*.pkl"))
    if not files:
        raise RuntimeError(f"Nenhum cache encontrado em {cache_dir}")
    return files


def _load_cache_sample(
    cache_dir: Path,
    feature_cols: List[str],
    label_col: str,
    max_rows: int,
    max_files: int,
    seed: int,
) -> pd.DataFrame:
    files = _list_cache_files(cache_dir)
    rnd = random.Random(seed)
    rnd.shuffle(files)
    frames: List[pd.DataFrame] = []
    total = 0
    for p in files[:max_files]:
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_pickle(p)
        except Exception:
            continue
        if label_col not in df.columns:
            continue
        cols_keep = [c for c in feature_cols if c in df.columns]
        if not cols_keep:
            continue
        sub = df[cols_keep + [label_col]].copy()
        sub = sub[sub[label_col].notna()]
        if sub.empty:
            continue
        frames.append(sub)
        total += len(sub)
        if total >= max_rows:
            break
    if not frames:
        raise RuntimeError("Nenhum dado valido encontrado no cache")
    out = pd.concat(frames, axis=0, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Implementacao simples de AP (average precision)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = 0.0
    fp = 0.0
    precisions: List[float] = []
    for i in range(y_true.size):
        if y_true[i] >= 0.5:
            tp += 1.0
        else:
            fp += 1.0
        if y_true[i] >= 0.5:
            precisions.append(tp / (tp + fp))
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def permutation_importance(
    bst: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    seed: int,
) -> List[Tuple[str, float]]:
    rng = np.random.default_rng(seed)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    base_pred = bst.predict(dmat)
    base_ap = average_precision(y, base_pred)

    importances: List[Tuple[str, float]] = []
    for j, name in enumerate(feature_cols):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        dmat_p = xgb.DMatrix(Xp, feature_names=feature_cols)
        pred_p = bst.predict(dmat_p)
        ap_p = average_precision(y, pred_p)
        importances.append((name, base_ap - ap_p))
    return importances


def main() -> None:
    # Defaults (rode sem flags)
    defaults = {
        "run_dir": "D:/astra/models_sniper/crypto/wf_002",
        "period": "period_0d",
        "model_subdir": "entry_model",
        "cache_dir": "",
        "label_col": "sniper_entry_label",
        "max_rows": 20000,
        "max_files": 200,
        "seed": 42,
        "out": "data/feature_importance_entry.csv",
        "out_xlsx": "data/feature_importance_entry.xlsx",
    }

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run-dir")
    ap.add_argument("--period")
    ap.add_argument("--model-subdir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--label-col")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--out-xlsx")
    args = ap.parse_args()

    # aplica defaults se flags nao forem passadas
    run_dir = args.run_dir or defaults["run_dir"]
    period = args.period or defaults["period"]
    model_subdir = args.model_subdir or defaults["model_subdir"]
    cache_dir_arg = args.cache_dir or defaults["cache_dir"]
    label_col = args.label_col or defaults["label_col"]
    max_rows = args.max_rows if args.max_rows is not None else defaults["max_rows"]
    max_files = args.max_files if args.max_files is not None else defaults["max_files"]
    seed = args.seed if args.seed is not None else defaults["seed"]
    out_csv = args.out or defaults["out"]
    out_xlsx = args.out_xlsx or defaults["out_xlsx"]

    run_dir = Path(run_dir)
    cache_dir = Path(cache_dir_arg) if cache_dir_arg else _default_cache_dir()

    meta = _load_meta(run_dir, period)
    feature_cols = _load_feature_cols(meta)
    bst = _load_model(run_dir, period, model_subdir)

    df = _load_cache_sample(cache_dir, feature_cols, label_col, max_rows, max_files, seed)
    cols_keep = [c for c in feature_cols if c in df.columns]
    if not cols_keep:
        raise RuntimeError("Nenhuma feature do modelo encontrada no cache")
    X = df[cols_keep].to_numpy(dtype=np.float32, copy=False)
    y = df[label_col].to_numpy(dtype=np.float32, copy=False)

    # Importancia nativa (gain/weight). Mapeia f0..fN para nomes reais.
    def _map_score(score_dict: dict) -> dict:
        if not score_dict:
            return {}
        out = {}
        for k, v in score_dict.items():
            if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feature_cols):
                    out[feature_cols[idx]] = float(v)
            else:
                out[str(k)] = float(v)
        return out

    try:
        gain_raw = bst.get_score(importance_type="gain")
    except Exception:
        gain_raw = {}
    try:
        weight_raw = bst.get_score(importance_type="weight")
    except Exception:
        weight_raw = {}
    gain = _map_score(gain_raw)
    weight = _map_score(weight_raw)
    gain_rows = [(f, float(gain.get(f, 0.0)), float(weight.get(f, 0.0))) for f in feature_cols]

    # Permutacao
    perm = permutation_importance(bst, X, y, cols_keep, seed)

    # Merge
    perm_map = dict(perm)
    rows = []
    for f, g, w in gain_rows:
        rows.append(
            {
                "feature": f,
                "gain": g,
                "weight": w,
                "perm_ap_drop": float(perm_map.get(f, 0.0)),
            }
        )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows).sort_values("perm_ap_drop", ascending=False)
    df_out.to_csv(out_path, index=False)
    # tenta salvar XLSX (se openpyxl estiver disponivel)
    try:
        out_xlsx_path = Path(out_xlsx)
        out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_excel(out_xlsx_path, index=False)
    except Exception:
        out_xlsx_path = None

    # Print top 20
    top = sorted(rows, key=lambda r: r["perm_ap_drop"], reverse=True)[:20]
    print("Top 20 (perm AP drop):")
    for r in top:
        print(f"- {r['feature']}: perm_ap_drop={r['perm_ap_drop']:.6f} gain={r['gain']:.3f}")
    print(f"\nSalvo em: {out_path}")
    if out_xlsx_path:
        print(f"Salvo em: {out_xlsx_path}")


if __name__ == "__main__":
    main()
