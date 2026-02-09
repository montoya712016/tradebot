# -*- coding: utf-8 -*-
"""
Dataflow específico do Sniper (regressão):
- calcula features completos via prepare_features
- gera dataset de Entry a partir de timing_label
- concatena símbolos e entrega arrays prontos para treino
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import os
import gc
import json
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

njit = None  # type: ignore
_HAS_NUMBA = False
_NUMBA_READY = False


def _ensure_numba() -> bool:
    global njit, _HAS_NUMBA, _NUMBA_READY
    if _NUMBA_READY:
        return _HAS_NUMBA
    _NUMBA_READY = True
    if os.getenv("SNIPER_DISABLE_NUMBA", "").strip().lower() in {"1", "true", "yes"}:
        _HAS_NUMBA = False
        return False
    try:
        from numba import njit as _njit  # type: ignore

        njit = _njit
        _HAS_NUMBA = True
    except Exception:
        njit = None  # type: ignore
        _HAS_NUMBA = False
    return _HAS_NUMBA

try:
    from prepare_features.prepare_features import (
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features import pf_config as cfg
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
    from prepare_features.labels import apply_timing_regression_labels
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    from prepare_features.prepare_features import (  # type: ignore[import]
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features import pf_config as cfg  # type: ignore[import]
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m  # type: ignore[import]
    from prepare_features.labels import apply_timing_regression_labels  # type: ignore[import]
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]


GLOBAL_FLAGS_FULL = build_flags(enable=FEATURE_KEYS, label=True)

def _stock_helper_bundle() -> tuple[dict | None, object | None, object | None, object | None, object | None]:
    try:
        from stocks import prepare_features_stocks as pfs  # type: ignore

        return (
            dict(getattr(pfs, "FLAGS_STOCKS", {})),
            getattr(pfs, "_apply_stock_windows", None),
            getattr(pfs, "_insert_daily_breaks", None),
            getattr(pfs, "_maybe_fill_day_start", None),
            getattr(pfs, "_add_time_features", None),
            getattr(pfs, "CFG_STOCK_WINDOWS", None),
        )
    except Exception:
        return None, None, None, None, None, None


def _default_flags_for_asset(asset_class: str) -> dict:
    asset = str(asset_class or "crypto").lower()
    if asset == "stocks":
        flags, _, _, _, _ = _stock_helper_bundle()
        if flags:
            return dict(flags)
    return dict(GLOBAL_FLAGS_FULL)


_STOCK_WINDOWS_APPLIED = False
_STOCK_WINDOWS_LOGGED = False


DROP_COL_PREFIXES = (
    "exit_",
    # IMPORTANT: evitar vazamento de label/simulação futura
    "sniper_",
    "label_",
    "timing_",  # timing regression labels
)
DROP_COLS_EXACT = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "gap_after",
    # colunas de suporte
    "ts",
    "sym_id",
}


def _frozen_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    cols = ["open", "high", "low", "close"]
    if df.empty or any(c not in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    cur = df[cols]
    prev = cur.shift(1)
    return (cur == prev).all(axis=1)


def _list_feature_columns(df: pd.DataFrame, mask: np.ndarray | None = None) -> List[str]:
    allow = getattr(cfg, "FEATURE_ALLOWLIST", None)
    allow_set = set(allow) if isinstance(allow, (list, tuple)) and allow else None
    cols: List[str] = []
    for c in df.columns:
        if c in DROP_COLS_EXACT:
            continue
        if any(c.startswith(pref) for pref in DROP_COL_PREFIXES):
            continue
        if allow_set is not None and c not in allow_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # evita colunas sempre NaN que zerariam o dataset no dropna
            if mask is None:
                has_any = df[c].notna().any()
            else:
                # usa máscara para não materializar entry_df completo
                col = df[c].to_numpy()
                if col.shape[0] != mask.shape[0]:
                    has_any = df[c].notna().any()
                else:
                    has_any = bool(np.any(pd.notna(col) & mask))
            if not has_any:
                continue
            cols.append(c)
    return cols


def _to_numpy(
    df: pd.DataFrame,
    feat_cols: Sequence[str],
    label_col: str,
    weight_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converte DataFrame para arrays numéricos usados no treino.
    Espera `label_col` presente e `weight_col` opcional.
    """
    if df.empty:
        return (
            np.empty((0, len(feat_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=np.int32),
        )
    X = df[list(feat_cols)].to_numpy(dtype=np.float32, copy=True)
    y = df[label_col].to_numpy(dtype=np.float32, copy=False)
    if weight_col and (weight_col in df.columns):
        w = df[weight_col].to_numpy(dtype=np.float32, copy=False)
    else:
        w = np.ones((df.shape[0],), dtype=np.float32)
    try:
        ts = pd.to_datetime(df.index).to_numpy(dtype="datetime64[ns]")
    except Exception:
        ts = np.array([], dtype="datetime64[ns]")
    if "sym_id" in df.columns:
        sym_id = df["sym_id"].to_numpy(dtype=np.int32, copy=False)
    else:
        sym_id = np.zeros((df.shape[0],), dtype=np.int32)
    return X, y, w, ts, sym_id


def _regression_target_from_close_numba(close: np.ndarray, window_bars: int) -> np.ndarray:
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan
    if window_bars <= 0:
        window_bars = 1
    if n == 0:
        return out
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for i in range(n):
        prefix[i + 1] = prefix[i] + close[i]
    for i in range(n):
        j0 = i + 1
        j1 = i + 1 + window_bars
        if j1 <= n:
            denom = close[i]
            if denom != 0.0 and np.isfinite(denom):
                mean = (prefix[j1] - prefix[j0]) / float(window_bars)
                out[i] = np.float32(mean / denom - 1.0)
    return out


_REG_TARGET_NUMBA = None


def _regression_target_from_close(close: np.ndarray, window_bars: int) -> np.ndarray:
    global _REG_TARGET_NUMBA
    if _ensure_numba():
        if _REG_TARGET_NUMBA is None and njit is not None:
            _REG_TARGET_NUMBA = njit(cache=True)(_regression_target_from_close_numba)
        if _REG_TARGET_NUMBA is not None:
            close_arr = np.asarray(close, dtype=np.float64)
            return _REG_TARGET_NUMBA(close_arr, int(window_bars))
    if window_bars <= 0:
        window_bars = 1
    s = pd.Series(close)
    fut = s.shift(-1).rolling(int(window_bars), min_periods=int(window_bars)).mean()
    y = (fut / s) - 1.0
    return y.to_numpy(dtype=np.float32, copy=False)


def _sample_indices_regression(
    y: np.ndarray,
    valid_mask: np.ndarray,
    bins: Sequence[float] | None,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if valid_mask.size == 0 or (not bool(np.any(valid_mask))):
        return np.empty((0,), dtype=np.int64)
    base_idx = np.flatnonzero(valid_mask)
    if max_rows <= 0 or base_idx.size <= max_rows:
        return base_idx
    yv = y[base_idx]
    edges = list(bins or [])
    edges = sorted(float(x) for x in edges)
    edges = [-np.inf] + edges + [np.inf]
    bins_n = max(1, len(edges) - 1)
    target_per_bin = max(1, int(round(max_rows / bins_n)))
    keep_idx: list[int] = []
    for i in range(bins_n):
        lo = edges[i]
        hi = edges[i + 1]
        sel = (yv >= lo) & (yv < hi)
        idx_bin = base_idx[sel]
        if idx_bin.size <= target_per_bin:
            keep_idx.extend(idx_bin.tolist())
        else:
            pick = rng.choice(idx_bin, size=target_per_bin, replace=False)
            keep_idx.extend(pick.tolist())
    if len(keep_idx) > max_rows:
        keep_idx = rng.choice(np.array(keep_idx), size=max_rows, replace=False).tolist()
    keep = np.array(sorted(set(keep_idx)), dtype=np.int64)
    return keep


@dataclass
class SniperBatch:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    ts: np.ndarray
    sym_id: np.ndarray
    feature_cols: List[str]


@dataclass
class SniperDataPack:
    entry: SniperBatch
    contract: TradeContract
    symbols: List[str]
    # símbolos efetivamente usados no pack (após skips)
    symbols_used: List[str] | None = None
    symbols_skipped: List[str] | None = None
    # timestamp final (UTC) do dataset de treino usado para este período (para walk-forward auditável)
    train_end_utc: pd.Timestamp | None = None
    entry_map: dict[str, SniperBatch] | None = None


def _repo_root() -> Path:
    # Mantém compat com layout antigo/novo:
    # queremos que cache_sniper fique no WORKSPACE (pai do repo_root),
    # e não dentro de modules/.
    try:
        from utils.paths import workspace_root  # type: ignore

        return workspace_root()
    except Exception:
        # fallback: heurística por parents (antigo: parents[2] era tradebot/)
        p = Path(__file__).resolve()
        for up in range(2, 8):
            try:
                cand = p.parents[up]
            except Exception:
                break
            if (cand / "models_sniper").exists() or (cand / "cache_sniper").exists():
                return cand
        return p.parents[2]


def _cache_dir(asset_class: str | None = None) -> Path:
    v = os.getenv("SNIPER_FEATURE_CACHE_DIR", "").strip()
    if v:
        return Path(v).expanduser().resolve()
    asset = str(asset_class or "crypto").lower()
    try:
        from utils.paths import feature_cache_root  # type: ignore

        base = feature_cache_root()
    except Exception:
        try:
            from utils.paths import feature_cache_root  # type: ignore[import]

            base = feature_cache_root()
        except Exception:
            # fallback extremo (mantém compat)
            base = _repo_root() / "cache_sniper" / "features_pf_1m"
    if asset and asset != "crypto":
        return base / asset
    return base


def _cache_refresh() -> bool:
    return _env_bool("SNIPER_CACHE_REFRESH", default=False)


def _cache_format() -> str:
    v = os.getenv("SNIPER_FEATURE_CACHE_FORMAT", "parquet").strip().lower()
    if v in {"parquet", "pq"}:
        return "parquet"
    if v in {"pickle", "pkl"}:
        return "pickle"
    return "parquet"


def _symbol_cache_paths(symbol: str, cache_dir: Path, fmt: str) -> Tuple[Path, Path]:
    safe = symbol.replace("/", "_").replace(":", "_")
    if fmt == "pickle":
        data_path = cache_dir / f"{safe}.pkl"
    else:
        data_path = cache_dir / f"{safe}.parquet"
    meta_path = cache_dir / f"{safe}.meta.json"
    return data_path, meta_path


def _try_downcast_df(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    # reduz bastante RAM/disco sem quebrar as labels
    if copy:
        df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df.loc[:, c] = df[c].astype(np.float32, copy=False)
    return df


def _load_feature_cache(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_parquet(path)

def _is_valid_cache_file(path: Path) -> bool:
    """
    Validação rápida para evitar ler parquet/pickle corrompido:
    - parquet: checa magic bytes "PAR1" no header e no footer
    - pickle: checa apenas tamanho > 0 (validação completa é custosa)
    """
    try:
        if not path.exists():
            return False
        if path.stat().st_size < 16:
            return False
        suf = path.suffix.lower()
        if suf == ".parquet":
            with path.open("rb") as f:
                head = f.read(4)
                if head != b"PAR1":
                    return False
                try:
                    f.seek(-4, os.SEEK_END)
                except Exception:
                    return False
                tail = f.read(4)
                return tail == b"PAR1"
        # pickle: no mínimo, arquivo não-vazio
        return True
    except Exception:
        return False


def _save_feature_cache(df: pd.DataFrame, path: Path, *, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        df.to_pickle(tmp)
        tmp.replace(path)
        return path
    try:
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        df.to_parquet(tmp, index=True)
        tmp.replace(path)
        return path
    except Exception:
        # fallback automático se parquet não estiver disponível no ambiente
        pkl = path.with_suffix(".pkl")
        tmp = pkl.with_suffix(pkl.suffix + f".tmp.{uuid.uuid4().hex}")
        df.to_pickle(tmp)
        tmp.replace(pkl)
        return pkl


def _build_stock_features(df_all: pd.DataFrame, flags: Dict[str, bool], contract: TradeContract) -> pd.DataFrame:
    flags_local = dict(flags)
    stock_flags, apply_windows, insert_breaks, fill_day_start, add_time_features, stock_windows = _stock_helper_bundle()
    global _STOCK_WINDOWS_APPLIED, _STOCK_WINDOWS_LOGGED
    # Ajusta janelas defaults dos indicadores de stocks (permite override por ENV) apenas 1x.
    if apply_windows and (not _STOCK_WINDOWS_APPLIED):
        try:
            apply_windows()
            _STOCK_WINDOWS_APPLIED = True
        except Exception:
            pass
    # Opcional: logar janelas ativas para auditoria (1x)
    if (not _STOCK_WINDOWS_LOGGED):
        try:
            if stock_windows and isinstance(stock_windows, dict):
                print("[stocks] CFG windows: " + ", ".join(f"{k}={v}" for k, v in stock_windows.items()), flush=True)
            _STOCK_WINDOWS_LOGGED = True
        except Exception:
            _STOCK_WINDOWS_LOGGED = True
    try:
        from prepare_features.features import make_features  # type: ignore
    except Exception:
        from prepare_features.features import make_features  # type: ignore

    df_day = insert_breaks(df_all) if insert_breaks else df_all.copy()
    quiet = bool(flags_local.get("_quiet") or flags_local.get("quiet"))
    make_features(df_day, flags_local, verbose=not quiet)
    new_cols = [c for c in df_day.columns if c not in df_all.columns]
    if new_cols:
        df_all[new_cols] = df_day.loc[df_all.index, new_cols].to_numpy()
        if fill_day_start:
            try:
                fill_day_start(df_all, new_cols)
            except Exception:
                pass
    if bool(flags_local.get("label", True)):
        candle_sec = int(getattr(contract, "timeframe_sec", 60) or 60)
        apply_timing_regression_labels(
            df_all,
            candle_sec=candle_sec,
        )
    if add_time_features:
        try:
            add_time_features(df_all)
        except Exception:
            pass
    return df_all


def _build_features_for_symbol(
    symbol: str,
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract,
    flags: Dict[str, bool],
    asset_class: str = "crypto",
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Carrega OHLC e calcula features/labels respeitando o asset_class.
    Retorna (df_pf, timings parciais).
    """
    asset = str(asset_class or "crypto").lower()
    t0 = time.perf_counter()
    if asset == "stocks":
        try:
            from prepare_features.data_stocks import load_ohlc_1m_series_stock  # type: ignore
        except Exception:
            raise
        raw = load_ohlc_1m_series_stock(symbol, int(total_days), remove_tail_days=0)
    else:
        raw = load_ohlc_1m_series(symbol, int(total_days), remove_tail_days=0)
    t_load = time.perf_counter()
    if raw.empty:
        raise RuntimeError("sem dados 1m no intervalo solicitado")

    df_all = to_ohlc_from_1m(raw, 60)
    if remove_tail_days > 0:
        cutoff = df_all.index[-1] - pd.Timedelta(days=int(remove_tail_days))
        df_all = df_all[df_all.index < cutoff]
    if df_all.empty or int(len(df_all)) < 500:
        raise RuntimeError(f"sem OHLC suficiente (rows={len(df_all)})")
    if df_all[["open", "high", "low", "close"]].isna().all(axis=None):
        raise RuntimeError("OHLC inválido (todos NaN)")
    t_ohlc = time.perf_counter()

    if asset == "stocks":
        df_pf = _build_stock_features(df_all, flags, contract)
    else:
        df_pf = pf_run(
            df_all,
            flags=flags,
            plot=False,
            trade_contract=contract,
        )
    t_feat = time.perf_counter()
    df_pf = _try_downcast_df(df_pf, copy=False)
    timings = {
        "load_s": float(t_load - t0),
        "ohlc_s": float(t_ohlc - t_load),
        "features_s": float(t_feat - t_ohlc),
    }
    return df_pf, timings


def ensure_feature_cache(
    symbols: Sequence[str],
    *,
    total_days: int,
    contract: TradeContract,
    flags: Dict[str, bool],
    cache_dir: Path | None = None,
    refresh: bool | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    # Se True, o cache só conta como "hit" se a meta indicar que foi construído
    # com `total_days` compatível com o solicitado (inclui o caso total_days<=0 => histórico completo).
    strict_total_days: bool = False,
    asset_class: str = "crypto",
    abort_ram_pct: float = 85.0,
) -> Dict[str, Path]:
    """
    Garante que existe um cache de features+labels Sniper por símbolo (computado 1x).
    Retorna mapa symbol -> caminho do arquivo cache.
    """
    cache_dir = cache_dir or _cache_dir(asset_class)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fmt = _cache_format()
    refresh = _cache_refresh() if refresh is None else bool(refresh)
    flags_run = dict(flags)
    # evita quebrar barra de progresso com logs de features; pode ser desativado via env
    log_timings = _env_bool("SNIPER_TIMINGS", default=False)
    want_timings = _env_bool("SNIPER_FEATURE_TIMINGS", default=False) or log_timings
    flags_run["_quiet"] = False if want_timings else True
    if want_timings:
        flags_run["_feature_timings"] = True
        os.environ.setdefault("PF_LOG_SUMMARY", "1")

    out: Dict[str, Path] = {}
    symbols = list(symbols)
    hits: List[str] = []
    to_build: List[str] = []
    skipped: List[str] = []
    timings: List[Dict[str, float | int | str]] = []
    t_lock = threading.Lock()

    for sym in symbols:
        data_path, meta_path = _symbol_cache_paths(sym, cache_dir, fmt)
        if data_path.exists() and (not refresh) and _is_valid_cache_file(data_path):
            if not meta_path.exists():
                try:
                    if data_path.suffix.lower() in {".pkl", ".pickle"}:
                        df_meta = pd.read_pickle(data_path)
                        real_fmt = "pickle"
                    else:
                        df_meta = pd.read_parquet(data_path)
                        real_fmt = "parquet"
                    idx = pd.to_datetime(df_meta.index, errors="coerce")
                    if isinstance(idx, pd.DatetimeIndex) and (idx.tz is not None):
                        idx = idx.tz_convert(None)
                    idx = idx[idx.notna()]
                    if len(idx) > 0:
                        start_ts = pd.Timestamp(idx.min()).tz_localize(None).to_pydatetime().isoformat()
                        end_ts = pd.Timestamp(idx.max()).tz_localize(None).to_pydatetime().isoformat()
                        total_days_meta = int(max(0, ((idx.max() - idx.min()) / pd.Timedelta(days=1))))
                        meta = {
                            "symbol": sym,
                            "rows": int(len(df_meta)),
                            "start_ts_utc": str(start_ts),
                            "end_ts_utc": str(end_ts),
                            "candle_sec": 60,
                            "total_days": int(total_days_meta),
                            "format": real_fmt,
                            "path": str(data_path),
                            "asset_class": asset_class,
                        }
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        out[sym] = data_path
                        hits.append(sym)
                        continue
                except Exception:
                    pass

        if data_path.exists() and meta_path.exists() and (not refresh) and _is_valid_cache_file(data_path):
            ok_hit = True
            if bool(strict_total_days):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta_days = meta.get("total_days", None)
                    req_days = int(total_days)
                    if meta_days is None:
                        ok_hit = False
                    else:
                        meta_days_i = int(meta_days)
                        if req_days <= 0:
                            # "máximo possível": cache deve ter sido construído com <=0 também
                            ok_hit = meta_days_i <= 0
                        else:
                            # Compat: meta_days<=0 significa "histórico completo".
                            # Nesse caso, ele satisfaz QUALQUER req_days positivo.
                            if meta_days_i <= 0:
                                ok_hit = True
                            else:
                                ok_hit = meta_days_i >= req_days
                except Exception:
                    ok_hit = False

            if ok_hit:
                out[sym] = data_path
                hits.append(sym)
                continue

        # miss / rebuild
        if True:
            # se estiver inválido/corrompido, limpa para forçar rebuild
            if data_path.exists() and (not _is_valid_cache_file(data_path)):
                try:
                    data_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            if meta_path.exists() and refresh:
                # refresh explícito: apaga meta (força rebuild completo)
                try:
                    meta_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            to_build.append(sym)

    print(
        f"[cache] features sniper: dir={cache_dir} fmt={fmt} refresh={bool(refresh)} asset={asset_class} | total={len(symbols)} hit={len(hits)} build={len(to_build)}",
        flush=True,
    )

    def _fmt_eta_s(seconds: float) -> str:
        s = int(max(0.0, float(seconds)))
        if s < 60:
            return f"{s:02d}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m:02d}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h:d}h{m:02d}m"

    def _bar(done: int, total: int, width: int = 26) -> str:
        total = max(1, int(total))
        done = int(max(0, min(done, total)))
        n = int(round(width * (done / total)))
        return ("#" * n) + ("-" * (width - n))

    def _build_one(sym: str) -> tuple[str, Path | None, str | None]:
        # retorna (sym, path|None, err|None)
        # Check RAM antes de processar
        if abort_ram_pct > 0 and psutil is not None:
            try:
                ram_used = float(psutil.virtual_memory().percent)
                if ram_used >= abort_ram_pct:
                    raise RuntimeError(f"RAM guard: {ram_used:.1f}% >= {abort_ram_pct:.1f}%")
            except RuntimeError:
                raise
            except Exception:
                pass
        data_path, meta_path = _symbol_cache_paths(sym, cache_dir, fmt)
        try:
            t0 = time.perf_counter()
            df_pf, t_stats = _build_features_for_symbol(
                sym,
                total_days=int(total_days),
                remove_tail_days=0,
                contract=contract,
                flags=flags_run,
                asset_class=asset_class,
            )
            t_pf = time.perf_counter()
            real_path = _save_feature_cache(df_pf, data_path, fmt=fmt)
            t_save = time.perf_counter()

            real_fmt = "pickle" if real_path.suffix.lower() in {".pkl", ".pickle"} else "parquet"
            meta = {
                "symbol": sym,
                "rows": int(len(df_pf)),
                "start_ts_utc": str(pd.Timestamp(df_pf.index.min()).tz_localize(None).to_pydatetime().isoformat()),
                "end_ts_utc": str(pd.Timestamp(df_pf.index.max()).tz_localize(None).to_pydatetime().isoformat()),
                "candle_sec": 60,
                "total_days": int(total_days),
                "format": real_fmt,
                "path": str(real_path),
                "asset_class": asset_class,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            with t_lock:
                row = {
                    "symbol": sym,
                    "rows": int(len(df_pf)),
                    "load_s": float(t_stats.get("load_s", 0.0)),
                    "ohlc_s": float(t_stats.get("ohlc_s", 0.0)),
                    "features_s": float(t_stats.get("features_s", 0.0)),
                    "save_s": float(t_save - t_pf),
                    "total_s": float(t_save - t0),
                }
                timings.append(row)
                if log_timings:
                    print(
                        "[cache][timing] "
                        f"{sym} rows={row['rows']} "
                        f"load={row['load_s']:.2f}s ohlc={row['ohlc_s']:.2f}s "
                        f"feat={row['features_s']:.2f}s save={row['save_s']:.2f}s total={row['total_s']:.2f}s",
                        flush=True,
                    )

            del df_pf
            gc.collect()
            return sym, real_path, None
        except Exception as e:
            try:
                del df_pf  # type: ignore[name-defined]
            except Exception:
                pass
            gc.collect()
            return sym, None, f"{type(e).__name__}: {e}"

    t_build0 = time.perf_counter()
    last_len = 0
    n_total = int(len(to_build))
    done = 0

    progress_every_s = 0.0
    try:
        v = os.getenv("SNIPER_CACHE_PROGRESS_EVERY_S", "").strip()
        progress_every_s = float(v) if v else 5.0
    except Exception:
        progress_every_s = 5.0
    last_progress_ts = 0.0

    def _print_progress(current_sym: str) -> None:
        nonlocal last_len
        nonlocal last_progress_ts
        elapsed = time.perf_counter() - t_build0
        avg = elapsed / max(1, done)
        eta = avg * max(0, n_total - done)
        pct = 100.0 * done / max(1, n_total)
        line = f"[cache] [{_bar(done, n_total)}] {done:>3}/{n_total:<3} {pct:5.1f}% ETA {_fmt_eta_s(eta)} | {current_sym}"
        if sys.stderr.isatty():
            sys.stderr.write("\r" + line + (" " * max(0, last_len - len(line))))
            sys.stderr.flush()
        else:
            now = time.perf_counter()
            if now - last_progress_ts >= max(1.0, progress_every_s):
                print(line, flush=True)
                last_progress_ts = now
        last_len = max(last_len, len(line))

    if parallel and n_total > 0:
        if max_workers is not None and int(max_workers) > 0:
            mw = int(max_workers)
        else:
            env_mw = os.getenv("SNIPER_CACHE_WORKERS", "").strip()
            if env_mw:
                try:
                    mw = int(env_mw)
                except Exception:
                    mw = 0
            else:
                # fallback conservador para evitar estourar RAM em historico completo
                if int(total_days) <= 0:
                    mw = min(4, int(os.cpu_count() or 4))
                else:
                    mw = min(8, int(os.cpu_count() or 8))
        mw = max(1, int(mw))
        # Nota: o gargalo aqui é misto (I/O MySQL + CPU features). Threads ajudam porque há muito I/O/espera.
        with ThreadPoolExecutor(max_workers=mw) as ex:
            futs = {ex.submit(_build_one, sym): sym for sym in to_build}
            for fut in as_completed(futs):
                sym = futs[fut]
                _print_progress(sym)
                sym, path, err = fut.result()
                if err:
                    skipped.append(sym)
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    print(f"[cache] SKIP {sym}: {err}", flush=True)
                elif path is not None:
                    out[sym] = path
                done += 1
                _print_progress(sym)
    else:
        for sym in to_build:
            _print_progress(sym)
            sym, path, err = _build_one(sym)
            if err:
                skipped.append(sym)
                sys.stderr.write("\n")
                sys.stderr.flush()
                print(f"[cache] SKIP {sym}: {err}", flush=True)
            elif path is not None:
                out[sym] = path
            done += 1
            _print_progress(sym)

    if to_build:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if timings:
        try:
            avg = {
                "load_s": 0.0,
                "ohlc_s": 0.0,
                "features_s": 0.0,
                "save_s": 0.0,
                "total_s": 0.0,
            }
            for t in timings:
                for k in avg:
                    avg[k] += float(t.get(k, 0.0))
            n_t = float(len(timings))
            for k in avg:
                avg[k] /= max(1.0, n_t)
            print(
                "[cache] avg (s): "
                f"load={avg['load_s']:.2f} ohlc={avg['ohlc_s']:.2f} "
                f"features={avg['features_s']:.2f} save={avg['save_s']:.2f} total={avg['total_s']:.2f}",
                flush=True,
            )
            if _env_bool("SNIPER_CACHE_TIMINGS", default=False):
                log_path = cache_dir / "_cache_timings.jsonl"
                with log_path.open("a", encoding="utf-8") as f:
                    for t in timings:
                        f.write(json.dumps(t, ensure_ascii=False) + "\n")
        except Exception:
            pass

    print(
        f"[cache] pronto: ok={len(out)} hit={len(hits)} built={len(to_build) - len(skipped)} skipped={len(skipped)}",
        flush=True,
    )

    return out


def _read_cache_meta_end_ts(symbol: str, cache_dir: Path) -> pd.Timestamp | None:
    fmt = _cache_format()
    _, meta_path = _symbol_cache_paths(symbol, cache_dir, fmt)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        v = meta.get("end_ts_utc", "")
        return pd.Timestamp(v) if v else None
    except Exception:
        return None


def _max_label_lookahead_bars(contract: TradeContract, candle_sec: int) -> int:
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    if windows:
        w_min = float(max(windows))
        label_bars = int(max(1, round((w_min * 60.0) / float(max(1, candle_sec)))))
    else:
        label_bars = 0
    return int(max(label_bars, contract.danger_horizon_bars(candle_sec)))


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _default_exit_margin_pct(contract: "TradeContract") -> float:
    """
    Default do Exit margin:
    - 0.2% (antigo) deixa o label_exit praticamente morto -> modelo com scores baixos.
    - valor fixo para evitar dependência de parâmetros do contrato.
    """
    return 0.008


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return bool(default)
    return v.strip().lower() not in {"0", "false", "no", "off"}

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)



def prepare_sniper_dataset(
    symbols: Sequence[str],
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract | None = None,
    entry_side: str = "reg",
    entry_reg_window_min: int = 0,
    entry_reg_weight_alpha: float = 4.0,
    entry_reg_weight_power: float = 1.2,
    entry_reg_balance_bins: Sequence[float] | None = None,
    max_rows_entry: int = 600_000,
    full_entry_pool: bool = False,
    seed: int = 42,
    feature_flags: Dict[str, bool] | None = None,
    asset_class: str = "crypto",
) -> SniperDataPack:
    contract = contract or DEFAULT_TRADE_CONTRACT
    symbols = list(symbols)
    if not symbols:
        raise RuntimeError("symbols vazio")
    flags = dict(feature_flags or _default_flags_for_asset(asset_class))

    rng = np.random.default_rng(int(seed) + int(remove_tail_days))
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    if bool(full_entry_pool) and int(max_rows_entry) <= 0:
        symbol_cap = 0

    window_min = int(entry_reg_window_min) if int(entry_reg_window_min) > 0 else 60
    preferred_name = f"ret_exp_{int(window_min)}m"
    tail_abs = float(_env_float("SNIPER_TAIL_ABS", 0.04))
    tail_mult = float(_env_float("SNIPER_TAIL_WEIGHT_MULT", 5.0))
    side = str(entry_side or "reg").strip().lower()

    entry_pool_map: dict[str, pd.DataFrame] = {preferred_name: pd.DataFrame()}
    entry_buf_map: dict[str, list[pd.DataFrame]] = {preferred_name: []}
    feat_cols_entry_map: dict[str, list[str]] = {}
    symbols_used: list[str] = []
    symbols_skipped: list[str] = []

    batch_flush_n = 8

    for sym_idx, symbol in enumerate(symbols):
        try:
            df_pf, _ = _build_features_for_symbol(
                symbol,
                total_days=int(total_days),
                remove_tail_days=int(remove_tail_days),
                contract=contract,
                flags=flags,
                asset_class=asset_class,
            )
            use_timing_weight = _env_bool("SNIPER_USE_TIMING_WEIGHT", default=True)
            timing_weight = None
            if side == "long" and "timing_label_long" in df_pf.columns:
                y = df_pf["timing_label_long"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight_long" in df_pf.columns:
                    timing_weight = df_pf["timing_weight_long"].to_numpy(copy=False).astype(np.float32)
            elif side == "short" and "timing_label_short" in df_pf.columns:
                y = df_pf["timing_label_short"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight_short" in df_pf.columns:
                    timing_weight = df_pf["timing_weight_short"].to_numpy(copy=False).astype(np.float32)
            elif "timing_label" in df_pf.columns:
                y = df_pf["timing_label"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight" in df_pf.columns:
                    timing_weight = df_pf["timing_weight"].to_numpy(copy=False).astype(np.float32)
            else:
                candle_sec = int(getattr(contract, "timeframe_sec", 60) or 60)
                window_bars = int(max(1, round((window_min * 60) / candle_sec)))
                y = _regression_target_from_close(df_pf["close"].to_numpy(copy=False), window_bars)

            valid_mask = np.isfinite(y)
            if valid_mask.size == 0 or (not bool(np.any(valid_mask))):
                symbols_skipped.append(symbol)
                continue
            feat_cols = _list_feature_columns(df_pf, mask=valid_mask)
            feat_cols_entry_map[preferred_name] = list(feat_cols)

            if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                keep_idx = np.flatnonzero(valid_mask)
            else:
                keep_idx = _sample_indices_regression(
                    y,
                    valid_mask,
                    entry_reg_balance_bins,
                    int(symbol_cap),
                    rng,
                )
            if keep_idx.size == 0:
                symbols_skipped.append(symbol)
                continue

            entry_df = df_pf.iloc[keep_idx][feat_cols].copy()
            entry_df["label_entry"] = y[keep_idx]
            if use_timing_weight and timing_weight is not None:
                entry_df["weight"] = timing_weight[keep_idx]
            else:
                entry_df["weight"] = (
                    1.0 + float(entry_reg_weight_alpha) * np.power(np.abs(y[keep_idx]), float(entry_reg_weight_power))
                ).astype(np.float32)
            label_scale = 100.0 if (side in {"long", "short"} and np.nanmax(y) > 1.0) else 1.0
            tail_abs_use = float(tail_abs) * float(label_scale)
            if tail_mult > 1.0:
                tail_mask = np.abs(entry_df["label_entry"].to_numpy(copy=False)) >= float(tail_abs_use)
                if np.any(tail_mask):
                    entry_df.loc[tail_mask, "weight"] = (
                        entry_df.loc[tail_mask, "weight"].to_numpy(copy=False) * float(tail_mult)
                    ).astype(np.float32)
            entry_df["sym_id"] = int(sym_idx)

            buf = entry_buf_map.get(preferred_name)
            if buf is not None:
                buf.append(entry_df)
                if len(buf) >= batch_flush_n:
                    pool_df = entry_pool_map.get(preferred_name)
                    combined = pd.concat(
                        ([pool_df] if pool_df is not None and not pool_df.empty else []) + buf,
                        axis=0,
                        ignore_index=False,
                    )
                    if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                        pool_df = combined
                    else:
                        y_all = combined["label_entry"].to_numpy(dtype=np.float32, copy=False)
                        vm = np.isfinite(y_all)
                        keep = _sample_indices_regression(y_all, vm, entry_reg_balance_bins, int(max_rows_entry), rng)
                        pool_df = combined.iloc[keep] if keep.size > 0 else combined.iloc[:0]
                    entry_pool_map[preferred_name] = pool_df
                    buf.clear()
                    del combined

            symbols_used.append(symbol)
        except Exception:
            symbols_skipped.append(symbol)

    # flush final buffers
    for name, buf in entry_buf_map.items():
        if not buf:
            continue
        pool_df = entry_pool_map.get(name)
        combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
        if bool(full_entry_pool) and int(max_rows_entry) <= 0:
            pool_df = combined
        else:
            y_all = combined["label_entry"].to_numpy(dtype=np.float32, copy=False)
            vm = np.isfinite(y_all)
            keep = _sample_indices_regression(y_all, vm, entry_reg_balance_bins, int(max_rows_entry), rng)
            pool_df = combined.iloc[keep] if keep.size > 0 else combined.iloc[:0]
        entry_pool_map[name] = pool_df
        buf.clear()
        del combined

    entry_batches: dict[str, SniperBatch] = {}
    for name, entry_df in entry_pool_map.items():
        entry_df = _try_downcast_df(entry_df, copy=False) if not entry_df.empty else entry_df
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        X, y, w, ts, sym_id = _to_numpy(entry_df, feat_cols, "label_entry", "weight")
        entry_batches[name] = SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))

    entry = entry_batches.get(preferred_name) or next(iter(entry_batches.values()))
    return SniperDataPack(
        entry=entry,
        contract=contract,
        symbols=list(symbols),
        symbols_used=symbols_used or None,
        symbols_skipped=symbols_skipped or None,
        train_end_utc=None,
        entry_map=entry_batches,
    )



def prepare_sniper_dataset_from_cache(
    symbols: Sequence[str],
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract | None = None,
    cache_map: Dict[str, Path] | None = None,
    entry_side: str = "reg",
    entry_reg_window_min: int = 0,
    entry_reg_weight_alpha: float = 4.0,
    entry_reg_weight_power: float = 1.2,
    entry_reg_balance_bins: Sequence[float] | None = None,
    max_rows_entry: int = 2_000_000,
    full_entry_pool: bool = False,
    seed: int = 42,
    feature_flags: Dict[str, bool] | None = None,
    asset_class: str = "crypto",
    parallel: bool = True,
    max_workers: int | None = None,
) -> SniperDataPack:
    """
    Calcula features 1x por s?mbolo (cache em disco) e, no walk-forward, apenas
    recorta o final por `remove_tail_days` antes de montar o dataset de entry.
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    flags = dict(feature_flags or _default_flags_for_asset(asset_class))
    symbols = list(symbols)
    if not symbols:
        raise RuntimeError("symbols vazio")

    cache_dir = _cache_dir(asset_class)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_map is None:
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(total_days),
            contract=contract,
            flags=flags,
            asset_class=asset_class,
        )
    symbols = [s for s in symbols if s in cache_map]
    if not symbols:
        raise RuntimeError("Nenhum simbolo com cache valido (todos falharam/foram pulados)")

    rng = np.random.default_rng(int(seed) + int(remove_tail_days))
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    if bool(full_entry_pool) and int(max_rows_entry) <= 0:
        symbol_cap = 0

    window_min = int(entry_reg_window_min) if int(entry_reg_window_min) > 0 else 60
    preferred_name = f"ret_exp_{int(window_min)}m"
    tail_abs = float(_env_float("SNIPER_TAIL_ABS", 0.04))
    tail_mult = float(_env_float("SNIPER_TAIL_WEIGHT_MULT", 5.0))
    side = str(entry_side or "reg").strip().lower()

    entry_pool_map: dict[str, pd.DataFrame] = {preferred_name: pd.DataFrame()}
    entry_buf_map: dict[str, list[pd.DataFrame]] = {preferred_name: []}
    feat_cols_entry_map: dict[str, list[str]] = {}
    symbols_used: list[str] = []
    symbols_skipped: list[str] = []

    def _process_symbol(sym: str, sym_idx: int) -> tuple[str, pd.DataFrame | None, list[str] | None]:
        try:
            data_path = cache_map.get(sym)
            if data_path is None or (not Path(data_path).exists()):
                return sym, None, None
            df = _load_feature_cache(Path(data_path))
            if df.empty:
                return sym, None, None
            if remove_tail_days > 0:
                cutoff = df.index[-1] - pd.Timedelta(days=int(remove_tail_days))
                df = df[df.index < cutoff]
            if df.empty:
                return sym, None, None
            df = _try_downcast_df(df, copy=False)
            frozen_mask = _frozen_ohlc_mask(df) if str(asset_class or "crypto").lower() == "stocks" else None

            use_timing_weight = _env_bool("SNIPER_USE_TIMING_WEIGHT", default=True)
            timing_weight_arr = None
            if side == "long" and "timing_label_long" in df.columns:
                y = df["timing_label_long"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight_long" in df.columns:
                    timing_weight_arr = df["timing_weight_long"].to_numpy(copy=False).astype(np.float32)
            elif side == "short" and "timing_label_short" in df.columns:
                y = df["timing_label_short"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight_short" in df.columns:
                    timing_weight_arr = df["timing_weight_short"].to_numpy(copy=False).astype(np.float32)
            elif "timing_label" in df.columns:
                y = df["timing_label"].to_numpy(copy=False).astype(np.float32)
                if use_timing_weight and "timing_weight" in df.columns:
                    timing_weight_arr = df["timing_weight"].to_numpy(copy=False).astype(np.float32)
            else:
                candle_sec = int(getattr(contract, "timeframe_sec", 60) or 60)
                window_bars = int(max(1, round((window_min * 60) / candle_sec)))
                y = _regression_target_from_close(df["close"].to_numpy(copy=False), window_bars)

            entry_mask = np.isfinite(y)
            if frozen_mask is not None:
                try:
                    fm = frozen_mask.to_numpy(dtype=bool, copy=False)
                    if fm.shape[0] == entry_mask.shape[0]:
                        entry_mask = entry_mask & (~fm)
                except Exception:
                    pass
            if entry_mask.size == 0 or (not bool(np.any(entry_mask))):
                return sym, None, None

            feat_cols = _list_feature_columns(df, mask=entry_mask)
            if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                keep_idx = np.flatnonzero(entry_mask)
            else:
                rng_sym = np.random.default_rng(int(seed) + int(remove_tail_days) + (sym_idx + 1) * 17)
                keep_idx = _sample_indices_regression(
                    y,
                    entry_mask,
                    entry_reg_balance_bins,
                    int(symbol_cap),
                    rng_sym,
                )
            if keep_idx.size == 0:
                return sym, None, None

            entry_df = df.iloc[keep_idx][feat_cols].copy()
            entry_df["label_entry"] = y[keep_idx]
            if use_timing_weight and timing_weight_arr is not None:
                entry_df["weight"] = timing_weight_arr[keep_idx]
            else:
                entry_df["weight"] = (
                    1.0 + float(entry_reg_weight_alpha) * np.power(np.abs(y[keep_idx]), float(entry_reg_weight_power))
                ).astype(np.float32)
            label_scale = 100.0 if (side in {"long", "short"} and np.nanmax(y) > 1.0) else 1.0
            tail_abs_use = float(tail_abs) * float(label_scale)
            if tail_mult > 1.0:
                tail_mask = np.abs(entry_df["label_entry"].to_numpy(copy=False)) >= float(tail_abs_use)
                if np.any(tail_mask):
                    entry_df.loc[tail_mask, "weight"] = (
                        entry_df.loc[tail_mask, "weight"].to_numpy(copy=False) * float(tail_mult)
                    ).astype(np.float32)
            entry_df["sym_id"] = int(sym_idx)
            return sym, entry_df, feat_cols
        except Exception:
            return sym, None, None

    if parallel and len(symbols) > 1:
        if max_workers is not None and int(max_workers) > 0:
            mw = int(max_workers)
        else:
            env_mw = os.getenv("SNIPER_DATASET_WORKERS", "").strip()
            mw = int(env_mw) if env_mw else min(8, int(os.cpu_count() or 4))
        mw = max(1, mw)
        with ThreadPoolExecutor(max_workers=mw) as ex:
            futures = {ex.submit(_process_symbol, sym, i): sym for i, sym in enumerate(symbols)}
            for fut in as_completed(futures):
                sym, entry_df, feat_cols = fut.result()
                if entry_df is None or entry_df.empty:
                    symbols_skipped.append(sym)
                    continue
                symbols_used.append(sym)
                if feat_cols:
                    feat_cols_entry_map[preferred_name] = list(feat_cols)
                buf = entry_buf_map.get(preferred_name)
                if buf is not None:
                    buf.append(entry_df)
                    if len(buf) >= 8:
                        pool_df = entry_pool_map.get(preferred_name)
                        combined = pd.concat(
                            ([pool_df] if pool_df is not None and not pool_df.empty else []) + buf,
                            axis=0,
                            ignore_index=False,
                        )
                        if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                            pool_df = combined
                        else:
                            y_all = combined["label_entry"].to_numpy(dtype=np.float32, copy=False)
                            vm = np.isfinite(y_all)
                            keep = _sample_indices_regression(y_all, vm, entry_reg_balance_bins, int(max_rows_entry), rng)
                            pool_df = combined.iloc[keep] if keep.size > 0 else combined.iloc[:0]
                        entry_pool_map[preferred_name] = pool_df
                        buf.clear()
                        del combined
    else:
        for sym_idx, sym in enumerate(symbols):
            sym, entry_df, feat_cols = _process_symbol(sym, sym_idx)
            if entry_df is None or entry_df.empty:
                symbols_skipped.append(sym)
                continue
            symbols_used.append(sym)
            if feat_cols:
                feat_cols_entry_map[preferred_name] = list(feat_cols)
            buf = entry_buf_map.get(preferred_name)
            if buf is not None:
                buf.append(entry_df)
                if len(buf) >= 8:
                    pool_df = entry_pool_map.get(preferred_name)
                    combined = pd.concat(
                        ([pool_df] if pool_df is not None and not pool_df.empty else []) + buf,
                        axis=0,
                        ignore_index=False,
                    )
                    if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                        pool_df = combined
                    else:
                        y_all = combined["label_entry"].to_numpy(dtype=np.float32, copy=False)
                        vm = np.isfinite(y_all)
                        keep = _sample_indices_regression(y_all, vm, entry_reg_balance_bins, int(max_rows_entry), rng)
                        pool_df = combined.iloc[keep] if keep.size > 0 else combined.iloc[:0]
                    entry_pool_map[preferred_name] = pool_df
                    buf.clear()
                    del combined

    # flush final buffers
    for name, buf in entry_buf_map.items():
        if not buf:
            continue
        pool_df = entry_pool_map.get(name)
        combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
        if bool(full_entry_pool) and int(max_rows_entry) <= 0:
            pool_df = combined
        else:
            y_all = combined["label_entry"].to_numpy(dtype=np.float32, copy=False)
            vm = np.isfinite(y_all)
            keep = _sample_indices_regression(y_all, vm, entry_reg_balance_bins, int(max_rows_entry), rng)
            pool_df = combined.iloc[keep] if keep.size > 0 else combined.iloc[:0]
        entry_pool_map[name] = pool_df
        buf.clear()
        del combined

    entry_batches: dict[str, SniperBatch] = {}
    for name, entry_df in entry_pool_map.items():
        entry_df = _try_downcast_df(entry_df, copy=False) if not entry_df.empty else entry_df
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        X, y, w, ts, sym_id = _to_numpy(entry_df, feat_cols, "label_entry", "weight")
        entry_batches[name] = SniperBatch(X=X, y=y, w=w, ts=ts, sym_id=sym_id, feature_cols=list(feat_cols))

    entry = entry_batches.get(preferred_name) or next(iter(entry_batches.values()))
    return SniperDataPack(
        entry=entry,
        contract=contract,
        symbols=list(symbols),
        symbols_used=symbols_used or None,
        symbols_skipped=symbols_skipped or None,
        train_end_utc=None,
        entry_map=entry_batches,
    )
