# -*- coding: utf-8 -*-
"""
Dataflow específico do Sniper:
- calcula features completos via prepare_features
- gera dataset de Entry usando build_sniper_datasets
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

import numpy as np
import pandas as pd
try:
    from utils.guarded_runner import GuardedParallelDefaults, GuardedRunner  # type: ignore
except Exception:
    from modules.utils.guarded_runner import GuardedParallelDefaults, GuardedRunner  # type: ignore[import]
try:
    from utils.progress import LineProgressPrinter  # type: ignore
except Exception:
    from modules.utils.progress import LineProgressPrinter  # type: ignore[import]
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    from prepare_features.prepare_features import (
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
    from prepare_features.sniper_dataset import build_sniper_datasets, warmup_sniper_dataset_numba
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    from prepare_features.prepare_features import (  # type: ignore[import]
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m  # type: ignore[import]
    from prepare_features.sniper_dataset import build_sniper_datasets, warmup_sniper_dataset_numba  # type: ignore[import]
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT  # type: ignore[import]


GLOBAL_FLAGS_FULL = build_flags(enable=FEATURE_KEYS, label=True)
_CACHE_GUARD = GuardedRunner(
    log_prefix="[sniper-data]",
    env_prefix="SNIPER_CACHE",
    defaults=GuardedParallelDefaults(
        max_ram_pct=78.0,
        min_free_mb=3072.0,
        per_worker_mem_mb=1024.0,
        critical_ram_pct=90.0,
        critical_min_free_mb=1536.0,
        abort_on_critical_ram=True,
        min_workers=1,
        poll_interval_s=0.3,
        log_every_s=20.0,
        throttle_sleep_s=3.0,
    ),
)
_DATASET_GUARD = GuardedRunner(
    log_prefix="[sniper-data]",
    env_prefix="SNIPER_DATASET",
    defaults=GuardedParallelDefaults(
        max_ram_pct=80.0,
        min_free_mb=3072.0,
        per_worker_mem_mb=1024.0,
        critical_ram_pct=90.0,
        critical_min_free_mb=1536.0,
        abort_on_critical_ram=True,
        min_workers=1,
        poll_interval_s=0.3,
        log_every_s=20.0,
        throttle_sleep_s=3.0,
    ),
)


def _thermal_wait(where: str) -> None:
    _CACHE_GUARD.thermal_wait(where)


def _is_guard_error_text(err: str | None) -> bool:
    return _CACHE_GUARD.is_guard_error(err)

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
    # estados internos do simulador de ciclo (nao sao sinais pre-trade)
    "cycle_",
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


def _list_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in DROP_COLS_EXACT:
            continue
        if any(c.startswith(pref) for pref in DROP_COL_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # evita colunas sempre NaN que zerariam o dataset no dropna
            if not df[c].notna().any():
                continue
            cols.append(c)
    return cols


def _entry_label_specs(contract: TradeContract) -> list[dict[str, str]]:
    # Label principal canônico (agregado OR entre janelas, quando habilitado no labels.py).
    return [
        {
            "name": "long",
            "label_col": "sniper_entry_label",
            "weight_col": "sniper_entry_weight",
            "exit_code_col": "",
            "suffix": "canonical",
        },
    ]


def _exit_target_spec(contract: TradeContract) -> dict[str, str]:
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    w = int(windows[0]) if windows else 360
    suffix = f"{w}m"
    return {
        "target_col": "sniper_exit_span_target",
        "weight_col": "sniper_exit_span_weight",
        "suffix": suffix,
    }


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
    entry_short: SniperBatch
    entry_mid: SniperBatch
    entry_long: SniperBatch
    danger: SniperBatch
    exit: SniperBatch
    contract: TradeContract
    symbols: List[str]
    # símbolos efetivamente usados no pack (após skips)
    symbols_used: List[str] | None = None
    symbols_skipped: List[str] | None = None
    # timestamp final (UTC) do dataset de treino usado para este período (para walk-forward auditável)
    train_end_utc: pd.Timestamp | None = None


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


def _timeframe_tag(candle_sec: int) -> str:
    candle_sec = int(max(1, candle_sec))
    if candle_sec % 60 == 0:
        return f"{int(candle_sec // 60)}m"
    return f"{int(candle_sec)}s"


def _infer_candle_sec_from_df(df: pd.DataFrame) -> int:
    try:
        idx = pd.DatetimeIndex(df.index)
        if len(idx) >= 2:
            dt = float((idx[1] - idx[0]).total_seconds())
            if np.isfinite(dt) and dt > 0.0:
                return int(round(dt))
    except Exception:
        pass
    return 60


def _cache_dir(asset_class: str | None = None, candle_sec: int | None = None) -> Path:
    v = os.getenv("SNIPER_FEATURE_CACHE_DIR", "").strip()
    if v:
        return Path(v).expanduser().resolve()
    asset = str(asset_class or "crypto").lower()
    candle_sec = int(max(1, candle_sec or _env_int("SNIPER_CANDLE_SEC", 60)))
    try:
        from utils.paths import cache_sniper_root  # type: ignore

        base = cache_sniper_root() / f"features_pf_{_timeframe_tag(candle_sec)}"
    except Exception:
        try:
            from utils.paths import cache_sniper_root  # type: ignore[import]

            base = cache_sniper_root() / f"features_pf_{_timeframe_tag(candle_sec)}"
        except Exception:
            # fallback extremo (mantém compat)
            base = _repo_root() / "cache_sniper" / f"features_pf_{_timeframe_tag(candle_sec)}"
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
        from prepare_features.labels import apply_trade_contract_labels  # type: ignore
    except Exception:
        from prepare_features.features import make_features  # type: ignore
        from prepare_features.labels import apply_trade_contract_labels  # type: ignore

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
        apply_trade_contract_labels(
            df_all,
            contract=contract,
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
    candle_sec = int(getattr(contract, "timeframe_sec", 60) or 60)
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

    df_all = to_ohlc_from_1m(raw, candle_sec)
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
    allow_build: bool = True,
    asset_class: str = "crypto",
    abort_ram_pct: float = 85.0,
) -> Dict[str, Path]:
    """
    Garante que existe um cache de features+labels Sniper por símbolo (computado 1x).
    Retorna mapa symbol -> caminho do arquivo cache.
    """
    desired_candle_sec = int(getattr(contract, "timeframe_sec", 60) or 60)
    cache_dir = cache_dir or _cache_dir(asset_class, desired_candle_sec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fmt = _cache_format()
    refresh = _cache_refresh() if refresh is None else bool(refresh)
    flags_run = dict(flags)
    # evita quebrar barra de progresso com logs de features; pode ser desativado via env
    want_timings = _env_bool("SNIPER_FEATURE_TIMINGS", default=False)
    flags_run["_quiet"] = False if want_timings else True
    if want_timings:
        flags_run["_feature_timings"] = True

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
                        inferred_candle_sec = _infer_candle_sec_from_df(df_meta)
                        meta = {
                            "symbol": sym,
                            "rows": int(len(df_meta)),
                            "start_ts_utc": str(start_ts),
                            "end_ts_utc": str(end_ts),
                            "candle_sec": int(inferred_candle_sec),
                            "total_days": int(total_days_meta),
                            "format": real_fmt,
                            "path": str(data_path),
                            "asset_class": asset_class,
                        }
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        if int(inferred_candle_sec) == int(desired_candle_sec):
                            out[sym] = data_path
                            hits.append(sym)
                            continue
                except Exception:
                    pass

        if data_path.exists() and meta_path.exists() and (not refresh) and _is_valid_cache_file(data_path):
            ok_hit = True
            meta = {}
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            try:
                meta_candle_sec = int(meta.get("candle_sec") or 0)
            except Exception:
                meta_candle_sec = 0
            meta_asset = str(meta.get("asset_class") or asset_class or "crypto").lower()
            if meta_candle_sec != int(desired_candle_sec):
                ok_hit = False
            if meta_asset != str(asset_class or "crypto").lower():
                ok_hit = False
            if bool(strict_total_days):
                try:
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

    if (not bool(allow_build)) and to_build:
        missing_preview = ",".join(str(x) for x in to_build[:12])
        if len(to_build) > 12:
            missing_preview += ",..."
        raise RuntimeError(
            f"feature cache missing for {len(to_build)} symbols in {cache_dir} "
            f"(timeframe={_timeframe_tag(desired_candle_sec)} allow_build=0 missing={missing_preview})"
        )

    def _build_one(sym: str) -> tuple[str, Path | None, str | None]:
        # retorna (sym, path|None, err|None)
        _thermal_wait(f"feature_cache_build:{sym}")
        if abort_ram_pct > 0 and psutil is not None:
            try:
                ram_used = float(psutil.virtual_memory().percent)
                if ram_used >= float(abort_ram_pct):
                    raise RuntimeError(f"RAM guard: {ram_used:.1f}% >= {float(abort_ram_pct):.1f}%")
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
                "candle_sec": int(desired_candle_sec),
                "total_days": int(total_days),
                "format": real_fmt,
                "path": str(real_path),
                "asset_class": asset_class,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            with t_lock:
                timings.append(
                    {
                        "symbol": sym,
                        "rows": int(len(df_pf)),
                        "load_s": float(t_stats.get("load_s", 0.0)),
                        "ohlc_s": float(t_stats.get("ohlc_s", 0.0)),
                        "features_s": float(t_stats.get("features_s", 0.0)),
                        "save_s": float(t_save - t_pf),
                        "total_s": float(t_save - t0),
                    }
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

    n_total = int(len(to_build))
    done = 0

    progress_every_s = 0.0
    try:
        v = os.getenv("SNIPER_CACHE_PROGRESS_EVERY_S", "").strip()
        progress_every_s = float(v) if v else 5.0
    except Exception:
        progress_every_s = 5.0
    prog_cache = LineProgressPrinter(
        prefix="cache",
        total=n_total,
        width=26,
        stream=sys.stderr,
        min_interval_s=max(1.0, progress_every_s),
        pad_seconds=True,
    )

    def _print_progress(current_sym: str) -> None:
        prog_cache.update(done, current=current_sym)

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
        cache_policy = _CACHE_GUARD.make_policy(
            max_ram_pct=float(os.getenv("SNIPER_CACHE_RAM_PCT", str(float(abort_ram_pct) if abort_ram_pct > 0 else 85.0)) or "85"),
        )
        print(
            f"[cache] workers={mw} ram_cap={cache_policy.max_ram_pct:.1f}% min_free_mb={cache_policy.min_free_mb:.0f} per_worker_mb={cache_policy.per_worker_mem_mb:.0f}",
            flush=True,
        )
        for sym_submitted, fut in _CACHE_GUARD.adaptive_map(
            to_build,
            _build_one,
            max_workers=mw,
            policy=cache_policy,
            task_name="feature-cache-build",
        ):
            _print_progress(sym_submitted)
            sym, path, err = fut.result()
            if err:
                if _is_guard_error_text(err):
                    skipped.append(sym)
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    raise RuntimeError(f"[cache] ABORT guard em {sym}: {err}")
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
            _thermal_wait(f"feature_cache_seq:{sym}")
            _print_progress(sym)
            sym, path, err = _build_one(sym)
            if err:
                if _is_guard_error_text(err):
                    skipped.append(sym)
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    raise RuntimeError(f"[cache] ABORT guard em {sym}: {err}")
                skipped.append(sym)
                sys.stderr.write("\n")
                sys.stderr.flush()
                print(f"[cache] SKIP {sym}: {err}", flush=True)
            elif path is not None:
                out[sym] = path
            done += 1
            _print_progress(sym)

    if to_build:
        prog_cache.close()

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
    # Novo label baseado em medias futuras usa janela independente do contrato (PF_LABEL_FUTURE_MIN).
    try:
        pf_future_min = float(_env_int("PF_LABEL_FUTURE_MIN", 0) or 0)
    except Exception:
        pf_future_min = 0.0
    pf_label_bars = int(max(0, round((pf_future_min * 60.0) / float(max(1, candle_sec))))) if pf_future_min > 0 else 0
    return int(max(pf_label_bars, contract.danger_horizon_bars(candle_sec)))


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


def _prepare_symbol(
    symbol: str,
    *,
    total_days: int,
    remove_tail_days: int,
    flags: Dict[str, bool],
    contract: TradeContract,
    asset_class: str = "crypto",
    entry_label_name: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, List[str]], List[str], List[str]]:
    df_pf, _ = _build_features_for_symbol(
        symbol,
        total_days=int(total_days),
        remove_tail_days=int(remove_tail_days),
        contract=contract,
        flags=flags,
        asset_class=asset_class,
    )
    frozen_mask = _frozen_ohlc_mask(df_pf) if str(asset_class or "crypto").lower() == "stocks" else None
    seed = _env_int("SNIPER_SEED", 1337)
    train_exit_model = str(os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    entry_specs = _entry_label_specs(contract)
    exit_spec = _exit_target_spec(contract)
    if entry_label_name:
        want = str(entry_label_name).strip().lower()
        entry_specs = [s for s in entry_specs if str(s.get("name", "")).lower() == want]
        if not entry_specs:
            raise ValueError(f"entry_label_name inv\u00e1lido: {entry_label_name}")
    entry_map: dict[str, pd.DataFrame] = {}
    feat_cols_entry_map: dict[str, List[str]] = {}
    danger_df: pd.DataFrame | None = None
    exit_df: pd.DataFrame | None = None
    for spec in entry_specs:
        sniper_ds = build_sniper_datasets(
            df_pf,
            contract=contract,
            entry_label_col=str(spec["label_col"]),
            entry_weight_col=str(spec.get("weight_col") or ""),
            exit_code_col=str(spec["exit_code_col"]),
            exit_target_col=str(exit_spec.get("target_col") or "sniper_exit_span_target"),
            exit_weight_col=str(exit_spec.get("weight_col") or "sniper_exit_span_weight"),
            seed=int(seed),
            enable_exit_dataset=bool(train_exit_model),
        )
        entry_df = sniper_ds.entry
        if frozen_mask is not None and not entry_df.empty:
            entry_df = entry_df.loc[~frozen_mask.reindex(entry_df.index).fillna(False)].copy()
        entry_map[str(spec["name"])] = entry_df
        feat_cols_entry_map[str(spec["name"])] = _list_feature_columns(entry_df)
        if danger_df is None:
            danger_df = sniper_ds.danger
            exit_df = sniper_ds.exit

    if danger_df is None or exit_df is None:
        danger_df = pd.DataFrame()
        exit_df = pd.DataFrame()

    feat_cols_danger = _list_feature_columns(danger_df)
    feat_cols_exit = _list_feature_columns(exit_df)
    # df_pf é enorme e não é necessário fora desta função -> libera cedo
    del df_pf
    return entry_map, danger_df, exit_df, feat_cols_entry_map, feat_cols_danger, feat_cols_exit


def _stack_batches(
    dfs: List[pd.DataFrame],
    feat_cols: List[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for df in dfs:
        if df.empty:
            continue
        mat = df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if mat.empty:
            continue
        idx = mat.index
        X_list.append(mat.to_numpy(np.float32, copy=True))
        y_list.append(df.loc[idx, label_col].astype(np.float32, copy=False).to_numpy())
        ts_list.append(idx.to_numpy(dtype="datetime64[ns]"))
    if not X_list:
        return (
            np.empty((0, len(feat_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    return (
        np.vstack(X_list),
        np.concatenate(y_list).astype(np.float32, copy=False),
        np.concatenate(ts_list),
    )


def prepare_sniper_dataset(
    symbols: Sequence[str],
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract | None = None,
    entry_label_name: str | None = None,
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
    max_rows_entry: int = 600_000,
    max_rows_exit: int = 600_000,
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
    # cap por simbolo: proporcional ao tamanho total para reduzir custo
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    train_exit_model = str(os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    per_sym_exit = int(max_rows_exit // max(1, len(symbols))) if max_rows_exit > 0 else 0
    min_cap_exit = _env_int("SNIPER_SYMBOL_CAP_EXIT_MIN", 12_000)
    max_cap_exit = _env_int("SNIPER_SYMBOL_CAP_EXIT_MAX", 80_000)
    symbol_cap_exit = 0 if (not train_exit_model) else max(int(min_cap_exit), min(int(max_cap_exit), per_sym_exit if per_sym_exit > 0 else int(max_cap_exit)))
    entry_specs = _entry_label_specs(contract)
    exit_spec = _exit_target_spec(contract)
    if entry_label_name:
        want = str(entry_label_name).strip().lower()
        entry_specs = [s for s in entry_specs if str(s.get("name", "")).lower() == want]
        if not entry_specs:
            raise ValueError(f"entry_label_name invalido: {entry_label_name}")
    entry_pool_map: dict[str, pd.DataFrame] = {str(s["name"]): pd.DataFrame() for s in entry_specs}
    entry_buf_map: dict[str, list[pd.DataFrame]] = {str(s["name"]): [] for s in entry_specs}
    feat_cols_entry_map: dict[str, List[str]] = {}
    exit_pool = pd.DataFrame()
    exit_buf: list[pd.DataFrame] = []
    feat_cols_exit: List[str] = []
    symbols_used: List[str] = []
    symbols_skipped: List[str] = []
    total_syms = len(symbols)

    prog_ds = LineProgressPrinter(
        prefix="sniper-ds",
        total=total_syms,
        width=24,
        stream=sys.stdout,
        min_interval_s=0.2,
    )

    def _print_progress(
        i: int,
        sym: str,
        *,
        pos_n: int | None = None,
        neg_n: int | None = None,
        pos_total: int | None = None,
        neg_total: int | None = None,
    ) -> None:
        if pos_n is None or neg_n is None:
            counts = "p- n-"
        else:
            counts = f"p{pos_n} n{neg_n}"
        if pos_total is None or neg_total is None:
            totals = "pt- nt-"
        else:
            totals = f"pt{pos_total} nt{neg_total}"
        prog_ds.update(int(i), current=sym, extra=f"{counts} | {totals}", force=True)

    def _sample_df(
        df_in: pd.DataFrame,
        label_col: str,
        ratio_neg_per_pos: float,
        max_rows: int,
        weight_col: str | None = None,
    ) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        y = df_in[label_col].to_numpy()
        pos_idx = np.flatnonzero(y >= 0.5)
        neg_idx = np.flatnonzero(y < 0.5)
        if pos_idx.size == 0:
            # sem positivos -> evita gerar dataset extremamente enviesado
            return df_in.iloc[0:0].copy()

        use_weight_bins = str(os.getenv("SNIPER_ENTRY_USE_WEIGHT_BINS", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        pos_favor_high = str(os.getenv("SNIPER_ENTRY_POS_FAVOR_HIGH", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}

        def _sample_weight_bins(
            idx_src: np.ndarray,
            target_n: int,
            *,
            favor_high: bool,
        ) -> np.ndarray:
            target_n = int(max(0, target_n))
            if target_n <= 0 or idx_src.size == 0:
                return np.empty((0,), dtype=np.int64)
            if target_n >= idx_src.size:
                out_all = np.asarray(idx_src, dtype=np.int64)
                out_all.sort()
                return out_all

            if (not use_weight_bins) or (not weight_col) or (weight_col not in df_in.columns):
                take = rng.choice(idx_src, size=target_n, replace=False)
                take.sort()
                return take.astype(np.int64, copy=False)

            w_all = pd.to_numeric(df_in[weight_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            try:
                bin_max = float(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_MAX", "7.0") or "7.0")
            except Exception:
                bin_max = 7.0
            if (not np.isfinite(bin_max)) or bin_max <= 0.0:
                bin_max = 7.0
            w = np.clip(np.nan_to_num(w_all[idx_src], nan=0.0, posinf=bin_max, neginf=0.0), 0.0, bin_max)
            # bins finos (step configuravel) em [0.0, bin_max]
            bin_step = float(_env_int("SNIPER_ENTRY_WEIGHT_BIN_STEP_X10", 1)) / 10.0
            if (not np.isfinite(bin_step)) or bin_step <= 0.0:
                bin_step = 0.5
            n_bins = int(max(8, round(bin_max / bin_step)))
            bins = np.floor(w / bin_step).astype(np.int32, copy=False)
            bins = np.clip(bins, 0, n_bins - 1)

            # Nao forca manter todo o bin de pico; isso colapsa o dataset em um unico bin.
            selected_rel: list[np.ndarray] = []
            rem = int(target_n)

            # Quotas por bin:
            # - favor_high=True: prioriza bins mais altos
            # - favor_high=False: prioriza bins mais baixos
            factors = np.arange(2, 2 + n_bins, dtype=np.float64)
            if not favor_high:
                factors = factors[::-1]
            counts = np.zeros(n_bins, dtype=np.int64)
            rel_by_bin: list[np.ndarray] = []
            for b in range(n_bins):
                rel = np.flatnonzero(bins == b)
                rel_by_bin.append(rel)
                counts[b] = int(rel.size)
            counts_eff = counts.copy()
            if int(counts_eff.sum()) <= 0:
                out = idx_src[np.concatenate(selected_rel)] if selected_rel else np.empty((0,), dtype=np.int64)
                out.sort()
                return out.astype(np.int64, copy=False)

            # Mistura prioridade por qualidade com cobertura dos bins.
            # O objetivo aqui nao e colapsar o pool nos bins de pico, e sim
            # manter bins medios/altos vivos para o modelo aprender trades
            # mais consistentes, nao apenas outliers.
            span = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
            if not favor_high:
                span = span[::-1]
            bias_strength = 1.25 if favor_high else 0.75
            bias = 1.0 + (bias_strength * span)
            score = bias * np.power(np.maximum(1.0, counts_eff.astype(np.float64)), 0.75)
            ssum = float(np.sum(score))
            if (not np.isfinite(ssum)) or ssum <= 0.0:
                pool_rel = np.arange(idx_src.size, dtype=np.int64)
                if rem >= pool_rel.size:
                    chosen_rel = pool_rel
                else:
                    chosen_rel = rng.choice(pool_rel, size=rem, replace=False)
                selected_rel.append(chosen_rel)
                out = idx_src[np.concatenate(selected_rel)]
                out.sort()
                return out.astype(np.int64, copy=False)

            quota = (score / ssum) * float(rem)
            take_n = np.floor(quota).astype(np.int64)
            take_n = np.minimum(take_n, counts_eff)
            used = int(np.sum(take_n))
            left = int(rem - used)
            if left > 0:
                frac = quota - np.floor(quota)
                order = np.argsort(-frac)
                for b in order:
                    if left <= 0:
                        break
                    cap = int(counts_eff[b] - take_n[b])
                    if cap <= 0:
                        continue
                    add = 1 if cap >= 1 else 0
                    take_n[b] += add
                    left -= add

            for b in range(n_bins):
                k = int(take_n[b])
                if k <= 0:
                    continue
                rel = rel_by_bin[b]
                if k >= rel.size:
                    selected_rel.append(rel)
                else:
                    selected_rel.append(rng.choice(rel, size=k, replace=False))

            out_rel = np.concatenate(selected_rel) if selected_rel else np.empty((0,), dtype=np.int64)
            if out_rel.size < target_n:
                chosen = np.zeros(idx_src.size, dtype=bool)
                chosen[out_rel] = True
                rem_pool = np.flatnonzero(~chosen)
                need = int(target_n - out_rel.size)
                if need > 0 and rem_pool.size > 0:
                    if need >= rem_pool.size:
                        extra = rem_pool
                    else:
                        extra = rng.choice(rem_pool, size=need, replace=False)
                    out_rel = np.concatenate([out_rel, extra])

            out = idx_src[out_rel]
            out.sort()
            if out.size > target_n:
                out = out[:target_n]
            return out.astype(np.int64, copy=False)

        ratio = float(max(0.0, ratio_neg_per_pos))
        max_rows = int(max_rows)
        if weight_col and weight_col in df_in.columns:
            w_all = pd.to_numeric(df_in[weight_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            try:
                pos_min_w = float(os.getenv("SNIPER_ENTRY_POS_MIN_WEIGHT", "0.0") or "0.0")
            except Exception:
                pos_min_w = 0.0
            if np.isfinite(pos_min_w) and pos_min_w > 0.0:
                wp = np.nan_to_num(w_all[pos_idx], nan=0.0, posinf=0.0, neginf=0.0)
                pos_idx = pos_idx[wp >= float(pos_min_w)]
                if pos_idx.size == 0:
                    return df_in.iloc[0:0].copy()
            try:
                neg_min_w = float(os.getenv("SNIPER_ENTRY_NEG_MIN_WEIGHT", "0.0") or "0.0")
            except Exception:
                neg_min_w = 0.0
            if np.isfinite(neg_min_w) and neg_min_w > 0.0:
                wn = np.nan_to_num(w_all[neg_idx], nan=0.0, posinf=0.0, neginf=0.0)
                neg_idx = neg_idx[wn >= float(neg_min_w)]
        pos_keep_n = int(pos_idx.size)
        try:
            pos_keep_frac = float(os.getenv("SNIPER_ENTRY_POS_KEEP_FRACTION", "1.0") or "1.0")
        except Exception:
            pos_keep_frac = 1.0
        neg_favor_high = str(os.getenv("SNIPER_ENTRY_NEG_FAVOR_HIGH", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        if (not np.isfinite(pos_keep_frac)) or pos_keep_frac <= 0.0:
            pos_keep_frac = 1.0
        if pos_keep_frac < 1.0:
            pos_keep_n = min(pos_keep_n, int(max(1, round(pos_idx.size * pos_keep_frac))))
        if max_rows > 0:
            max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
            pos_keep_n = min(pos_keep_n, max_pos)

        # Positivos: prioriza weights altos para reduzir ruido de retornos marginais.
        pos_keep = _sample_weight_bins(pos_idx, pos_keep_n, favor_high=bool(pos_favor_high))

        neg_target = int(round(pos_keep_n * ratio))
        if max_rows > 0:
            max_neg_allowed = max_rows - pos_keep_n
            if max_neg_allowed < neg_target:
                neg_target = max(0, max_neg_allowed)

        if neg_target <= 0 or neg_idx.size == 0:
            keep = pos_keep
        else:
            # Negativos: quando habilitado, prioriza bins altos para reforcar falsos positivos caros.
            neg_keep = _sample_weight_bins(neg_idx, neg_target, favor_high=bool(neg_favor_high))
            keep = np.concatenate([pos_keep, neg_keep])
        return df_in.iloc[np.sort(keep)].copy()


    def _sample_reg_df(
        df_in: pd.DataFrame,
        label_col: str,
        max_rows: int,
        weight_col: str | None = None,
    ) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        if int(max_rows) <= 0 or len(df_in) <= int(max_rows):
            return df_in
        idx_all = np.arange(len(df_in), dtype=np.int64)
        if weight_col and weight_col in df_in.columns:
            w = pd.to_numeric(df_in[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
            w = np.clip(w, 1e-6, None)
            sw = float(np.sum(w))
            if np.isfinite(sw) and sw > 0.0:
                p = w / sw
                take = rng.choice(idx_all, size=int(max_rows), replace=False, p=p)
                take.sort()
                return df_in.iloc[take].copy()
        take = rng.choice(idx_all, size=int(max_rows), replace=False)
        take.sort()
        return df_in.iloc[take].copy()
    def _to_numpy(
        df: pd.DataFrame,
        feats: List[str],
        label_col: str,
        weight_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if df.empty:
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat = df.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
        mask = mat.notnull().all(axis=1).to_numpy(dtype=bool, copy=False)
        if mask.size == 0 or (not bool(np.any(mask))):
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat2 = mat.to_numpy(np.float32, copy=True)
        X = mat2[mask]
        y = df[label_col].to_numpy(dtype=np.float32, copy=False)[mask]
        if weight_col and weight_col in df.columns:
            w = df[weight_col].to_numpy(dtype=np.float32, copy=False)[mask]
        else:
            w = np.ones(int(y.shape[0]), dtype=np.float32)
        ts = df.index.to_numpy(dtype="datetime64[ns]")[mask]
        if "sym_id" in df.columns:
            sym_arr = df["sym_id"].to_numpy(dtype=np.int32, copy=False)[mask]
        else:
            sym_arr = np.zeros(int(ts.shape[0]), dtype=np.int32)
        return X, y, w, ts, sym_arr

    preferred_name = "long" if any(str(s["name"]) == "long" for s in entry_specs) else str(entry_specs[0]["name"])

    total_pos = 0
    total_neg = 0
    entry_raw_pos_map: dict[str, int] = {}
    entry_raw_total_map: dict[str, int] = {}
    batch_flush_n = 8
    for sym_idx, symbol in enumerate(symbols):
        df_pf = None
        pos_n = None
        neg_n = None
        fallback_pos = None
        fallback_neg = None
        try:
            df_pf, _ = _build_features_for_symbol(
                symbol,
                total_days=int(total_days),
                remove_tail_days=int(remove_tail_days),
                contract=contract,
                flags=flags,
                asset_class=asset_class,
            )
            exit_df_local: pd.DataFrame | None = None
            for spec in entry_specs:
                name = str(spec["name"])
                weight_col = str(spec.get("weight_col") or "")
                sniper_ds = build_sniper_datasets(
                    df_pf,
                    contract=contract,
                    entry_label_col=str(spec["label_col"]),
                    entry_weight_col=str(spec.get("weight_col") or ""),
                    exit_code_col=str(spec["exit_code_col"]),
                    exit_target_col=str(exit_spec.get("target_col") or "sniper_exit_span_target"),
                    exit_weight_col=str(exit_spec.get("weight_col") or "sniper_exit_span_weight"),
                    seed=int(seed),
                    enable_exit_dataset=bool(symbol_cap_exit > 0),
                )
                entry_df = sniper_ds.entry
                if entry_df.empty:
                    continue
                entry_df = _try_downcast_df(entry_df, copy=False)
                try:
                    y_raw = entry_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                    entry_raw_pos_map[name] = int(entry_raw_pos_map.get(name, 0)) + int(np.sum(y_raw >= 0.5))
                    entry_raw_total_map[name] = int(entry_raw_total_map.get(name, 0)) + int(y_raw.shape[0])
                except Exception:
                    pass
                entry_df["sym_id"] = int(sym_idx)
                feat_cols = feat_cols_entry_map.get(name)
                if not feat_cols:
                    feat_cols = _list_feature_columns(entry_df)
                    feat_cols_entry_map[name] = list(feat_cols)
                cols_keep = list(feat_cols) + ["label_entry", "sym_id"]
                if weight_col and weight_col in entry_df.columns:
                    cols_keep.append(weight_col)
                entry_df = entry_df.reindex(columns=cols_keep)
                entry_df = _sample_df(
                    entry_df,
                    "label_entry",
                    float(entry_ratio_neg_per_pos),
                    int(symbol_cap),
                    weight_col=weight_col,
                )
                if not entry_df.empty and "label_entry" in entry_df.columns:
                    y = entry_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                    if name == preferred_name:
                        pos_n = int(np.sum(y >= 0.5))
                        neg_n = int(np.sum(y < 0.5))
                    if fallback_pos is None:
                        fallback_pos = int(np.sum(y >= 0.5))
                        fallback_neg = int(np.sum(y < 0.5))
                buf = entry_buf_map.get(name)
                if buf is not None:
                    buf.append(entry_df)
                    if len(buf) >= batch_flush_n:
                        pool_df = entry_pool_map.get(name)
                        combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
                        pool_df = _sample_df(
                            combined,
                            "label_entry",
                            float(entry_ratio_neg_per_pos),
                            int(max_rows_entry),
                            weight_col=weight_col,
                        )
                        entry_pool_map[name] = pool_df
                        buf.clear()
                        del combined
                if exit_df_local is None:
                    try:
                        exit_df_local = sniper_ds.exit
                    except Exception:
                        exit_df_local = None
            if (symbol_cap_exit > 0) and (exit_df_local is not None) and (not exit_df_local.empty):
                ex = _try_downcast_df(exit_df_local, copy=False)
                ex["sym_id"] = int(sym_idx)
                if not feat_cols_exit:
                    feat_cols_exit = _list_feature_columns(ex)
                cols_keep_ex = list(feat_cols_exit) + ["label_exit", "sym_id"]
                exit_weight_col = "label_exit_weight" if "label_exit_weight" in ex.columns else ""
                if exit_weight_col:
                    cols_keep_ex.append(exit_weight_col)
                ex = ex.reindex(columns=cols_keep_ex)
                ex = _sample_reg_df(
                    ex,
                    "label_exit",
                    int(symbol_cap_exit),
                    weight_col=exit_weight_col or None,
                )
                exit_buf.append(ex)
                if len(exit_buf) >= batch_flush_n:
                    combined_ex = pd.concat(([exit_pool] if (exit_pool is not None and not exit_pool.empty) else []) + exit_buf, axis=0, ignore_index=False)
                    exit_pool = _sample_reg_df(
                        combined_ex,
                        "label_exit",
                        int(max_rows_exit),
                        weight_col=("label_exit_weight" if "label_exit_weight" in combined_ex.columns else None),
                    )
                    exit_buf.clear()
                    del combined_ex
            symbols_used.append(symbol)
        except Exception:
            symbols_skipped.append(symbol)
            continue
        finally:
            try:
                del df_pf
            except Exception:
                pass
        if pos_n is None and fallback_pos is not None:
            pos_n, neg_n = fallback_pos, (fallback_neg if fallback_neg is not None else 0)
        if pos_n is None or neg_n is None:
            pos_n, neg_n = 0, 0
        total_pos += int(pos_n)
        total_neg += int(neg_n)
        _print_progress(
            sym_idx + 1,
            symbol,
            pos_n=pos_n,
            neg_n=neg_n,
            pos_total=total_pos,
            neg_total=total_neg,
        )
        if (sym_idx + 1) % 10 == 0:
            gc.collect()
    prog_ds.close()
    # flush remaining buffers
    for name, buf in entry_buf_map.items():
        if not buf:
            continue
        weight_col = ""
        for spec in entry_specs:
            if str(spec.get("name")) == name:
                weight_col = str(spec.get("weight_col") or "")
                break
        pool_df = entry_pool_map.get(name)
        combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
        pool_df = _sample_df(
            combined,
            "label_entry",
            float(entry_ratio_neg_per_pos),
            int(max_rows_entry),
            weight_col=weight_col,
        )
        entry_pool_map[name] = pool_df
        buf.clear()
        del combined
    if (symbol_cap_exit > 0) and exit_buf:
        combined_ex = pd.concat(([exit_pool] if (exit_pool is not None and not exit_pool.empty) else []) + exit_buf, axis=0, ignore_index=False)
        exit_pool = _sample_reg_df(
            combined_ex,
            "label_exit",
            int(max_rows_exit),
            weight_col=("label_exit_weight" if "label_exit_weight" in combined_ex.columns else None),
        )
        exit_buf.clear()
        del combined_ex

    entry_batches: dict[str, SniperBatch] = {}
    for spec in entry_specs:
        name = str(spec["name"])
        entry_df = entry_pool_map.get(name, pd.DataFrame())
        if not entry_df.empty:
            entry_df.sort_index(inplace=True)
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        weight_col = str(spec.get("weight_col") or "")
        Xe, ye, we, tse, syme = _to_numpy(entry_df, list(feat_cols), "label_entry", weight_col=weight_col)
        batch = SniperBatch(X=Xe, y=ye, w=we, ts=tse, sym_id=syme, feature_cols=list(feat_cols))
        rt = int(entry_raw_total_map.get(name, 0) or 0)
        if rt > 0:
            setattr(batch, "natural_pos_rate", float(entry_raw_pos_map.get(name, 0) or 0) / float(rt))
            setattr(batch, "natural_pos_rows", rt)
        entry_batches[name] = batch

    entry_short = entry_batches.get("short")
    entry_long = entry_batches.get("long")
    entry_mid = entry_batches.get("mid", entry_long if entry_long is not None else entry_short)
    entry_batch = entry_mid if entry_mid is not None else SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
    )
    if exit_pool is not None and (not exit_pool.empty):
        try:
            exit_pool.sort_index(inplace=True)
        except Exception:
            pass
    Xe_x, ye_x, we_x, tse_x, syme_x = _to_numpy(
        exit_pool if exit_pool is not None else pd.DataFrame(),
        list(feat_cols_exit),
        "label_exit",
        weight_col=("label_exit_weight" if (exit_pool is not None and "label_exit_weight" in getattr(exit_pool, "columns", [])) else None),
    )
    exit_batch = SniperBatch(
        X=Xe_x,
        y=ye_x,
        w=we_x,
        ts=tse_x,
        sym_id=syme_x,
        feature_cols=list(feat_cols_exit),
    )
    empty_batch = SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
    )
    train_end = None
    try:
        ts_entry_mid = entry_mid.ts if entry_mid is not None else np.empty((0,), dtype="datetime64[ns]")
        if ts_entry_mid.size:
            train_end = pd.to_datetime(ts_entry_mid.max())
    except Exception:
        train_end = None

    return SniperDataPack(
        entry=entry_batch,
        entry_short=entry_short,
        entry_mid=entry_mid,
        entry_long=entry_long,
        danger=empty_batch,
        exit=exit_batch,
        contract=contract,
        symbols=list(symbols_used or symbols),
        symbols_used=symbols_used or None,
        symbols_skipped=symbols_skipped or None,
        train_end_utc=train_end,
    )


def prepare_sniper_dataset_from_cache(
    symbols: Sequence[str],
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract | None = None,
    cache_map: Dict[str, Path] | None = None,
    entry_label_name: str | None = None,
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
    max_rows_entry: int = 2_000_000,
    max_rows_exit: int = 2_000_000,
    seed: int = 42,
    feature_flags: Dict[str, bool] | None = None,
    asset_class: str = "crypto",
    parallel: bool = True,
    max_workers: int | None = None,
) -> SniperDataPack:
    """
    Calcula features 1x por simbolo (cache em disco) e, no walk-forward, apenas
    recorta o final por `remove_tail_days` antes de montar o dataset de entry.
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    flags = dict(feature_flags or _default_flags_for_asset(asset_class))
    symbols = list(symbols)
    if not symbols:
        raise RuntimeError("symbols vazio")

    cache_dir = _cache_dir(asset_class, int(getattr(contract, "timeframe_sec", 60) or 60))
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
    # cap por simbolo: proporcional ao tamanho total para reduzir custo
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    train_exit_model = str(os.getenv("SNIPER_TRAIN_EXIT_MODEL", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    per_sym_exit = int(max_rows_exit // max(1, len(symbols))) if max_rows_exit > 0 else 0
    min_cap_exit = _env_int("SNIPER_SYMBOL_CAP_EXIT_MIN", 12_000)
    max_cap_exit = _env_int("SNIPER_SYMBOL_CAP_EXIT_MAX", 80_000)
    symbol_cap_exit = 0 if (not train_exit_model) else max(int(min_cap_exit), min(int(max_cap_exit), per_sym_exit if per_sym_exit > 0 else int(max_cap_exit)))
    entry_specs = _entry_label_specs(contract)
    exit_spec = _exit_target_spec(contract)
    if entry_label_name:
        want = str(entry_label_name).strip().lower()
        entry_specs = [s for s in entry_specs if str(s.get("name", "")).lower() == want]
        if not entry_specs:
            raise ValueError(f"entry_label_name invalido: {entry_label_name}")
    entry_pool_map: dict[str, pd.DataFrame] = {str(s["name"]): pd.DataFrame() for s in entry_specs}
    entry_buf_map: dict[str, list[pd.DataFrame]] = {str(s["name"]): [] for s in entry_specs}
    feat_cols_entry_map: dict[str, List[str]] = {}
    exit_pool = pd.DataFrame()
    exit_buf: list[pd.DataFrame] = []
    feat_cols_exit: List[str] = []
    symbols_used: List[str] = []
    symbols_skipped: List[str] = []
    total_syms = len(symbols)

    prog_ds = LineProgressPrinter(
        prefix="sniper-ds",
        total=total_syms,
        width=24,
        stream=sys.stdout,
        min_interval_s=0.2,
    )

    def _print_progress(
        i: int,
        sym: str,
        *,
        pos_n: int | None = None,
        neg_n: int | None = None,
        pos_total: int | None = None,
        neg_total: int | None = None,
    ) -> None:
        if pos_n is None or neg_n is None:
            counts = "p- n-"
        else:
            counts = f"p{pos_n} n{neg_n}"
        if pos_total is None or neg_total is None:
            totals = "pt- nt-"
        else:
            totals = f"pt{pos_total} nt{neg_total}"
        prog_ds.update(int(i), current=sym, extra=f"{counts} | {totals}", force=True)

    def _sample_df(
        df_in: pd.DataFrame,
        label_col: str,
        ratio_neg_per_pos: float,
        max_rows: int,
        weight_col: str | None = None,
    ) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        y = df_in[label_col].to_numpy()
        pos_idx = np.flatnonzero(y >= 0.5)
        neg_idx = np.flatnonzero(y < 0.5)
        if pos_idx.size == 0:
            # sem positivos -> evita gerar dataset extremamente enviesado
            return df_in.iloc[0:0].copy()

        use_weight_bins = str(os.getenv("SNIPER_ENTRY_USE_WEIGHT_BINS", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        pos_favor_high = str(os.getenv("SNIPER_ENTRY_POS_FAVOR_HIGH", "1") or "1").strip().lower() in {"1", "true", "yes", "y", "on"}

        def _sample_weight_bins(
            idx_src: np.ndarray,
            target_n: int,
            *,
            favor_high: bool,
        ) -> np.ndarray:
            target_n = int(max(0, target_n))
            if target_n <= 0 or idx_src.size == 0:
                return np.empty((0,), dtype=np.int64)
            if target_n >= idx_src.size:
                out_all = np.asarray(idx_src, dtype=np.int64)
                out_all.sort()
                return out_all

            if (not use_weight_bins) or (not weight_col) or (weight_col not in df_in.columns):
                take = rng.choice(idx_src, size=target_n, replace=False)
                take.sort()
                return take.astype(np.int64, copy=False)

            w_all = pd.to_numeric(df_in[weight_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            try:
                bin_max = float(os.getenv("SNIPER_ENTRY_WEIGHT_BIN_MAX", "7.0") or "7.0")
            except Exception:
                bin_max = 7.0
            if (not np.isfinite(bin_max)) or bin_max <= 0.0:
                bin_max = 7.0
            w = np.clip(np.nan_to_num(w_all[idx_src], nan=0.0, posinf=bin_max, neginf=0.0), 0.0, bin_max)
            bin_step = float(_env_int("SNIPER_ENTRY_WEIGHT_BIN_STEP_X10", 1)) / 10.0
            if (not np.isfinite(bin_step)) or bin_step <= 0.0:
                bin_step = 0.5
            n_bins = int(max(8, round(bin_max / bin_step)))
            bins = np.floor(w / bin_step).astype(np.int32, copy=False)
            bins = np.clip(bins, 0, n_bins - 1)

            selected_rel: list[np.ndarray] = []
            rem = int(target_n)

            factors = np.arange(2, 2 + n_bins, dtype=np.float64)
            if not favor_high:
                factors = factors[::-1]
            counts = np.zeros(n_bins, dtype=np.int64)
            rel_by_bin: list[np.ndarray] = []
            for b in range(n_bins):
                rel = np.flatnonzero(bins == b)
                rel_by_bin.append(rel)
                counts[b] = int(rel.size)
            counts_eff = counts.copy()
            if int(counts_eff.sum()) <= 0:
                out = idx_src[np.concatenate(selected_rel)] if selected_rel else np.empty((0,), dtype=np.int64)
                out.sort()
                return out.astype(np.int64, copy=False)

            span = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
            if not favor_high:
                span = span[::-1]
            bias_strength = 1.25 if favor_high else 0.75
            bias = 1.0 + (bias_strength * span)
            score = bias * np.power(np.maximum(1.0, counts_eff.astype(np.float64)), 0.75)
            ssum = float(np.sum(score))
            if (not np.isfinite(ssum)) or ssum <= 0.0:
                pool_rel = np.arange(idx_src.size, dtype=np.int64)
                if rem >= pool_rel.size:
                    chosen_rel = pool_rel
                else:
                    chosen_rel = rng.choice(pool_rel, size=rem, replace=False)
                selected_rel.append(chosen_rel)
                out = idx_src[np.concatenate(selected_rel)]
                out.sort()
                return out.astype(np.int64, copy=False)

            quota = (score / ssum) * float(rem)
            take_n = np.floor(quota).astype(np.int64)
            take_n = np.minimum(take_n, counts_eff)
            used = int(np.sum(take_n))
            left = int(rem - used)
            if left > 0:
                frac = quota - np.floor(quota)
                order = np.argsort(-frac)
                for b in order:
                    if left <= 0:
                        break
                    cap = int(counts_eff[b] - take_n[b])
                    if cap <= 0:
                        continue
                    add = 1 if cap >= 1 else 0
                    take_n[b] += add
                    left -= add

            for b in range(n_bins):
                k = int(take_n[b])
                if k <= 0:
                    continue
                rel = rel_by_bin[b]
                if k >= rel.size:
                    selected_rel.append(rel)
                else:
                    selected_rel.append(rng.choice(rel, size=k, replace=False))

            out_rel = np.concatenate(selected_rel) if selected_rel else np.empty((0,), dtype=np.int64)
            if out_rel.size < target_n:
                chosen = np.zeros(idx_src.size, dtype=bool)
                chosen[out_rel] = True
                rem_pool = np.flatnonzero(~chosen)
                need = int(target_n - out_rel.size)
                if need > 0 and rem_pool.size > 0:
                    if need >= rem_pool.size:
                        extra = rem_pool
                    else:
                        extra = rng.choice(rem_pool, size=need, replace=False)
                    out_rel = np.concatenate([out_rel, extra])

            out = idx_src[out_rel]
            out.sort()
            if out.size > target_n:
                out = out[:target_n]
            return out.astype(np.int64, copy=False)

        ratio = float(max(0.0, ratio_neg_per_pos))
        max_rows = int(max_rows)
        if weight_col and weight_col in df_in.columns:
            w_all = pd.to_numeric(df_in[weight_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            try:
                pos_min_w = float(os.getenv("SNIPER_ENTRY_POS_MIN_WEIGHT", "0.0") or "0.0")
            except Exception:
                pos_min_w = 0.0
            if np.isfinite(pos_min_w) and pos_min_w > 0.0:
                wp = np.nan_to_num(w_all[pos_idx], nan=0.0, posinf=0.0, neginf=0.0)
                pos_idx = pos_idx[wp >= float(pos_min_w)]
                if pos_idx.size == 0:
                    return df_in.iloc[0:0].copy()
            try:
                neg_min_w = float(os.getenv("SNIPER_ENTRY_NEG_MIN_WEIGHT", "0.0") or "0.0")
            except Exception:
                neg_min_w = 0.0
            if np.isfinite(neg_min_w) and neg_min_w > 0.0:
                wn = np.nan_to_num(w_all[neg_idx], nan=0.0, posinf=0.0, neginf=0.0)
                neg_idx = neg_idx[wn >= float(neg_min_w)]
        pos_keep_n = int(pos_idx.size)
        try:
            pos_keep_frac = float(os.getenv("SNIPER_ENTRY_POS_KEEP_FRACTION", "1.0") or "1.0")
        except Exception:
            pos_keep_frac = 1.0
        neg_favor_high = str(os.getenv("SNIPER_ENTRY_NEG_FAVOR_HIGH", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        if (not np.isfinite(pos_keep_frac)) or pos_keep_frac <= 0.0:
            pos_keep_frac = 1.0
        if pos_keep_frac < 1.0:
            pos_keep_n = min(pos_keep_n, int(max(1, round(pos_idx.size * pos_keep_frac))))
        if max_rows > 0:
            max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
            pos_keep_n = min(pos_keep_n, max_pos)

        pos_keep = _sample_weight_bins(pos_idx, pos_keep_n, favor_high=bool(pos_favor_high))

        neg_target = int(round(pos_keep_n * ratio))
        if max_rows > 0:
            max_neg_allowed = max_rows - pos_keep_n
            if max_neg_allowed < neg_target:
                neg_target = max(0, max_neg_allowed)

        if neg_target <= 0 or neg_idx.size == 0:
            keep = pos_keep
        else:
            neg_keep = _sample_weight_bins(neg_idx, neg_target, favor_high=bool(neg_favor_high))
            keep = np.concatenate([pos_keep, neg_keep])
        return df_in.iloc[np.sort(keep)].copy()
    def _to_numpy(
        df: pd.DataFrame,
        feats: List[str],
        label_col: str,
        weight_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if df.empty:
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat = df.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
        mask = mat.notnull().all(axis=1).to_numpy(dtype=bool, copy=False)
        if mask.size == 0 or (not bool(np.any(mask))):
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat2 = mat.to_numpy(np.float32, copy=True)
        X = mat2[mask]
        y = df[label_col].to_numpy(dtype=np.float32, copy=False)[mask]
        if weight_col and weight_col in df.columns:
            w = df[weight_col].to_numpy(dtype=np.float32, copy=False)[mask]
        else:
            w = np.ones(int(y.shape[0]), dtype=np.float32)
        ts = df.index.to_numpy(dtype="datetime64[ns]")[mask]
        if "sym_id" in df.columns:
            sym_arr = df["sym_id"].to_numpy(dtype=np.int32, copy=False)[mask]
        else:
            sym_arr = np.zeros(int(ts.shape[0]), dtype=np.int32)
        return X, y, w, ts, sym_arr

    def _sample_reg_df(
        df_in: pd.DataFrame,
        label_col: str,
        max_rows: int,
        weight_col: str | None = None,
    ) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        if int(max_rows) <= 0 or len(df_in) <= int(max_rows):
            return df_in
        idx_all = np.arange(len(df_in), dtype=np.int64)
        if weight_col and weight_col in df_in.columns:
            w = pd.to_numeric(df_in[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
            w = np.clip(w, 1e-6, None)
            sw = float(np.sum(w))
            if np.isfinite(sw) and sw > 0.0:
                p = w / sw
                take = rng.choice(idx_all, size=int(max_rows), replace=False, p=p)
                take.sort()
                return df_in.iloc[take].copy()
        take = rng.choice(idx_all, size=int(max_rows), replace=False)
        take.sort()
        return df_in.iloc[take].copy()

    def _process_symbol(sym_idx: int, symbol: str):
        _thermal_wait(f"dataset_from_cache:{symbol}")
        data_path = cache_map.get(symbol)
        if data_path is None or (not Path(data_path).exists()):
            raise RuntimeError(f"{symbol}: cache nao encontrado")
        df = _load_feature_cache(Path(data_path))
        if df.empty:
            raise RuntimeError(f"{symbol}: cache vazio")
        if remove_tail_days > 0:
            cutoff = df.index[-1] - pd.Timedelta(days=int(remove_tail_days))
            df = df[df.index < cutoff]
        if df.empty:
            raise RuntimeError(f"{symbol}: sem dados apos corte")
        df = _try_downcast_df(df, copy=False)
        frozen_mask = _frozen_ohlc_mask(df) if str(asset_class or "crypto").lower() == "stocks" else None

        frames_local: dict[str, tuple[pd.DataFrame, List[str]]] = {}
        raw_counts_local: dict[str, tuple[int, int]] = {}
        exit_local: pd.DataFrame | None = None
        for spec in entry_specs:
            name = str(spec["name"])
            weight_col = str(spec.get("weight_col") or "")
            sniper_ds = build_sniper_datasets(
                df,
                contract=contract,
                entry_label_col=str(spec["label_col"]),
                entry_weight_col=str(spec.get("weight_col") or ""),
                exit_code_col=str(spec["exit_code_col"]),
                exit_target_col=str(exit_spec.get("target_col") or "sniper_exit_span_target"),
                exit_weight_col=str(exit_spec.get("weight_col") or "sniper_exit_span_weight"),
                seed=int(seed),
                enable_exit_dataset=bool(symbol_cap_exit > 0),
            )
            entry_df = sniper_ds.entry
            if entry_df.empty:
                continue
            if frozen_mask is not None:
                entry_df = entry_df.loc[~frozen_mask.reindex(entry_df.index).fillna(False)].copy()
                if entry_df.empty:
                    continue
            entry_df = _try_downcast_df(entry_df, copy=False)
            try:
                y_raw = entry_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                raw_counts_local[name] = (int(np.sum(y_raw >= 0.5)), int(y_raw.shape[0]))
            except Exception:
                raw_counts_local[name] = (0, int(entry_df.shape[0]))
            entry_df["sym_id"] = int(sym_idx)
            feat_cols = _list_feature_columns(entry_df)
            cols_keep = list(feat_cols) + ["label_entry", "sym_id"]
            if weight_col and weight_col in entry_df.columns:
                cols_keep.append(weight_col)
            entry_df = entry_df.reindex(columns=cols_keep)
            # cap por sÌmbolo para evitar combo gigante
            entry_df = _sample_df(
                entry_df,
                "label_entry",
                float(entry_ratio_neg_per_pos),
                int(symbol_cap),
                weight_col=weight_col,
            )
            frames_local[name] = (entry_df, feat_cols)
            if exit_local is None:
                try:
                    exit_local = sniper_ds.exit
                except Exception:
                    exit_local = None
        exit_payload: tuple[pd.DataFrame, List[str]] | None = None
        if (symbol_cap_exit > 0) and (exit_local is not None) and (not exit_local.empty):
            ex = _try_downcast_df(exit_local, copy=False)
            if frozen_mask is not None:
                ex = ex.loc[~frozen_mask.reindex(ex.index).fillna(False)].copy()
            if not ex.empty:
                ex["sym_id"] = int(sym_idx)
                feat_ex = _list_feature_columns(ex)
                cols_keep_ex = list(feat_ex) + ["label_exit", "sym_id"]
                if "label_exit_weight" in ex.columns:
                    cols_keep_ex.append("label_exit_weight")
                ex = ex.reindex(columns=cols_keep_ex)
                ex = _sample_reg_df(
                    ex,
                    "label_exit",
                    int(symbol_cap_exit),
                    weight_col=("label_exit_weight" if "label_exit_weight" in ex.columns else None),
                )
                exit_payload = (ex, feat_ex)
        return sym_idx, symbol, frames_local, exit_payload, raw_counts_local

    preferred_name = "long" if any(str(s["name"]) == "long" for s in entry_specs) else str(entry_specs[0]["name"])
    total_pos = 0
    total_neg = 0
    entry_raw_pos_map: dict[str, int] = {}
    entry_raw_total_map: dict[str, int] = {}
    batch_flush_n = 8
    done = 0

    def _consume_symbol_result(
        res_symbol: str,
        frames_local: dict[str, tuple[pd.DataFrame, List[str]]] | None,
        exit_payload: tuple[pd.DataFrame, List[str]] | None,
        raw_counts_local: dict[str, tuple[int, int]] | None,
        *,
        fallback_symbol: str,
    ) -> tuple[int, int]:
        nonlocal exit_pool
        pos_n: int | None = None
        neg_n: int | None = None
        if not frames_local:
            symbols_skipped.append(res_symbol or fallback_symbol)
            return 0, 0
        for _name, (_rp, _rt) in (raw_counts_local or {}).items():
            try:
                entry_raw_pos_map[_name] = int(entry_raw_pos_map.get(_name, 0)) + int(_rp)
                entry_raw_total_map[_name] = int(entry_raw_total_map.get(_name, 0)) + int(_rt)
            except Exception:
                pass

        pref = frames_local.get(preferred_name)
        if pref is not None:
            pref_df = pref[0]
            if not pref_df.empty and "label_entry" in pref_df.columns:
                y = pref_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                pos_n = int(np.sum(y >= 0.5))
                neg_n = int(np.sum(y < 0.5))
        if pos_n is None or neg_n is None:
            for _name, (entry_df, _feat_cols) in frames_local.items():
                if not entry_df.empty and "label_entry" in entry_df.columns:
                    y = entry_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                    pos_n = int(np.sum(y >= 0.5))
                    neg_n = int(np.sum(y < 0.5))
                    break

        for name, (entry_df, feat_cols) in frames_local.items():
            if name not in feat_cols_entry_map or not feat_cols_entry_map[name]:
                feat_cols_entry_map[name] = list(feat_cols)
            weight_col = ""
            for spec in entry_specs:
                if str(spec["name"]) == name:
                    weight_col = str(spec.get("weight_col") or "")
                    break
            buf = entry_buf_map.get(name)
            if buf is not None:
                buf.append(entry_df)
                if len(buf) >= batch_flush_n:
                    pool_df = entry_pool_map.get(name)
                    combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
                    pool_df = _sample_df(
                        combined,
                        "label_entry",
                        float(entry_ratio_neg_per_pos),
                        int(max_rows_entry),
                        weight_col=weight_col,
                    )
                    entry_pool_map[name] = pool_df
                    buf.clear()
                    del combined
        if (symbol_cap_exit > 0) and (exit_payload is not None):
            ex_df, ex_cols = exit_payload
            if ex_cols and (not feat_cols_exit):
                feat_cols_exit.extend(list(ex_cols))
            if ex_df is not None and (not ex_df.empty):
                exit_buf.append(ex_df)
                if len(exit_buf) >= batch_flush_n:
                    combined_ex = pd.concat(([exit_pool] if (exit_pool is not None and not exit_pool.empty) else []) + exit_buf, axis=0, ignore_index=False)
                    exit_pool_new = _sample_reg_df(
                        combined_ex,
                        "label_exit",
                        int(max_rows_exit),
                        weight_col=("label_exit_weight" if "label_exit_weight" in combined_ex.columns else None),
                    )
                    exit_buf.clear()
                    del combined_ex
                    exit_pool = exit_pool_new
        symbols_used.append(res_symbol or fallback_symbol)
        return int(pos_n or 0), int(neg_n or 0)

    if parallel and len(symbols) > 1:
        if max_workers is not None and int(max_workers) > 0:
            mw = int(max_workers)
        else:
            env_mw = os.getenv("SNIPER_DATASET_WORKERS", "").strip()
            mw = int(env_mw) if env_mw else min(8, int(os.cpu_count() or 4))
        mw = max(1, mw)
        dataset_policy = _DATASET_GUARD.make_policy()
        print(
            f"[sniper-data] dataset_workers={mw} ram_cap={dataset_policy.max_ram_pct:.1f}% min_free_mb={dataset_policy.min_free_mb:.0f} per_worker_mb={dataset_policy.per_worker_mem_mb:.0f}",
            flush=True,
        )
        for pair_submitted, fut in _DATASET_GUARD.adaptive_map(
            list(enumerate(symbols)),
            lambda pair: _process_symbol(pair[0], pair[1]),
            max_workers=mw,
            policy=dataset_policy,
            task_name="dataset-build",
        ):
            _sym_idx, submitted_symbol = pair_submitted
            pos_n = 0
            neg_n = 0
            current_symbol = submitted_symbol
            try:
                _, res_symbol, frames_local, exit_payload, raw_counts_local = fut.result()
                current_symbol = res_symbol or submitted_symbol
                pos_n, neg_n = _consume_symbol_result(
                    current_symbol,
                    frames_local,
                    exit_payload,
                    raw_counts_local,
                    fallback_symbol=submitted_symbol,
                )
            except Exception:
                symbols_skipped.append(submitted_symbol)
            done += 1
            total_pos += int(pos_n)
            total_neg += int(neg_n)
            _print_progress(
                done,
                current_symbol,
                pos_n=pos_n,
                neg_n=neg_n,
                pos_total=total_pos,
                neg_total=total_neg,
            )
            if done % 5 == 0:
                gc.collect()
    else:
        for sym_idx, symbol in enumerate(symbols):
            pos_n = 0
            neg_n = 0
            current_symbol = symbol
            try:
                _, res_symbol, frames_local, exit_payload, raw_counts_local = _process_symbol(sym_idx, symbol)
                current_symbol = res_symbol or symbol
                pos_n, neg_n = _consume_symbol_result(
                    current_symbol,
                    frames_local,
                    exit_payload,
                    raw_counts_local,
                    fallback_symbol=symbol,
                )
            except Exception:
                symbols_skipped.append(symbol)
            done += 1
            total_pos += int(pos_n)
            total_neg += int(neg_n)
            _print_progress(
                done,
                current_symbol,
                pos_n=pos_n,
                neg_n=neg_n,
                pos_total=total_pos,
                neg_total=total_neg,
            )
            if done % 5 == 0:
                gc.collect()
    prog_ds.close()
    # flush remaining buffers
    for name, buf in entry_buf_map.items():
        if not buf:
            continue
        weight_col = ""
        for spec in entry_specs:
            if str(spec["name"]) == name:
                weight_col = str(spec.get("weight_col") or "")
                break
        pool_df = entry_pool_map.get(name)
        combined = pd.concat(([pool_df] if pool_df is not None and not pool_df.empty else []) + buf, axis=0, ignore_index=False)
        pool_df = _sample_df(
            combined,
            "label_entry",
            float(entry_ratio_neg_per_pos),
            int(max_rows_entry),
            weight_col=weight_col,
        )
        entry_pool_map[name] = pool_df
        buf.clear()
        del combined
    if (symbol_cap_exit > 0) and exit_buf:
        combined_ex = pd.concat(([exit_pool] if (exit_pool is not None and not exit_pool.empty) else []) + exit_buf, axis=0, ignore_index=False)
        exit_pool = _sample_reg_df(
            combined_ex,
            "label_exit",
            int(max_rows_exit),
            weight_col=("label_exit_weight" if "label_exit_weight" in combined_ex.columns else None),
        )
        exit_buf.clear()
        del combined_ex

    entry_batches: dict[str, SniperBatch] = {}
    for spec in entry_specs:
        name = str(spec["name"])
        entry_df = entry_pool_map.get(name, pd.DataFrame())
        if not entry_df.empty:
            entry_df.sort_index(inplace=True)
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        weight_col = str(spec.get("weight_col") or "")
        Xe, ye, we, tse, syme = _to_numpy(entry_df, list(feat_cols), "label_entry", weight_col=weight_col)
        batch = SniperBatch(X=Xe, y=ye, w=we, ts=tse, sym_id=syme, feature_cols=list(feat_cols))
        rt = int(entry_raw_total_map.get(name, 0) or 0)
        if rt > 0:
            setattr(batch, "natural_pos_rate", float(entry_raw_pos_map.get(name, 0) or 0) / float(rt))
            setattr(batch, "natural_pos_rows", rt)
        entry_batches[name] = batch

    entry_short = entry_batches.get("short")
    entry_long = entry_batches.get("long")
    entry_mid = entry_batches.get("mid", entry_long if entry_long is not None else entry_short)
    entry_batch = entry_mid if entry_mid is not None else SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
    )
    if exit_pool is not None and (not exit_pool.empty):
        try:
            exit_pool.sort_index(inplace=True)
        except Exception:
            pass
    Xe_x, ye_x, we_x, tse_x, syme_x = _to_numpy(
        exit_pool if exit_pool is not None else pd.DataFrame(),
        list(feat_cols_exit),
        "label_exit",
        weight_col=("label_exit_weight" if (exit_pool is not None and "label_exit_weight" in getattr(exit_pool, "columns", [])) else None),
    )
    exit_batch = SniperBatch(
        X=Xe_x,
        y=ye_x,
        w=we_x,
        ts=tse_x,
        sym_id=syme_x,
        feature_cols=list(feat_cols_exit),
    )
    empty_batch = SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
    )
    train_end = None
    try:
        ts_entry_mid = entry_mid.ts if entry_mid is not None else np.empty((0,), dtype="datetime64[ns]")
        if ts_entry_mid.size:
            train_end = pd.to_datetime(ts_entry_mid.max())
    except Exception:
        train_end = None

    return SniperDataPack(
        entry=entry_batch,
        entry_short=entry_short,
        entry_mid=entry_mid,
        entry_long=entry_long,
        danger=empty_batch,
        exit=exit_batch,
        contract=contract,
        symbols=list(symbols_used or symbols),
        symbols_used=symbols_used or None,
        symbols_skipped=symbols_skipped or None,
        train_end_utc=train_end,
    )

