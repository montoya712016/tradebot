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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

try:
    from prepare_features.prepare_features import (
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features import pf_config as cfg
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m
    from prepare_features.sniper_dataset import build_sniper_datasets, warmup_sniper_dataset_numba
    from prepare_features.labels import apply_trade_contract_labels
    from trade_contract import TradeContract, DEFAULT_TRADE_CONTRACT
except Exception:
    from prepare_features.prepare_features import (  # type: ignore[import]
        run as pf_run,
        FEATURE_KEYS,
        build_flags,
    )
    from prepare_features import pf_config as cfg  # type: ignore[import]
    from prepare_features.data import load_ohlc_1m_series, to_ohlc_from_1m  # type: ignore[import]
    from prepare_features.sniper_dataset import build_sniper_datasets, warmup_sniper_dataset_numba  # type: ignore[import]
    from prepare_features.labels import apply_trade_contract_labels  # type: ignore[import]
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


def _normalize_entry_label_sides(sides: Sequence[str] | None) -> list[str]:
    if not sides:
        return ["long"]
    out: list[str] = []
    for side in sides:
        s = str(side or "").strip().lower()
        if s in {"long", "short"} and s not in out:
            out.append(s)
    return out or ["long"]


def _entry_label_specs(contract: TradeContract, *, entry_label_sides: Sequence[str] | None = None) -> list[dict[str, str]]:
    windows = list(getattr(contract, "entry_label_windows_minutes", []) or [])
    if len(windows) < 1:
        raise ValueError("entry_label_windows_minutes deve ter ao menos 1 valor")
    sides = _normalize_entry_label_sides(entry_label_sides)
    specs: list[dict[str, str]] = []
    for side in sides:
        prefix = "sniper_long_" if side == "long" else "sniper_short_"
        name_prefix = "long" if side == "long" else "short"
        for w in windows:
            suf = f"{int(w)}m"
            specs.append(
                {
                    "name": f"{name_prefix}_w{int(w)}",
                    "side": side,
                    "window": int(w),
                    "label_col": f"{prefix}label_{suf}",
                    "weight_col": f"{prefix}weight_{suf}",
                    "exit_code_col": f"sniper_exit_code_{suf}",
                    "suffix": suf,
                }
            )
    return specs


def entry_label_specs(contract: TradeContract, *, entry_label_sides: Sequence[str] | None = None) -> list[dict[str, str]]:
    """
    Public helper para manter consistência entre dataflow e treino.
    """
    return _entry_label_specs(contract, entry_label_sides=entry_label_sides)


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


def _prepare_symbol(
    symbol: str,
    *,
    total_days: int,
    remove_tail_days: int,
    flags: Dict[str, bool],
    contract: TradeContract,
    asset_class: str = "crypto",
    entry_label_name: str | None = None,
    entry_label_sides: Sequence[str] | None = None,
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
    entry_specs = _entry_label_specs(contract, entry_label_sides=entry_label_sides)
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
            exit_code_col=str(spec["exit_code_col"]),
            seed=int(seed),
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
    entry_label_sides: Sequence[str] | None = None,
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
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
    debug_ds = os.getenv("SNIPER_DS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_ds = os.getenv("SNIPER_DS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_ds = os.getenv("SNIPER_DS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_ds = os.getenv("SNIPER_DS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    # cap por simbolo: proporcional ao tamanho total para reduzir custo
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    entry_specs = _entry_label_specs(contract, entry_label_sides=entry_label_sides)
    if entry_label_name:
        want = str(entry_label_name).strip().lower()
        entry_specs = [s for s in entry_specs if str(s.get("name", "")).lower() == want]
        if not entry_specs:
            raise ValueError(f"entry_label_name invalido: {entry_label_name}")
    entry_pool_map: dict[str, pd.DataFrame] = {str(s["name"]): pd.DataFrame() for s in entry_specs}
    feat_cols_entry_map: dict[str, List[str]] = {}
    symbols_used: List[str] = []
    symbols_skipped: List[str] = []
    total_syms = len(symbols)

    def _fmt_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        m, s = divmod(int(seconds + 0.5), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}h{m:02d}m"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    start_time = time.time()

    last_len = 0

    def _print_progress(
        i: int,
        sym: str,
        *,
        pos_n: int | None = None,
        neg_n: int | None = None,
        pos_total: int | None = None,
        neg_total: int | None = None,
    ) -> None:
        if total_syms <= 1:
            return
        nonlocal last_len
        w = 24
        done = min(max(int(round(w * (i / total_syms))), 0), w)
        bar = "#" * done + "-" * (w - done)
        elapsed = max(time.time() - start_time, 0.0)
        eta_sec = ((elapsed / i) * (total_syms - i)) if i > 0 else 0.0
        eta = _fmt_eta(eta_sec)
        if pos_n is None or neg_n is None:
            counts = "p- n-"
        else:
            counts = f"p{pos_n} n{neg_n}"
        if pos_total is None or neg_total is None:
            totals = "pt- nt-"
        else:
            totals = f"pt{pos_total} nt{neg_total}"
        line = f"[sniper-ds] [{bar}] {i}/{total_syms} | {sym} | eta {eta} | {counts} | {totals}"
        if len(line) < last_len:
            line = line + (" " * (last_len - len(line)))
        last_len = max(last_len, len(line))
        if debug_ds:
            print(line, flush=True)
        else:
            print(line, end="\r", flush=True)

    def _sample_df(
        df_in: pd.DataFrame,
        label_col: str,
        ratio_neg_per_pos: float,
        max_rows: int,
        weight_col: str | None = None,
        rng_local: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        rng_use = rng_local if rng_local is not None else rng
        y = df_in[label_col].to_numpy()
        pos_idx = np.flatnonzero(y >= 0.5)
        neg_idx = np.flatnonzero(y < 0.5)
        if pos_idx.size == 0:
            # sem positivos -> evita gerar dataset extremamente enviesado
            return df_in.iloc[0:0].copy()

        ratio = float(max(0.0, ratio_neg_per_pos))
        max_rows = int(max_rows)
        pos_keep_n = int(pos_idx.size)
        if max_rows > 0:
            max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
            pos_keep_n = min(pos_keep_n, max_pos)

        if pos_keep_n < pos_idx.size:
            pos_keep = rng_use.choice(pos_idx, size=pos_keep_n, replace=False)
        else:
            pos_keep = pos_idx

        neg_target = int(round(pos_keep_n * ratio))
        if max_rows > 0:
            max_neg_allowed = max_rows - pos_keep_n
            if max_neg_allowed < neg_target:
                neg_target = max(0, max_neg_allowed)

        if neg_target <= 0 or neg_idx.size == 0:
            keep = pos_keep
        else:
            if neg_target >= neg_idx.size:
                neg_keep = neg_idx
            else:
                neg_keep = rng_use.choice(neg_idx, size=neg_target, replace=False)
            keep = np.concatenate([pos_keep, neg_keep])
        return df_in.iloc[np.sort(keep)].copy()

    def _sample_indices(
        labels: np.ndarray,
        valid_mask: np.ndarray,
        ratio_neg_per_pos: float,
        max_rows: int,
        rng_local: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Retorna índices (sobre o df original) já sampleados, sem materializar entry_df inteiro.
        """
        if valid_mask.size == 0 or (not bool(np.any(valid_mask))):
            return np.empty((0,), dtype=np.int64)
        rng_use = rng_local if rng_local is not None else rng
        base_idx = np.flatnonzero(valid_mask)
        y = labels[base_idx]
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
            pos_keep = rng_use.choice(pos_idx, size=pos_keep_n, replace=False)
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
                neg_keep = rng_use.choice(neg_idx, size=neg_target, replace=False)
            keep_local = np.concatenate([pos_keep, neg_keep])
        keep = base_idx[np.sort(keep_local)]
        return keep
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

    preferred_name = "mid" if any(str(s["name"]) == "mid" for s in entry_specs) else str(entry_specs[0]["name"])

    total_pos = 0
    total_neg = 0
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
            for spec in entry_specs:
                name = str(spec["name"])
                weight_col = str(spec.get("weight_col") or "")
                sniper_ds = build_sniper_datasets(
                    df_pf,
                    contract=contract,
                    entry_label_col=str(spec["label_col"]),
                    exit_code_col=str(spec["exit_code_col"]),
                    seed=int(seed),
                )
                entry_df = sniper_ds.entry
                if entry_df.empty:
                    continue
                entry_df = _try_downcast_df(entry_df, copy=False)
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
                        combined = pd.concat(
                            ([pool_df] if pool_df is not None and not pool_df.empty else []) + buf,
                            axis=0,
                            ignore_index=False,
                        )
                        if bool(full_entry_pool) and int(max_rows_entry) <= 0:
                            pool_df = combined
                        else:
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
    print("", flush=True)
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
        if bool(full_entry_pool) and int(max_rows_entry) <= 0:
            pool_df = combined
        else:
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

    entry_batches: dict[str, SniperBatch] = {}
    for spec in entry_specs:
        name = str(spec["name"])
        entry_df = entry_pool_map.get(name, pd.DataFrame())
        if not entry_df.empty:
            entry_df.sort_index(inplace=True)
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        weight_col = str(spec.get("weight_col") or "")
        Xe, ye, we, tse, syme = _to_numpy(entry_df, list(feat_cols), "label_entry", weight_col=weight_col)
        entry_batches[name] = SniperBatch(X=Xe, y=ye, w=we, ts=tse, sym_id=syme, feature_cols=list(feat_cols))

    base_name = str(entry_specs[0]["name"])
    entry_short = entry_batches.get("short")
    entry_mid = entry_batches.get("mid", entry_short)
    entry_long = entry_batches.get("long", entry_mid)
    if entry_mid is None and base_name in entry_batches:
        entry_mid = entry_batches.get(base_name)
    entry_batch = entry_mid if entry_mid is not None else entry_batches.get(base_name)
    if entry_batch is None:
        entry_batch = SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
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
        ts_entry_mid = entry_mid.ts if entry_mid is not None else (entry_batch.ts if entry_batch is not None else np.empty((0,), dtype="datetime64[ns]"))
        if ts_entry_mid.size:
            train_end = pd.to_datetime(ts_entry_mid.max())
    except Exception:
        train_end = None

    return SniperDataPack(
        entry=entry_batch,
        entry_short=entry_short,
        entry_mid=entry_mid,
        entry_long=entry_long,
        entry_map=entry_batches,
        danger=empty_batch,
        exit=empty_batch,
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
    entry_label_sides: Sequence[str] | None = None,
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
    max_rows_entry: int = 2_000_000,
    full_entry_pool: bool = False,
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
    debug_ds = os.getenv("SNIPER_DS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    # cap por simbolo: proporcional ao tamanho total para reduzir custo
    per_sym = int(max_rows_entry // max(1, len(symbols))) if max_rows_entry > 0 else 0
    min_cap = _env_int("SNIPER_SYMBOL_CAP_MIN", 20_000)
    max_cap = _env_int("SNIPER_SYMBOL_CAP_MAX", 100_000)
    symbol_cap = max(int(min_cap), min(int(max_cap), per_sym if per_sym > 0 else int(max_cap)))
    if bool(full_entry_pool):
        symbol_cap = 0
    entry_specs = _entry_label_specs(contract, entry_label_sides=entry_label_sides)
    if entry_label_name:
        want = str(entry_label_name).strip().lower()
        entry_specs = [s for s in entry_specs if str(s.get("name", "")).lower() == want]
        if not entry_specs:
            raise ValueError(f"entry_label_name invalido: {entry_label_name}")
    entry_pool_map: dict[str, pd.DataFrame] = {str(s["name"]): pd.DataFrame() for s in entry_specs}
    feat_cols_entry_map: dict[str, List[str]] = {}
    symbols_used: List[str] = []
    symbols_skipped: List[str] = []
    total_syms = len(symbols)

    def _fmt_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        m, s = divmod(int(seconds + 0.5), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}h{m:02d}m"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    start_time = time.time()

    last_len = 0

    def _print_progress(
        i: int,
        sym: str,
        *,
        pos_n: int | None = None,
        neg_n: int | None = None,
        pos_total: int | None = None,
        neg_total: int | None = None,
    ) -> None:
        if total_syms <= 1:
            return
        nonlocal last_len
        w = 24
        done = min(max(int(round(w * (i / total_syms))), 0), w)
        bar = "#" * done + "-" * (w - done)
        elapsed = max(time.time() - start_time, 0.0)
        eta = _fmt_eta((elapsed / i) * (total_syms - i)) if i > 0 else "?"
        if pos_n is None or neg_n is None:
            counts = "p- n-"
        else:
            counts = f"p{pos_n} n{neg_n}"
        if pos_total is None or neg_total is None:
            totals = "pt- nt-"
        else:
            totals = f"pt{pos_total} nt{neg_total}"
        line = f"[sniper-ds] [{bar}] {i}/{total_syms} | {sym} | eta {eta} | {counts} | {totals}"
        if len(line) < last_len:
            line = line + (" " * (last_len - len(line)))
        last_len = max(last_len, len(line))
        if debug_ds:
            print(line, flush=True)
        else:
            print(line, end="\r", flush=True)

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
            keep = pos_keep
        else:
            if neg_target >= neg_idx.size:
                neg_keep = neg_idx
            else:
                neg_keep = rng.choice(neg_idx, size=neg_target, replace=False)
            keep = np.concatenate([pos_keep, neg_keep])
        return df_in.iloc[np.sort(keep)].copy()

    def _sample_indices(
        labels: np.ndarray,
        valid_mask: np.ndarray,
        ratio_neg_per_pos: float,
        max_rows: int,
        rng_local: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Retorna indices (sobre o df original) ja sampleados, sem materializar entry_df inteiro.
        """
        if valid_mask.size == 0 or (not bool(np.any(valid_mask))):
            return np.empty((0,), dtype=np.int64)
        rng_use = rng_local if rng_local is not None else rng
        base_idx = np.flatnonzero(valid_mask)
        y = labels[base_idx]
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
            pos_keep = rng_use.choice(pos_idx, size=pos_keep_n, replace=False)
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
                neg_keep = rng_use.choice(neg_idx, size=neg_target, replace=False)
            keep_local = np.concatenate([pos_keep, neg_keep])
        keep = base_idx[np.sort(keep_local)]
        return keep

    def _merge_sample_pool(
        pool_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        label_col: str,
        ratio_neg_per_pos: float,
        max_rows: int,
    ) -> pd.DataFrame:
        """
        Faz o mesmo sampling que _sample_df em (pool+entry), mas sem concatenar
        o dataframe inteiro (economia de RAM).
        """
        if entry_df is None or entry_df.empty:
            return pool_df
        if pool_df is None or pool_df.empty:
            return _sample_df(entry_df, label_col, ratio_neg_per_pos, max_rows)
        y_pool = pool_df[label_col].to_numpy()
        y_new = entry_df[label_col].to_numpy()
        pos_pool = np.flatnonzero(y_pool >= 0.5)
        neg_pool = np.flatnonzero(y_pool < 0.5)
        pos_new = np.flatnonzero(y_new >= 0.5)
        neg_new = np.flatnonzero(y_new < 0.5)
        total_pos = pos_pool.size + pos_new.size
        if total_pos == 0:
            return pool_df.iloc[0:0].copy()
        ratio = float(max(0.0, ratio_neg_per_pos))
        max_rows = int(max_rows)
        pos_keep_n = int(total_pos)
        if max_rows > 0:
            max_pos = int(max(1, round(max_rows / (1.0 + ratio))))
            pos_keep_n = min(pos_keep_n, max_pos)
        # sample positivos no espaço combinado
        if pos_keep_n < total_pos:
            pos_keep = rng.choice(np.arange(total_pos), size=pos_keep_n, replace=False)
        else:
            pos_keep = np.arange(total_pos, dtype=np.int64)
        pos_keep.sort()
        pos_keep_pool = pos_keep[pos_keep < pos_pool.size]
        pos_keep_new = pos_keep[pos_keep >= pos_pool.size] - pos_pool.size
        # alvo de negativos
        neg_target = int(round(pos_keep_n * ratio))
        if max_rows > 0:
            max_neg_allowed = max_rows - pos_keep_n
            if max_neg_allowed < neg_target:
                neg_target = max(0, max_neg_allowed)
        if neg_target <= 0 or (neg_pool.size + neg_new.size) == 0:
            keep_pool = pos_pool[pos_keep_pool]
            keep_new = pos_new[pos_keep_new]
            out = pd.concat(
                [pool_df.iloc[keep_pool], entry_df.iloc[keep_new]],
                axis=0,
                ignore_index=False,
            )
            return out.copy()
        total_neg = neg_pool.size + neg_new.size
        if neg_target >= total_neg:
            neg_keep = np.arange(total_neg, dtype=np.int64)
        else:
            neg_keep = rng.choice(np.arange(total_neg), size=neg_target, replace=False)
        neg_keep.sort()
        neg_keep_pool = neg_keep[neg_keep < neg_pool.size]
        neg_keep_new = neg_keep[neg_keep >= neg_pool.size] - neg_pool.size
        keep_pool = np.concatenate([pos_pool[pos_keep_pool], neg_pool[neg_keep_pool]])
        keep_new = np.concatenate([pos_new[pos_keep_new], neg_new[neg_keep_new]])
        # mantém ordem relativa (pool vem antes do entry)
        out = pd.concat(
            [pool_df.iloc[np.sort(keep_pool)], entry_df.iloc[np.sort(keep_new)]],
            axis=0,
            ignore_index=False,
        )
        return out.copy()
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

    def _process_symbol(sym_idx: int, symbol: str):
        rng_local = np.random.default_rng(int(seed) + int(remove_tail_days) + (sym_idx + 1) * 101)
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
        allow = getattr(cfg, "FEATURE_ALLOWLIST", None)
        allow_set = set(allow) if isinstance(allow, (list, tuple)) and allow else None
        cycle_cols = [
            "cycle_is_add",
            "cycle_num_adds",
            "cycle_time_in_trade",
            "cycle_dd_pct",
            "cycle_avg_entry_price",
            "cycle_last_fill_price",
        ]
        # garante labels 1x se alguma janela estiver ausente
        if any(str(spec["label_col"]) not in df.columns for spec in entry_specs):
            apply_trade_contract_labels(
                df,
                contract=contract,
                candle_sec=int(getattr(contract, "timeframe_sec", 60) or 60),
            )
        base_spec = entry_specs[0]
        base_label_col = str(base_spec["label_col"])
        if base_label_col not in df.columns:
            raise RuntimeError(f"{symbol}: label base ausente ({base_label_col})")
        label_cols = [str(spec["label_col"]) for spec in entry_specs if str(spec["label_col"]) in df.columns]
        weight_cols = [str(spec.get("weight_col") or "") for spec in entry_specs if str(spec.get("weight_col") or "") in df.columns]
        entry_mask = np.zeros((len(df),), dtype=bool)
        for lc in label_cols:
            entry_mask |= df[lc].notna().to_numpy(dtype=bool, copy=False)
        if frozen_mask is not None:
            try:
                fm = frozen_mask.to_numpy(dtype=bool, copy=False)
                if fm.shape[0] == entry_mask.shape[0]:
                    entry_mask = entry_mask & (~fm)
            except Exception:
                pass
        if entry_mask.size == 0 or (not bool(np.any(entry_mask))):
            if debug_ds:
                print(f"[sniper-ds][debug] {symbol}: entry_mask vazio", flush=True)
            return sym_idx, symbol, {}
        base_feat_cols = _list_feature_columns(df, mask=entry_mask)
        feat_cols = list(base_feat_cols)
        for cc in cycle_cols:
            if allow_set is None or cc in allow_set:
                feat_cols.append(cc)
        labels_base = df[base_label_col].to_numpy(copy=False)
        if debug_ds and sym_idx < 3:
            try:
                base_idx = np.flatnonzero(entry_mask)
                pos_any = np.zeros((base_idx.size,), dtype=bool)
                for lc in label_cols:
                    arr = df[lc].to_numpy(copy=False)
                    pos_any |= (arr[base_idx] >= 0.5)
                pos0 = int(np.sum(pos_any))
                neg0 = int(base_idx.size - pos0)
                print(
                    f"[sniper-ds][debug] {symbol}: mask={int(entry_mask.sum())} pos={pos0} neg={neg0}",
                    flush=True,
                )
            except Exception:
                pass
        if full_entry_pool:
            # mantém todos os positivos (união das janelas) e amostra negativos na razão pos x alpha
            base_idx = np.flatnonzero(entry_mask)
            pos_any = np.zeros((base_idx.size,), dtype=bool)
            for lc in label_cols:
                arr = df[lc].to_numpy(copy=False)
                pos_any |= (arr[base_idx] >= 0.5)
            pos_idx = np.flatnonzero(pos_any)
            neg_idx = np.flatnonzero(~pos_any)
            if pos_idx.size == 0:
                keep_local = base_idx
            else:
                target_neg = int(round(pos_idx.size * float(entry_ratio_neg_per_pos)))
                target_neg = min(int(neg_idx.size), max(0, target_neg))
                if target_neg > 0:
                    # usa o maior peso entre as janelas para amostrar negativos
                    if weight_cols:
                        wmax = np.zeros((base_idx.size,), dtype=np.float64)
                        for wc in weight_cols:
                            w = df[wc].to_numpy(dtype=np.float64, copy=False)[base_idx]
                            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                            wmax = np.maximum(wmax, w)
                        wneg = wmax[neg_idx]
                        if float(wneg.sum()) > 0.0:
                            p = wneg / float(wneg.sum())
                            neg_keep = rng_local.choice(neg_idx, size=target_neg, replace=False, p=p)
                        else:
                            neg_keep = rng_local.choice(neg_idx, size=target_neg, replace=False)
                    else:
                        neg_keep = rng_local.choice(neg_idx, size=target_neg, replace=False)
                    keep_local = np.concatenate([pos_idx, neg_keep])
                else:
                    keep_local = pos_idx
            keep_idx = base_idx[np.sort(keep_local)]
        else:
            keep_idx = _sample_indices(
                labels_base,
                entry_mask,
                float(entry_ratio_neg_per_pos),
                int(symbol_cap),
                rng_local=rng_local,
            )
        if keep_idx.size == 0:
            if debug_ds:
                try:
                    y0 = labels_base[entry_mask]
                    pos0 = int(np.sum(y0 >= 0.5))
                    neg0 = int(np.sum(y0 < 0.5))
                    print(
                        f"[sniper-ds][debug] {symbol}: keep_idx vazio | mask={int(entry_mask.sum())} pos={pos0} neg={neg0}",
                        flush=True,
                    )
                except Exception:
                    print(f"[sniper-ds][debug] {symbol}: keep_idx vazio", flush=True)
            return sym_idx, symbol, {}
        for spec in entry_specs:
            name = str(spec["name"])
            weight_col = str(spec.get("weight_col") or "")
            entry_label_col = str(spec["label_col"])
            if entry_label_col not in df.columns:
                continue
            cols_keep = list(base_feat_cols)
            if "close" not in cols_keep:
                cols_keep.append("close")
            if entry_label_col not in cols_keep:
                cols_keep.append(entry_label_col)
            if weight_col and weight_col in df.columns and weight_col not in cols_keep:
                cols_keep.append(weight_col)
            entry_df = df.iloc[keep_idx][cols_keep].copy()
            if entry_df.empty:
                continue
            entry_df = _try_downcast_df(entry_df, copy=False)
            entry_df["sym_id"] = int(sym_idx)
            entry_df["cycle_is_add"] = 0
            entry_df["cycle_num_adds"] = 0
            entry_df["cycle_time_in_trade"] = 0
            entry_df["cycle_dd_pct"] = 0.0
            entry_df["cycle_avg_entry_price"] = entry_df["close"]
            entry_df["cycle_last_fill_price"] = entry_df["close"]
            entry_df["label_entry"] = np.nan_to_num(entry_df[entry_label_col].to_numpy(dtype=np.float32, copy=False), nan=0.0).astype(np.uint8)
            cols_keep_final = list(feat_cols) + ["label_entry", "sym_id"]
            if weight_col and weight_col in entry_df.columns:
                entry_df[weight_col] = np.nan_to_num(entry_df[weight_col].to_numpy(dtype=np.float32, copy=False), nan=0.0)
                cols_keep_final.append(weight_col)
            entry_df = entry_df.reindex(columns=cols_keep_final)
            frames_local[name] = (entry_df, feat_cols)
        try:
            del df
        except Exception:
            pass
        return sym_idx, symbol, frames_local

    preferred_name = "mid" if any(str(s["name"]) == "mid" for s in entry_specs) else str(entry_specs[0]["name"])
    total_pos = 0
    total_neg = 0
    done = 0

    def _handle_symbol_result(symbol: str, frames_local: dict[str, tuple[pd.DataFrame, List[str]]]):
        nonlocal total_pos, total_neg, done
        pos_n = None
        neg_n = None
        if not frames_local:
            symbols_skipped.append(symbol)
            if debug_ds and done < 3:
                print(f"[sniper-ds][debug] {symbol}: frames_local vazio", flush=True)
        else:
            pref = frames_local.get(preferred_name)
            if pref is not None:
                pref_df = pref[0]
                if not pref_df.empty and "label_entry" in pref_df.columns:
                    y = pref_df["label_entry"].to_numpy(dtype=np.float32, copy=False)
                    pos_n = int(np.sum(y >= 0.5))
                    neg_n = int(np.sum(y < 0.5))
                    if debug_ds and done < 3:
                        try:
                            print(
                                f"[sniper-ds][debug] {symbol}: entry_rows={int(len(pref_df))} pos_kept={pos_n} neg_kept={neg_n}",
                                flush=True,
                            )
                        except Exception:
                            pass
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
                pool_df = entry_pool_map.get(name)
                if full_entry_pool:
                    if pool_df is None or pool_df.empty:
                        entry_pool_map[name] = entry_df.copy()
                    else:
                        entry_pool_map[name] = pd.concat([pool_df, entry_df], axis=0, ignore_index=False).copy()
                else:
                    pool_df = _merge_sample_pool(
                        pool_df,
                        entry_df,
                        "label_entry",
                        float(entry_ratio_neg_per_pos),
                        int(max_rows_entry),
                    )
                    entry_pool_map[name] = pool_df
                # liberar refs o quanto antes
                try:
                    del entry_df
                except Exception:
                    pass
            symbols_used.append(symbol)
        done += 1
        if pos_n is None or neg_n is None:
            pos_n, neg_n = 0, 0
        total_pos += int(pos_n)
        total_neg += int(neg_n)
        _print_progress(
            done,
            symbol,
            pos_n=pos_n,
            neg_n=neg_n,
            pos_total=total_pos,
            neg_total=total_neg,
        )
        if done % 5 == 0:
            gc.collect()

    use_parallel = bool(parallel)
    if max_workers is None:
        env_w = os.getenv("SNIPER_DATASET_WORKERS", "").strip()
        if env_w:
            try:
                max_workers = int(env_w)
            except Exception:
                max_workers = None
    if max_workers is None:
        max_workers = min(8, int(os.cpu_count() or 8))
    if (not use_parallel) or int(max_workers or 0) <= 1:
        for sym_idx, symbol in enumerate(symbols):
            try:
                _, res_symbol, frames_local = _process_symbol(sym_idx, symbol)
                _handle_symbol_result(res_symbol, frames_local)
            except Exception:
                _handle_symbol_result(symbol, {})
    else:
        # Mantem o numero de resultados "em voo" limitado a max_workers
        # para evitar acumulo de frames_local grandes em memoria.
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            futures = {}
            it = iter(enumerate(symbols))
            for _ in range(int(max_workers)):
                try:
                    sym_idx, symbol = next(it)
                except StopIteration:
                    break
                futures[ex.submit(_process_symbol, sym_idx, symbol)] = symbol
            while futures:
                done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    symbol = futures.pop(fut, "?")
                    try:
                        _sym_idx, res_symbol, frames_local = fut.result()
                        _handle_symbol_result(res_symbol, frames_local)
                    except Exception:
                        _handle_symbol_result(symbol, {})
                    try:
                        sym_idx, symbol = next(it)
                    except StopIteration:
                        continue
                    futures[ex.submit(_process_symbol, sym_idx, symbol)] = symbol
    print("", flush=True)
    entry_batches: dict[str, SniperBatch] = {}
    for spec in entry_specs:
        name = str(spec["name"])
        entry_df = entry_pool_map.get(name, pd.DataFrame())
        if not entry_df.empty:
            entry_df.sort_index(inplace=True)
        feat_cols = feat_cols_entry_map.get(name, []) if not entry_df.empty else []
        weight_col = str(spec.get("weight_col") or "")
        Xe, ye, we, tse, syme = _to_numpy(entry_df, list(feat_cols), "label_entry", weight_col=weight_col)
        entry_batches[name] = SniperBatch(X=Xe, y=ye, w=we, ts=tse, sym_id=syme, feature_cols=list(feat_cols))

    base_name = str(entry_specs[0]["name"])
    entry_short = entry_batches.get("short")
    entry_mid = entry_batches.get("mid", entry_short)
    entry_long = entry_batches.get("long", entry_mid)
    if entry_mid is None and base_name in entry_batches:
        entry_mid = entry_batches.get(base_name)
    entry_batch = entry_mid if entry_mid is not None else entry_batches.get(base_name)
    if entry_batch is None:
        entry_batch = SniperBatch(
        X=np.empty((0, 0), dtype=np.float32),
        y=np.empty((0,), dtype=np.float32),
        w=np.empty((0,), dtype=np.float32),
        ts=np.empty((0,), dtype="datetime64[ns]"),
        sym_id=np.empty((0,), dtype=np.int32),
        feature_cols=[],
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
        ts_entry_mid = entry_mid.ts if entry_mid is not None else (entry_batch.ts if entry_batch is not None else np.empty((0,), dtype="datetime64[ns]"))
        if ts_entry_mid.size:
            train_end = pd.to_datetime(ts_entry_mid.max())
    except Exception:
        train_end = None

    return SniperDataPack(
        entry=entry_batch,
        entry_short=entry_short,
        entry_mid=entry_mid,
        entry_long=entry_long,
        entry_map=entry_batches,
        danger=empty_batch,
        exit=empty_batch,
        contract=contract,
        symbols=list(symbols_used or symbols),
        symbols_used=symbols_used or None,
        symbols_skipped=symbols_skipped or None,
        train_end_utc=train_end,
    )

