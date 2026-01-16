# -*- coding: utf-8 -*-
"""
Dataflow específico do Sniper:
- calcula features completos via prepare_features
- gera datasets Entry/Add e Danger usando build_sniper_datasets
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

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
}


def _list_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in DROP_COLS_EXACT:
            continue
        if any(c.startswith(pref) for pref in DROP_COL_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


@dataclass
class SniperBatch:
    X: np.ndarray
    y: np.ndarray
    ts: np.ndarray
    sym_id: np.ndarray
    feature_cols: List[str]


@dataclass
class SniperDataPack:
    entry: SniperBatch
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


def _cache_dir() -> Path:
    v = os.getenv("SNIPER_FEATURE_CACHE_DIR", "").strip()
    if v:
        return Path(v).expanduser().resolve()
    try:
        from utils.paths import feature_cache_root  # type: ignore

        return feature_cache_root()
    except Exception:
        try:
            from utils.paths import feature_cache_root  # type: ignore[import]

            return feature_cache_root()
        except Exception:
            # fallback extremo (mantém compat)
            return _repo_root() / "cache_sniper" / "features_pf_1m"


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


def _try_downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    # reduz bastante RAM/disco sem quebrar as labels
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32, copy=False)
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
) -> Dict[str, Path]:
    """
    Garante que existe um cache de features+labels Sniper por símbolo (computado 1x).
    Retorna mapa symbol -> caminho do arquivo cache.
    """
    cache_dir = cache_dir or _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fmt = _cache_format()
    refresh = _cache_refresh() if refresh is None else bool(refresh)
    flags_run = dict(flags)
    # evita quebrar barra de progresso com logs de features
    flags_run["_quiet"] = True

    out: Dict[str, Path] = {}
    symbols = list(symbols)
    hits: List[str] = []
    to_build: List[str] = []
    skipped: List[str] = []
    timings: List[Dict[str, float | int | str]] = []
    t_lock = threading.Lock()

    for sym in symbols:
        data_path, meta_path = _symbol_cache_paths(sym, cache_dir, fmt)
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
        f"[cache] features sniper: dir={cache_dir} fmt={fmt} refresh={bool(refresh)} | total={len(symbols)} hit={len(hits)} build={len(to_build)}",
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
            raw = load_ohlc_1m_series(sym, int(total_days), remove_tail_days=0)
            t_load = time.perf_counter()
            if raw.empty:
                raise RuntimeError("sem dados 1m no intervalo solicitado")
            df_all = to_ohlc_from_1m(raw, 60)
            t_ohlc = time.perf_counter()
            if df_all.empty or int(len(df_all)) < 500:
                raise RuntimeError(f"sem OHLC suficiente (rows={len(df_all)})")
            if df_all[["open", "high", "low", "close"]].isna().all(axis=None):
                raise RuntimeError("OHLC inválido (todos NaN)")

            df_pf = pf_run(
                df_all,
                flags=flags_run,
                plot=False,
                trade_contract=contract,
            )
            df_pf = _try_downcast_df(df_pf)
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
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            with t_lock:
                timings.append(
                    {
                        "symbol": sym,
                        "rows": int(len(df_pf)),
                        "load_s": float(t_load - t0),
                        "ohlc_s": float(t_ohlc - t_load),
                        "features_s": float(t_pf - t_ohlc),
                        "save_s": float(t_save - t_pf),
                        "total_s": float(t_save - t0),
                    }
                )

            del df_pf, df_all, raw
            gc.collect()
            return sym, real_path, None
        except Exception as e:
            try:
                del df_pf  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                del df_all  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                del raw  # type: ignore[name-defined]
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
    return int(max(contract.timeout_bars(candle_sec), contract.danger_horizon_bars(candle_sec)))


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
    - usamos um default ligado ao contrato (tp/sl), com um piso mínimo.
    """
    try:
        tp = float(getattr(contract, "tp_min_pct", 0.02) or 0.02)
    except Exception:
        tp = 0.02
    try:
        sl = float(getattr(contract, "sl_pct", 0.03) or 0.03)
    except Exception:
        sl = 0.03
    # NOTE: tp/sl podem ser 5% dependendo do contrato; usar frações menores pra não inflar demais o label_exit.
    # Limiar mais permissivo para gerar mais positivos no ExitScore.
    return float(max(0.008, 0.20 * tp, 0.12 * sl))


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    raw = load_ohlc_1m_series(symbol, int(total_days), remove_tail_days=0)
    if raw.empty:
        raise RuntimeError(f"{symbol}: sem dados 1m no intervalo solicitado")
    df_all = to_ohlc_from_1m(raw, 60)
    if remove_tail_days > 0:
        cutoff = df_all.index[-1] - pd.Timedelta(days=int(remove_tail_days))
        df_all = df_all[df_all.index < cutoff]
    # evita rodar features em df vazio (gera rows=0 e atrapalha o treino)
    if df_all.empty or int(len(df_all)) < 500:
        raise RuntimeError(f"{symbol}: sem OHLC suficiente após corte (rows={len(df_all)})")
    # se OHLC estiver todo NaN, pula
    if df_all[["open","high","low","close"]].isna().all(axis=None):
        raise RuntimeError(f"{symbol}: OHLC inválido (todos NaN) após corte")

    df_pf = pf_run(
        df_all,
        flags=flags,
        plot=False,
        trade_contract=contract,
    )
    # Limita geração de ADD contexts (evita O(N*timeout) em séries enormes)
    max_add_starts = _env_int("SNIPER_MAX_ADD_STARTS", 20_000)
    seed = _env_int("SNIPER_SEED", 1337)
    # Snapshots para EXIT model (mais barato que features, mas pode crescer se o stride for pequeno)
    max_exit_starts = _env_int("SNIPER_MAX_EXIT_STARTS", 8_000)
    exit_stride_bars = _env_int("SNIPER_EXIT_STRIDE_BARS", 15)
    exit_lookahead_bars = _env_int("SNIPER_EXIT_LOOKAHEAD_BARS", 0)
    try:
        env_exit_margin = os.getenv("SNIPER_EXIT_MARGIN_PCT")
        if env_exit_margin is None or str(env_exit_margin).strip() == "":
            exit_margin_pct = _default_exit_margin_pct(contract)
        else:
            exit_margin_pct = float(env_exit_margin)
    except Exception:
        exit_margin_pct = _default_exit_margin_pct(contract)
    sniper_ds = build_sniper_datasets(
        df_pf,
        contract=contract,
        max_add_starts=int(max_add_starts),
        max_exit_starts=int(max_exit_starts),
        exit_stride_bars=int(exit_stride_bars),
        exit_lookahead_bars=(int(exit_lookahead_bars) if int(exit_lookahead_bars) > 0 else None),
        exit_margin_pct=float(exit_margin_pct),
        seed=int(seed),
    )
    entry_df = pd.concat([sniper_ds.entry, sniper_ds.add], axis=0, ignore_index=False)
    entry_df.sort_index(inplace=True)
    danger_df = sniper_ds.danger
    exit_df = sniper_ds.exit

    feat_cols_entry = _list_feature_columns(entry_df)
    feat_cols_danger = _list_feature_columns(danger_df)
    feat_cols_exit = _list_feature_columns(exit_df)
    # df_pf é enorme e não é necessário fora desta função -> libera cedo
    del df_pf
    return entry_df, danger_df, exit_df, feat_cols_entry, feat_cols_danger, feat_cols_exit


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
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
    danger_ratio_neg_per_pos: float = 4.0,
    exit_ratio_neg_per_pos: float = 4.0,
    max_rows_entry: int = 600_000,
    max_rows_danger: int = 600_000,
    max_rows_exit: int = 600_000,
    # regressões (lucro/tempo)
    seed: int = 42,
) -> SniperDataPack:
    contract = contract or DEFAULT_TRADE_CONTRACT
    # Warmup do Numba (evita aparência de travamento na 1a compilação do kernel de ADD)
    try:
        warmup_sniper_dataset_numba()
    except Exception:
        pass
    entry_frames: List[pd.DataFrame] = []
    danger_frames: List[pd.DataFrame] = []
    exit_frames: List[pd.DataFrame] = []
    feat_cols_entry: List[str] | None = None
    feat_cols_danger: List[str] | None = None
    feat_cols_exit: List[str] | None = None
    sym_ids_entry: List[np.ndarray] = []
    sym_ids_danger: List[np.ndarray] = []
    sym_ids_exit: List[np.ndarray] = []
    ts_entry_parts: List[np.ndarray] = []
    ts_danger_parts: List[np.ndarray] = []
    ts_exit_parts: List[np.ndarray] = []
    X_entry_parts: List[np.ndarray] = []
    X_danger_parts: List[np.ndarray] = []
    X_exit_parts: List[np.ndarray] = []
    y_entry_parts: List[np.ndarray] = []
    y_danger_parts: List[np.ndarray] = []
    y_exit_parts: List[np.ndarray] = []

    rng = np.random.default_rng(int(seed) + int(remove_tail_days))

    def _sample_df(df_in: pd.DataFrame, label_col: str, ratio_neg_per_pos: float, max_rows: int) -> pd.DataFrame:
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        y = df_in[label_col].to_numpy()
        pos_idx = np.flatnonzero(y >= 0.5)
        neg_idx = np.flatnonzero(y < 0.5)
        max_rows = int(max_rows)
        if max_rows <= 0:
            return df_in

        # Estratégia robusta:
        # - tenta aproximar a proporção neg:pos (ratio_neg_per_pos) dentro de max_rows
        # - se pos for muito maior que max_rows, também faz downsample de positivos
        if pos_idx.size == 0:
            keep = neg_idx
            if keep.size > max_rows:
                keep = rng.choice(keep, size=max_rows, replace=False)
            return df_in.iloc[np.sort(keep)].copy()

        # alvo de positivos para caber no orçamento total (ex.: ratio=6 => ~1/7 do total)
        denom = 1.0 + float(max(0.0, ratio_neg_per_pos))
        pos_target = int(max(1, round(max_rows / denom)))
        pos_keep_n = int(min(pos_idx.size, pos_target))

        if pos_idx.size > pos_keep_n:
            pos_keep = rng.choice(pos_idx, size=pos_keep_n, replace=False)
        else:
            pos_keep = pos_idx

        remain = int(max_rows - pos_keep.size)
        if remain <= 0:
            return df_in.iloc[np.sort(pos_keep)].copy()

        # tenta preencher o resto com negativos (até remain)
        neg_target = int(min(neg_idx.size, remain))
        if neg_target > 0:
            if neg_idx.size > neg_target:
                neg_keep = rng.choice(neg_idx, size=neg_target, replace=False)
            else:
                neg_keep = neg_idx
            keep = np.unique(np.concatenate([pos_keep, neg_keep]))
        else:
            keep = pos_keep

        # se ainda sobrar espaço (poucos negativos), completa com mais positivos
        if keep.size < max_rows and pos_idx.size > keep.size:
            extra = int(min(max_rows - keep.size, pos_idx.size - pos_keep.size))
            if extra > 0:
                remaining_pos = np.setdiff1d(pos_idx, pos_keep, assume_unique=False)
                if remaining_pos.size > extra:
                    extra_pos = rng.choice(remaining_pos, size=extra, replace=False)
                else:
                    extra_pos = remaining_pos
                keep = np.unique(np.concatenate([keep, extra_pos]))

        if keep.size > max_rows:
            keep = rng.choice(keep, size=max_rows, replace=False)
        return df_in.iloc[np.sort(keep)].copy()

    def _sample_arrays(
        X: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        sym: np.ndarray,
        *,
        ratio_neg_per_pos: float,
        max_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.size == 0 or y.size == 0:
            return X, y, ts, sym
        n = int(y.shape[0])
        if max_rows <= 0 or n <= max_rows:
            return X, y, ts, sym
        pos = np.flatnonzero(y >= 0.5)
        neg = np.flatnonzero(y < 0.5)
        max_rows = int(max_rows)

        if pos.size == 0:
            keep = neg
            if keep.size > max_rows:
                keep = rng.choice(keep, size=max_rows, replace=False)
            keep = np.sort(keep)
            return X[keep], y[keep], ts[keep], sym[keep]

        denom = 1.0 + float(max(0.0, ratio_neg_per_pos))
        pos_target = int(max(1, round(max_rows / denom)))
        pos_keep_n = int(min(pos.size, pos_target))
        pos_keep = rng.choice(pos, size=pos_keep_n, replace=False) if pos.size > pos_keep_n else pos

        remain = int(max_rows - pos_keep.size)
        if remain <= 0:
            keep = np.sort(pos_keep)
            return X[keep], y[keep], ts[keep], sym[keep]

        neg_keep_n = int(min(neg.size, remain))
        if neg_keep_n > 0:
            neg_keep = rng.choice(neg, size=neg_keep_n, replace=False) if neg.size > neg_keep_n else neg
            keep = np.unique(np.concatenate([pos_keep, neg_keep]))
        else:
            keep = pos_keep

        # completa com mais positivos se faltar (poucos negativos)
        if keep.size < max_rows and pos.size > pos_keep.size:
            extra = int(min(max_rows - keep.size, pos.size - pos_keep.size))
            if extra > 0:
                remaining_pos = np.setdiff1d(pos, pos_keep, assume_unique=False)
                extra_pos = rng.choice(remaining_pos, size=extra, replace=False) if remaining_pos.size > extra else remaining_pos
                keep = np.unique(np.concatenate([keep, extra_pos]))

        if keep.size > max_rows:
            keep = rng.choice(keep, size=max_rows, replace=False)

        keep = np.sort(keep)
        return X[keep], y[keep], ts[keep], sym[keep]

    symbols = list(symbols)
    if not symbols:
        raise RuntimeError("symbols vazio")

    def _to_numpy(df: pd.DataFrame, feats: List[str], label_col: str, sym_idx: int):
        # IMPORTANTE: df pode ter index duplicado (especialmente em ADD contexts).
        # Então NÃO use df.loc[idx] (pode duplicar linhas). Use máscara posicional.
        mat = df.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
        mask = mat.notnull().all(axis=1).to_numpy(dtype=bool, copy=False)
        if mask.size == 0 or (not bool(np.any(mask))):
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat2 = mat.to_numpy(np.float32, copy=True)
        X = mat2[mask]
        y = df[label_col].to_numpy(dtype=np.float32, copy=False)[mask]
        ts = df.index.to_numpy(dtype="datetime64[ns]")[mask]
        sym_arr = np.full(int(ts.shape[0]), sym_idx, dtype=np.int32)
        return X, y, ts, sym_arr

    # 1) primeiro símbolo sequencial para fixar feature_cols
    entry_df0, danger_df0, exit_df0, feat_entry0, feat_danger0, feat_exit0 = _prepare_symbol(
        symbols[0],
        total_days=total_days,
        remove_tail_days=remove_tail_days,
        flags=GLOBAL_FLAGS_FULL,
        contract=contract,
    )
    # IMPORTANTE: sampling é GLOBAL (por período), então aqui limitamos por símbolo só para evitar RAM explodir
    per_sym_entry_max = int(min(int(max_rows_entry), max(50_000, (int(max_rows_entry) // max(1, len(symbols))) * 2)))
    per_sym_danger_max = int(min(int(max_rows_danger), max(50_000, (int(max_rows_danger) // max(1, len(symbols))) * 2)))
    per_sym_exit_max = int(min(int(max_rows_exit), max(50_000, (int(max_rows_exit) // max(1, len(symbols))) * 2)))
    entry_df0 = _sample_df(entry_df0, "label_entry", entry_ratio_neg_per_pos, per_sym_entry_max)
    danger_df0 = _sample_df(danger_df0, "label_danger", danger_ratio_neg_per_pos, per_sym_danger_max)
    exit_df0 = _sample_df(exit_df0, "label_exit", exit_ratio_neg_per_pos, per_sym_exit_max)
    feat_cols_entry = list(feat_entry0)
    feat_cols_danger = list(feat_danger0)
    feat_cols_exit = list(feat_exit0)

    Xe, ye, tse, syme = _to_numpy(entry_df0, feat_cols_entry, "label_entry", 0)
    Xd, yd, tsd, symd = _to_numpy(danger_df0, feat_cols_danger, "label_danger", 0)
    Xx, yx, tsx, symx = _to_numpy(exit_df0, feat_cols_exit, "label_exit", 0)
    X_entry_parts.append(Xe); y_entry_parts.append(ye); ts_entry_parts.append(tse); sym_ids_entry.append(syme)
    X_danger_parts.append(Xd); y_danger_parts.append(yd); ts_danger_parts.append(tsd); sym_ids_danger.append(symd)
    X_exit_parts.append(Xx); y_exit_parts.append(yx); ts_exit_parts.append(tsx); sym_ids_exit.append(symx)

    # 2) demais símbolos em paralelo (threads) — usa 32 threads da CPU sem cópia entre processos
    use_parallel = _env_bool("PF_SYMBOL_PARALLEL", default=True)
    # default conservador em RAM (cada símbolo pode ter milhões de linhas)
    max_workers = _env_int("PF_SYMBOL_WORKERS", min(4, int(os.cpu_count() or 4)))

    def _worker(sym_idx: int, symbol: str):
        entry_df, danger_df, exit_df, _feat_entry, _feat_danger, _feat_exit = _prepare_symbol(
            symbol,
            total_days=total_days,
            remove_tail_days=remove_tail_days,
            flags=GLOBAL_FLAGS_FULL,
            contract=contract,
        )
        entry_df = _sample_df(entry_df, "label_entry", entry_ratio_neg_per_pos, per_sym_entry_max)
        danger_df = _sample_df(danger_df, "label_danger", danger_ratio_neg_per_pos, per_sym_danger_max)
        exit_df = _sample_df(exit_df, "label_exit", exit_ratio_neg_per_pos, per_sym_exit_max)
        Xe, ye, tse, syme = _to_numpy(entry_df, feat_cols_entry, "label_entry", sym_idx)
        Xd, yd, tsd, symd = _to_numpy(danger_df, feat_cols_danger, "label_danger", sym_idx)
        Xx, yx, tsx, symx = _to_numpy(exit_df, feat_cols_exit, "label_exit", sym_idx)
        # ajuda a liberar RAM entre threads
        del entry_df, danger_df, exit_df
        gc.collect()
        return Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx

    rest = list(enumerate(symbols[1:], start=1))
    if rest:
        if use_parallel and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(rest))) as ex:
                futs = {ex.submit(_worker, si, sym): (si, sym) for (si, sym) in rest}
                for fut in as_completed(futs):
                    try:
                        Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx = fut.result()
                    except Exception as e:
                        _si, _sym = futs.get(fut, (-1, "?"))
                        continue
                    X_entry_parts.append(Xe); y_entry_parts.append(ye); ts_entry_parts.append(tse); sym_ids_entry.append(syme)
                    X_danger_parts.append(Xd); y_danger_parts.append(yd); ts_danger_parts.append(tsd); sym_ids_danger.append(symd)
                    X_exit_parts.append(Xx); y_exit_parts.append(yx); ts_exit_parts.append(tsx); sym_ids_exit.append(symx)
        else:
            for si, sym in rest:
                try:
                    Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx = _worker(si, sym)
                except Exception as e:
                    continue
                X_entry_parts.append(Xe); y_entry_parts.append(ye); ts_entry_parts.append(tse); sym_ids_entry.append(syme)
                X_danger_parts.append(Xd); y_danger_parts.append(yd); ts_danger_parts.append(tsd); sym_ids_danger.append(symd)
                X_exit_parts.append(Xx); y_exit_parts.append(yx); ts_exit_parts.append(tsx); sym_ids_exit.append(symx)

    def _combine(parts: List[np.ndarray], axis=0):
        if not parts:
            return np.empty((0,), dtype=np.float32)
        if axis == 0:
            return np.concatenate(parts)
        return np.vstack(parts)

    X_entry = np.vstack(X_entry_parts) if X_entry_parts else np.empty((0, len(feat_cols_entry or [])), dtype=np.float32)
    y_entry = _combine(y_entry_parts)
    ts_entry = _combine(ts_entry_parts)
    sym_entry = _combine(sym_ids_entry)

    X_danger = np.vstack(X_danger_parts) if X_danger_parts else np.empty((0, len(feat_cols_danger or [])), dtype=np.float32)
    y_danger = _combine(y_danger_parts)
    ts_danger = _combine(ts_danger_parts)
    sym_danger = _combine(sym_ids_danger)

    X_exit = np.vstack(X_exit_parts) if X_exit_parts else np.empty((0, len(feat_cols_exit or [])), dtype=np.float32)
    y_exit = _combine(y_exit_parts)
    ts_exit = _combine(ts_exit_parts)
    sym_exit = _combine(sym_ids_exit)

    # Sampling GLOBAL (controla VRAM/tempo final de treino)
    X_entry, y_entry, ts_entry, sym_entry = _sample_arrays(
        X_entry, y_entry, ts_entry, sym_entry,
        ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
        max_rows=int(max_rows_entry),
    )
    X_danger, y_danger, ts_danger, sym_danger = _sample_arrays(
        X_danger, y_danger, ts_danger, sym_danger,
        ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
        max_rows=int(max_rows_danger),
    )
    X_exit, y_exit, ts_exit, sym_exit = _sample_arrays(
        X_exit, y_exit, ts_exit, sym_exit,
        ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
        max_rows=int(max_rows_exit),
    )

    entry_batch = SniperBatch(
        X=X_entry,
        y=y_entry,
        ts=ts_entry,
        sym_id=sym_entry,
        feature_cols=feat_cols_entry or [],
    )
    danger_batch = SniperBatch(
        X=X_danger,
        y=y_danger,
        ts=ts_danger,
        sym_id=sym_danger,
        feature_cols=feat_cols_danger or [],
    )
    exit_batch = SniperBatch(
        X=X_exit,
        y=y_exit,
        ts=ts_exit,
        sym_id=sym_exit,
        feature_cols=feat_cols_exit or [],
    )
    train_end: pd.Timestamp | None = None
    try:
        if ts_entry.size and ts_danger.size and ts_exit.size:
            train_end = pd.to_datetime(min(ts_entry.max(), ts_danger.max(), ts_exit.max()))
        elif ts_entry.size and ts_danger.size:
            train_end = pd.to_datetime(min(ts_entry.max(), ts_danger.max()))
        elif ts_entry.size and ts_exit.size:
            train_end = pd.to_datetime(min(ts_entry.max(), ts_exit.max()))
        elif ts_danger.size and ts_exit.size:
            train_end = pd.to_datetime(min(ts_danger.max(), ts_exit.max()))
        elif ts_entry.size:
            train_end = pd.to_datetime(ts_entry.max())
        elif ts_danger.size:
            train_end = pd.to_datetime(ts_danger.max())
        elif ts_exit.size:
            train_end = pd.to_datetime(ts_exit.max())
    except Exception:
        train_end = None
    return SniperDataPack(
        entry=entry_batch,
        danger=danger_batch,
        exit=exit_batch,
        contract=contract,
        symbols=list(symbols),
        train_end_utc=train_end,
    )


def prepare_sniper_dataset_from_cache(
    symbols: Sequence[str],
    *,
    total_days: int,
    remove_tail_days: int,
    contract: TradeContract | None = None,
    cache_map: Dict[str, Path] | None = None,
    # controle de tamanho (VRAM / tempo)
    entry_ratio_neg_per_pos: float = 6.0,
    danger_ratio_neg_per_pos: float = 4.0,
    exit_ratio_neg_per_pos: float = 4.0,
    max_rows_entry: int = 2_000_000,
    max_rows_danger: int = 1_200_000,
    max_rows_exit: int = 1_200_000,
    seed: int = 42,
) -> SniperDataPack:
    """
    Calcula features 1x por símbolo (cache em disco) e, no walk-forward, apenas
    recorta o final por `remove_tail_days` antes de montar o dataset.

    Importante: para evitar leakage perto do cutoff, removemos também os últimos
    `max(timeout_bars, danger_horizon_bars)` candles após o recorte.
    """
    contract = contract or DEFAULT_TRADE_CONTRACT
    symbols = list(symbols)
    if not symbols:
        raise RuntimeError("symbols vazio")

    # warmup do numba (add snapshots)
    try:
        warmup_sniper_dataset_numba()
    except Exception:
        pass

    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_map is None:
        cache_map = ensure_feature_cache(
            symbols,
            total_days=int(total_days),
            contract=contract,
            flags=GLOBAL_FLAGS_FULL,
        )
    # só mantém símbolos com cache válido
    symbols = [s for s in symbols if s in cache_map]
    if not symbols:
        raise RuntimeError("Nenhum símbolo com cache válido (todos falharam/foram pulados)")

    # cutoff global consistente: menor end_ts (quando disponível)
    end_ts_list: List[pd.Timestamp] = []
    for s in symbols:
        ts = _read_cache_meta_end_ts(s, cache_dir)
        if ts is not None:
            end_ts_list.append(ts)
    global_end_ts = min(end_ts_list) if end_ts_list else None
    # train_end determinístico (não depende do sampling): cutoff - lookahead
    train_end_det: pd.Timestamp | None = None
    if global_end_ts is not None:
        cutoff = pd.Timestamp(global_end_ts) - pd.Timedelta(days=int(remove_tail_days))
        lookahead = _max_label_lookahead_bars(contract, 60)
        train_end_det = cutoff - pd.Timedelta(seconds=int(lookahead) * 60)

    rng = np.random.default_rng(int(seed) + int(remove_tail_days))

    def _sample_df(df_in: pd.DataFrame, label_col: str, ratio_neg_per_pos: float, max_rows: int) -> pd.DataFrame:
        # mesmo sampler robusto do prepare_sniper_dataset (duplicado aqui para independência)
        if df_in.empty or label_col not in df_in.columns:
            return df_in
        y = df_in[label_col].to_numpy()
        pos_idx = np.flatnonzero(y >= 0.5)
        neg_idx = np.flatnonzero(y < 0.5)
        max_rows = int(max_rows)
        if max_rows <= 0:
            return df_in
        if pos_idx.size == 0:
            keep = neg_idx
            if keep.size > max_rows:
                keep = rng.choice(keep, size=max_rows, replace=False)
            return df_in.iloc[np.sort(keep)].copy()

        denom = 1.0 + float(max(0.0, ratio_neg_per_pos))
        pos_target = int(max(1, round(max_rows / denom)))
        pos_keep_n = int(min(pos_idx.size, pos_target))
        pos_keep = rng.choice(pos_idx, size=pos_keep_n, replace=False) if pos_idx.size > pos_keep_n else pos_idx

        remain = int(max_rows - pos_keep.size)
        if remain <= 0:
            return df_in.iloc[np.sort(pos_keep)].copy()

        neg_target = int(min(neg_idx.size, remain))
        if neg_target > 0:
            neg_keep = rng.choice(neg_idx, size=neg_target, replace=False) if neg_idx.size > neg_target else neg_idx
            keep = np.unique(np.concatenate([pos_keep, neg_keep]))
        else:
            keep = pos_keep

        if keep.size < max_rows and pos_idx.size > pos_keep.size:
            extra = int(min(max_rows - keep.size, pos_idx.size - pos_keep.size))
            if extra > 0:
                remaining_pos = np.setdiff1d(pos_idx, pos_keep, assume_unique=False)
                extra_pos = rng.choice(remaining_pos, size=extra, replace=False) if remaining_pos.size > extra else remaining_pos
                keep = np.unique(np.concatenate([keep, extra_pos]))

        if keep.size > max_rows:
            keep = rng.choice(keep, size=max_rows, replace=False)
        return df_in.iloc[np.sort(keep)].copy()

    def _sample_arrays(
        X: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        sym: np.ndarray,
        *,
        ratio_neg_per_pos: float,
        max_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.size == 0 or y.size == 0:
            return X, y, ts, sym
        n = int(y.shape[0])
        max_rows = int(max_rows)
        if max_rows <= 0 or n <= max_rows:
            return X, y, ts, sym
        pos = np.flatnonzero(y >= 0.5)
        neg = np.flatnonzero(y < 0.5)
        if pos.size == 0:
            keep = neg
            if keep.size > max_rows:
                keep = rng.choice(keep, size=max_rows, replace=False)
            keep = np.sort(keep)
            return X[keep], y[keep], ts[keep], sym[keep]

        denom = 1.0 + float(max(0.0, ratio_neg_per_pos))
        pos_target = int(max(1, round(max_rows / denom)))
        pos_keep_n = int(min(pos.size, pos_target))
        pos_keep = rng.choice(pos, size=pos_keep_n, replace=False) if pos.size > pos_keep_n else pos

        remain = int(max_rows - pos_keep.size)
        if remain <= 0:
            keep = np.sort(pos_keep)
            return X[keep], y[keep], ts[keep], sym[keep]

        neg_keep_n = int(min(neg.size, remain))
        if neg_keep_n > 0:
            neg_keep = rng.choice(neg, size=neg_keep_n, replace=False) if neg.size > neg_keep_n else neg
            keep = np.unique(np.concatenate([pos_keep, neg_keep]))
        else:
            keep = pos_keep

        if keep.size < max_rows and pos.size > pos_keep.size:
            extra = int(min(max_rows - keep.size, pos.size - pos_keep.size))
            if extra > 0:
                remaining_pos = np.setdiff1d(pos, pos_keep, assume_unique=False)
                extra_pos = rng.choice(remaining_pos, size=extra, replace=False) if remaining_pos.size > extra else remaining_pos
                keep = np.unique(np.concatenate([keep, extra_pos]))

        if keep.size > max_rows:
            keep = rng.choice(keep, size=max_rows, replace=False)

        keep = np.sort(keep)
        return X[keep], y[keep], ts[keep], sym[keep]

    def _to_numpy(df: pd.DataFrame, feats: List[str], label_col: str, sym_idx: int):
        if label_col not in df.columns:
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat = df.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
        mat = mat.infer_objects(copy=False)
        mask = mat.notnull().all(axis=1).to_numpy(dtype=bool, copy=False)
        if mask.size == 0 or (not bool(np.any(mask))):
            return (
                np.empty((0, len(feats)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), dtype=np.int32),
            )
        mat2 = mat.to_numpy(np.float32, copy=True)
        X = mat2[mask]
        y = df[label_col].to_numpy(dtype=np.float32, copy=False)[mask]
        ts = df.index.to_numpy(dtype="datetime64[ns]")[mask]
        sym_arr = np.full(int(ts.shape[0]), sym_idx, dtype=np.int32)
        return X, y, ts, sym_arr

    def _cut_df(df_pf: pd.DataFrame) -> pd.DataFrame:
        if df_pf.empty:
            return df_pf
        # normaliza index para evitar erro com tz-aware vs naive
        try:
            idx = pd.to_datetime(df_pf.index)
            if isinstance(idx, pd.DatetimeIndex) and (idx.tz is not None):
                idx = idx.tz_localize(None)
            if not idx.equals(df_pf.index):
                df_pf = df_pf.copy()
                df_pf.index = idx
        except Exception:
            pass
        end_ts = pd.Timestamp(df_pf.index.max())
        if global_end_ts is not None:
            end_ts = min(end_ts, global_end_ts)
        cutoff = end_ts - pd.Timedelta(days=int(remove_tail_days))
        df_cut = df_pf[df_pf.index < cutoff]
        lookahead = _max_label_lookahead_bars(contract, 60)
        if len(df_cut) > lookahead + 10:
            df_cut = df_cut.iloc[: -lookahead]
        return df_cut

    # limites por símbolo enquanto concatena (protege RAM: sampling global já vai recortar depois)
    n_syms = max(1, int(len(symbols)))
    per_sym_entry_max = int(min(int(max_rows_entry), max(20_000, (int(max_rows_entry) // n_syms) * 2)))
    per_sym_danger_max = int(min(int(max_rows_danger), max(15_000, (int(max_rows_danger) // n_syms) * 2)))
    per_sym_exit_max = int(min(int(max_rows_exit), max(15_000, (int(max_rows_exit) // n_syms) * 2)))

    def _build_symbol(
        sym_idx: int,
        symbol: str,
        *,
        feats_entry: List[str] | None = None,
        feats_danger: List[str] | None = None,
        feats_exit: List[str] | None = None,
    ):
        df_pf = _load_feature_cache(cache_map[symbol])
        df_pf = _cut_df(df_pf)
        if df_pf.empty or int(len(df_pf)) < 500:
            raise RuntimeError(f"{symbol}: df_pf vazio após corte (rows={len(df_pf)})")

        max_add_starts = _env_int("SNIPER_MAX_ADD_STARTS", 20_000)
        max_exit_starts = _env_int("SNIPER_MAX_EXIT_STARTS", 8_000)
        exit_stride_bars = _env_int("SNIPER_EXIT_STRIDE_BARS", 15)
        exit_lookahead_bars = _env_int("SNIPER_EXIT_LOOKAHEAD_BARS", 0)
        try:
            env_exit_margin = os.getenv("SNIPER_EXIT_MARGIN_PCT")
            if env_exit_margin is None or str(env_exit_margin).strip() == "":
                exit_margin_pct = _default_exit_margin_pct(contract)
            else:
                exit_margin_pct = float(env_exit_margin)
        except Exception:
            exit_margin_pct = _default_exit_margin_pct(contract)
        seed_local = _env_int("SNIPER_SEED", 1337) + int(remove_tail_days) + sym_idx
        sniper_ds = build_sniper_datasets(
            df_pf,
            contract=contract,
            max_add_starts=int(max_add_starts),
            max_exit_starts=int(max_exit_starts),
            exit_stride_bars=int(exit_stride_bars),
            exit_lookahead_bars=(int(exit_lookahead_bars) if int(exit_lookahead_bars) > 0 else None),
            exit_margin_pct=float(exit_margin_pct),
            seed=int(seed_local),
        )

        frames = [sniper_ds.entry, sniper_ds.add]
        frames = [f for f in frames if f is not None and (not f.empty)]
        if frames:
            entry_df = pd.concat(frames, axis=0, ignore_index=False)
        else:
            entry_df = sniper_ds.entry.copy()
        entry_df.sort_index(inplace=True)
        danger_df = sniper_ds.danger
        exit_df = sniper_ds.exit

        # IMPORTANT: os modelos precisam de um espaço de features consistente entre símbolos.
        # - No seed, inferimos as colunas.
        # - Nos demais símbolos, usamos as colunas do seed (reindex cuida de colunas faltantes).
        local_feat_cols_entry = _list_feature_columns(entry_df)
        local_feat_cols_danger = _list_feature_columns(danger_df)
        local_feat_cols_exit = _list_feature_columns(exit_df)
        feat_cols_entry_eff = list(local_feat_cols_entry) if feats_entry is None else list(feats_entry)
        feat_cols_danger_eff = list(local_feat_cols_danger) if feats_danger is None else list(feats_danger)
        feat_cols_exit_eff = list(local_feat_cols_exit) if feats_exit is None else list(feats_exit)

        entry_df = _sample_df(entry_df, "label_entry", float(entry_ratio_neg_per_pos), per_sym_entry_max)
        danger_df = _sample_df(danger_df, "label_danger", float(danger_ratio_neg_per_pos), per_sym_danger_max)
        exit_df = _sample_df(exit_df, "label_exit", float(exit_ratio_neg_per_pos), per_sym_exit_max)

        Xe, ye, tse, syme = _to_numpy(entry_df, feat_cols_entry_eff, "label_entry", sym_idx)
        Xd, yd, tsd, symd = _to_numpy(danger_df, feat_cols_danger_eff, "label_danger", sym_idx)
        Xx, yx, tsx, symx = _to_numpy(exit_df, feat_cols_exit_eff, "label_exit", sym_idx)

        del df_pf, entry_df, danger_df, exit_df
        gc.collect()
        return (
            Xe,
            ye,
            tse,
            syme,
            Xd,
            yd,
            tsd,
            symd,
            Xx,
            yx,
            tsx,
            symx,
            feat_cols_entry_eff,
            feat_cols_danger_eff,
            feat_cols_exit_eff,
        )

    # coleta de símbolos realmente usados
    symbols_total = list(symbols)
    symbols_used: List[str] = []
    symbols_skipped: List[str] = []

    total_syms = int(len(symbols_total))
    processed = 0
    progress_every_s = 3.0
    try:
        v = os.getenv("SNIPER_DATAFLOW_PROGRESS_EVERY_S", "").strip()
        progress_every_s = float(v) if v else 3.0
    except Exception:
        progress_every_s = 3.0
    last_progress_ts = 0.0

    def _bar(done: int, total: int, width: int = 26) -> str:
        total = max(1, int(total))
        done = int(max(0, min(done, total)))
        n = int(round(width * (done / total)))
        return ("#" * n) + ("-" * (width - n))

    def _print_progress(sym: str) -> None:
        nonlocal last_progress_ts
        if total_syms <= 0:
            return
        line = f"[dataflow] [{_bar(processed, total_syms)}] {processed:>3}/{total_syms:<3} | {sym}"
        if sys.stderr.isatty():
            sys.stderr.write("\r" + line)
            sys.stderr.flush()
        else:
            now = time.perf_counter()
            if now - last_progress_ts >= max(1.0, progress_every_s):
                print(line, flush=True)
                last_progress_ts = now

    # 1) achar um primeiro símbolo válido para definir feature_cols
    feat_cols_entry: List[str] = []
    feat_cols_danger: List[str] = []
    feat_cols_exit: List[str] = []

    # buffers (para flush em blocos) + pools (mantidos sempre <= max_rows_*)
    X_entry_parts: List[np.ndarray] = []
    y_entry_parts: List[np.ndarray] = []
    ts_entry_parts: List[np.ndarray] = []
    sym_ids_entry: List[np.ndarray] = []
    buf_entry_rows = 0

    X_danger_parts: List[np.ndarray] = []
    y_danger_parts: List[np.ndarray] = []
    ts_danger_parts: List[np.ndarray] = []
    sym_ids_danger: List[np.ndarray] = []
    buf_danger_rows = 0

    X_exit_parts: List[np.ndarray] = []
    y_exit_parts: List[np.ndarray] = []
    ts_exit_parts: List[np.ndarray] = []
    sym_ids_exit: List[np.ndarray] = []
    buf_exit_rows = 0

    # pools (inicializados após achar seed_symbol/feature_cols)
    X_entry_pool: np.ndarray | None = None
    y_entry_pool: np.ndarray | None = None
    ts_entry_pool: np.ndarray | None = None
    sym_entry_pool: np.ndarray | None = None

    X_danger_pool: np.ndarray | None = None
    y_danger_pool: np.ndarray | None = None
    ts_danger_pool: np.ndarray | None = None
    sym_danger_pool: np.ndarray | None = None

    X_exit_pool: np.ndarray | None = None
    y_exit_pool: np.ndarray | None = None
    ts_exit_pool: np.ndarray | None = None
    sym_exit_pool: np.ndarray | None = None

    def _flush_cls(
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        ts_pool: np.ndarray,
        sym_pool: np.ndarray,
        X_parts: List[np.ndarray],
        y_parts: List[np.ndarray],
        ts_parts: List[np.ndarray],
        sym_parts: List[np.ndarray],
        *,
        ratio_neg_per_pos: float,
        max_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not X_parts:
            return X_pool, y_pool, ts_pool, sym_pool
        X_new = (np.vstack([X_pool] + X_parts) if X_pool.size else np.vstack(X_parts))
        y_new = (np.concatenate([y_pool] + y_parts) if y_pool.size else np.concatenate(y_parts))
        ts_new = (np.concatenate([ts_pool] + ts_parts) if ts_pool.size else np.concatenate(ts_parts))
        sym_new = (np.concatenate([sym_pool] + sym_parts) if sym_pool.size else np.concatenate(sym_parts))
        X_new, y_new, ts_new, sym_new = _sample_arrays(
            X_new,
            y_new,
            ts_new,
            sym_new,
            ratio_neg_per_pos=float(ratio_neg_per_pos),
            max_rows=int(max_rows),
        )
        X_parts.clear(); y_parts.clear(); ts_parts.clear(); sym_parts.clear()
        return X_new, y_new, ts_new, sym_new

    seed_symbol = None
    first_err: str | None = None
    for sym in symbols_total:
        try:
            (
                Xe0,
                ye0,
                tse0,
                syme0,
                Xd0,
                yd0,
                tsd0,
                symd0,
                Xx0,
                yx0,
                tsx0,
                symx0,
                feat_entry0,
                feat_danger0,
                feat_exit0,
            ) = _build_symbol(
                0, sym, feats_entry=None, feats_danger=None, feats_exit=None
            )
            feat_cols_entry = list(feat_entry0)
            feat_cols_danger = list(feat_danger0)
            feat_cols_exit = list(feat_exit0)
            # inicializa pools com shapes corretos
            X_entry_pool = np.empty((0, len(feat_cols_entry)), dtype=np.float32)
            y_entry_pool = np.empty((0,), dtype=np.float32)
            ts_entry_pool = np.empty((0,), dtype="datetime64[ns]")
            sym_entry_pool = np.empty((0,), dtype=np.int32)

            X_danger_pool = np.empty((0, len(feat_cols_danger)), dtype=np.float32)
            y_danger_pool = np.empty((0,), dtype=np.float32)
            ts_danger_pool = np.empty((0,), dtype="datetime64[ns]")
            sym_danger_pool = np.empty((0,), dtype=np.int32)

            X_exit_pool = np.empty((0, len(feat_cols_exit)), dtype=np.float32)
            y_exit_pool = np.empty((0,), dtype=np.float32)
            ts_exit_pool = np.empty((0,), dtype="datetime64[ns]")
            sym_exit_pool = np.empty((0,), dtype=np.int32)

            # joga o seed nos buffers e dá flush imediato
            X_entry_parts.append(Xe0); y_entry_parts.append(ye0); ts_entry_parts.append(tse0); sym_ids_entry.append(syme0); buf_entry_rows += int(Xe0.shape[0])
            X_danger_parts.append(Xd0); y_danger_parts.append(yd0); ts_danger_parts.append(tsd0); sym_ids_danger.append(symd0); buf_danger_rows += int(Xd0.shape[0])
            X_exit_parts.append(Xx0); y_exit_parts.append(yx0); ts_exit_parts.append(tsx0); sym_ids_exit.append(symx0); buf_exit_rows += int(Xx0.shape[0])

            X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool = _flush_cls(
                X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool,
                X_entry_parts, y_entry_parts, ts_entry_parts, sym_ids_entry,
                ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
                max_rows=int(max_rows_entry),
            )
            X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool = _flush_cls(
                X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool,
                X_danger_parts, y_danger_parts, ts_danger_parts, sym_ids_danger,
                ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
                max_rows=int(max_rows_danger),
            )
            X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool = _flush_cls(
                X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool,
                X_exit_parts, y_exit_parts, ts_exit_parts, sym_ids_exit,
                ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
                max_rows=int(max_rows_exit),
            )
            buf_entry_rows = buf_danger_rows = buf_exit_rows = 0
            symbols_used.append(sym)
            seed_symbol = sym
            processed += 1
            _print_progress(sym)
            break
        except Exception as e:
            symbols_skipped.append(sym)
            if first_err is None:
                first_err = f"{sym}: {e}"
            processed += 1
            _print_progress(sym)
            continue

    if seed_symbol is None:
        extra = f" | first_error={first_err}" if first_err else ""
        raise RuntimeError(f"Nenhum símbolo válido após corte (todos SKIP){extra}")

    # 2) restantes (threads)
    use_parallel = _env_bool("PF_SYMBOL_PARALLEL", default=True)
    max_workers = _env_int("PF_SYMBOL_WORKERS", min(4, int(os.cpu_count() or 4)))

    def _worker(si: int, sym: str):
        Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx, _f1, _f2, _f3 = _build_symbol(
            si,
            sym,
            feats_entry=feat_cols_entry,
            feats_danger=feat_cols_danger,
            feats_exit=feat_cols_exit,
        )
        # robustez final: se ainda assim houver mismatch (não deveria), pula o símbolo
        if Xe.size and Xe.shape[1] != len(feat_cols_entry):
            raise RuntimeError(f"{sym}: entry features mismatch {Xe.shape[1]} != {len(feat_cols_entry)}")
        if Xd.size and Xd.shape[1] != len(feat_cols_danger):
            raise RuntimeError(f"{sym}: danger features mismatch {Xd.shape[1]} != {len(feat_cols_danger)}")
        if Xx.size and Xx.shape[1] != len(feat_cols_exit):
            raise RuntimeError(f"{sym}: exit features mismatch {Xx.shape[1]} != {len(feat_cols_exit)}")
        return Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx

    # 2) restantes (threads) — índices compactos (sym_id não é usado no treino, mas mantemos consistente)
    rest_symbols = [s for s in symbols_total if s != seed_symbol]
    rest = list(enumerate(rest_symbols, start=1))
    if rest:
        if use_parallel and max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(rest))) as ex:
                futs = {ex.submit(_worker, si, sym): (si, sym) for (si, sym) in rest}
                for fut in as_completed(futs):
                    try:
                        Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx = fut.result()
                    except Exception as e:
                        _si, _sym = futs.get(fut, (-1, "?"))
                        symbols_skipped.append(str(_sym))
                        processed += 1
                        _print_progress(str(_sym))
                        continue
                    symbols_used.append(str(futs[fut][1]))
                    processed += 1
                    _print_progress(str(futs[fut][1]))
                    X_entry_parts.append(Xe); y_entry_parts.append(ye); ts_entry_parts.append(tse); sym_ids_entry.append(syme); buf_entry_rows += int(Xe.shape[0])
                    X_danger_parts.append(Xd); y_danger_parts.append(yd); ts_danger_parts.append(tsd); sym_ids_danger.append(symd); buf_danger_rows += int(Xd.shape[0])
                    X_exit_parts.append(Xx); y_exit_parts.append(yx); ts_exit_parts.append(tsx); sym_ids_exit.append(symx); buf_exit_rows += int(Xx.shape[0])

                    # flush em blocos para evitar vstack gigante
                    if buf_entry_rows >= int(max(50_000, min(250_000, int(max_rows_entry) // 4))):
                        X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool = _flush_cls(
                            X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool,
                            X_entry_parts, y_entry_parts, ts_entry_parts, sym_ids_entry,
                            ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
                            max_rows=int(max_rows_entry),
                        )
                        buf_entry_rows = 0
                    if buf_danger_rows >= int(max(50_000, min(200_000, int(max_rows_danger) // 4))):
                        X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool = _flush_cls(
                            X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool,
                            X_danger_parts, y_danger_parts, ts_danger_parts, sym_ids_danger,
                            ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
                            max_rows=int(max_rows_danger),
                        )
                        buf_danger_rows = 0
                    if buf_exit_rows >= int(max(50_000, min(200_000, int(max_rows_exit) // 4))):
                        X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool = _flush_cls(
                            X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool,
                            X_exit_parts, y_exit_parts, ts_exit_parts, sym_ids_exit,
                            ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
                            max_rows=int(max_rows_exit),
                        )
                        buf_exit_rows = 0
        else:
            for si, sym in rest:
                try:
                    Xe, ye, tse, syme, Xd, yd, tsd, symd, Xx, yx, tsx, symx = _worker(si, sym)
                except Exception as e:
                    symbols_skipped.append(str(sym))
                    processed += 1
                    _print_progress(str(sym))
                    continue
                symbols_used.append(str(sym))
                processed += 1
                _print_progress(str(sym))
                X_entry_parts.append(Xe); y_entry_parts.append(ye); ts_entry_parts.append(tse); sym_ids_entry.append(syme); buf_entry_rows += int(Xe.shape[0])
                X_danger_parts.append(Xd); y_danger_parts.append(yd); ts_danger_parts.append(tsd); sym_ids_danger.append(symd); buf_danger_rows += int(Xd.shape[0])
                X_exit_parts.append(Xx); y_exit_parts.append(yx); ts_exit_parts.append(tsx); sym_ids_exit.append(symx); buf_exit_rows += int(Xx.shape[0])

                if buf_entry_rows >= int(max(50_000, min(250_000, int(max_rows_entry) // 4))):
                    X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool = _flush_cls(
                        X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool,
                        X_entry_parts, y_entry_parts, ts_entry_parts, sym_ids_entry,
                        ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
                        max_rows=int(max_rows_entry),
                    )
                    buf_entry_rows = 0
                if buf_danger_rows >= int(max(50_000, min(200_000, int(max_rows_danger) // 4))):
                    X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool = _flush_cls(
                        X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool,
                        X_danger_parts, y_danger_parts, ts_danger_parts, sym_ids_danger,
                        ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
                        max_rows=int(max_rows_danger),
                    )
                    buf_danger_rows = 0
                if buf_exit_rows >= int(max(50_000, min(200_000, int(max_rows_exit) // 4))):
                    X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool = _flush_cls(
                        X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool,
                        X_exit_parts, y_exit_parts, ts_exit_parts, sym_ids_exit,
                        ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
                        max_rows=int(max_rows_exit),
                    )
                    buf_exit_rows = 0

    # flush final (restos nos buffers)
    if X_entry_pool is None or y_entry_pool is None or ts_entry_pool is None or sym_entry_pool is None:
        raise RuntimeError("Estado inválido: pools não inicializados")

    X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool = _flush_cls(
        X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool,
        X_entry_parts, y_entry_parts, ts_entry_parts, sym_ids_entry,
        ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
        max_rows=int(max_rows_entry),
    )
    X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool = _flush_cls(
        X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool,
        X_danger_parts, y_danger_parts, ts_danger_parts, sym_ids_danger,
        ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
        max_rows=int(max_rows_danger),
    )
    X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool = _flush_cls(
        X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool,
        X_exit_parts, y_exit_parts, ts_exit_parts, sym_ids_exit,
        ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
        max_rows=int(max_rows_exit),
    )

    # (opcional) recorte final adicional (idempotente)
    X_entry, y_entry, ts_entry, sym_entry = _sample_arrays(
        X_entry_pool, y_entry_pool, ts_entry_pool, sym_entry_pool,
        ratio_neg_per_pos=float(entry_ratio_neg_per_pos),
        max_rows=int(max_rows_entry),
    )
    X_danger, y_danger, ts_danger, sym_danger = _sample_arrays(
        X_danger_pool, y_danger_pool, ts_danger_pool, sym_danger_pool,
        ratio_neg_per_pos=float(danger_ratio_neg_per_pos),
        max_rows=int(max_rows_danger),
    )
    X_exit, y_exit, ts_exit, sym_exit = _sample_arrays(
        X_exit_pool, y_exit_pool, ts_exit_pool, sym_exit_pool,
        ratio_neg_per_pos=float(exit_ratio_neg_per_pos),
        max_rows=int(max_rows_exit),
    )

    if total_syms > 0:
        try:
            if sys.stderr.isatty():
                sys.stderr.write("\n")
                sys.stderr.flush()
        except Exception:
            pass
        print(
            f"[dataflow] symbols: processed={processed} ok={len(symbols_used)} skipped={len(symbols_skipped)}",
            flush=True,
        )

    entry_batch = SniperBatch(X=X_entry, y=y_entry, ts=ts_entry, sym_id=sym_entry, feature_cols=feat_cols_entry)
    danger_batch = SniperBatch(X=X_danger, y=y_danger, ts=ts_danger, sym_id=sym_danger, feature_cols=feat_cols_danger)
    exit_batch = SniperBatch(X=X_exit, y=y_exit, ts=ts_exit, sym_id=sym_exit, feature_cols=feat_cols_exit)
    if train_end_det is None:
        try:
            if ts_entry.size and ts_danger.size and ts_exit.size:
                train_end_det = pd.to_datetime(min(ts_entry.max(), ts_danger.max(), ts_exit.max()))
            elif ts_entry.size and ts_danger.size:
                train_end_det = pd.to_datetime(min(ts_entry.max(), ts_danger.max()))
            elif ts_entry.size and ts_exit.size:
                train_end_det = pd.to_datetime(min(ts_entry.max(), ts_exit.max()))
            elif ts_danger.size and ts_exit.size:
                train_end_det = pd.to_datetime(min(ts_danger.max(), ts_exit.max()))
            elif ts_entry.size:
                train_end_det = pd.to_datetime(ts_entry.max())
            elif ts_danger.size:
                train_end_det = pd.to_datetime(ts_danger.max())
            elif ts_exit.size:
                train_end_det = pd.to_datetime(ts_exit.max())
        except Exception:
            train_end_det = None
    return SniperDataPack(
        entry=entry_batch,
        danger=danger_batch,
        exit=exit_batch,
        contract=contract,
        symbols=list(symbols_total),
        symbols_used=sorted(set(symbols_used)),
        symbols_skipped=sorted(set(symbols_skipped)),
        train_end_utc=train_end_det,
    )


__all__ = [
    "SniperDataPack",
    "SniperBatch",
    "prepare_sniper_dataset",
    "prepare_sniper_dataset_from_cache",
    "ensure_feature_cache",
]

