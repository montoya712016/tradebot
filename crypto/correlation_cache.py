# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    njit = None  # type: ignore[assignment]
    _HAS_NUMBA = False


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

from prepare_features.data import load_ohlc_1m_series  # type: ignore
from utils.progress import progress as _progress_base  # type: ignore
try:
    from plotting import plot_time_series as _plot_time_series  # type: ignore
except Exception:
    try:
        from modules.plotting import plot_time_series as _plot_time_series  # type: ignore
    except Exception:
        _plot_time_series = None  # type: ignore[assignment]


def _progress(it, *, total: int | None = None, desc: str = ""):
    return _progress_base(it, total=total, desc=desc, prefix="corr", fallback="eta")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tradebot":
            return p
    return here.parent


def _cache_root() -> Path:
    root = _repo_root()
    return root.parent / "cache_sniper" / "corr"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name, str(default)) or "").strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name, str(default)) or "").strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return str(v).strip() if v is not None else str(default)


def _parse_symbols_env() -> list[str]:
    raw = (os.getenv("SYMBOLS") or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for tok in raw.replace(";", ",").split(","):
        s = tok.strip().upper()
        if s:
            out.append(s)
    return out


def _canonical_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        su = str(s).strip().upper()
        if not su or su in seen:
            continue
        seen.add(su)
        out.append(su)
    return out


def _freq_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()
    if tf.endswith("min"):
        return int(tf[:-3])
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"timeframe invalido: {timeframe}")


def _timeframe_to_candle_sec(timeframe: str) -> int:
    return int(_freq_to_minutes(timeframe) * 60)


def _cache_key(
    *,
    symbols: list[str],
    timeframe: str,
    window_bars: int,
    method: str,
    stats_mode: str,
    engine: str,
    min_obs_ratio: float,
    source_days: int,
    version: str = "v2",
) -> str:
    payload = {
        "version": version,
        "symbols_sorted": sorted(_canonical_symbols(symbols)),
        "timeframe": str(timeframe).lower(),
        "window_bars": int(window_bars),
        "method": str(method).lower(),
        "stats_mode": str(stats_mode).lower(),
        "engine": str(engine).lower(),
        "min_obs_ratio": float(min_obs_ratio),
        "source_days": int(source_days),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha1(raw).hexdigest()[:16]


def _cache_paths(
    *,
    symbols: list[str],
    timeframe: str,
    window_bars: int,
    method: str,
    stats_mode: str,
    engine: str,
    min_obs_ratio: float,
    source_days: int,
) -> tuple[Path, Path]:
    root = _ensure_dir(_cache_root())
    key = _cache_key(
        symbols=symbols,
        timeframe=timeframe,
        window_bars=window_bars,
        method=method,
        stats_mode=stats_mode,
        engine=engine,
        min_obs_ratio=min_obs_ratio,
        source_days=source_days,
    )
    base = root / f"corr_{timeframe}_w{int(window_bars)}_{str(stats_mode).lower()}_{key}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


@dataclass(slots=True)
class CorrBuildResult:
    metrics_path: Path
    meta_path: Path
    metrics: pd.DataFrame
    metadata: dict[str, Any]
    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None


def _load_symbol_close_resampled(sym: str, *, source_days: int, timeframe: str) -> pd.Series:
    df = load_ohlc_1m_series(sym, days=int(source_days), remove_tail_days=0)
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype="float64", name=sym)
    candle_sec = _timeframe_to_candle_sec(timeframe)
    if candle_sec == 60:
        ser = pd.to_numeric(df["close"], errors="coerce")
    else:
        rule = f"{int(candle_sec)}s"
        ser = pd.to_numeric(df["close"], errors="coerce").resample(rule).last()
    ser = ser.astype("float64").sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    ser = ser.ffill()
    ser.name = str(sym).upper()
    return ser


def build_price_panel(
    symbols: list[str],
    *,
    source_days: int = 0,
    timeframe: str = "5m",
    min_symbol_rows: int = 200,
) -> pd.DataFrame:
    syms = _canonical_symbols(symbols)
    if not syms:
        return pd.DataFrame()

    cols: list[pd.Series] = []
    for sym in _progress(syms, total=len(syms), desc="load ohlc"):
        try:
            ser = _load_symbol_close_resampled(sym, source_days=int(source_days), timeframe=timeframe)
        except Exception as e:
            print(f"[corr] FAIL load {sym}: {type(e).__name__}: {e}", flush=True)
            continue
        if ser.empty:
            print(f"[corr] skip {sym}: sem dados", flush=True)
            continue
        if int(len(ser)) < int(min_symbol_rows):
            print(f"[corr] skip {sym}: poucos pontos ({len(ser)})", flush=True)
            continue
        cols.append(ser)

    if not cols:
        return pd.DataFrame()

    px = pd.concat(cols, axis=1, join="outer").sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.dropna(how="all")
    px = px.ffill()
    return px


def build_returns_panel(
    prices: pd.DataFrame,
    *,
    min_obs_ratio: float = 0.80,
) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    px = prices.copy()
    for c in px.columns:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    rets = np.log(px).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    if len(rets) == 0:
        return rets

    min_rows = max(2, int(math.ceil(float(min_obs_ratio) * max(1, len(rets)))))
    rets = rets.dropna(axis=1, thresh=min_rows)
    rets = rets.dropna(how="all")
    return rets


def _corr_stats_from_window(dfw: pd.DataFrame, *, method: str = "pearson") -> dict[str, float | int]:
    if dfw.empty:
        return {
            "avg_corr": np.nan,
            "median_corr": np.nan,
            "p90_corr": np.nan,
            "max_corr": np.nan,
            "min_corr": np.nan,
            "n_assets": 0,
            "n_pairs": 0,
            "n_obs": 0,
        }

    dense = dfw.dropna(axis=0, how="any")
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        return {
            "avg_corr": np.nan,
            "median_corr": np.nan,
            "p90_corr": np.nan,
            "max_corr": np.nan,
            "min_corr": np.nan,
            "n_assets": int(dense.shape[1]),
            "n_pairs": 0,
            "n_obs": int(dense.shape[0]),
        }

    # Remove colunas sem variacao nesta janela (std=0), que quebram a normalizacao
    # do np.corrcoef e geram RuntimeWarning/NaN em massa.
    stdv = np.nanstd(dense.to_numpy(dtype=np.float64, copy=False), axis=0)
    keep = np.isfinite(stdv) & (stdv > 0.0)
    if not bool(np.all(keep)):
        dense = dense.iloc[:, keep]
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        return {
            "avg_corr": np.nan,
            "median_corr": np.nan,
            "p90_corr": np.nan,
            "max_corr": np.nan,
            "min_corr": np.nan,
            "n_assets": int(dense.shape[1]),
            "n_pairs": 0,
            "n_obs": int(dense.shape[0]),
        }

    x = dense.to_numpy(dtype=np.float64, copy=False)
    if method.lower() == "pearson":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.corrcoef(x, rowvar=False)
    else:
        c = dense.corr(method=method).to_numpy(dtype=np.float64, copy=False)
    if c.ndim != 2 or c.shape[0] < 2:
        return {
            "avg_corr": np.nan,
            "median_corr": np.nan,
            "p90_corr": np.nan,
            "max_corr": np.nan,
            "min_corr": np.nan,
            "n_assets": int(dense.shape[1]),
            "n_pairs": 0,
            "n_obs": int(dense.shape[0]),
        }
    mask = ~np.eye(c.shape[0], dtype=bool)
    vals = c[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "avg_corr": np.nan,
            "median_corr": np.nan,
            "p90_corr": np.nan,
            "max_corr": np.nan,
            "min_corr": np.nan,
            "n_assets": int(dense.shape[1]),
            "n_pairs": 0,
            "n_obs": int(dense.shape[0]),
        }
    return {
        "avg_corr": float(np.mean(vals)),
        "median_corr": float(np.median(vals)),
        "p90_corr": float(np.quantile(vals, 0.90)),
        "max_corr": float(np.max(vals)),
        "min_corr": float(np.min(vals)),
        "n_assets": int(c.shape[0]),
        "n_pairs": int(vals.size // 2),
        "n_obs": int(dense.shape[0]),
    }


def _corr_avg_from_dense_numpy(x: np.ndarray) -> tuple[float, int, int, int]:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return (float("nan"), int(max(0, x.shape[1] if x.ndim == 2 else 0)), 0, int(max(0, x.shape[0] if x.ndim == 2 else 0)))
    m = int(x.shape[0])
    mu = np.mean(x, axis=0, dtype=np.float64)
    xc = x - mu
    ss = np.sum(xc * xc, axis=0, dtype=np.float64)
    valid = np.isfinite(ss) & (ss > 0.0)
    k = int(np.sum(valid))
    if k < 2:
        return (float("nan"), k, 0, m)
    z = xc[:, valid] / np.sqrt(ss[valid] / max(1.0, float(m - 1)))
    v = np.sum(z, axis=1, dtype=np.float64)
    sum_all = float(np.dot(v, v) / max(1.0, float(m - 1)))
    pairs = int(k * (k - 1) // 2)
    avg = float((sum_all - float(k)) / max(1.0, float(k * (k - 1))))
    return (avg, k, pairs, m)


def _corr_avg_stats_from_window(dfw: pd.DataFrame, *, min_w_obs: int) -> dict[str, float | int]:
    if dfw.empty:
        return {"avg_corr": np.nan, "n_assets": 0, "n_pairs": 0, "n_obs": 0}
    dfx = dfw.dropna(axis=1, thresh=int(min_w_obs))
    if dfx.shape[1] < 2:
        return {"avg_corr": np.nan, "n_assets": int(dfx.shape[1]), "n_pairs": 0, "n_obs": 0}
    dense = dfx.dropna(axis=0, how="any")
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        return {"avg_corr": np.nan, "n_assets": int(dense.shape[1]), "n_pairs": 0, "n_obs": int(dense.shape[0])}
    avg, k, pairs, n_obs = _corr_avg_from_dense_numpy(dense.to_numpy(dtype=np.float64, copy=False))
    return {"avg_corr": float(avg), "n_assets": int(k), "n_pairs": int(pairs), "n_obs": int(n_obs)}


def _rolling_avg_corr_sparse_numpy_prefix(
    x: np.ndarray,
    index: pd.Index,
    *,
    endpoints: np.ndarray,
    window_bars: int,
    min_w_obs: int,
    chunk_windows: int = 2048,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if endpoints.size == 0:
        return rows
    w = int(window_bars)
    step = int(max(1, int(chunk_windows)))
    ranges = range(0, int(endpoints.size), step)
    for s in _progress(ranges, total=((int(endpoints.size) + step - 1) // step), desc="corr sparse prefix"):
        e = min(int(endpoints.size), int(s) + step)
        ep_chunk = endpoints[int(s) : int(e)]
        for end_i in ep_chunk:
            si = int(end_i) - w + 1
            wi = x[si : int(end_i) + 1]  # (w, n)
            finite = np.isfinite(wi)
            cnt = np.sum(finite, axis=0)
            keep = cnt >= int(min_w_obs)
            k0 = int(np.sum(keep))
            if k0 < 2:
                rows.append(
                    {
                        "ts": pd.to_datetime(index[int(end_i)]),
                        "avg_corr": np.nan,
                        "n_assets": k0,
                        "n_pairs": 0,
                        "n_obs": 0,
                    }
                )
                continue
            wk = wi[:, keep]
            row_good = np.all(np.isfinite(wk), axis=1)
            dense = wk[row_good]
            if dense.shape[0] < 2 or dense.shape[1] < 2:
                rows.append(
                    {
                        "ts": pd.to_datetime(index[int(end_i)]),
                        "avg_corr": np.nan,
                        "n_assets": int(dense.shape[1]),
                        "n_pairs": 0,
                        "n_obs": int(dense.shape[0]),
                    }
                )
                continue
            avg, k, pairs, n_obs = _corr_avg_from_dense_numpy(np.asarray(dense, dtype=np.float64))
            rows.append(
                {
                    "ts": pd.to_datetime(index[int(end_i)]),
                    "avg_corr": float(avg),
                    "n_assets": int(k),
                    "n_pairs": int(pairs),
                    "n_obs": int(n_obs),
                }
            )
    return rows


def _dense_suffix_start(x: np.ndarray) -> int:
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return 0
    row_all = np.isfinite(x).all(axis=1)
    bad = np.flatnonzero(~row_all)
    if bad.size == 0:
        return 0
    return int(bad[-1] + 1)


def _rolling_avg_corr_dense_numpy_chunked(
    x: np.ndarray,
    *,
    window_bars: int,
    chunk_windows: int = 256,
    show_progress: bool = False,
) -> dict[str, np.ndarray]:
    t = int(x.shape[0])
    n = int(x.shape[1]) if x.ndim == 2 else 0
    w = int(max(2, int(window_bars)))
    m = t - w + 1
    if m <= 0 or n <= 0:
        return {
            "avg_corr": np.empty((0,), dtype=np.float64),
            "n_assets": np.empty((0,), dtype=np.int32),
            "n_pairs": np.empty((0,), dtype=np.int32),
            "n_obs": np.empty((0,), dtype=np.int32),
        }

    from numpy.lib.stride_tricks import sliding_window_view

    v = sliding_window_view(x, window_shape=w, axis=0)
    # numpy shape pode sair (m, n, w); padroniza para (m, w, n)
    if v.ndim == 3 and v.shape[1] != w and v.shape[2] == w:
        v = np.swapaxes(v, 1, 2)
    if v.ndim != 3 or v.shape[1] != w:
        raise RuntimeError(f"shape inesperado de sliding_window_view: {getattr(v, 'shape', None)}")

    out_avg = np.full((m,), np.nan, dtype=np.float64)
    out_k = np.zeros((m,), dtype=np.int32)
    out_pairs = np.zeros((m,), dtype=np.int32)
    out_nobs = np.full((m,), int(w), dtype=np.int32)

    step = int(max(1, int(chunk_windows)))
    ranges = range(0, m, step)
    it = _progress(ranges, total=((m + step - 1) // step), desc="corr dense numpy") if show_progress else ranges
    for s in it:
        e = min(m, int(s) + step)
        wc = v[int(s) : int(e)]  # (b, w, n)
        mu = np.mean(wc, axis=1, dtype=np.float64)  # (b, n)
        xc = wc - mu[:, None, :]
        ss = np.sum(xc * xc, axis=1, dtype=np.float64)  # (b, n)
        std = np.sqrt(ss / max(1.0, float(w - 1)))
        bad = (~np.isfinite(std)) | (std <= 0.0)
        std = std.astype(np.float64, copy=False)
        std[bad] = np.nan
        z = xc / std[:, None, :]
        vrow = np.nansum(z, axis=2, dtype=np.float64)  # (b, w)
        sum_all = np.sum(vrow * vrow, axis=1, dtype=np.float64) / max(1.0, float(w - 1))
        k = np.sum(np.isfinite(std), axis=1, dtype=np.int32)
        pairs = (k * (k - 1)) // 2
        out_avg[int(s) : int(e)] = np.nan
        good = pairs > 0
        if np.any(good):
            kg = k[good].astype(np.float64, copy=False)
            tmp = out_avg[int(s) : int(e)].copy()
            tmp[good] = (sum_all[good] - kg) / (kg * (kg - 1.0))
            out_avg[int(s) : int(e)] = tmp
        out_k[int(s) : int(e)] = k
        out_pairs[int(s) : int(e)] = pairs.astype(np.int32, copy=False)

    return {
        "avg_corr": out_avg,
        "n_assets": out_k,
        "n_pairs": out_pairs,
        "n_obs": out_nobs,
    }


if _HAS_NUMBA:
    @njit(cache=True)
    def _rolling_avg_corr_dense_numba_core(x: np.ndarray, w: int):
        t, n = x.shape
        m = t - w + 1
        out_avg = np.empty(m, dtype=np.float64)
        out_k = np.empty(m, dtype=np.int32)
        out_pairs = np.empty(m, dtype=np.int32)
        out_nobs = np.empty(m, dtype=np.int32)

        sx = np.zeros(n, dtype=np.float64)
        sxx = np.zeros(n, dtype=np.float64)
        sxy = np.zeros((n, n), dtype=np.float64)

        for r in range(w):
            row = x[r]
            for i in range(n):
                xi = row[i]
                sx[i] += xi
                sxx[i] += xi * xi
            for i in range(n):
                xi = row[i]
                for j in range(i, n):
                    sxy[i, j] += xi * row[j]

        eps = 1e-30
        den = np.empty(n, dtype=np.float64)
        valid = np.empty(n, dtype=np.uint8)
        wf = float(w)

        for kidx in range(m):
            kval = 0
            for i in range(n):
                d = wf * sxx[i] - sx[i] * sx[i]
                den[i] = d
                if d > eps:
                    valid[i] = 1
                    kval += 1
                else:
                    valid[i] = 0

            pairs = 0
            sum_corr = 0.0
            if kval >= 2:
                for i in range(n):
                    if valid[i] == 0:
                        continue
                    di = den[i]
                    for j in range(i + 1, n):
                        if valid[j] == 0:
                            continue
                        dj = den[j]
                        denom = di * dj
                        if denom <= eps:
                            continue
                        num = wf * sxy[i, j] - sx[i] * sx[j]
                        r = num / np.sqrt(denom)
                        if np.isfinite(r):
                            if r > 1.0:
                                r = 1.0
                            elif r < -1.0:
                                r = -1.0
                            sum_corr += r
                            pairs += 1

            out_k[kidx] = np.int32(kval)
            out_pairs[kidx] = np.int32(pairs)
            out_nobs[kidx] = np.int32(w)
            if pairs > 0:
                out_avg[kidx] = sum_corr / float(pairs)
            else:
                out_avg[kidx] = np.nan

            if kidx + 1 < m:
                r_out = x[kidx]
                r_in = x[kidx + w]
                for i in range(n):
                    xo = r_out[i]
                    xi = r_in[i]
                    sx[i] += xi - xo
                    sxx[i] += xi * xi - xo * xo
                for i in range(n):
                    xo_i = r_out[i]
                    xi_i = r_in[i]
                    for j in range(i, n):
                        sxy[i, j] += xi_i * r_in[j] - xo_i * r_out[j]

        return out_avg, out_k, out_pairs, out_nobs


def _rolling_avg_corr_dense_numba(
    x: np.ndarray,
    *,
    window_bars: int,
) -> dict[str, np.ndarray]:
    if not _HAS_NUMBA:
        raise RuntimeError("numba indisponivel")
    t = int(x.shape[0])
    n = int(x.shape[1]) if x.ndim == 2 else 0
    w = int(max(2, int(window_bars)))
    if t - w + 1 <= 0 or n <= 0:
        return {
            "avg_corr": np.empty((0,), dtype=np.float64),
            "n_assets": np.empty((0,), dtype=np.int32),
            "n_pairs": np.empty((0,), dtype=np.int32),
            "n_obs": np.empty((0,), dtype=np.int32),
        }
    avg, k, pairs, nobs = _rolling_avg_corr_dense_numba_core(np.asarray(x, dtype=np.float64), int(w))
    return {"avg_corr": avg, "n_assets": k, "n_pairs": pairs, "n_obs": nobs}


def benchmark_corr_engines(
    returns: pd.DataFrame,
    *,
    window_bars: int,
    sample_windows: int = 4096,
    chunk_windows: int = 256,
) -> dict[str, float]:
    rets = returns.sort_index()
    x = rets.to_numpy(dtype=np.float64, copy=False)
    dense_start = _dense_suffix_start(x)
    xd = x[dense_start:]
    w = int(max(2, int(window_bars)))
    if xd.shape[0] < w + 16:
        return {}
    max_rows = min(int(xd.shape[0]), int(sample_windows) + w - 1)
    xs = np.asarray(xd[:max_rows], dtype=np.float64)
    out: dict[str, float] = {}

    t0 = time.perf_counter()
    _ = _rolling_avg_corr_dense_numpy_chunked(xs, window_bars=w, chunk_windows=int(chunk_windows), show_progress=False)
    out["numpy_chunk_s"] = float(time.perf_counter() - t0)

    if _HAS_NUMBA:
        # warmup/compile
        _ = _rolling_avg_corr_dense_numba(xs[: max(w + 4, min(xs.shape[0], w + 32))], window_bars=w)
        t1 = time.perf_counter()
        _ = _rolling_avg_corr_dense_numba(xs, window_bars=w)
        out["numba_s"] = float(time.perf_counter() - t1)
    return out


def _rolling_correlation_metrics_baseline(
    returns: pd.DataFrame,
    *,
    window_bars: int,
    method: str = "pearson",
    chunk_points: int = 256,
    min_window_obs_ratio: float = 0.80,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame(
            columns=[
                "avg_corr",
                "median_corr",
                "p90_corr",
                "max_corr",
                "min_corr",
                "n_assets",
                "n_pairs",
                "n_obs",
            ]
        )

    rets = returns.sort_index()
    w = int(max(2, int(window_bars)))
    if len(rets) < w:
        return pd.DataFrame(index=rets.index[:0])

    min_w_obs = max(2, int(math.ceil(float(min_window_obs_ratio) * w)))
    endpoints = np.arange(w - 1, len(rets), dtype=np.int64)
    chunk_n = int(max(1, int(chunk_points)))
    chunks = [endpoints[i : i + chunk_n] for i in range(0, len(endpoints), chunk_n)]

    rows: list[dict[str, Any]] = []
    for chunk in _progress(chunks, total=len(chunks), desc="corr chunks"):
        for end_i in chunk:
            start_i = int(end_i) - w + 1
            dfw = rets.iloc[start_i : int(end_i) + 1]
            # remove symbols sem observacoes suficientes neste trecho
            dfw = dfw.dropna(axis=1, thresh=min_w_obs)
            stats = _corr_stats_from_window(dfw, method=method)
            rows.append({"ts": rets.index[int(end_i)], **stats})

    if not rows:
        return pd.DataFrame(index=rets.index[:0])
    out = pd.DataFrame(rows).set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def rolling_correlation_metrics(
    returns: pd.DataFrame,
    *,
    window_bars: int,
    method: str = "pearson",
    chunk_points: int = 256,
    min_window_obs_ratio: float = 0.80,
    engine: str = "auto",
    stats_mode: str = "avg_only",
    benchmark: bool = True,
) -> pd.DataFrame:
    """
    Rolling correlation do universo.

    stats_mode:
      - avg_only (rapido): calcula avg_corr + contagens
      - full (lento): avg/median/p90/max/min via corrcoef por janela (baseline)
    engine (para avg_only+pearson):
      - auto | numpy | numba | baseline
    """
    if returns is None or returns.empty:
        return pd.DataFrame()

    method_l = str(method).strip().lower()
    stats_l = str(stats_mode).strip().lower()
    eng_l = str(engine).strip().lower()
    if stats_l in {"full", "all"} or method_l != "pearson" or eng_l == "baseline":
        t0 = time.perf_counter()
        out = _rolling_correlation_metrics_baseline(
            returns,
            window_bars=int(window_bars),
            method=method_l,
            chunk_points=int(chunk_points),
            min_window_obs_ratio=float(min_window_obs_ratio),
        )
        try:
            out.attrs["corr_engine_used"] = "baseline"
            out.attrs["corr_stats_mode"] = "full"
        except Exception:
            pass
        print(f"[corr] calc baseline(full) sec={time.perf_counter() - t0:.2f} rows={len(out)}", flush=True)
        return out

    rets = returns.sort_index()
    x = rets.to_numpy(dtype=np.float64, copy=False)
    t = int(x.shape[0])
    n = int(x.shape[1]) if x.ndim == 2 else 0
    w = int(max(2, int(window_bars)))
    if t < w or n < 1:
        return pd.DataFrame(index=rets.index[:0], columns=["avg_corr", "n_assets", "n_pairs", "n_obs"])

    min_w_obs = max(2, int(math.ceil(float(min_window_obs_ratio) * w)))
    endpoints = np.arange(w - 1, t, dtype=np.int64)
    dense_start = _dense_suffix_start(x)
    dense_first_end = max(w - 1, int(dense_start) + w - 1)
    prefix_mask = endpoints < dense_first_end
    suffix_mask = ~prefix_mask

    if eng_l == "auto":
        eng_l = "numpy"
        if _HAS_NUMBA:
            if bool(benchmark):
                bench = benchmark_corr_engines(rets, window_bars=w, sample_windows=min(4096, len(endpoints)), chunk_windows=int(chunk_points))
                if bench:
                    bn = bench.get("numpy_chunk_s", float("inf"))
                    bj = bench.get("numba_s", float("inf"))
                    eng_l = "numba" if bj < bn else "numpy"
                    print(
                        f"[corr] bench engines sample: numpy={bn:.3f}s numba={bj:.3f}s -> engine={eng_l}",
                        flush=True,
                    )
            else:
                eng_l = "numba"
    if eng_l == "numba" and (not _HAS_NUMBA):
        eng_l = "numpy"

    t_calc = time.perf_counter()
    rows: list[dict[str, Any]] = []

    # Prefixo (janelas com NaN, tipicamente no inicio) — exato, porém mais caro.
    prefix_eps = endpoints[prefix_mask]
    if prefix_eps.size > 0:
        t_pref = time.perf_counter()
        rows.extend(
            _rolling_avg_corr_sparse_numpy_prefix(
                x,
                rets.index,
                endpoints=prefix_eps,
                window_bars=w,
                min_w_obs=int(min_w_obs),
                chunk_windows=max(512, int(chunk_points) * 4),
            )
        )
        print(f"[corr] sparse prefix numpy sec={time.perf_counter() - t_pref:.2f} windows={int(prefix_eps.size)}", flush=True)

    # Sufixo denso (sem NaN em todas as colunas) — usa engine rapido.
    suffix_eps = endpoints[suffix_mask]
    if suffix_eps.size > 0:
        xd = np.asarray(x[int(dense_start) :], dtype=np.float64)
        if eng_l == "numba":
            t_eng = time.perf_counter()
            dense_stats = _rolling_avg_corr_dense_numba(xd, window_bars=w)
            print(f"[corr] dense numba sec={time.perf_counter() - t_eng:.2f} windows={len(dense_stats['avg_corr'])}", flush=True)
        else:
            t_eng = time.perf_counter()
            dense_stats = _rolling_avg_corr_dense_numpy_chunked(
                xd,
                window_bars=w,
                chunk_windows=int(chunk_points),
                show_progress=True,
            )
            print(f"[corr] dense numpy sec={time.perf_counter() - t_eng:.2f} windows={len(dense_stats['avg_corr'])}", flush=True)

        # Map local dense windows -> timestamps globais.
        dense_endpoints_global = np.arange(int(dense_start) + w - 1, t, dtype=np.int64)
        for i, end_i in enumerate(dense_endpoints_global):
            rows.append(
                {
                    "ts": rets.index[int(end_i)],
                    "avg_corr": float(dense_stats["avg_corr"][i]),
                    "n_assets": int(dense_stats["n_assets"][i]),
                    "n_pairs": int(dense_stats["n_pairs"][i]),
                    "n_obs": int(dense_stats["n_obs"][i]),
                }
            )

    if not rows:
        return pd.DataFrame(index=rets.index[:0], columns=["avg_corr", "n_assets", "n_pairs", "n_obs"])

    out = pd.DataFrame(rows).set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    try:
        out.attrs["corr_engine_used"] = str(eng_l)
        out.attrs["corr_stats_mode"] = str(stats_l)
    except Exception:
        pass
    print(
        f"[corr] calc fast stats_mode={stats_l} engine={eng_l} sec={time.perf_counter() - t_calc:.2f} "
        f"rows={len(out)} prefix={int(prefix_eps.size)} suffix={int(suffix_eps.size)} dense_start={dense_start}",
        flush=True,
    )
    return out


def build_and_cache_correlation(
    symbols: list[str],
    *,
    source_days: int = 0,
    timeframe: str = "5m",
    window_bars: int = 12 * 60 // 5,
    method: str = "pearson",
    engine: str = "auto",
    stats_mode: str = "avg_only",
    min_obs_ratio: float = 0.80,
    min_window_obs_ratio: float = 0.80,
    chunk_points: int = 256,
    min_symbol_rows: int = 200,
    refresh: bool = False,
    return_panels: bool = False,
) -> CorrBuildResult:
    syms = _canonical_symbols(symbols)
    if not syms:
        raise ValueError("lista de simbolos vazia")

    data_path, meta_path = _cache_paths(
        symbols=syms,
        timeframe=timeframe,
        window_bars=int(window_bars),
        method=method,
        stats_mode=str(stats_mode),
        engine=str(engine),
        min_obs_ratio=float(min_obs_ratio),
        source_days=int(source_days),
    )
    if data_path.exists() and meta_path.exists() and (not bool(refresh)):
        print(f"[corr] cache hit {data_path}", flush=True)
        metrics = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return CorrBuildResult(
            metrics_path=data_path,
            meta_path=meta_path,
            metrics=metrics,
            metadata=meta,
        )

    t0 = time.time()
    print(
        f"[corr] build symbols={len(syms)} timeframe={timeframe} window_bars={int(window_bars)} "
        f"method={method} stats_mode={stats_mode} engine={engine} source_days={int(source_days)}",
        flush=True,
    )
    t_load = time.perf_counter()
    prices = build_price_panel(
        syms,
        source_days=int(source_days),
        timeframe=str(timeframe),
        min_symbol_rows=int(min_symbol_rows),
    )
    print(f"[corr] stage prices sec={time.perf_counter() - t_load:.2f} rows={len(prices)} cols={len(prices.columns)}", flush=True)
    if prices.empty:
        raise RuntimeError("nenhum preco carregado para o universo")
    t_ret = time.perf_counter()
    returns = build_returns_panel(prices, min_obs_ratio=float(min_obs_ratio))
    print(f"[corr] stage returns sec={time.perf_counter() - t_ret:.2f} rows={len(returns)} cols={len(returns.columns)}", flush=True)
    t_corr = time.perf_counter()
    metrics = rolling_correlation_metrics(
        returns,
        window_bars=int(window_bars),
        method=str(method),
        chunk_points=int(chunk_points),
        min_window_obs_ratio=float(min_window_obs_ratio),
        engine=str(engine),
        stats_mode=str(stats_mode),
        benchmark=True,
    )
    print(f"[corr] stage corr sec={time.perf_counter() - t_corr:.2f} rows={len(metrics)}", flush=True)

    meta = {
        "version": "v2",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "symbols_requested": syms,
        "symbols_used": [str(c) for c in prices.columns],
        "timeframe": str(timeframe),
        "window_bars": int(window_bars),
        "window_minutes": int(_freq_to_minutes(str(timeframe))) * int(window_bars),
        "method": str(method),
        "engine_requested": str(engine),
        "engine_used": str(getattr(metrics, "attrs", {}).get("corr_engine_used", str(engine))),
        "stats_mode": str(getattr(metrics, "attrs", {}).get("corr_stats_mode", str(stats_mode))),
        "source_days": int(source_days),
        "min_obs_ratio": float(min_obs_ratio),
        "min_window_obs_ratio": float(min_window_obs_ratio),
        "chunk_points": int(chunk_points),
        "prices_rows": int(len(prices)),
        "returns_rows": int(len(returns)),
        "metrics_rows": int(len(metrics)),
        "start_ts": None if prices.empty else str(pd.to_datetime(prices.index.min())),
        "end_ts": None if prices.empty else str(pd.to_datetime(prices.index.max())),
        "elapsed_sec": float(time.time() - t0),
    }
    metrics.to_parquet(data_path, index=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[corr] saved {data_path}", flush=True)

    return CorrBuildResult(
        metrics_path=data_path,
        meta_path=meta_path,
        metrics=metrics,
        metadata=meta,
        prices=prices if return_panels else None,
        returns=returns if return_panels else None,
    )


def load_corr_cache(
    symbols: list[str],
    *,
    source_days: int = 0,
    timeframe: str = "5m",
    window_bars: int = 12 * 60 // 5,
    method: str = "pearson",
    engine: str = "auto",
    stats_mode: str = "avg_only",
    min_obs_ratio: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    data_path, meta_path = _cache_paths(
        symbols=symbols,
        timeframe=timeframe,
        window_bars=int(window_bars),
        method=method,
        stats_mode=str(stats_mode),
        engine=str(engine),
        min_obs_ratio=float(min_obs_ratio),
        source_days=int(source_days),
    )
    metrics = pd.read_parquet(data_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return metrics, meta, data_path, meta_path


def plot_corr_metrics(
    metrics: pd.DataFrame,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    cols: tuple[str, ...] = ("avg_corr", "median_corr", "p90_corr"),
) -> Path | None:
    if metrics is None or metrics.empty:
        print("[corr] plot skip: metrics vazio", flush=True)
        return None

    use_cols = [c for c in cols if c in metrics.columns]
    if not use_cols:
        raise ValueError(f"nenhuma coluna encontrada para plotar: {cols}")

    out_path: Path | None = None
    plot_df = metrics[use_cols].copy()
    # evita salvar/plotar linhas totalmente vazias em modo avg_only+full colunas pedidas
    plot_df = plot_df.dropna(axis=1, how="all")
    if plot_df.empty:
        print("[corr] plot skip: colunas selecionadas vazias", flush=True)
        return None

    if save_path is not None:
        out_path = Path(save_path)
        _ensure_dir(out_path.parent)
    if _plot_time_series is not None:
        _ = _plot_time_series(
            plot_df,
            columns=list(plot_df.columns),
            title=str(title or "Correlacao rolling do universo"),
            y_title="correlacao",
            save_path=out_path,
            show=bool(show),
        )
        if out_path is not None:
            print(f"[corr] plot saved {out_path}", flush=True)
        return out_path

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[corr] plotting indisponivel: {type(e).__name__}: {e}", flush=True)
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    for c in plot_df.columns:
        ax.plot(plot_df.index, plot_df[c].astype(float), label=str(c), linewidth=1.2)
    ax.set_xlabel("tempo")
    ax.set_ylabel("correlacao")
    ax.set_title(title or "Correlacao rolling do universo")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
        print(f"[corr] plot saved {out_path}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _select_symbols_from_train_cfg(
    *,
    max_symbols: int,
    mcap_min_usd: float,
    mcap_max_usd: float,
) -> list[str]:
    from train.sniper_trainer import TrainConfig, _select_symbols  # type: ignore

    cfg = TrainConfig(
        asset_class="crypto",
        mcap_min_usd=float(mcap_min_usd),
        mcap_max_usd=float(mcap_max_usd),
        max_symbols=int(max_symbols),
    )
    syms = [str(s).upper() for s in _select_symbols(cfg)]
    if int(max_symbols) > 0:
        syms = syms[: int(max_symbols)]
    return _canonical_symbols(syms)


def run_build(
    *,
    symbols: list[str] | None = None,
    max_symbols: int = 0,
    mcap_min_usd: float = 100_000_000.0,
    mcap_max_usd: float = 150_000_000_000.0,
    source_days: int = 0,
    timeframe: str = "5m",
    window_bars: int = 12 * 60 // 5,
    method: str = "pearson",
    engine: str = "auto",
    stats_mode: str = "avg_only",
    min_obs_ratio: float = 0.80,
    min_window_obs_ratio: float = 0.80,
    chunk_points: int = 256,
    min_symbol_rows: int = 200,
    refresh: bool = False,
    plot: bool = True,
    show_plot: bool = True,
) -> CorrBuildResult:
    syms = _canonical_symbols(symbols or [])
    if not syms:
        syms = _select_symbols_from_train_cfg(
            max_symbols=int(max_symbols),
            mcap_min_usd=float(mcap_min_usd),
            mcap_max_usd=float(mcap_max_usd),
        )
    if not syms:
        raise RuntimeError("nao foi possivel montar universo de simbolos")

    res = build_and_cache_correlation(
        syms,
        source_days=int(source_days),
        timeframe=str(timeframe),
        window_bars=int(window_bars),
        method=str(method),
        engine=str(engine),
        stats_mode=str(stats_mode),
        min_obs_ratio=float(min_obs_ratio),
        min_window_obs_ratio=float(min_window_obs_ratio),
        chunk_points=int(chunk_points),
        min_symbol_rows=int(min_symbol_rows),
        refresh=bool(refresh),
        return_panels=False,
    )
    if plot:
        plot_path = res.metrics_path.with_suffix(".html")
        plot_corr_metrics(
            res.metrics,
            title=f"Rolling corr | {timeframe} | w={int(window_bars)} | n={len(res.metadata.get('symbols_used', []))}",
            save_path=plot_path,
            show=bool(show_plot),
        )
    return res


def main() -> None:
    symbols = _parse_symbols_env()
    max_symbols = _env_int("MAX_SYMBOLS", 0)
    mcap_min_usd = _env_float("MCAP_MIN_USD", 100_000_000.0)
    mcap_max_usd = _env_float("MCAP_MAX_USD", 150_000_000_000.0)
    source_days = _env_int("DAYS", 0)  # 0 = historico completo
    timeframe = _env_str("CORR_TIMEFRAME", "5m")
    window_bars = _env_int("CORR_WINDOW_BARS", 12 * 60 // 5)
    method = _env_str("CORR_METHOD", "pearson")
    engine = _env_str("CORR_ENGINE", "auto")
    stats_mode = _env_str("CORR_STATS_MODE", "avg_only")
    min_obs_ratio = _env_float("CORR_MIN_OBS_RATIO", 0.80)
    min_window_obs_ratio = _env_float("CORR_MIN_WINDOW_OBS_RATIO", 0.80)
    chunk_points = _env_int("CORR_CHUNK_POINTS", 256)
    min_symbol_rows = _env_int("CORR_MIN_SYMBOL_ROWS", 200)
    refresh = _env_bool("CORR_CACHE_REFRESH", False)
    plot = _env_bool("CORR_PLOT", True)
    show_plot = _env_bool("CORR_PLOT_SHOW", True)

    _ = run_build(
        symbols=symbols,
        max_symbols=int(max_symbols),
        mcap_min_usd=float(mcap_min_usd),
        mcap_max_usd=float(mcap_max_usd),
        source_days=int(source_days),
        timeframe=timeframe,
        window_bars=int(window_bars),
        method=method,
        engine=engine,
        stats_mode=stats_mode,
        min_obs_ratio=float(min_obs_ratio),
        min_window_obs_ratio=float(min_window_obs_ratio),
        chunk_points=int(chunk_points),
        min_symbol_rows=int(min_symbol_rows),
        refresh=bool(refresh),
        plot=bool(plot),
        show_plot=bool(show_plot),
    )


if __name__ == "__main__":
    main()
