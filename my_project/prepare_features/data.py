# -*- coding: utf-8 -*-
import os, time, threading
from typing import Any
import numpy as np, pandas as pd
import mysql.connector

# Configuração padrão dos bancos (pode sobrescrever por ENV)
DB_CFG_1S = dict(
    host=os.getenv("PF_DB1S_HOST", "localhost"),
    user=os.getenv("PF_DB1S_USER", "root"),
    password=os.getenv("PF_DB1S_PASS", "2017"),
    database=os.getenv("PF_DB1S_NAME", "crypto_second"),
)
DB_CFG_1M = dict(
    host=os.getenv("PF_DB1M_HOST", "localhost"),
    user=os.getenv("PF_DB1M_USER", "root"),
    password=os.getenv("PF_DB1M_PASS", "2017"),
    database=os.getenv("PF_DB1M_NAME", "crypto"),
)

# Gap padrão ao costurar base 1s
DEFAULT_MAX_GAP_SEC = int(os.getenv("PF_MAX_GAP_SEC", str(60*15)))

PREFER_PURE_FOR_THREADS = True


def _open_conn(db_cfg: dict[str, Any], *, want_c_ext: bool = True, use_compress: bool = False, timeout: int = 8):
    t0 = time.perf_counter()
    try:
        conn = mysql.connector.connect(
            autocommit=True,
            use_pure=not want_c_ext,
            compress=use_compress,
            connection_timeout=timeout,
            **db_cfg,
        )
        conn.ping(reconnect=False, attempts=1, delay=0)
        _ = time.perf_counter() - t0
        return conn
    except Exception as e:
        if want_c_ext:
            return _open_conn(db_cfg, want_c_ext=False, use_compress=True, timeout=timeout)
        raise


def _now_ms_from_env() -> int:
    v = os.getenv("PF_NOW_MS")
    if v:
        try:
            return int(v)
        except Exception:
            pass
    return int(time.time() * 1000)


def load_close_series(sym: str, days: int, *, remove_tail_days: int = 0) -> pd.Series:
    """Lê série de close em 1s do schema 'crypto_second' (tabelas em lower-case)."""
    now_ms = _now_ms_from_env()
    now_s = now_ms / 1000.0
    start_ms = int((now_s - days*86400) * 1000)
    end_ms = None
    if remove_tail_days and remove_tail_days > 0:
        end_ms = int((now_s - remove_tail_days*86400) * 1000)

    force = (os.getenv("PF_FORCE_DRIVER") or "").strip().lower()
    if force == "cext":
        want_c_ext = True
    elif force == "pure":
        want_c_ext = False
    else:
        many_threads = threading.active_count() > 1
        want_c_ext = not (PREFER_PURE_FOR_THREADS and many_threads)

    conn = _open_conn(DB_CFG_1S, want_c_ext=want_c_ext, use_compress=False, timeout=8)
    try:
        use_raw = (not want_c_ext)
        cur = conn.cursor(buffered=False, raw=use_raw)
        try:
            with conn.cursor() as c2:
                c2.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
        except Exception:
            pass
        if end_ms is None:
            sql = (f"SELECT dates, closing_prices FROM `{sym.lower()}` FORCE INDEX (PRIMARY) "
                   "WHERE dates >= %s ORDER BY dates")
            params = (start_ms,)
        else:
            sql = (f"SELECT dates, closing_prices FROM `{sym.lower()}` FORCE INDEX (PRIMARY) "
                   "WHERE dates >= %s AND dates < %s ORDER BY dates")
            params = (start_ms, end_ms)
        cur.execute(sql, params)
        CHUNK = 1_000_000
        ts_list, px_list = [], []
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            if use_raw:
                ts_list.extend(int(d) for d, _ in rows)
                px_list.extend(float(p) for _, p in rows)
            else:
                for d, p in rows:
                    ts_list.append(int(d)); px_list.append(float(p))
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

    if not ts_list:
        return pd.Series(dtype="float64")
    ts = pd.to_datetime(np.asarray(ts_list, dtype=np.int64), unit="ms")
    px = np.asarray(px_list, dtype=np.float32)
    ser = pd.Series(px.astype(np.float64, copy=False), index=ts, name="close")
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    if getattr(ser.index, "tz", None) is not None:
        ser.index = ser.index.tz_localize(None)
    return ser


def load_ohlc_1m_series(sym: str, days: int, *, remove_tail_days: int = 0) -> pd.DataFrame:
    """Lê OHLCV 1-min do schema 'crypto' (tabelas em lower-case)."""
    now_ms = _now_ms_from_env()
    now_s = now_ms / 1000.0
    start_ms = int((now_s - days*86400) * 1000)
    end_ms = None
    if remove_tail_days and remove_tail_days > 0:
        end_ms = int((now_s - remove_tail_days*86400) * 1000)

    force = (os.getenv("PF_FORCE_DRIVER") or "").strip().lower()
    if force == "cext":
        want_c_ext = True
    elif force == "pure":
        want_c_ext = False
    else:
        many_threads = threading.active_count() > 1
        want_c_ext = not (PREFER_PURE_FOR_THREADS and many_threads)

    conn = _open_conn(DB_CFG_1M, want_c_ext=want_c_ext, use_compress=False, timeout=8)
    try:
        use_raw = (not want_c_ext)
        cur = conn.cursor(buffered=False, raw=use_raw)
        try:
            with conn.cursor() as c2:
                c2.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
        except Exception:
            pass
        if end_ms is None:
            sql = (f"SELECT dates, open_prices, high_prices, low_prices, closing_prices, volume "
                   f"FROM `{sym.lower()}` WHERE dates >= %s ORDER BY dates")
            params = (start_ms,)
        else:
            sql = (f"SELECT dates, open_prices, high_prices, low_prices, closing_prices, volume "
                   f"FROM `{sym.lower()}` WHERE dates >= %s AND dates < %s ORDER BY dates")
            params = (start_ms, end_ms)
        cur.execute(sql, params)
        CHUNK = 1_000_000
        ts_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            if use_raw:
                for r in rows:
                    ts_list.append(int(r[0]))
                    o_list.append(float(r[1]) if r[1] is not None else np.nan)
                    h_list.append(float(r[2]) if r[2] is not None else np.nan)
                    l_list.append(float(r[3]) if r[3] is not None else np.nan)
                    c_list.append(float(r[4]) if r[4] is not None else np.nan)
                    v_list.append(float(r[5]) if r[5] is not None else 0.0)
            else:
                for d, o, h, l, c, v in rows:
                    ts_list.append(int(d))
                    o_list.append(float(o) if o is not None else np.nan)
                    h_list.append(float(h) if h is not None else np.nan)
                    l_list.append(float(l) if l is not None else np.nan)
                    c_list.append(float(c) if c is not None else np.nan)
                    v_list.append(float(v) if v is not None else 0.0)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

    if not ts_list:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    ts = pd.to_datetime(np.asarray(ts_list, dtype=np.int64), unit="ms")
    df = pd.DataFrame({
        "open":  np.asarray(o_list, dtype=np.float64),
        "high":  np.asarray(h_list, dtype=np.float64),
        "low":   np.asarray(l_list, dtype=np.float64),
        "close": np.asarray(c_list, dtype=np.float64),
        "volume":np.asarray(v_list, dtype=np.float64),
    }, index=ts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def to_ohlc_from_1m(df_1m: pd.DataFrame, candle_sec: int) -> pd.DataFrame:
    """Reamostra OHLC 1m para janelas de múltiplos de minuto, mantendo OHLC padrão."""
    assert candle_sec >= 60 and candle_sec % 60 == 0
    if candle_sec == 60:
        df = df_1m.copy()
    else:
        rule = f"{candle_sec}s"
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df_1m.columns:
            agg["volume"] = "sum"
        df = df_1m.resample(rule).agg(agg)
    df = df.dropna(subset=["open","high","low","close"])
    df["gap_after"] = False
    return df


def to_ohlc_gapfill(obj: pd.Series | pd.DataFrame, *, candle_sec: int, max_gap_sec: int = DEFAULT_MAX_GAP_SEC) -> pd.DataFrame:
    """
    Se obj for DataFrame OHLC, reamostra (1m→candle_sec) sem costura de gaps.
    Se obj for Series de close (1s), reamostra para OHLC e costura gaps curtos.
    """
    if isinstance(obj, pd.DataFrame) and {"open","high","low","close"}.issubset(obj.columns):
        return to_ohlc_from_1m(obj, candle_sec)

    ser = obj
    rule = f"{candle_sec}s"
    ohlc = ser.resample(rule).ohlc()
    full = pd.date_range(ohlc.index[0], ohlc.index[-1], freq=rule)
    ohlc = ohlc.reindex(full)
    ohlc["close"] = ohlc["close"].ffill()
    # classifica gaps
    max_c = max(1, max_gap_sec // candle_sec)
    mask = ohlc["open"].isna()
    grp = (mask != mask.shift()).cumsum()
    run_len = mask.groupby(grp).transform("size")
    short_gap = mask & (run_len <= max_c)
    fillv = ohlc.loc[short_gap, "close"].to_numpy()
    ohlc.loc[short_gap, ["open","high","low","close"]] = np.c_[fillv, fillv, fillv, fillv]
    long_gap = mask & (~short_gap)
    gap_after = (~mask) & long_gap.shift(-1).fillna(False)
    ohlc["gap_after"] = gap_after.astype(np.uint8)
    return ohlc


__all__ = [
    "DB_CFG_1S","DB_CFG_1M","DEFAULT_MAX_GAP_SEC",
    "load_close_series","load_ohlc_1m_series","to_ohlc_from_1m","to_ohlc_gapfill",
]


