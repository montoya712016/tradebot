# -*- coding: utf-8 -*-
from __future__ import annotations

"""
MySQL loader for stocks (1m OHLCV).
"""

import os
import time
import threading
from typing import Any
import numpy as np
import pandas as pd
import mysql.connector

# Config defaults (override via ENV)
DB_CFG_STOCKS = dict(
    host=os.getenv("PF_DB_STOCKS_HOST", "localhost"),
    user=os.getenv("PF_DB_STOCKS_USER", "root"),
    password=os.getenv("PF_DB_STOCKS_PASS", "2017"),
    database=os.getenv("PF_DB_STOCKS_NAME", "stocks_us"),
)

# Prefira driver C mesmo com várias threads (mais rápido para volumes grandes).
_prefer = os.getenv("PF_PREFER_PURE_FOR_THREADS", "0").strip().lower()
PREFER_PURE_FOR_THREADS = _prefer in {"1", "true", "yes", "y", "on"}


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


def _norm_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    return s.replace(".", "-")


def load_ohlc_1m_series_stock(sym: str, days: int, *, remove_tail_days: int = 0) -> pd.DataFrame:
    """Load 1m OHLCV from schema 'stocks_us' (tables in uppercase)."""
    now_ms = _now_ms_from_env()
    now_s = now_ms / 1000.0
    start_ms = 0 if int(days) <= 0 else int((now_s - days * 86400) * 1000)
    end_ms = None
    if remove_tail_days and remove_tail_days > 0:
        end_ms = int((now_s - remove_tail_days * 86400) * 1000)

    force = (os.getenv("PF_FORCE_DRIVER") or "").strip().lower()
    if force == "cext":
        want_c_ext = True
    elif force == "pure":
        want_c_ext = False
    else:
        many_threads = threading.active_count() > 1
        want_c_ext = not (PREFER_PURE_FOR_THREADS and many_threads)

    sym_tbl = _norm_symbol(sym)

    conn = _open_conn(DB_CFG_STOCKS, want_c_ext=want_c_ext, use_compress=False, timeout=8)
    try:
        use_raw = (not want_c_ext)
        cur = conn.cursor(buffered=False, raw=use_raw)
        try:
            with conn.cursor() as c2:
                c2.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
        except Exception:
            pass
        if end_ms is None:
            sql = (
                f"SELECT dates, open_prices, high_prices, low_prices, closing_prices, volume "
                f"FROM `{sym_tbl}` WHERE dates >= %s ORDER BY dates"
            )
            params = (start_ms,)
        else:
            sql = (
                f"SELECT dates, open_prices, high_prices, low_prices, closing_prices, volume "
                f"FROM `{sym_tbl}` WHERE dates >= %s AND dates < %s ORDER BY dates"
            )
            params = (start_ms, end_ms)
        try:
            cur.execute(sql, params)
        except mysql.connector.Error as exc:
            # tabela inexistente ou outro erro -> retorna vazio sem quebrar pipeline
            try:
                code = getattr(exc, "errno", None)
            except Exception:
                code = None
            if code == 1146:  # table doesn't exist
                cur.close()
                conn.close()
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            raise
        CHUNK = 2_000_000
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
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(np.asarray(ts_list, dtype=np.int64), unit="ms")
    df = pd.DataFrame(
        {
            "open": np.asarray(o_list, dtype=np.float64),
            "high": np.asarray(h_list, dtype=np.float64),
            "low": np.asarray(l_list, dtype=np.float64),
            "close": np.asarray(c_list, dtype=np.float64),
            "volume": np.asarray(v_list, dtype=np.float64),
        },
        index=ts,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


__all__ = ["DB_CFG_STOCKS", "load_ohlc_1m_series_stock"]
