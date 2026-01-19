# -*- coding: utf-8 -*-
"""
Label-only stats for stocks:
- counts label=1 per symbol
- counts label=1 per day (aggregate)
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
from typing import Iterable

import numpy as np
import pandas as pd
import mysql.connector


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if p.name.lower() == "tradebot":
            root = p
            break
    if root:
        for cand in (root / "modules", root):
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)


_add_repo_paths()

from modules.prepare_features.data_stocks import load_ohlc_1m_series_stock, DB_CFG_STOCKS
from modules.prepare_features.labels import apply_trade_contract_labels
from stocks.trade_contract import DEFAULT_TRADE_CONTRACT as STOCKS_CONTRACT


def _read_symbols_from_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out = []
    seen = set()
    for ln in lines:
        s = ln.strip().upper()
        if not s or s.startswith("#"):
            continue
        s = s.replace(".", "-")
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_symbols_from_env() -> list[str]:
    raw = os.getenv("PF_LABEL_SYMBOLS", "").strip()
    if not raw:
        return []
    out = []
    seen = set()
    for tok in raw.replace(";", ",").split(","):
        s = tok.strip().upper()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_symbols_from_db(limit: int | None = None) -> list[str]:
    cfg = dict(DB_CFG_STOCKS)
    conn = mysql.connector.connect(**cfg)
    try:
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        rows = cur.fetchall() or []
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
    syms = [str(r[0]).upper() for r in rows]
    syms.sort()
    if limit and limit > 0:
        syms = syms[:limit]
    return syms


def _pick_label_col(df: pd.DataFrame) -> str:
    windows = list(getattr(STOCKS_CONTRACT, "entry_label_windows_minutes", []) or [])
    if windows:
        col = f"sniper_entry_label_{int(windows[0])}m"
        if col in df.columns:
            return col
    if "sniper_entry_label" in df.columns:
        return "sniper_entry_label"
    for c in df.columns:
        if c.startswith("sniper_entry_label_"):
            return c
    return "sniper_entry_label"


def _iter_symbols() -> list[str]:
    file_path = os.getenv("PF_LABEL_SYMBOLS_FILE", "").strip()
    if file_path:
        syms = _read_symbols_from_file(file_path)
        if syms:
            return syms
    syms = _read_symbols_from_env()
    if syms:
        return syms
    limit = int(os.getenv("PF_LABEL_MAX_SYMBOLS", "0") or 0)
    return _read_symbols_from_db(limit=limit if limit > 0 else None)


def main() -> None:
    days = int(os.getenv("PF_LABEL_DAYS", "180") or 180)
    remove_tail = int(os.getenv("PF_LABEL_TAIL_DAYS", "0") or 0)
    symbols = _iter_symbols()
    if not symbols:
        print("Nenhum simbolo encontrado (file/env/db).", flush=True)
        return

    print(f"[labels] symbols={len(symbols)} days={days}", flush=True)

    per_symbol = []
    per_day_all: dict = {}

    for i, sym in enumerate(symbols, 1):
        try:
            df = load_ohlc_1m_series_stock(sym, int(days), remove_tail_days=int(remove_tail))
            if df.empty:
                df = load_ohlc_1m_series_stock(sym, 0, remove_tail_days=int(remove_tail))
        except Exception as exc:
            print(f"[labels] {sym}: erro {type(exc).__name__}: {exc}", flush=True)
            continue
        if df.empty:
            print(f"[labels] {sym}: sem dados (mesmo sem filtro de dias)", flush=True)
            continue

        df = apply_trade_contract_labels(df, contract=STOCKS_CONTRACT, candle_sec=60)
        label_col = _pick_label_col(df)
        lbl = df[label_col].astype(int)
        total = int(lbl.sum())

        day_counts = lbl.groupby(df.index.date).sum()
        for d, v in day_counts.items():
            per_day_all[d] = per_day_all.get(d, 0) + int(v)

        per_symbol.append((sym, total, int(day_counts.count())))
        if i % 25 == 0 or i == len(symbols):
            print(f"[labels] progresso {i}/{len(symbols)}", flush=True)

    if not per_symbol:
        print("Sem resultados de labels.", flush=True)
        return

    total_labels = sum(v for _, v, _ in per_symbol)
    unique_days = len(per_day_all)
    avg_per_day = (float(total_labels) / float(unique_days)) if unique_days else 0.0
    print(f"[labels] total_labels={total_labels} unique_days={unique_days} avg_per_day={avg_per_day:.2f}", flush=True)

    per_symbol.sort(key=lambda x: x[1], reverse=True)
    print("[labels] top symbols:", flush=True)
    for sym, total, days_with_data in per_symbol[:20]:
        print(f"  {sym}: labels={total} days={days_with_data}", flush=True)

    try:
        import plotly.graph_objects as go

        days_sorted = sorted(per_day_all.keys())
        counts = [per_day_all[d] for d in days_sorted]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days_sorted, y=counts, mode="markers+lines", name="label_1_count"))
        fig.update_layout(
            title="Label=1 por dia (total)",
            xaxis_title="Date",
            yaxis_title="Count",
            template="plotly_white",
        )
        fig.show()
    except Exception as exc:
        print(f"[labels] plot erro: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
