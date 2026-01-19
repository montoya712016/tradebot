# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Plot all available close data for a single stock from MySQL.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

import mysql.connector
import pandas as pd


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if p.name.lower() == "tradebot":
            root = p
            break
    if root:
        for cand in (root, root / "modules"):
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)


try:
    from modules.plotting import plot_all
except Exception:
    _add_repo_paths()
    try:
        from modules.plotting import plot_all  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Este script requer o modulo plotting (plotly).") from e


# permitir rodar como script direto (sem PYTHONPATH)
if __package__ in (None, ""):
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break


# Simple config (edit and run)
SYMBOL = "AAPL"
DB_NAME = "stocks_us"
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "2017"
OUTPUT_HTML = "D:/MySQL/plots/stock_plot.html"
AUTO_OPEN = True


def main() -> None:
    sym = str(SYMBOL).strip().upper().replace(".", "-")

    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT dates, open_prices, high_prices, low_prices, closing_prices FROM `{sym}` ORDER BY dates"
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        print(f"Sem dados para {sym} no intervalo.")
        return

    idx = [datetime.fromtimestamp(int(r[0]) / 1000, tz=timezone.utc) for r in rows]
    df = pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
        },
        index=pd.to_datetime(idx),
    )

    plot_all(
        df,
        flags={"label": False, "plot_candles": True},
        candle_sec=60,
        plot_candles=True,
        max_points=0,    # 0 => sem downsample
        max_candles=None,  # None => sempre velas
        show=bool(AUTO_OPEN),
        save_path=str(OUTPUT_HTML),
    )
    print(f"Plot salvo em: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
