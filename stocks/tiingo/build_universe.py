from __future__ import annotations

"""
Build a seed universe of US stocks from public index lists (SP500 + Nasdaq100),
then enrich with Tiingo metadata and market cap.

Output: CSV and TXT (tickers), suitable as a starting universe for research.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any


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


if __package__ in (None, ""):
    _add_repo_paths()


def _load_token() -> str:
    token = os.getenv("TIINGO_API_KEY", "").strip()
    if token:
        return token
    try:
        from config import secrets as sec  # type: ignore

        token = str(getattr(sec, "TIINGO_API_KEY", "") or "").strip()
    except Exception:
        token = ""
    if not token:
        raise RuntimeError("Missing TIINGO_API_KEY (env or modules/config/secrets.py).")
    return token


def _get_json(url: str) -> Any:
    token = _load_token()
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    import json

    return json.loads(raw.decode("utf-8"))


def _normalize_ticker(sym: str) -> str:
    s = str(sym or "").strip().upper()
    # Tiingo uses '-' for class shares (e.g., BRK-B)
    return s.replace(".", "-")


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; tiingo-universe/1.0; +https://tiingo.com/)",
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _read_html_tables(html: str) -> list[Any]:
    import pandas as pd  # type: ignore
    from io import StringIO

    return pd.read_html(StringIO(html))


def _fetch_sp500() -> list[dict[str, str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _fetch_html(url)
    tables = _read_html_tables(html)
    # Table 0: S&P 500 constituents
    df = tables[0]
    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "ticker": str(row.get("Symbol", "")).strip(),
                "name": str(row.get("Security", "")).strip(),
                "security": str(row.get("Security", "")).strip(),
                "sector": str(row.get("GICS Sector", "")).strip(),
                "source": "sp500",
            }
        )
    return out


def _fetch_nasdaq100() -> list[dict[str, str]]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _fetch_html(url)
    tables = _read_html_tables(html)
    # One of the tables contains tickers; find it by column presence.
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "ticker" in cols or "ticker symbol" in cols:
            df = t
            break
    if df is None:
        return []

    # Normalize column name
    col_ticker = "Ticker" if "Ticker" in df.columns else "Ticker symbol"
    col_name = "Company" if "Company" in df.columns else None
    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "ticker": str(row.get(col_ticker, "")).strip(),
                "name": str(row.get(col_name, "")).strip() if col_name else "",
                "security": "Common Stock",
                "sector": "",
                "source": "nasdaq100",
            }
        )
    return out


def _fetch_meta(ticker: str) -> dict[str, Any] | None:
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}"
    try:
        return _get_json(url)
    except Exception:
        return None


def _fetch_market_cap(ticker: str) -> tuple[float | None, str | None]:
    # Latest market cap from fundamentals daily endpoint
    url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/daily"
    try:
        data = _get_json(url)
        if isinstance(data, list) and data:
            last = data[-1]
            mcap = last.get("marketCap")
            dt = last.get("date")
            try:
                mcap = float(mcap) if mcap is not None else None
            except Exception:
                mcap = None
            return mcap, str(dt) if dt else None
    except Exception:
        return None, None
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a US stocks seed universe.")
    parser.add_argument("--out-csv", default="data/generated/tiingo_universe_seed.csv")
    parser.add_argument("--out-txt", default="data/generated/tiingo_universe_seed.txt")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers (0 = no limit).")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep between API calls (seconds).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV if present.")
    parser.add_argument("--checkpoint", type=int, default=20, help="Flush progress every N tickers.")
    args = parser.parse_args()

    rows = _fetch_sp500() + _fetch_nasdaq100()

    # Deduplicate by ticker (prefer SP500 row if present)
    by_ticker: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = _normalize_ticker(r.get("ticker", ""))
        if not t:
            continue
        existing = by_ticker.get(t)
        if existing is None or existing.get("source") != "sp500":
            by_ticker[t] = {**r, "ticker": t}

    tickers = sorted(by_ticker.keys())
    if args.limit and int(args.limit) > 0:
        tickers = tickers[: int(args.limit)]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and out_csv.exists():
        try:
            with out_csv.open("r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    t = str(row.get("ticker", "")).strip().upper()
                    if t:
                        existing[t] = row
        except Exception:
            existing = {}

    out_rows: list[dict[str, Any]] = list(existing.values())
    existing_set = set(existing.keys())
    total = len(tickers)
    to_process = [t for t in tickers if t not in existing_set]
    for i, t in enumerate(to_process, 1):
        meta = _fetch_meta(t)
        mcap, mcap_date = _fetch_market_cap(t)
        base = by_ticker.get(t, {})
        row = {
            "ticker": t,
            "name": (meta or {}).get("name") or base.get("name", ""),
            "exchange": (meta or {}).get("exchangeCode", ""),
            "security_type": base.get("security", "Common Stock"),
            "sector": base.get("sector", ""),
            "source": base.get("source", ""),
            "market_cap": mcap,
            "market_cap_date": mcap_date,
        }
        out_rows.append(row)

        # Incremental append (resume-friendly)
        if args.resume:
            write_header = not out_csv.exists()
            with out_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ticker",
                        "name",
                        "exchange",
                        "security_type",
                        "sector",
                        "source",
                        "market_cap",
                        "market_cap_date",
                    ],
                )
                if write_header:
                    w.writeheader()
                w.writerow(row)

        if i % int(max(1, args.checkpoint)) == 0:
            print(f"[universe] processed {len(existing_set) + i}/{total}", flush=True)
        if args.sleep:
            time.sleep(float(args.sleep))

    if not args.resume:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ticker",
                    "name",
                    "exchange",
                    "security_type",
                    "sector",
                    "source",
                    "market_cap",
                    "market_cap_date",
                ],
            )
            w.writeheader()
            for r in out_rows:
                w.writerow(r)

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join([r["ticker"] for r in out_rows]), encoding="utf-8")

    print(f"[universe] saved CSV: {out_csv}", flush=True)
    print(f"[universe] saved TXT: {out_txt}", flush=True)
    print(f"[universe] total tickers: {len(out_rows)}", flush=True)


if __name__ == "__main__":
    main()
