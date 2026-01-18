from __future__ import annotations

"""
Lightweight probe for Tiingo data availability and column schema.

This script is intentionally small and safe: it fetches minimal data so we can
inspect available fields before running any bulk download.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any


BASE_URL = "https://api.tiingo.com"


def _load_token() -> str:
    token = os.getenv("TIINGO_API_KEY", "").strip()
    if token:
        return token

    # Try local config (modules/config/secrets.py)
    try:
        # allow running as script without PYTHONPATH
        here = Path(__file__).resolve()
        for p in here.parents:
            if p.name.lower() == "modules":
                sp = str(p)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                break
        from config import secrets as sec  # type: ignore

        token = str(getattr(sec, "TIINGO_API_KEY", "") or "").strip()
    except Exception:
        token = ""
    if not token:
        raise RuntimeError("Missing TIINGO_API_KEY (env or modules/config/secrets.py).")
    return token


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    token = _load_token()
    qp = urllib.parse.urlencode(params or {})
    url = f"{BASE_URL}/{path.lstrip('/')}"
    if qp:
        url = f"{url}?{qp}"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _key_union(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            keys.update(r.keys())
    return sorted(keys)


def fetch_ticker_universe(limit: int | None = 2000) -> list[dict[str, Any]]:
    """
    Returns the full Tiingo daily ticker list (metadata). You can filter after.
    """
    rows = _get_json("tiingo/daily")
    if isinstance(rows, list) and limit:
        return rows[: int(limit)]
    return rows if isinstance(rows, list) else []


def fetch_daily_meta(symbol: str) -> dict[str, Any]:
    return _get_json(f"tiingo/daily/{symbol}")


def fetch_daily_prices(symbol: str, *, start: str, end: str) -> list[dict[str, Any]]:
    return _get_json(
        f"tiingo/daily/{symbol}/prices",
        params={"startDate": start, "endDate": end},
    )


def fetch_intraday_prices(symbol: str, *, start: str, end: str, freq: str = "1min") -> list[dict[str, Any]]:
    # IEX intraday endpoint (resample for minute bars)
    return _get_json(
        f"iex/{symbol}/prices",
        params={"startDate": start, "endDate": end, "resampleFreq": freq},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Tiingo availability and columns.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker to sample.")
    parser.add_argument("--days", type=int, default=5, help="Days of sample data.")
    parser.add_argument("--intraday", action="store_true", help="Also probe IEX intraday (1min).")
    parser.add_argument("--universe", action="store_true", help="Fetch ticker universe metadata.")
    parser.add_argument("--universe-limit", type=int, default=2000, help="Max tickers to display.")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=max(1, int(args.days)))
    start_s = start.isoformat()
    end_s = end.isoformat()

    print(f"[tiingo] sample symbol={args.symbol} start={start_s} end={end_s}", flush=True)

    meta = fetch_daily_meta(args.symbol)
    print("[tiingo] daily meta keys:", ", ".join(sorted(meta.keys())), flush=True)

    daily = fetch_daily_prices(args.symbol, start=start_s, end=end_s)
    daily_cols = _key_union(daily if isinstance(daily, list) else [])
    print("[tiingo] daily prices columns:", ", ".join(daily_cols), flush=True)

    if args.intraday:
        intraday = fetch_intraday_prices(args.symbol, start=start_s, end=end_s, freq="1min")
        intraday_cols = _key_union(intraday if isinstance(intraday, list) else [])
        print("[tiingo] intraday (1min) columns:", ", ".join(intraday_cols), flush=True)

    if args.universe:
        try:
            rows = fetch_ticker_universe(limit=args.universe_limit)
            cols = _key_union(rows if isinstance(rows, list) else [])
            print("[tiingo] universe columns:", ", ".join(cols), flush=True)
            if rows:
                print(f"[tiingo] universe sample size={len(rows)}", flush=True)
        except Exception as exc:
            print(f"[tiingo] universe fetch failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
