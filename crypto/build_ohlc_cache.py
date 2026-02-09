# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys
import time


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

from train.sniper_trainer import TrainConfig, _select_symbols  # type: ignore
from prepare_features.data import load_ohlc_1m_series  # type: ignore
from prepare_features.data import _ohlc_cache_paths  # type: ignore


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)

def _cache_ok(sym: str) -> bool:
    cache_path, meta_path = _ohlc_cache_paths(sym)
    if not cache_path.exists():
        return False
    try:
        if meta_path.exists():
            meta = meta_path.read_text(encoding="utf-8")
            if meta:
                import json
                data = json.loads(meta)
                rows = int(data.get("rows") or 0)
                return rows > 0
    except Exception:
        pass
    try:
        return cache_path.stat().st_size > 4096
    except Exception:
        return False


def main() -> None:
    # garante cache ligado para o build
    os.environ.setdefault("PF_OHLC_CACHE", "1")
    # force refresh se quiser rebuild completo
    refresh = os.getenv("PF_OHLC_CACHE_REFRESH", "0").strip().lower()
    os.environ["PF_OHLC_CACHE_REFRESH"] = "1" if refresh in {"1", "true", "yes", "on"} else "0"

    max_symbols = _env_int("MAX_SYMBOLS", 0)
    days = _env_int("DAYS", 0)  # 0 = histórico completo

    cfg = TrainConfig(
        asset_class="crypto",
        mcap_min_usd=50_000_000.0,
        mcap_max_usd=150_000_000_000.0,
        max_symbols=max_symbols,
    )
    symbols = _select_symbols(cfg)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    print(f"[ohlc-cache] symbols={len(symbols)} days={days} refresh={os.environ['PF_OHLC_CACHE_REFRESH']}", flush=True)
    t0 = time.time()
    do_refresh = os.environ["PF_OHLC_CACHE_REFRESH"] == "1"
    for i, sym in enumerate(symbols, 1):
        t1 = time.time()
        if (not do_refresh) and _cache_ok(sym):
            print(f"[ohlc-cache] {i}/{len(symbols)} {sym} skip (cached)", flush=True)
            continue
        df = load_ohlc_1m_series(sym, days=days, remove_tail_days=0)
        dt = time.time() - t1
        print(f"[ohlc-cache] {i}/{len(symbols)} {sym} rows={len(df):,} dt={dt:.2f}s", flush=True)
    print(f"[ohlc-cache] done sec={time.time()-t0:.2f}", flush=True)


if __name__ == "__main__":
    main()
