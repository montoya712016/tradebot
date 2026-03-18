# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any
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
from utils.guarded_runner import GuardedParallelDefaults, GuardedRunner  # type: ignore
from utils.progress import LineProgressPrinter  # type: ignore


_GUARD = GuardedRunner(
    log_prefix="[ohlc-cache]",
    env_prefix="OHLC_CACHE",
    defaults=GuardedParallelDefaults(
        max_ram_pct=78.0,
        min_free_mb=3072.0,
        per_worker_mem_mb=768.0,
        critical_ram_pct=92.0,
        critical_min_free_mb=1024.0,
        abort_on_critical_ram=True,
        min_workers=1,
        poll_interval_s=0.3,
        log_every_s=20.0,
        throttle_sleep_s=3.0,
    ),
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return float(default)


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


def run_build(
    *,
    max_symbols: int = 0,
    days: int = 0,
    mcap_min_usd: float = 100_000_000.0,
    mcap_max_usd: float = 150_000_000_000.0,
    refresh: bool = False,
    workers: int = 0,
) -> dict[str, Any]:
    os.environ.setdefault("PF_OHLC_CACHE", "1")
    os.environ["PF_OHLC_CACHE_REFRESH"] = "1" if bool(refresh) else "0"

    cfg = TrainConfig(
        asset_class="crypto",
        mcap_min_usd=float(mcap_min_usd),
        mcap_max_usd=float(mcap_max_usd),
        max_symbols=int(max_symbols),
    )
    symbols = _select_symbols(cfg)
    if int(max_symbols) > 0:
        symbols = symbols[: int(max_symbols)]

    print(
        f"[ohlc-cache] symbols={len(symbols)} days={int(days)} refresh={os.environ['PF_OHLC_CACHE_REFRESH']} "
        f"mcap=[{float(mcap_min_usd):.0f}..{float(mcap_max_usd):.0f}]",
        flush=True,
    )
    t0 = time.time()
    total = len(symbols)
    done = 0
    ok = 0
    fail = 0
    skipped = 0
    prog = LineProgressPrinter(prefix="ohlc-cache", total=total, width=26, stream=sys.stderr, min_interval_s=1.0)

    if int(workers) <= 0:
        workers = _env_int("OHLC_CACHE_WORKERS", min(4, int(os.cpu_count() or 4)))
    workers = max(1, int(workers))
    policy = _GUARD.make_policy()
    print(
        f"[ohlc-cache] workers={workers} ram_cap={policy.max_ram_pct:.1f}% "
        f"min_free_mb={policy.min_free_mb:.0f} per_worker_mb={policy.per_worker_mem_mb:.0f}",
        flush=True,
    )

    def _worker(sym: str) -> dict[str, Any]:
        _GUARD.thermal_wait(f"ohlc:{sym}")
        if (not refresh) and _cache_ok(sym):
            return {"sym": sym, "skip": True, "rows": 0, "sec": 0.0}
        t1 = time.time()
        df = load_ohlc_1m_series(sym, days=int(days), remove_tail_days=0)
        return {"sym": sym, "skip": False, "rows": int(len(df)), "sec": float(time.time() - t1)}

    for sym_submitted, fut in _GUARD.adaptive_map(
        symbols,
        _worker,
        max_workers=workers,
        policy=policy,
        task_name="ohlc-cache",
    ):
        cur = sym_submitted
        try:
            out = fut.result()
            cur = str(out.get("sym", sym_submitted))
            if bool(out.get("skip", False)):
                skipped += 1
            else:
                ok += 1
        except Exception as e:
            if _GUARD.is_guard_error(e):
                print(f"[ohlc-cache] ABORT guard {sym_submitted}: {type(e).__name__}: {e}", flush=True)
                raise
            fail += 1
            print(f"[ohlc-cache] FAIL {sym_submitted}: {type(e).__name__}: {e}", flush=True)
        done += 1
        prog.update(done, current=cur, force=True)

    prog.close()
    dt = time.time() - t0
    print(f"[ohlc-cache] done ok={ok} skip={skipped} fail={fail} sec={dt:.2f}", flush=True)
    return {
        "ok": int(ok),
        "skip": int(skipped),
        "fail": int(fail),
        "total": int(total),
        "seconds": float(dt),
    }


def main() -> None:
    refresh = os.getenv("PF_OHLC_CACHE_REFRESH", "0").strip().lower() in {"1", "true", "yes", "on"}
    max_symbols = _env_int("MAX_SYMBOLS", 0)
    days = _env_int("DAYS", 0)  # 0 = historico completo
    mcap_min = _env_float("MCAP_MIN_USD", 100_000_000.0)
    mcap_max = _env_float("MCAP_MAX_USD", 150_000_000_000.0)
    workers = _env_int("OHLC_CACHE_WORKERS", 0)
    _ = run_build(
        max_symbols=max_symbols,
        days=days,
        mcap_min_usd=mcap_min,
        mcap_max_usd=mcap_max,
        refresh=bool(refresh),
        workers=int(workers),
    )


if __name__ == "__main__":
    main()
