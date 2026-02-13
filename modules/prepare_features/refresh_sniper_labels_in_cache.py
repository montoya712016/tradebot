# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Recalcula labels supervisionados (timing, edge e entry_gate) dentro do cache de features,
sem recomputar features.

Uso:
python modules/prepare_features/refresh_sniper_labels_in_cache.py
"""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd

# permitir rodar como script direto (sem PYTHONPATH)
_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if _p.name.lower() == "modules":
        _repo = _p.parent
        for _cand in (_repo, _p):
            _sp = str(_cand)
            if _sp not in sys.path:
                sys.path.insert(0, _sp)
        break

from config.symbols import default_top_market_cap_path, load_market_caps  # noqa: E402
from utils.progress import ProgressPrinter  # noqa: E402
from utils.adaptive_parallel import AdaptiveParallelPolicy, run_adaptive_thread_map  # noqa: E402
from utils.thermal_guard import ThermalGuard  # noqa: E402
from train.sniper_dataflow import _cache_dir, _cache_format, _symbol_cache_paths  # type: ignore  # noqa: E402
from prepare_features.labels import (  # noqa: E402
    apply_timing_regression_labels,
    TIMING_HORIZON_PROFIT,
    TIMING_K_LOOKAHEAD,
    TIMING_TOP_N,
    TIMING_ALPHA,
    TIMING_LABEL_CLIP,
    TIMING_WEIGHT_LABEL_MULT,
    TIMING_WEIGHT_VOL_MULT,
    TIMING_WEIGHT_MIN,
    TIMING_WEIGHT_MAX,
    TIMING_VOL_WINDOW,
    TIMING_SIDE_MAE_PENALTY,
    TIMING_SIDE_TIME_PENALTY,
    TIMING_SIDE_GIVEBACK_PENALTY,
    TIMING_SIDE_CROSS_PENALTY,
    TIMING_SIDE_REV_LOOKBACK_MIN,
    TIMING_SIDE_CHASE_PENALTY,
    TIMING_SIDE_REVERSAL_BONUS,
    TIMING_SIDE_CONFIRM_MIN,
    TIMING_SIDE_CONFIRM_MOVE,
    TIMING_SIDE_PRECONF_SUPPRESS,
)

_THERMAL_GUARD = ThermalGuard.from_env()


def _thermal_wait(where: str) -> None:
    _THERMAL_GUARD.wait_until_safe(
        where=where,
        force_sample=False,
        logger=lambda msg: print(f"[labels-refresh] {msg}", flush=True),
    )


@dataclass
class RefreshLabelsSettings:
    # 0 => todos
    limit: int = 0
    # se não vazio, processa só estes
    symbols: list[str] | None = None
    # filtro por market cap (se symbols vazio)
    symbols_file: Path | None = None
    mcap_min_usd: float = 50_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    max_symbols: int = 0
    candle_sec: int = 60
    # overrides opcionais do timing label
    horizon_profit: int | None = None
    k_lookahead: int | None = None
    top_n: int | None = None
    alpha: float | None = None
    label_clip: float | None = None
    weight_label_mult: float | None = None
    weight_vol_mult: float | None = None
    weight_min: float | None = None
    weight_max: float | None = None
    vol_window: int | None = None
    # centraliza labels ao redor de 0 (reduz bias direcional)
    label_center: bool | None = None
    # usa movimento dominante (pos/neg) quando mais forte
    use_dominant: bool | None = None
    # mistura entre profit_now e dominante (0..1)
    dominant_mix: float | None = None
    # penalidades do label por lado (anti-entrada antecipada)
    side_mae_penalty: float | None = None
    side_time_penalty: float | None = None
    side_giveback_penalty: float | None = None
    side_cross_penalty: float | None = None
    side_rev_lookback_min: int | None = None
    side_chase_penalty: float | None = None
    side_reversal_bonus: float | None = None
    side_confirm_min: int | None = None
    side_confirm_move: float | None = None
    side_preconfirm_suppress: float | None = None
    # se True, imprime 1 linha por símbolo
    verbose: bool = True
    # workers para refresh paralelo (0 => auto)
    workers: int = 0
    # segurança de recursos (não aborta; reduz paralelismo)
    max_ram_pct: float = 85.0
    min_free_mb: float = 1024.0
    per_worker_mem_mb: float = 512.0


def _atomic_save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    if path.suffix.lower() == ".parquet":
        df.to_parquet(tmp, index=True)
    else:
        df.to_pickle(tmp)
    tmp.replace(path)


def _list_symbols_from_cache(cache_dir: Path, fmt: str) -> list[str]:
    suf = ".parquet" if fmt == "parquet" else ".pkl"
    out = []
    for p in sorted(cache_dir.glob(f"*{suf}")):
        sym = p.stem.upper()
        if sym:
            out.append(sym)
    return out


def _select_symbols_by_market_cap(s: RefreshLabelsSettings) -> list[str]:
    path = Path(s.symbols_file) if s.symbols_file is not None else default_top_market_cap_path()
    caps = load_market_caps(path)
    if not caps:
        return []
    lo = min(float(s.mcap_min_usd), float(s.mcap_max_usd))
    hi = max(float(s.mcap_min_usd), float(s.mcap_max_usd))
    ranked = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    for sym, cap in ranked:
        if cap < lo or cap > hi:
            continue
        s_sym = str(sym).upper()
        if not s_sym.endswith("USDT"):
            s_sym = s_sym + "USDT"
        out.append(s_sym)
        if int(s.max_symbols) > 0 and len(out) >= int(s.max_symbols):
            break
    return out


def run(settings: RefreshLabelsSettings | None = None) -> dict:
    """
    Recalcula labels supervisionados (timing_*, edge_* e entry_gate_*) no cache,
    sem recomputar features.
    """
    s = settings or RefreshLabelsSettings()
    try:
        env_max = os.getenv("SNIPER_REFRESH_MAX_SYMBOLS", "").strip()
        if env_max and int(s.max_symbols) <= 0:
            s.max_symbols = int(env_max)
    except Exception:
        pass
    asset_class = os.getenv("SNIPER_ASSET_CLASS", "").strip().lower()
    cache_dir = _cache_dir(asset_class or None)
    fmt = _cache_format()

    symbols = [x.strip().upper() for x in (s.symbols or []) if str(x).strip()]
    if not symbols:
        symbols = _list_symbols_from_cache(cache_dir, fmt)
        if not symbols:
            symbols = _select_symbols_by_market_cap(s)
    if int(s.max_symbols) > 0:
        symbols = symbols[: int(s.max_symbols)]
    if int(s.limit) > 0:
        symbols = symbols[: int(s.limit)]

    print(
        f"[labels-refresh] cache_dir={cache_dir} fmt={fmt} asset={asset_class or 'crypto'} symbols={len(symbols)} candle_sec={s.candle_sec} "
        f"mcap_min={float(s.mcap_min_usd):.0f} mcap_max={float(s.mcap_max_usd):.0f} max_symbols={int(s.max_symbols)}",
        flush=True,
    )
    print(
        "[labels-refresh] timing_label: "
        f"horizon_profit={s.horizon_profit or TIMING_HORIZON_PROFIT} "
        f"k_lookahead={s.k_lookahead or TIMING_K_LOOKAHEAD} "
        f"top_n={s.top_n or TIMING_TOP_N} "
        f"alpha={s.alpha or TIMING_ALPHA} "
        f"clip={s.label_clip or TIMING_LABEL_CLIP} "
        f"side_mae={s.side_mae_penalty if s.side_mae_penalty is not None else TIMING_SIDE_MAE_PENALTY} "
        f"side_time={s.side_time_penalty if s.side_time_penalty is not None else TIMING_SIDE_TIME_PENALTY} "
        f"side_giveback={s.side_giveback_penalty if s.side_giveback_penalty is not None else TIMING_SIDE_GIVEBACK_PENALTY} "
        f"side_cross={s.side_cross_penalty if s.side_cross_penalty is not None else TIMING_SIDE_CROSS_PENALTY} "
        f"side_rev_lb={s.side_rev_lookback_min if s.side_rev_lookback_min is not None else TIMING_SIDE_REV_LOOKBACK_MIN} "
        f"side_chase={s.side_chase_penalty if s.side_chase_penalty is not None else TIMING_SIDE_CHASE_PENALTY} "
        f"side_rev_bonus={s.side_reversal_bonus if s.side_reversal_bonus is not None else TIMING_SIDE_REVERSAL_BONUS} "
        f"side_confirm_min={s.side_confirm_min if s.side_confirm_min is not None else TIMING_SIDE_CONFIRM_MIN} "
        f"side_confirm_move={s.side_confirm_move if s.side_confirm_move is not None else TIMING_SIDE_CONFIRM_MOVE} "
        f"side_preconfirm_sup={s.side_preconfirm_suppress if s.side_preconfirm_suppress is not None else TIMING_SIDE_PRECONF_SUPPRESS} "
        f"center={s.label_center if s.label_center is not None else 'env'} "
        f"dominant={s.use_dominant if s.use_dominant is not None else 'env'} "
        f"dominant_mix={s.dominant_mix if s.dominant_mix is not None else 'env'}",
        flush=True,
    )

    t0 = time.perf_counter()
    ok = 0
    fail = 0
    rows_total = 0
    sum_nz_long = 0.0
    sum_nz_short = 0.0
    sum_overlap_20 = 0.0
    sum_mean_long = 0.0
    sum_mean_short = 0.0
    sum_edge_long = 0.0
    sum_edge_short = 0.0
    sum_gate_long = 0.0
    sum_gate_short = 0.0
    p95_long_vals: list[float] = []
    p95_short_vals: list[float] = []
    total = len(symbols)
    progress = ProgressPrinter(prefix="[labels-refresh]", total=total, stream=sys.stderr, print_every_s=5.0)
    workers = int(getattr(s, "workers", 0) or 0)
    if workers <= 0:
        workers = int(os.getenv("SNIPER_LABELS_REFRESH_WORKERS", "0") or "0")
    if workers <= 0:
        workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    workers = max(1, min(32, workers))
    policy = AdaptiveParallelPolicy(
        max_ram_pct=float(getattr(s, "max_ram_pct", 85.0)),
        min_free_mb=float(getattr(s, "min_free_mb", 1024.0)),
        per_worker_mem_mb=float(getattr(s, "per_worker_mem_mb", 512.0)),
        min_workers=1,
        poll_interval_s=0.5,
        log_every_s=10.0,
    )
    print(
        f"[labels-refresh] workers={workers} ram_cap={policy.max_ram_pct:.1f}% min_free_mb={policy.min_free_mb:.0f} per_worker_mb={policy.per_worker_mem_mb:.0f}",
        flush=True,
    )

    def _process_symbol(sym: str) -> dict:
        _thermal_wait(f"refresh_symbol:{sym}")
        data_path, meta_path = _symbol_cache_paths(sym, cache_dir, fmt)
        if not data_path.exists():
            return {"sym": sym, "ok": False, "skip": True}
        df = pd.read_parquet(data_path) if data_path.suffix.lower() == ".parquet" else pd.read_pickle(data_path)
        if df is None or df.empty:
            raise RuntimeError("df vazio")
        if "close" not in df.columns:
            raise RuntimeError("faltam colunas: close")

        # recalcula timing labels
        df_lab = df[["close"]].copy()
        apply_timing_regression_labels(
            df_lab,
            candle_sec=int(s.candle_sec),
            horizon_profit=s.horizon_profit,
            k_lookahead=s.k_lookahead,
            top_n=s.top_n,
            alpha=s.alpha,
            label_clip=s.label_clip,
            weight_label_mult=s.weight_label_mult,
            weight_vol_mult=s.weight_vol_mult,
            weight_min=s.weight_min,
            weight_max=s.weight_max,
            vol_window=s.vol_window,
            label_center=s.label_center,
            use_dominant=s.use_dominant,
            dominant_mix=s.dominant_mix,
            side_mae_penalty=s.side_mae_penalty,
            side_time_penalty=s.side_time_penalty,
            side_giveback_penalty=s.side_giveback_penalty,
            side_cross_penalty=s.side_cross_penalty,
            side_rev_lookback_min=s.side_rev_lookback_min,
            side_chase_penalty=s.side_chase_penalty,
            side_reversal_bonus=s.side_reversal_bonus,
            side_confirm_min=s.side_confirm_min,
            side_confirm_move=s.side_confirm_move,
            side_preconfirm_suppress=s.side_preconfirm_suppress,
        )
        for c in [
            "timing_label",
            "timing_label_pct",
            "timing_profit_now",
            "timing_profit_now_pct",
            "timing_weight",
            "timing_label_long",
            "timing_label_short",
            "timing_weight_long",
            "timing_weight_short",
            "edge_label_long",
            "edge_label_short",
            "edge_weight_long",
            "edge_weight_short",
            "entry_gate_long",
            "entry_gate_short",
            "entry_gate_weight",
            "entry_gate_weight_long",
            "entry_gate_weight_short",
        ]:
            if c in df_lab.columns:
                df[c] = df_lab[c]

        # diagnóstico agregado dos labels por lado (0..100)
        n = 0
        nz_l = nz_s = ov20 = m_l = m_s = p95_l = p95_s = float("nan")
        edge_l = edge_s = gate_l = gate_s = float("nan")
        try:
            y_long = df_lab["timing_label_long"].to_numpy(dtype=float, copy=False)
            y_short = df_lab["timing_label_short"].to_numpy(dtype=float, copy=False)
            n = int(min(y_long.size, y_short.size))
            if n > 0:
                yl = y_long[:n]
                ys = y_short[:n]
                nz_l = float((yl > 0.0).mean())
                nz_s = float((ys > 0.0).mean())
                ov20 = float(((yl >= 20.0) & (ys >= 20.0)).mean())
                m_l = float(np.mean(yl))
                m_s = float(np.mean(ys))
                p95_l = float(np.quantile(yl, 0.95))
                p95_s = float(np.quantile(ys, 0.95))
                if "edge_label_long" in df_lab.columns:
                    edge_l = float(np.nanmean(df_lab["edge_label_long"].to_numpy(dtype=float, copy=False)[:n]))
                if "edge_label_short" in df_lab.columns:
                    edge_s = float(np.nanmean(df_lab["edge_label_short"].to_numpy(dtype=float, copy=False)[:n]))
                if "entry_gate_long" in df_lab.columns:
                    gate_l = float(np.nanmean((df_lab["entry_gate_long"].to_numpy(dtype=float, copy=False)[:n]) >= 50.0))
                if "entry_gate_short" in df_lab.columns:
                    gate_s = float(np.nanmean((df_lab["entry_gate_short"].to_numpy(dtype=float, copy=False)[:n]) >= 50.0))
        except Exception:
            pass

        # remove colunas legadas (labels antigos)
        legacy_prefixes = ("sniper_long_", "sniper_short_", "sniper_exit_", "sniper_mae_", "sniper_entry_")
        legacy_cols = [c for c in df.columns if str(c).startswith(legacy_prefixes)]
        if legacy_cols:
            df.drop(columns=legacy_cols, inplace=True, errors="ignore")

        _atomic_save_df(df, data_path)
        try:
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = dict(meta or {})
            meta["labels_refreshed_utc"] = pd.Timestamp.utcnow().tz_localize(None).isoformat()
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return {
            "sym": sym,
            "ok": True,
            "skip": False,
            "n": n,
            "nz_l": nz_l,
            "nz_s": nz_s,
            "ov20": ov20,
            "m_l": m_l,
            "m_s": m_s,
            "p95_l": p95_l,
            "p95_s": p95_s,
            "edge_l": edge_l,
            "edge_s": edge_s,
            "gate_l": gate_l,
            "gate_s": gate_s,
        }

    done = 0
    for sym, fut in run_adaptive_thread_map(
        symbols,
        _process_symbol,
        max_workers=workers,
        policy=policy,
        task_name="labels-refresh",
    ):
        try:
            out = fut.result()
            if out.get("ok"):
                ok += 1
                n = int(out.get("n", 0) or 0)
                if n > 0:
                    def _sf(v: float) -> float:
                        try:
                            vv = float(v)
                            return vv if np.isfinite(vv) else 0.0
                        except Exception:
                            return 0.0
                    rows_total += n
                    sum_nz_long += _sf(out.get("nz_l", 0.0)) * n
                    sum_nz_short += _sf(out.get("nz_s", 0.0)) * n
                    sum_overlap_20 += _sf(out.get("ov20", 0.0)) * n
                    sum_mean_long += _sf(out.get("m_l", 0.0)) * n
                    sum_mean_short += _sf(out.get("m_s", 0.0)) * n
                    sum_edge_long += _sf(out.get("edge_l", 0.0)) * n
                    sum_edge_short += _sf(out.get("edge_s", 0.0)) * n
                    sum_gate_long += _sf(out.get("gate_l", 0.0)) * n
                    sum_gate_short += _sf(out.get("gate_s", 0.0)) * n
                    p95_long_vals.append(float(out.get("p95_l", float("nan"))))
                    p95_short_vals.append(float(out.get("p95_s", float("nan"))))
        except Exception as e:
            fail += 1
            print(f"[labels-refresh] FAIL {sym}: {type(e).__name__}: {e}", flush=True)
        done += 1
        if s.verbose:
            progress.update(done, suffix=sym)

    dt = time.perf_counter() - t0
    if s.verbose:
        progress.close()
    if rows_total > 0:
        nz_long = sum_nz_long / float(rows_total)
        nz_short = sum_nz_short / float(rows_total)
        overlap_20 = sum_overlap_20 / float(rows_total)
        mean_long = sum_mean_long / float(rows_total)
        mean_short = sum_mean_short / float(rows_total)
        mean_edge_long = sum_edge_long / float(rows_total)
        mean_edge_short = sum_edge_short / float(rows_total)
        gate_long_rate = sum_gate_long / float(rows_total)
        gate_short_rate = sum_gate_short / float(rows_total)
        p95_long_med = float(np.median(np.asarray(p95_long_vals, dtype=float))) if p95_long_vals else float("nan")
        p95_short_med = float(np.median(np.asarray(p95_short_vals, dtype=float))) if p95_short_vals else float("nan")
        print(
            "[labels-refresh] diag: "
            f"rows={rows_total} "
            f"nz_long={nz_long:.3f} nz_short={nz_short:.3f} "
            f"overlap20={overlap_20:.3f} "
            f"mean_long={mean_long:.3f} mean_short={mean_short:.3f} "
            f"edge_long={mean_edge_long:.3f} edge_short={mean_edge_short:.3f} "
            f"gate_long@50={gate_long_rate:.3f} gate_short@50={gate_short_rate:.3f} "
            f"p95_long_med={p95_long_med:.2f} p95_short_med={p95_short_med:.2f}",
            flush=True,
        )
    print(f"[labels-refresh] done ok={ok} fail={fail} sec={dt:.2f}", flush=True)
    return {
        "ok": int(ok),
        "fail": int(fail),
        "total": int(total),
        "seconds": float(dt),
        "cache_dir": str(cache_dir),
        "format": str(fmt),
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
