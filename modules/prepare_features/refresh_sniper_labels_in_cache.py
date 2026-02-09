# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Recalcula APENAS os labels de regressão (timing_label/weight) dentro do cache de features,
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
    TIMING_SIDE_CROSS_PENALTY,
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
    side_cross_penalty: float | None = None
    # se True, imprime 1 linha por símbolo
    verbose: bool = True


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
    Recalcula labels de regressão (timing_*) no cache, sem recomputar features.
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
        f"side_cross={s.side_cross_penalty if s.side_cross_penalty is not None else TIMING_SIDE_CROSS_PENALTY} "
        f"center={s.label_center if s.label_center is not None else 'env'} "
        f"dominant={s.use_dominant if s.use_dominant is not None else 'env'} "
        f"dominant_mix={s.dominant_mix if s.dominant_mix is not None else 'env'}",
        flush=True,
    )

    t0 = time.perf_counter()
    ok = 0
    fail = 0
    total = len(symbols)
    progress = ProgressPrinter(prefix="[labels-refresh]", total=total, stream=sys.stderr, print_every_s=5.0)
    for i, sym in enumerate(symbols, start=1):
        if s.verbose:
            progress.update(ok + fail, suffix=sym)
        data_path, meta_path = _symbol_cache_paths(sym, cache_dir, fmt)
        if not data_path.exists():
            continue
        try:
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
                side_cross_penalty=s.side_cross_penalty,
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
            ]:
                if c in df_lab.columns:
                    df[c] = df_lab[c]

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

            ok += 1
            if s.verbose:
                progress.update(ok + fail, suffix=sym)
        except Exception as e:
            fail += 1
            print(f"[labels-refresh] FAIL {sym}: {type(e).__name__}: {e}", flush=True)

    dt = time.perf_counter() - t0
    if s.verbose:
        progress.close()
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
