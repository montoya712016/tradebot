# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Recalcula APENAS os labels sniper_* dentro do cache de features (parquet/pickle),
sem recomputar features.

Útil quando ajustamos a lógica de Danger/Entry/Exit em `prepare_features/labels.py`
ou os parâmetros do `TradeContract` (ex.: entry_label_windows_minutes / exit_ema_span).

Uso:
python modules/prepare_features/refresh_sniper_labels_in_cache.py

Observação:
- Isso reescreve os arquivos do cache (operação atômica: escreve tmp e renomeia).
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

_asset = os.getenv("SNIPER_ASSET_CLASS", "").strip().lower()
if _asset == "stocks":
    from stocks.trade_contract import (  # noqa: E402
        DEFAULT_TRADE_CONTRACT,
        TradeContract,
        exit_ema_span_from_window,
    )
else:
    from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract, exit_ema_span_from_window  # noqa: E402

# Contrato padrão (crypto) alinhado com o treino atual
ENTRY_LABEL_WINDOWS_MIN = (60,)
ENTRY_LABEL_MIN_PROFIT_PCTS = (0.03,)
ENTRY_LABEL_MAX_DD_PCTS = (0.03,)
ENTRY_LABEL_WEIGHT_ALPHA = 0.5
EXIT_EMA_INIT_OFFSET_PCT = 0.002

if _asset != "stocks":
    DEFAULT_TRADE_CONTRACT = TradeContract(
        entry_label_windows_minutes=ENTRY_LABEL_WINDOWS_MIN,
        entry_label_min_profit_pcts=ENTRY_LABEL_MIN_PROFIT_PCTS,
        entry_label_max_dd_pcts=ENTRY_LABEL_MAX_DD_PCTS,
        entry_label_weight_alpha=ENTRY_LABEL_WEIGHT_ALPHA,
        exit_ema_init_offset_pct=EXIT_EMA_INIT_OFFSET_PCT,
    )
from config.symbols import default_top_market_cap_path, load_market_caps  # noqa: E402
from utils.progress import ProgressPrinter  # noqa: E402
from train.sniper_dataflow import _cache_dir, _cache_format, _symbol_cache_paths  # type: ignore  # noqa: E402
from prepare_features.labels import apply_trade_contract_labels  # noqa: E402


@dataclass
class RefreshLabelsSettings:
    # 0 => todos
    limit: int = 0
    # se não vazio, processa só estes
    symbols: list[str] | None = None
    # filtro por market cap (se symbols vazio)
    symbols_file: Path | None = None
    mcap_min_usd: float = 100_000_000.0
    mcap_max_usd: float = 150_000_000_000.0
    max_symbols: int = 0
    candle_sec: int = 60
    contract: TradeContract = DEFAULT_TRADE_CONTRACT
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
    Recalcula labels `sniper_*` no cache, sem recomputar features.
    Retorna métricas básicas para logging/integração.
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
        # crypto: prefer cache list (reflete exatamente o que já foi gerado)
        if asset_class == "stocks":
            symbols = _list_symbols_from_cache(cache_dir, fmt)
        else:
            symbols = _list_symbols_from_cache(cache_dir, fmt)
            if not symbols:
                symbols = _select_symbols_by_market_cap(s)
    if int(s.limit) > 0:
        symbols = symbols[: int(s.limit)]

    print(
        f"[labels-refresh] cache_dir={cache_dir} fmt={fmt} asset={asset_class or 'crypto'} symbols={len(symbols)} candle_sec={s.candle_sec} "
        f"mcap_min={float(s.mcap_min_usd):.0f} mcap_max={float(s.mcap_max_usd):.0f} max_symbols={int(s.max_symbols)}",
        flush=True,
    )
    eff_ema_span = exit_ema_span_from_window(s.contract, int(s.candle_sec))
    print(
        "[labels-refresh] contract: "
        f"entry_label_windows_minutes={s.contract.entry_label_windows_minutes} "
        f"entry_label_min_profit_pcts={s.contract.entry_label_min_profit_pcts} "
        f"exit_ema_span={eff_ema_span} "
        f"exit_ema_init_offset_pct={s.contract.exit_ema_init_offset_pct}",
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
            need = {"close", "high", "low"}
            if not need.issubset(df.columns):
                raise RuntimeError(f"faltam colunas: {sorted(need - set(df.columns))}")

            # recalcula labels (somente sniper_*)
            df_lab = apply_trade_contract_labels(df[["close", "high", "low"]].copy(), contract=s.contract, candle_sec=int(s.candle_sec))
            cols = [
                "sniper_long_label",
                "sniper_long_weight",
                "sniper_mae_pct",
                "sniper_long_ret_pct",
                "sniper_short_label",
                "sniper_short_weight",
                "sniper_short_ret_pct",
                "sniper_exit_code",
                "sniper_exit_wait_bars",
            ]
            windows = list(getattr(s.contract, "entry_label_windows_minutes", []) or [])
            for w in windows:
                suf = f"{int(w)}m"
                cols.extend(
                    [
                        f"sniper_long_label_{suf}",
                        f"sniper_long_weight_{suf}",
                        f"sniper_mae_pct_{suf}",
                        f"sniper_long_ret_pct_{suf}",
                        f"sniper_short_label_{suf}",
                        f"sniper_short_weight_{suf}",
                        f"sniper_mae_pct_short_{suf}",
                        f"sniper_short_ret_pct_{suf}",
                        f"sniper_exit_code_{suf}",
                        f"sniper_exit_wait_bars_{suf}",
                    ]
                )
            for c in cols:
                if c in df_lab.columns:
                    df[c] = df_lab[c]
            # remove colunas antigas/legadas
            old_cols = [c for c in df.columns if str(c).startswith("sniper_entry_")]
            old_cols.extend([c for c in df.columns if str(c).endswith("_short") and str(c).startswith("sniper_")])
            if old_cols:
                df.drop(columns=old_cols, inplace=True, errors="ignore")

            _atomic_save_df(df, data_path)
            # atualiza meta com um carimbo (opcional)
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

