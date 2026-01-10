# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Recalcula APENAS os labels sniper_* dentro do cache de features (parquet/pickle),
sem recomputar features.

Útil quando ajustamos a lógica de Danger/Entry/Exit em `prepare_features/labels.py`
ou os parâmetros do `TradeContract` (ex.: danger_drop_pct / danger_timeout_hours).

Uso:
python modules/prepare_features/refresh_sniper_labels_in_cache.py

Observação:
- Isso reescreve os arquivos do cache (operação atômica: escreve tmp e renomeia).
"""

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time
import uuid

import numpy as np
import pandas as pd

# permitir rodar como script direto (sem PYTHONPATH)
_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if _p.name.lower() == "modules":
        _sp = str(_p)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
        break

from trade_contract import DEFAULT_TRADE_CONTRACT, TradeContract  # noqa: E402
from train.sniper_dataflow import _cache_dir, _cache_format, _symbol_cache_paths  # type: ignore  # noqa: E402
from prepare_features.labels import apply_trade_contract_labels  # noqa: E402


@dataclass
class RefreshLabelsSettings:
    # 0 => todos
    limit: int = 0
    # se não vazio, processa só estes
    symbols: list[str] | None = None
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


def run(settings: RefreshLabelsSettings | None = None) -> dict:
    """
    Recalcula labels `sniper_*` no cache, sem recomputar features.
    Retorna métricas básicas para logging/integração.
    """
    s = settings or RefreshLabelsSettings()
    cache_dir = _cache_dir()
    fmt = _cache_format()

    symbols = [x.strip().upper() for x in (s.symbols or []) if str(x).strip()]
    if not symbols:
        symbols = _list_symbols_from_cache(cache_dir, fmt)
    if int(s.limit) > 0:
        symbols = symbols[: int(s.limit)]

    print(f"[labels-refresh] cache_dir={cache_dir} fmt={fmt} symbols={len(symbols)} candle_sec={s.candle_sec}", flush=True)
    print(
        f"[labels-refresh] contract: danger_drop_pct={s.contract.danger_drop_pct} danger_timeout_hours={s.contract.danger_timeout_hours} "
        f"danger_recovery_pct={s.contract.danger_recovery_pct}",
        flush=True,
    )

    t0 = time.perf_counter()
    ok = 0
    fail = 0
    total = len(symbols)
    for i, sym in enumerate(symbols, start=1):
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
            for c in [
                "sniper_entry_label",
                "sniper_mae_pct",
                "sniper_exit_code",
                "sniper_exit_wait_bars",
                "sniper_danger_label",
                "sniper_mfe_safe_pct",
                "sniper_mfe_safe_wait_bars",
            ]:
                if c in df_lab.columns:
                    df[c] = df_lab[c]

            _atomic_save_df(df, data_path)
            # atualiza meta com um carimbo (opcional)
            try:
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = dict(meta or {})
                meta["labels_refreshed_utc"] = pd.Timestamp.utcnow().tz_localize(None).isoformat()
                meta["danger_drop_pct"] = float(s.contract.danger_drop_pct)
                meta["danger_timeout_hours"] = float(s.contract.danger_timeout_hours)
                meta["danger_recovery_pct"] = float(s.contract.danger_recovery_pct)
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

            ok += 1
            if s.verbose:
                rate = float(np.mean(df["sniper_danger_label"].to_numpy(dtype=np.float64))) if "sniper_danger_label" in df.columns else float("nan")
                print(f"[labels-refresh] {i}/{total} OK {sym} danger_rate={rate:.4f}", flush=True)
        except Exception as e:
            fail += 1
            print(f"[labels-refresh] {i}/{total} FAIL {sym}: {type(e).__name__}: {e}", flush=True)

    dt = time.perf_counter() - t0
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

