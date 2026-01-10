# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Leitores/normalizadores de listas de símbolos.

Fonte atual:
- `top_market_cap.txt` (formato: SYMBOLUSDT: 123456789)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    # modules/config/symbols.py -> config -> modules
    return Path(__file__).resolve().parents[1]


def default_top_market_cap_path() -> Path:
    # compat: aceita tanto na raiz quanto em data/
    root = _repo_root()
    candidates = [
        root / "top_market_cap.txt",
        root / "data" / "top_market_cap.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: primeiro candidato
    return candidates[0]


def load_market_caps(path: str | Path | None = None) -> dict[str, float]:
    p = Path(path) if path is not None else default_top_market_cap_path()
    if not p.exists():
        return {}
    out: dict[str, float] = {}
    txt = p.read_text(encoding="utf-8")
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        sym, val = line.split(":", 1)
        sym = sym.strip().upper()
        v = val.strip().replace(".", "").replace(",", "")
        try:
            out[sym] = float(int(v))
        except Exception:
            continue
    return out


def load_top_market_cap_symbols(
    *,
    path: str | Path | None = None,
    limit: int | None = None,
    ensure_usdt: bool = True,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    caps = load_market_caps(path)
    if not caps:
        return []
    exclude_set = {str(s).upper() for s in (exclude or [])}
    ranked = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    for sym, _cap in ranked:
        s = str(sym).upper()
        if ensure_usdt and (not s.endswith("USDT")):
            s = s + "USDT"
        if s in exclude_set:
            continue
        out.append(s)
        if limit is not None and len(out) >= int(limit):
            break
    return out

