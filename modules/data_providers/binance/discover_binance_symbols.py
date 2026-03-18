# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Descoberta de símbolos (Top-N por market cap) que têm par USDT spot na Binance.

Regras do repositório:
- parâmetros definidos em código (sem depender de ENV)
- salva saída em `data/generated/` para não poluir a raiz
"""

from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import json
import time

import requests

# permitir rodar como script direto (sem PYTHONPATH)
if __package__ in (None, ""):
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break

from utils.paths import resolve_generated_path

try:
    from rich.console import Console
    from rich.table import Table
except Exception:
    Console = None  # type: ignore
    Table = None  # type: ignore


COINGECKO_API = "https://api.coingecko.com/api/v3"


@dataclass
class DiscoverSettings:
    limit: int = 300  # <= 1000 recomendado
    quote: str = "USDT"
    # Exclusões básicas (stablecoins e wrappers frequentes)
    exclude_bases: set[str] = None  # type: ignore[assignment]
    # Mapeamentos conhecidos de símbolo CGK -> Binance base (para divergências)
    overrides: dict[str, str] = None  # type: ignore[assignment]
    # Saída
    save_outputs: bool = True
    out_dir: str | None = None  # None => repo_root/data/generated/

    def __post_init__(self) -> None:
        if self.exclude_bases is None:
            self.exclude_bases = {
                "USDT",
                "USDC",
                "FDUSD",
                "TUSD",
                "BUSD",
                "DAI",
                "UST",
                "USTC",
                "USD",
                "USD1",
                "WBTC",
                "BNSOL",
                "PAXG",
                "GUSD",
                "USDD",
                "SUSD",
                "LUSD",
                "EUR",
                "GBP",
            }
        if self.overrides is None:
            self.overrides = {
                "RNDR": "RENDER",
                "MIOTA": "IOTA",
            }


def _default_out_dir() -> Path:
    return resolve_generated_path(".")


def _console() -> Console | None:
    try:
        return Console() if Console is not None else None
    except Exception:
        return None


def fetch_coingecko_ids_by_symbol() -> Dict[str, List[str]]:
    """Retorna {symbol_lower: [ids...]} usando /coins/list."""
    url = f"{COINGECKO_API}/coins/list"
    r = requests.get(url, params={"include_platform": "false"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: Dict[str, List[str]] = {}
    for row in data:
        sym = str(row.get("symbol", "")).lower()
        cid = str(row.get("id", ""))
        if not sym or not cid:
            continue
        out.setdefault(sym, []).append(cid)
    return out


def fetch_market_caps_by_ids(ids: List[str]) -> Dict[str, Dict[str, object]]:
    """Consulta /coins/markets por batches de ids.
    Retorna {id: {"symbol": sym, "name": name, "market_cap": mc}}.
    """
    out: Dict[str, Dict[str, object]] = {}
    if not ids:
        return out
    max_page_size = 250  # lim CoinGecko
    for i in range(0, len(ids), max_page_size):
        batch = ids[i : i + max_page_size]
        url = f"{COINGECKO_API}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(batch),
            "order": "market_cap_desc",
            "per_page": len(batch),
            "page": 1,
            "price_change_percentage": "",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for d in data:
            cid = str(d.get("id", ""))
            out[cid] = {
                "symbol": str(d.get("symbol", "")).upper(),
                "name": str(d.get("name", "")),
                "market_cap": float(d.get("market_cap") or 0.0),
            }
        time.sleep(0.2)
    return out


def get_binance_quote_bases(quote: str) -> Dict[str, str]:
    """Retorna {baseAsset: symbol} somente para pares SPOT/QUOTE ativos."""
    r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    info = r.json()
    bases: Dict[str, str] = {}
    for s in info.get("symbols", []):
        try:
            if s.get("status") != "TRADING":
                continue
            if not s.get("isSpotTradingAllowed", True):
                continue
            if s.get("quoteAsset") != quote:
                continue
            base = str(s.get("baseAsset", "")).upper()
            bases[base] = str(s.get("symbol"))
        except Exception:
            continue
    return bases


def discover_on_binance_with_market_caps(settings: DiscoverSettings) -> Tuple[List[Tuple[str, float]], List[str]]:
    """Cruza: bases Binance (QUOTE) vs market cap CoinGecko.
    Retorna [(SYMBOLQUOTE, market_cap)], ordenado desc. Ignorados retornam em skipped.
    """
    quote = settings.quote.strip().upper()
    bin_bases = get_binance_quote_bases(quote)  # {BASE: BASEQUOTE}
    bases = [b for b in bin_bases.keys() if b not in settings.exclude_bases]
    bases = sorted(set(bases))

    ids_by_sym = fetch_coingecko_ids_by_symbol()

    cand_ids: List[str] = []
    map_sym_to_ids: Dict[str, List[str]] = {}
    for base in bases:
        sym_lo = base.lower()
        ids = ids_by_sym.get(sym_lo, [])
        if not ids:
            continue
        map_sym_to_ids[base] = ids
        cand_ids.extend(ids)

    markets = fetch_market_caps_by_ids(list(dict.fromkeys(cand_ids)))

    out_pairs: List[Tuple[str, float]] = []
    skipped: List[str] = []
    for base in bases:
        ids = map_sym_to_ids.get(base)
        if not ids:
            skipped.append(base)
            continue
        best_mc = 0.0
        for cid in ids:
            info = markets.get(cid)
            if not info:
                continue
            mc = float(info.get("market_cap", 0.0))
            if mc > best_mc:
                best_mc = mc
        if best_mc <= 0.0:
            skipped.append(base)
            continue
        out_pairs.append((bin_bases[base], best_mc))

    out_pairs.sort(key=lambda t: t[1], reverse=True)
    if settings.limit and settings.limit > 0:
        out_pairs = out_pairs[: int(settings.limit)]
    return out_pairs, skipped


def run(settings: DiscoverSettings | None = None) -> list[str]:
    settings = settings or DiscoverSettings()
    con = _console()

    if con:
        con.print(f"[bold]Descobrindo Top-{settings.limit} (Binance {settings.quote}) e consultando market caps (CoinGecko)...[/]")

    ranked, skipped = discover_on_binance_with_market_caps(settings)
    syms = [s for (s, _) in ranked]

    if con:
        con.print(f"[green]Encontrados na Binance:[/] {len(syms)} | [yellow]Sem market cap mapeado:[/] {len(skipped)}")

    # imprime um array Python copiável e um top com market caps
    arr = "SYMBOLS_TOP_BINANCE_USDT = " + json.dumps(syms, ensure_ascii=False, indent=2)
    print(arr)
    print("\n# Top por market cap (USD):")
    for s, mc in ranked[: min(1000, len(ranked))]:
        print(f"{s}: {int(mc):,}".replace(",", "."))

    if settings.save_outputs:
        out_dir = resolve_generated_path(Path(settings.out_dir).expanduser()) if settings.out_dir else _default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        (out_dir / "symbols_top_binance_usdt.json").write_text(json.dumps(syms, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "symbols_top_binance_usdt.py").write_text("SYMBOLS_TOP_BINANCE_USDT = " + json.dumps(syms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"symbols_top_binance_usdt_{stamp}.json").write_text(json.dumps(syms, ensure_ascii=False, indent=2), encoding="utf-8")

    return syms


def main() -> None:
    # Edite os parâmetros aqui:
    settings = DiscoverSettings(limit=500, quote="USDT", save_outputs=True)
    run(settings)


if __name__ == "__main__":
    main()

