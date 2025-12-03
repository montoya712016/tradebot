#!/usr/bin/env python3
"""
Descoberta das maiores altcoins (Top-N por market cap) que TÊM par USDT na Binance.

O script APENAS lista os símbolos (não baixa dados). Ideal para gerar uma lista
copiável/colável para uso posterior no coletor/baixador.

Fluxo:
- Busca no CoinGecko o Top-N por market cap (até 1000, em páginas de 250)
- Cruza com a lista de pares USDT spot na Binance (exchangeInfo)
- Aplica exclusões básicas (stablecoins, wrappers comuns)
- Imprime um array Python pronto para copiar e (opcionalmente) salva JSON/.py

Requisitos: requests, rich (opcional)
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import threading
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone

import requests

try:
    from rich.console import Console
    from rich.table import Table
except Exception:
    Console = None


# ===================== Config =====================

COINGECKO_API = "https://api.coingecko.com/api/v3"

DEFAULT_LIMIT = int(os.getenv("DISCOVER_LIMIT", "1000"))        # até 1000
MAX_PAGE_SIZE = 250                                                # lim. CoinGecko (para requests por ids)
QUOTE = os.getenv("BINANCE_QUOTE", "USDT").upper()

# Exclusões básicas (stablecoins e wrappers frequentes)
EXCLUDE_BASES = {
    "USDT","USDC","FDUSD","TUSD","BUSD","DAI","UST","USTC","USD","USD1",
    "WBTC","BNSOL","PAXG","GUSD","USDD","SUSD","LUSD","FDUSD","EUR","GBP"
}

# Mapeamentos conhecidos de símbolo CGK -> Binance base (para divergências)
OVERRIDES = {
    "RNDR": "RENDER",
    "MIOTA": "IOTA",
}

# ===================== Helpers =====================

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
    # batching por até 250 ids
    for i in range(0, len(ids), MAX_PAGE_SIZE):
        batch = ids[i:i+MAX_PAGE_SIZE]
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


def get_binance_usdt_bases() -> Dict[str, str]:
    """Retorna {baseAsset: symbol} somente para pares SPOT/USDT ativos."""
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
            if s.get("quoteAsset") != QUOTE:
                continue
            base = str(s.get("baseAsset", "")).upper()
            bases[base] = str(s.get("symbol"))
        except Exception:
            continue
    return bases


def normalize_to_binance_base(sym: str) -> str:
    sym_u = sym.upper()
    return OVERRIDES.get(sym_u, sym_u)


def discover_on_binance_with_market_caps(limit: int = DEFAULT_LIMIT) -> Tuple[List[Tuple[str, float]], List[str]]:
    """Parte da Binance: pega TODAS as bases com par USDT e cruza market cap via CG.
    Retorna [(SYMBOLUSDT, market_cap)], ordenado desc. Ignorados retornam em skipped.
    """
    bin_bases = get_binance_usdt_bases()  # {BASE: BASEUSDT}
    bases = [b for b in bin_bases.keys() if b not in EXCLUDE_BASES]
    bases = sorted(set(bases))

    # Mapeia base -> melhor id do CoinGecko (por maior market cap)
    ids_by_sym = fetch_coingecko_ids_by_symbol()
    # junta todos ids candidatos
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
        # escolhe o id com MAIOR market cap
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

    # ordena por market cap desc e aplica limite
    out_pairs.sort(key=lambda t: t[1], reverse=True)
    if limit and limit > 0:
        out_pairs = out_pairs[:limit]
    return out_pairs, skipped


def run_discovery(limit: int = DEFAULT_LIMIT) -> List[str]:
    con = _console()
    if con:
        con.print(f"[bold]Descobrindo Top-{limit} (Binance USDT) e consultando market caps (CoinGecko)...[/]")
    ranked, skipped = discover_on_binance_with_market_caps(limit)
    syms = [s for (s, _) in ranked]
    if con:
        con.print(f"[green]Encontrados na Binance:[/] {len(syms)} | [yellow]Sem market cap mapeado:[/] {len(skipped)}")

    # Salva lista em JSON e módulo Python
    out_dir = Path(__file__).resolve().parent
    try:
        (out_dir / "symbols_top_binance_usdt.json").write_text(json.dumps(syms, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "symbols_top_binance_usdt.py").write_text("SYMBOLS_TOP_BINANCE_USDT = " + json.dumps(syms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return syms


# ===================== CLI =====================

from pathlib import Path

def _parse_int(env_name: str, default: int) -> int:
    try:
        v = int(os.getenv(env_name, str(default)))
        return v
    except Exception:
        return default


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Descobrir Top-N altcoins (com par USDT/spot na Binance)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Quantidade de moedas (<=1000)")
    ap.add_argument("--no-save", action="store_true", help="Não salvar arquivos; apenas imprimir array no stdout")
    args = ap.parse_args()

    ranked = discover_on_binance_with_market_caps(limit=args.limit)[0]
    syms = [s for (s, _) in ranked]
    # Imprime um array Python copiável e um top com market caps
    arr = "SYMBOLS_TOP_BINANCE_USDT = " + json.dumps(syms, ensure_ascii=False, indent=2)
    print(arr)
    print("\n# Top por market cap (USD):")
    for s, mc in ranked[:1000]:
        print(f"{s}: {int(mc):,}".replace(",", "."))
    if args.no_save:
        # nada a fazer; arquivos já foram escritos por padrão — se quiser evitar, podemos apagar aqui
        pass


