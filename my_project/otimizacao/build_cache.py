# -*- coding: utf-8 -*-
"""
Utilitário simples para (re)gerar os caches de previsões usados pelo GA e
pelos testes em lote (teste_multiplo.py).

Depois de treinar novos modelos, rode:

    python my_project/otimizacao/build_cache.py --run wf_001

Isso recalcula as previsões de todos os símbolos usados no treinamento e grava
os parquet em `my_project/otimizacao/cache/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ga import _predict_symbol_once, SAVE_DIR

try:
    from ..test.load_models import choose_run, preload_boosters
except Exception:
    from my_project.test.load_models import choose_run, preload_boosters


TOP_FILE = Path(__file__).resolve().parents[1] / "top_market_cap.txt"


def _load_symbols(run_dir: Path) -> list[str]:
    sym_map_path = run_dir / "dataset" / "sym_map.json"
    if sym_map_path.exists():
        try:
            data = json.loads(sym_map_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(s).upper() for s in data]
        except Exception:
            pass
    if TOP_FILE.exists():
        cap_map: dict[str, float] = {}
        for line in TOP_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            sym, val = line.split(":", 1)
            sym = sym.strip().upper()
            if not sym.endswith("USDT"):
                sym = sym + "USDT"
            try:
                cap_map[sym] = float(int(val.strip().replace(".", "").replace(",", "")))
            except Exception:
                continue
        return sorted(cap_map.keys())
    raise RuntimeError("Não foi possível descobrir a lista de símbolos do treinamento.")


def build_cache(
    *,
    run_hint: str | None = None,
    days: int = 360,
    skip_days: int = 360,
    symbols: list[str] | None = None,
    force_recalc: bool = False,
) -> None:
    run_dir, periods = choose_run(SAVE_DIR, run_hint)
    periods_use = sorted(int(p) for p in periods if int(p) > 0)
    if not periods_use:
        raise RuntimeError("Nenhum período válido encontrado para o run escolhido.")

    print(f"[cache-builder] run_dir={run_dir} | períodos={periods_use}")

    preload_boosters(run_dir, periods_use, print_timing=True)

    if not symbols:
        symbols = _load_symbols(run_dir)
    symbols = [s.upper() for s in symbols]
    print(f"[cache-builder] {len(symbols)} símbolos para processar...")

    ok = 0
    for sym in symbols:
        try:
            df = _predict_symbol_once(
                sym,
                run_dir,
                periods_use,
                int(days),
                skip_days=int(skip_days),
                require_cache=False,
                bypass_cache_on_recalc=bool(force_recalc),
            )
            if df is not None and not df.empty:
                ok += 1
                print(f"[cache-builder] {sym}: {len(df)} linhas (cache atualizado)")
        except Exception as exc:
            print(f"[cache-builder] {sym}: falhou ({exc})")
    print(f"[cache-builder] concluído: {ok}/{len(symbols)} símbolos com cache válido.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regera caches de previsões usados pelo GA/teste_multiplo.")
    parser.add_argument("--run", dest="run_hint", type=str, default=None, help="Nome do run (ex: wf_001).")
    parser.add_argument("--days", type=int, default=360, help="Janela de dias usada pelo GA/backtest.")
    parser.add_argument("--skip-days", type=int, default=360, help="Dias removidos do final (OOS).")
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Lista específica de símbolos.")
    parser.add_argument("--force", action="store_true", help="Recalcula mesmo que o cache já exista.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(
        run_hint=args.run_hint,
        days=args.days,
        skip_days=args.skip_days,
        symbols=args.symbols,
        force_recalc=bool(args.force),
    )

