#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))

_add_repo_paths()

try:
    from realtime.dashboard_server import create_app
except ImportError:
    # fallback se modules/ estiver em outro lugar
    from modules.realtime.dashboard_server import create_app

def main() -> None:
    ap = argparse.ArgumentParser(description="Dashboard em tempo real (Flask).")
    ap.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "5000")))
    ap.add_argument("--demo", action="store_true", help="Habilita modo demo (dados fake + animação).")
    ap.add_argument("--no-demo", dest="demo", action="store_false", help="Desabilita modo demo.")
    ap.set_defaults(demo=True)
    ap.add_argument("--refresh-sec", type=float, default=float(os.getenv("DASHBOARD_REFRESH_SEC", "2.0")))
    args = ap.parse_args()

    app, _store = create_app(demo=bool(args.demo), refresh_sec=float(args.refresh_sec))
    print(f"[OK] Dashboard online em http://{args.host}:{args.port}")
    app.run(host=args.host, port=int(args.port), debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
