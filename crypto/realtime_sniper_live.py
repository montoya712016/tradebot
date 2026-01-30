# -*- coding: utf-8 -*-
"""
Shim para compatibilidade: importa de realtime.bot.sniper.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que o root do projeto esteja no path
def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_paths()

from realtime.bot.sniper import LiveDecisionBot, main
from realtime.bot.settings import LiveSettings

if __name__ == "__main__":
    main()
    