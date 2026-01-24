# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import os
import sys


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def main() -> None:
    _add_repo_paths()
    os.environ.setdefault("SNIPER_ASSET_CLASS", "stocks")
    from prepare_features.refresh_sniper_labels_in_cache import main as refresh_main  # type: ignore

    refresh_main()


if __name__ == "__main__":
    main()
