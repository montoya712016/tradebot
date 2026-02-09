# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shim para executar o pipeline híbrido WF (supervisionado + RL) via pasta `crypto/`.
"""

from pathlib import Path
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


_add_repo_paths()

from rl.run_hybrid_wf_pipeline import main  # type: ignore


if __name__ == "__main__":
    main()

