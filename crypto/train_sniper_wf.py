# -*- coding: utf-8 -*-
from __future__ import annotations

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

from train.train_sniper_wf import TrainSniperWFSettings, run  # type: ignore


def main() -> None:
    settings = TrainSniperWFSettings(asset_class="crypto")
    run_dir = run(settings)
    print(f"[train-wf-crypto] run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

