# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Paths do projeto.

Layout:
repo_root/
  modules/

Artefatos grandes (ex.: cache_sniper/) ficam fora do repo.
"""

import os
from pathlib import Path
from typing import Union


def modules_root() -> Path:
    # modules/utils/paths.py -> utils -> modules
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    # modules/utils/paths.py -> utils -> modules -> repo_root
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    # workspace_root contem models_sniper/ e cache_sniper/
    return repo_root().parent


def models_root() -> Path:
    return workspace_root() / "models_sniper"


def storage_root() -> Path:
    """
    Pasta externa para artefatos grandes (cache, etc.) fora do repo/pasta tradebot.

    Prioridade:
    - `TRADEBOT_STORAGE_ROOT` (ou `MY_PROJECT_STORAGE_ROOT`) se definido
    - Windows: `%LOCALAPPDATA%\\tradebot`
    - Outros: `~/.tradebot`
    """
    v = (os.getenv("TRADEBOT_STORAGE_ROOT") or os.getenv("MY_PROJECT_STORAGE_ROOT") or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return (base / "tradebot").resolve()
    return (Path.home() / ".tradebot").resolve()


def cache_sniper_root() -> Path:
    # por padrao, cache fica no workspace (pai do repo) para nao entrar no git
    return workspace_root() / "cache_sniper"


def generated_root() -> Path:
    """
    Pasta para outputs/artefatos gerados por backtests/GA/analises.
    """
    v = (os.getenv("TRADEBOT_GENERATED_ROOT") or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    return (repo_root() / "data" / "generated").resolve()


def feature_cache_root() -> Path:
    # Cache grande (varios anos x varias criptos) -> fora do tradebot por padrao
    v = (os.getenv("SNIPER_FEATURE_CACHE_DIR") or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    return cache_sniper_root() / "features_pf_1m"


def ohlc_cache_root() -> Path:
    return cache_sniper_root() / "ohlc_1m"


PathLike = Union[str, Path]


def resolve_repo_path(p: PathLike) -> Path:
    """
    Resolve caminhos relativos usando a raiz do repo (onde fica `modules/`).
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root() / pp).resolve()


def resolve_workspace_path(p: PathLike) -> Path:
    """
    Resolve caminhos relativos usando a raiz do workspace (pai do repo_root).
    Util para artefatos grandes: models_sniper/, cache_sniper/, etc.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (workspace_root() / pp).resolve()


def resolve_generated_path(p: PathLike) -> Path:
    """
    Resolve caminhos relativos usando a pasta externa `generated_root()`.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (generated_root() / pp).resolve()


__all__ = [
    "modules_root",
    "repo_root",
    "workspace_root",
    "models_root",
    "storage_root",
    "cache_sniper_root",
    "generated_root",
    "feature_cache_root",
    "ohlc_cache_root",
    "resolve_repo_path",
    "resolve_workspace_path",
    "resolve_generated_path",
    "PathLike",
]
