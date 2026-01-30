# -*- coding: utf-8 -*-
"""
Shim para compatibilidade: importa de core.utils.paths.
"""
from __future__ import annotations

from core.utils.paths import *

__all__ = [
    "modules_root",
    "repo_root",
    "workspace_root",
    "models_root",
    "models_root_for_asset",
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
