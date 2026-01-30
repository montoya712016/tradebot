# -*- coding: utf-8 -*-
"""
Shim para compatibilidade: importa de core.utils.notify.
"""
from __future__ import annotations

from core.utils.notify import (
    PushoverConfig,
    load_from_env,
    load_from_local_secrets,
    load_default,
    send_pushover,
)

__all__ = [
    "PushoverConfig",
    "load_from_env",
    "load_from_local_secrets",
    "load_default",
    "send_pushover",
]
