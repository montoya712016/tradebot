# -*- coding: utf-8 -*-
"""
Executores de ordens - Interface e implementações.
"""

from core.executors.paper import PaperExecutor
from core.executors.live import LiveExecutor

__all__ = ["PaperExecutor", "LiveExecutor"]
