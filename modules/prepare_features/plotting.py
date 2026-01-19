# -*- coding: utf-8 -*-
"""
Wrapper for centralized plotting (plotly).
"""

from __future__ import annotations

try:
    from plotting import plot_all  # type: ignore[import]
except Exception:
    try:
        import sys
        from pathlib import Path

        here = Path(__file__).resolve()
        for p in here.parents:
            if p.name.lower() == "modules":
                sp = str(p)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                break
        from plotting import plot_all  # type: ignore[import]
    except Exception:
        from plotting import plot_all  # type: ignore[import]

__all__ = ["plot_all"]
