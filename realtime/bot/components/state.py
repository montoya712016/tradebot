from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional, Dict, List, Any
import psutil

from realtime.bot.settings import LiveSettings

log = logging.getLogger("realtime.components.state")

class StateManager:
    """
    Manages state persistence (JSON) and Dashboard updates.
    """
    def __init__(self, settings: LiveSettings, feature_flags: dict):
        self.settings = settings
        self.feature_flags = feature_flags
        self.last_dashboard_push = 0.0
        self.dashboard_url = "http://127.0.0.1:5000/api/update"

    def load_persisted_state(self, state_file: str) -> dict:
        """Loads state from JSON file."""
        if not os.path.exists(state_file):
            log.info("[state] arquivo %s não existe", state_file)
            return {}
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("[state] carregado %d chaves de %s", len(data), state_file)
            return data
        except Exception as e:
            log.warning("[state] erro ao ler %s: %s", state_file, e)
            return {}

    def persist_state(self, state_file: str, data: dict):
        """Saves state to JSON file."""
        try:
            tmp_file = state_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, state_file)
        except Exception as e:
            log.error("[state] erro ao salvar state: %s", e)

    def push_dashboard_state(self, state_payload: dict, force: bool = False):
        """Pushes state to the local dashboard server."""
        if not self.settings.dashboard_enabled:
            return
        
        now = time.time()
        # Rate limit: max 1 request every 1.5s unless forced
        if not force and (now - self.last_dashboard_push < 1.5):
            return

        try:
            import requests
            try:
                # Add system metrics
                mem = psutil.virtual_memory()
                state_payload["meta"]["system"] = {
                    "cpu_pct": psutil.cpu_percent(interval=None),
                    "mem_pct": mem.percent,
                }
            except Exception:
                pass

            requests.post(self.dashboard_url, json=state_payload, timeout=0.1)
            self.last_dashboard_push = now
        except Exception:
            # Silent fail for dashboard connection
            pass
