from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, render_template, request

from .dashboard_state import DemoStateGenerator, StateStore, create_demo_state


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def create_app(*, demo: bool = True, refresh_sec: float = 2.0) -> tuple[Flask, StateStore]:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["JSON_SORT_KEYS"] = False

    store = StateStore(create_demo_state())
    demo_gen: DemoStateGenerator | None = None
    if demo:
        demo_gen = DemoStateGenerator(store, refresh_sec=refresh_sec)
        demo_gen.start()

    @app.get("/")
    def index() -> Any:
        st = store.get().to_dict()
        return render_template("index.html", initial_state=st)

    @app.get("/api/state")
    def api_state() -> Any:
        return jsonify(store.get().to_dict())

    @app.post("/api/update")
    def api_update() -> Any:
        """
        Endpoint pra integração futura com o robô:
        POST JSON com summary/positions/trades/allocation/meta.
        """
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload_must_be_object"}), 400
        try:
            store.update_from_payload(payload)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    @app.get("/api/health")
    def api_health() -> Any:
        return jsonify({"ok": True})

    @app.get("/api/config")
    def api_config() -> Any:
        return jsonify(
            {
                "demo": bool(demo),
                "refresh_sec": float(refresh_sec),
                "server": "realtime_dashboard",
            }
        )

    # Só pra evitar "unused variable" e deixar explícito:
    _ = asdict  # noqa: F841
    _ = _bool_env  # noqa: F841
    _ = demo_gen  # noqa: F841

    return app, store

