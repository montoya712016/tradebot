from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, render_template, request

from .dashboard_state import DemoStateGenerator, StateStore, create_demo_state
# trade contract (para EMA de saída)
try:  # pragma: no cover
    from trade_contract import DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
except Exception:  # pragma: no cover
    DEFAULT_TRADE_CONTRACT = None  # type: ignore
    exit_ema_span_from_window = None  # type: ignore

# OHLC helpers (opcionais)
try:  # pragma: no cover - dependência dinâmica
    from prepare_features.data import load_ohlc_1m_series
except Exception:  # pragma: no cover
    load_ohlc_1m_series = None  # type: ignore
try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


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

    @app.get("/api/ohlc_window")
    def api_ohlc_window() -> Any:
        """
        Retorna OHLC+EMA para um símbolo.
        Params:
        - symbol (obrigatório)
        - end_ms (epoch ms, opcional; default=agora)
        - lookback_min (minutos, opcional; default=30)
        """
        if load_ohlc_1m_series is None or pd is None:
            return jsonify({"ok": False, "error": "ohlc_loader_unavailable"}), 500
        sym = (request.args.get("symbol") or "").upper().strip()
        if not sym:
            return jsonify({"ok": False, "error": "symbol_required"}), 400
        try:
            end_ms = int(request.args.get("end_ms") or int(pd.Timestamp.utcnow().value // 1_000_000))
        except Exception:
            end_ms = int(pd.Timestamp.utcnow().value // 1_000_000)
        try:
            lookback_min = int(request.args.get("lookback_min") or 30)
        except Exception:
            lookback_min = 30
        start_ms = end_ms - max(1, lookback_min) * 60_000
        days = max(1, int((lookback_min / 1440.0) + 2))
        df = load_ohlc_1m_series(sym, days, remove_tail_days=0)
        if df is None or df.empty:
            return jsonify({"ok": False, "error": "no_data"}), 404
        df = df[(df.index.view("int64") // 1_000_000 >= start_ms) & (df.index.view("int64") // 1_000_000 <= end_ms)]
        if df.empty:
            # fallback: usa os últimos 120 candles disponíveis
            df = load_ohlc_1m_series(sym, days, remove_tail_days=0)
            if df is None or df.empty:
                return jsonify({"ok": False, "error": "no_data_in_range"}), 404
            df = df.sort_index().tail(120)
        df = df.sort_index()
        if (DEFAULT_TRADE_CONTRACT is not None) and (exit_ema_span_from_window is not None):
            span = exit_ema_span_from_window(DEFAULT_TRADE_CONTRACT, 60)
            offset = float(getattr(DEFAULT_TRADE_CONTRACT, "exit_ema_init_offset_pct", 0.0) or 0.0)
        else:
            span = 55
            offset = 0.0
        payload = [
            {
                "ts": int(ts.value // 1_000_000),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for ts, r in df.iterrows()
        ]
        return jsonify(
            {
                "ok": True,
                "symbol": sym,
                "data": payload,
                "ema_span": int(span),
                "ema_offset_pct": float(offset),
            }
        )

    # Só pra evitar "unused variable" e deixar explícito:
    _ = asdict  # noqa: F841
    _ = _bool_env  # noqa: F841
    _ = demo_gen  # noqa: F841

    return app, store
