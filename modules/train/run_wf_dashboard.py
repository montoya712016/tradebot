# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

try:
    from .wf_dashboard_server import DashboardConfig, create_app
except Exception:
    from wf_dashboard_server import DashboardConfig, create_app


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--out-root", type=str, default="wf_random_loop")
    ap.add_argument("--results-csv", type=str, default="random_runs.csv")
    ap.add_argument("--max-rows", type=int, default=200)
    ap.add_argument("--max-images", type=int, default=24)
    ap.add_argument("--refresh-sec", type=float, default=6.0)
    ap.add_argument("--demo-csv", action="store_true", default=_env_bool("WF_DASH_DEMO_CSV", False))
    ap.add_argument("--loop-log-path", type=str, default=os.getenv("WF_DASH_LOOP_LOG", ""))
    ap.add_argument("--dash-log-path", type=str, default=os.getenv("WF_DASH_LOG", ""))
    ap.add_argument("--log-max-lines", type=int, default=int(os.getenv("WF_DASH_LOG_MAX_LINES", "180")))
    args = ap.parse_args()

    cfg = DashboardConfig(
        out_root=str(args.out_root),
        results_csv=str(args.results_csv),
        max_rows=int(args.max_rows),
        max_images=int(args.max_images),
        refresh_sec=float(args.refresh_sec),
        demo_csv=bool(args.demo_csv),
        loop_log_path=str(args.loop_log_path) if str(args.loop_log_path).strip() else None,
        dash_log_path=str(args.dash_log_path) if str(args.dash_log_path).strip() else None,
        log_max_lines=int(args.log_max_lines),
    )
    app, _ = create_app(cfg)
    app.run(host=str(args.host), port=int(args.port), debug=False)


if __name__ == "__main__":
    main()
