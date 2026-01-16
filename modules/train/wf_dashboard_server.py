# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import csv
import io
import random
import time
import sys

from flask import Flask, abort, jsonify, render_template, request, send_file


def _ensure_modules_on_sys_path() -> None:
    if __package__ not in (None, ""):
        return
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return


_ensure_modules_on_sys_path()

from utils.paths import resolve_generated_path  # noqa: E402


@dataclass
class DashboardConfig:
    out_root: str = "wf_random_loop"
    results_csv: str = "random_runs.csv"
    max_rows: int = 200
    max_images: int = 24
    refresh_sec: float = 6.0
    demo_csv: bool = False
    loop_log_path: str | None = None
    dash_log_path: str | None = None
    log_max_lines: int = 180


def _to_float(v: object, default: float | None = None) -> float | None:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _get_results_header() -> list[str]:
    try:
        from train import wf_random_loop as _wfrl  # type: ignore

        header = getattr(_wfrl, "RESULTS_HEADER", None)
        if isinstance(header, list) and header:
            return list(header)
    except Exception:
        pass
    return [
        "train_id",
        "backtest_id",
        "stage",
        "status",
        "start_utc",
        "end_utc",
        "duration_sec",
        "seed",
        "ret_pct",
        "max_dd",
        "win_rate",
        "profit_factor",
        "trades",
        "tau_entry",
        "tau_danger",
    ]


def _build_sample_rows(count: int = 6) -> list[dict]:
    rng = random.Random(42)
    now = time.time()
    rows = []
    for i in range(max(1, int(count))):
        start = now - (i + 1) * 3600
        end = start + rng.randint(900, 4200)
        ret = rng.uniform(-3.0, 12.0)
        dd = rng.uniform(0.03, 0.18)
        pf = rng.uniform(0.9, 2.2)
        win = rng.uniform(0.38, 0.62)
        trades = rng.randint(40, 380)
        tau_e = rng.uniform(0.78, 0.90)
        tau_d = rng.uniform(0.75, 0.92)
        rows.append(
            {
                "train_id": f"demo_{i + 1:02d}",
                "backtest_id": f"bt_{i + 1:02d}",
                "stage": "backtest",
                "status": "ok",
                "start_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start)),
                "end_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(end)),
                "duration_sec": int(end - start),
                "seed": 42 + i,
                "ret_pct": round(ret, 3),
                "max_dd": round(dd, 4),
                "win_rate": round(win, 4),
                "profit_factor": round(pf, 3),
                "trades": trades,
                "tau_entry": round(tau_e, 3),
                "tau_danger": round(tau_d, 3),
                "bt_years": 6,
                "bt_step_days": 90,
                "bt_bar_stride": 1,
            }
        )
    return rows


def _ensure_sample_csv(path: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    if path.exists() and path.stat().st_size > 0:
        return
    header = _get_results_header()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _build_sample_rows()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in header}
            w.writerow(out)


def _read_csv_tail(path: Path, max_rows: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            return []
        header = header.strip("\r\n")
        if not header:
            return []
        if max_rows > 0:
            tail = deque(maxlen=max_rows)
            for line in f:
                tail.append(line)
            lines = list(tail)
        else:
            lines = list(f)
    data = header + "\n" + "".join(lines)
    reader = csv.DictReader(io.StringIO(data))
    out = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        out.append(row)
    return out


def _filter_real_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        tid = str(row.get("train_id") or "").strip().lower()
        if not tid:
            continue
        if tid.startswith("demo_"):
            continue
        out.append(row)
    return out


def _csv_from_rows(rows: list[dict], header: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    for row in rows:
        out = {k: row.get(k, "") for k in header}
        w.writerow(out)
    return buf.getvalue().encode("utf-8")


def _resolve_log_path(path_str: str | None, root: Path, default_name: str) -> Path:
    if path_str:
        p = Path(path_str)
        if not p.is_absolute():
            return (root / p).resolve()
        return p.resolve()
    return (root / default_name).resolve()


def _read_log_tail(path: Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            if max_lines > 0:
                tail = deque(maxlen=max_lines)
                for line in f:
                    tail.append(line)
                return list(tail)
            return list(f)
    except Exception:
        return []


def _safe_relpath(path_str: str, root: Path) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    root_resolved = root.resolve()
    try:
        rel = p.relative_to(root_resolved)
    except Exception:
        return None
    return rel.as_posix()


class LoopStateLoader:
    def __init__(self, cfg: DashboardConfig) -> None:
        self.cfg = cfg
        self.out_root = resolve_generated_path(cfg.out_root)
        self.csv_path = self.out_root / cfg.results_csv
        self.loop_log_path = _resolve_log_path(cfg.loop_log_path, self.out_root, "loop.log")
        self.dash_log_path = _resolve_log_path(cfg.dash_log_path, self.out_root, "dash.log")
        self._last_key: tuple[float | None, float | None, float | None] | None = None
        self._cache: dict | None = None

    def load(self) -> dict:
        _ensure_sample_csv(self.csv_path, enabled=bool(self.cfg.demo_csv))
        try:
            csv_mtime = self.csv_path.stat().st_mtime
        except Exception:
            csv_mtime = None
        try:
            loop_mtime = self.loop_log_path.stat().st_mtime
        except Exception:
            loop_mtime = None
        try:
            dash_mtime = self.dash_log_path.stat().st_mtime
        except Exception:
            dash_mtime = None
        cache_key = (csv_mtime, loop_mtime, dash_mtime)
        if self._last_key == cache_key and self._cache is not None:
            return self._cache

        rows = _read_csv_tail(self.csv_path, int(self.cfg.max_rows))
        if not self.cfg.demo_csv:
            rows = _filter_real_rows(rows)
        state = self._build_state(rows)
        state["logs"] = {
            "loop": _read_log_tail(self.loop_log_path, int(self.cfg.log_max_lines)),
            "dash": _read_log_tail(self.dash_log_path, int(self.cfg.log_max_lines)),
        }
        self._last_key = cache_key
        self._cache = state
        return state

    def _build_state(self, rows: list[dict]) -> dict:
        now_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        backtests = [r for r in rows if (r.get("stage") or "").lower() == "backtest"]
        has_real = any((r.get("train_id") or "").lower() not in ("", "demo") for r in backtests)
        ok_backtests = [
            r
            for r in backtests
            if (r.get("status") or "").lower() == "ok"
            and (not has_real or (r.get("train_id") or "").lower() not in ("demo",))
        ]
        err_backtests = [r for r in backtests if (r.get("status") or "").lower() != "ok"]
        train_ids = {r.get("train_id") for r in rows if r.get("train_id")}
        last = rows[-1] if rows else {}

        def pick_max(key: str) -> dict | None:
            best = None
            best_v = None
            for r in ok_backtests:
                v = _to_float(r.get(key), None)
                if v is None:
                    continue
                if best_v is None or v > best_v:
                    best_v = v
                    best = r
            return best

        def pick_min(key: str) -> dict | None:
            best = None
            best_v = None
            for r in ok_backtests:
                v = _to_float(r.get(key), None)
                if v is None:
                    continue
                if best_v is None or v < best_v:
                    best_v = v
                    best = r
            return best

        best_ret = pick_max("ret_pct")
        best_pf = pick_max("profit_factor")
        best_win = pick_max("win_rate")
        best_dd = pick_min("max_dd")

        images = []
        for r in reversed(backtests):
            rel = _safe_relpath(str(r.get("equity_png") or ""), self.out_root)
            if not rel:
                continue
            images.append(
                {
                    "train_id": r.get("train_id") or "",
                    "backtest_id": r.get("backtest_id") or "",
                    "ret_pct": _to_float(r.get("ret_pct"), 0.0) or 0.0,
                    "max_dd": _to_float(r.get("max_dd"), 0.0) or 0.0,
                    "profit_factor": _to_float(r.get("profit_factor"), 0.0) or 0.0,
                    "img": rel,
                }
            )
            if len(images) >= int(self.cfg.max_images):
                break

        return {
            "meta": {
                "out_root": str(self.out_root),
                "csv_path": str(self.csv_path),
                "updated_at_utc": now_utc,
                "rows_total": len(rows),
                "last_end_utc": last.get("end_utc") or "",
                "last_status": (last.get("status") or "").lower(),
                "refresh_sec": float(self.cfg.refresh_sec),
            },
            "summary": {
                "total_trains": len(train_ids),
                "total_backtests": len(backtests),
                "ok_backtests": len(ok_backtests),
                "err_backtests": len(err_backtests),
                "best_ret": best_ret or {},
                "best_pf": best_pf or {},
                "best_win": best_win or {},
                "best_dd": best_dd or {},
            },
            "rows": rows,
            "images": images,
        }


def create_app(cfg: DashboardConfig | None = None) -> tuple[Flask, LoopStateLoader]:
    cfg = cfg or DashboardConfig()
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["JSON_SORT_KEYS"] = False
    loader = LoopStateLoader(cfg)

    @app.get("/")
    def index() -> str:
        st = loader.load()
        return render_template("index.html", initial_state=st, refresh_sec=cfg.refresh_sec)

    @app.get("/api/state")
    def api_state() -> object:
        return jsonify(loader.load())

    @app.get("/api/health")
    def api_health() -> object:
        return jsonify({"ok": True})

    @app.get("/img/<path:relpath>")
    def img(relpath: str) -> object:
        root = loader.out_root.resolve()
        target = (root / relpath).resolve()
        try:
            target.relative_to(root)
        except Exception:
            abort(404)
        if not target.exists():
            abort(404)
        return send_file(target, mimetype="image/png")

    @app.get("/api/config")
    def api_config() -> object:
        return jsonify(
            {
                "out_root": str(loader.out_root),
                "results_csv": str(loader.csv_path),
                "refresh_sec": float(cfg.refresh_sec),
                "max_rows": int(cfg.max_rows),
                "max_images": int(cfg.max_images),
                "demo_csv": bool(cfg.demo_csv),
                "loop_log_path": str(loader.loop_log_path),
                "dash_log_path": str(loader.dash_log_path),
                "log_max_lines": int(cfg.log_max_lines),
                "server": "wf_dashboard",
            }
        )

    @app.get("/csv")
    def csv_inline() -> object:
        _ensure_sample_csv(loader.csv_path, enabled=bool(loader.cfg.demo_csv))
        if loader.cfg.demo_csv:
            if not loader.csv_path.exists():
                abort(404)
            return send_file(loader.csv_path, mimetype="text/csv")
        header = _get_results_header()
        if not loader.csv_path.exists():
            payload = _csv_from_rows([], header)
            return send_file(io.BytesIO(payload), mimetype="text/csv")
        rows = _read_csv_tail(loader.csv_path, max_rows=0)
        real_rows = _filter_real_rows(rows)
        payload = _csv_from_rows(real_rows, header)
        return send_file(io.BytesIO(payload), mimetype="text/csv")

    @app.get("/download/csv")
    def csv_download() -> object:
        _ensure_sample_csv(loader.csv_path, enabled=bool(loader.cfg.demo_csv))
        if loader.cfg.demo_csv:
            if not loader.csv_path.exists():
                abort(404)
            return send_file(
                loader.csv_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=loader.csv_path.name,
            )
        header = _get_results_header()
        if not loader.csv_path.exists():
            payload = _csv_from_rows([], header)
            return send_file(
                io.BytesIO(payload),
                mimetype="text/csv",
                as_attachment=True,
                download_name=loader.csv_path.name,
            )
        rows = _read_csv_tail(loader.csv_path, max_rows=0)
        real_rows = _filter_real_rows(rows)
        payload = _csv_from_rows(real_rows, header)
        return send_file(
            io.BytesIO(payload),
            mimetype="text/csv",
            as_attachment=True,
            download_name=loader.csv_path.name,
        )

    return app, loader
