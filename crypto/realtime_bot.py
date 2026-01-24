# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Realtime bot (stage 1): REST backfill + WebSocket live klines (1m).

Behavior:
- On startup, backfill missing 1m candles via REST (last + 1m -> last closed minute).
  If table is empty, bootstrap last N days.
- Keep WS stream (kline_1m) for live data.
- If a gap is detected on WS, enqueue a REST backfill for the missing range.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import queue
import sys
import threading
import time

import mysql.connector
from mysql.connector import pooling
import requests
import websocket


def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    repo_root = here
    for p in here.parents:
        if p.name.lower() == "tradebot":
            repo_root = p
            break
    for cand in (repo_root, repo_root / "modules"):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_add_repo_paths()

try:
    from modules.config.symbols import default_top_market_cap_path
except Exception:
    from config.symbols import default_top_market_cap_path  # type: ignore[import]


@dataclass
class MySQLConfig:
    host: str = "localhost"
    user: str = "root"
    password: str = "2017"
    database: str = "crypto"
    autocommit: bool = False
    pool_size: int = 8


@dataclass
class RealtimeSettings:
    # Binance
    api_key: str = ""
    interval: str = "1m"
    page_limit: int = 1000

    # Bootstrap (only if table is empty)
    bootstrap_days: int = 3

    # WS
    use_websocket: bool = True
    ws_chunk: int = 100
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10

    # REST gap backfill
    gap_workers: int = 2
    rest_timeout_sec: float = 20.0
    rest_retry_sleep_sec: float = 0.5

    # Symbols
    quote_symbols_file: str = ""

    # DB
    db: MySQLConfig = field(default_factory=MySQLConfig)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _last_closed_minute_ms(now_ms: Optional[int] = None) -> int:
    now_ms = int(now_ms or _now_ms())
    return (now_ms // 60_000 - 1) * 60_000


def load_symbols(path: str) -> List[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out, seen = [], set()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        sym = s.split(":", 1)[0].strip()
        if sym.endswith("USDT") and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


class MySQLStore:
    def __init__(self, cfg: MySQLConfig):
        self.cfg = cfg
        try:
            self.pool = pooling.MySQLConnectionPool(pool_name="realtime_pool", pool_size=int(cfg.pool_size), **cfg.__dict__)
        except Exception:
            self.pool = None

    def get_conn(self):
        if self.pool:
            return self.pool.get_connection()
        return mysql.connector.connect(**self.cfg.__dict__)

    def ensure_table(self, sym: str) -> None:
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{sym}` (
                dates BIGINT PRIMARY KEY,
                open_prices DOUBLE,
                high_prices DOUBLE,
                low_prices DOUBLE,
                closing_prices DOUBLE,
                volume DOUBLE
            ) ENGINE=InnoDB;
            """
        )
        conn.commit()
        cur.close()
        conn.close()

    def mysql_max_date(self, sym: str) -> Optional[int]:
        conn = self.get_conn()
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT MAX(dates) FROM `{sym}`")
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()
            conn.close()

    def insert_batch(self, sym: str, rows: List[Tuple]) -> int:
        if not rows:
            return 0
        conn = self.get_conn()
        cur = conn.cursor()
        try:
            cur.executemany(
                f"INSERT IGNORE INTO `{sym}` (dates, open_prices, high_prices, low_prices, closing_prices, volume) "
                f"VALUES (%s,%s,%s,%s,%s,%s)",
                rows,
            )
            conn.commit()
            return cur.rowcount
        finally:
            cur.close()
            conn.close()


def _normalize_last_ts(last_raw: Optional[int], end_ms_closed: int) -> Optional[int]:
    if last_raw is None:
        return None
    last = int(last_raw)
    if last < 10_000_000_000:
        last *= 1000
    if last > end_ms_closed * 10:
        last //= 1000
    if last > end_ms_closed:
        last = end_ms_closed - 60_000
    return last


class RestBackfillQueue:
    def __init__(self, store: MySQLStore, settings: RealtimeSettings):
        self.store = store
        self.settings = settings
        self.q: "queue.Queue[tuple[str, int, int]]" = queue.Queue()
        self.inflight: set[tuple[str, int, int]] = set()
        self.locks: Dict[str, threading.Lock] = {}
        self.threads: List[threading.Thread] = []
        self.stop_evt = threading.Event()
        self.stats_lock = threading.Lock()
        self.stats_started_at = time.time()
        self.stats_total_minutes = 0
        self.stats_done_minutes = 0
        self._warned: set[str] = set()

    def start(self) -> None:
        for i in range(int(max(1, self.settings.gap_workers))):
            th = threading.Thread(target=self._worker, name=f"rest-gap-{i}", daemon=True)
            th.start()
            self.threads.append(th)

    def stop(self) -> None:
        self.stop_evt.set()

    def _sym_lock(self, sym: str) -> threading.Lock:
        if sym not in self.locks:
            self.locks[sym] = threading.Lock()
        return self.locks[sym]

    def enqueue(self, sym: str, start_ms: int, end_ms: int) -> None:
        if end_ms < start_ms:
            return
        key = (sym, int(start_ms), int(end_ms))
        if key in self.inflight:
            return
        self.inflight.add(key)
        minutes = int((int(end_ms) - int(start_ms)) // 60_000 + 1)
        with self.stats_lock:
            self.stats_total_minutes += max(0, minutes)
        self.q.put(key)

    def _worker(self) -> None:
        while not self.stop_evt.is_set():
            try:
                sym, start_ms, end_ms = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with self._sym_lock(sym):
                    self._rest_backfill_range(sym, int(start_ms), int(end_ms))
            except Exception as e:
                print(f"[rest][err] {sym} {type(e).__name__}: {e}", flush=True)
            finally:
                self.inflight.discard((sym, int(start_ms), int(end_ms)))
                self.q.task_done()

    def _rest_backfill_range(self, sym: str, start_ms: int, end_ms: int) -> None:
        end_ms = min(int(end_ms), int(_last_closed_minute_ms()))
        if end_ms < start_ms:
            return
        url = "https://api.binance.com/api/v3/klines"
        session = requests.Session()
        ts = int(start_ms)
        fetched_minutes = 0
        failures = 0
        while ts <= end_ms:
            params = {"symbol": sym, "interval": self.settings.interval, "startTime": ts, "limit": int(self.settings.page_limit)}
            try:
                resp = session.get(url, params=params, timeout=float(self.settings.rest_timeout_sec))
            except Exception:
                failures += 1
                if failures <= 3:
                    print(f"[rest][warn] {sym} request error ts={ts} end={end_ms}", flush=True)
                time.sleep(float(self.settings.rest_retry_sleep_sec))
                continue
            if resp.status_code == 429:
                failures += 1
                if failures <= 3:
                    print(f"[rest][warn] {sym} HTTP 429 rate limit ts={ts}", flush=True)
                time.sleep(float(self.settings.rest_retry_sleep_sec))
                continue
            if resp.status_code >= 500:
                failures += 1
                if failures <= 3:
                    print(f"[rest][warn] {sym} HTTP {resp.status_code} ts={ts}", flush=True)
                time.sleep(float(self.settings.rest_retry_sleep_sec))
                continue
            try:
                data = resp.json()
            except Exception:
                failures += 1
                if failures <= 3:
                    print(f"[rest][warn] {sym} JSON parse error ts={ts}", flush=True)
                data = []
            if not data:
                if failures <= 3:
                    print(f"[rest][warn] {sym} empty response ts={ts} end={end_ms}", flush=True)
                break
            rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in data]
            self.store.insert_batch(sym, rows)
            last_open = int(data[-1][0])
            fetched_minutes += len(data)
            ts = last_open + 60_000
            if last_open >= end_ms:
                break
        if fetched_minutes > 0:
            with self.stats_lock:
                self.stats_done_minutes += int(fetched_minutes)
        else:
            if sym not in self._warned:
                self._warned.add(sym)
                print(
                    f"[rest][warn] {sym} backfill sem progresso (fetched=0 failures={failures}) "
                    f"range=[{start_ms}..{end_ms}]",
                    flush=True,
                )

    def snapshot_stats(self) -> dict:
        with self.stats_lock:
            total = int(self.stats_total_minutes)
            done = int(self.stats_done_minutes)
            elapsed = max(1e-6, time.time() - float(self.stats_started_at))
        remain = max(0, total - done)
        rate = done / elapsed
        eta = (remain / rate) if rate > 0 else 0.0
        return {
            "total_minutes": total,
            "done_minutes": done,
            "remain_minutes": remain,
            "rate_per_sec": rate,
            "eta_sec": eta,
            "queue_size": int(self.q.qsize()),
            "inflight": int(len(self.inflight)),
        }


class WsKlineIngestor:
    def __init__(self, symbols: List[str], store: MySQLStore, backfill: RestBackfillQueue):
        self.symbols = symbols
        self.store = store
        self.backfill = backfill
        self.last_ts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.cur_minute_ts: Optional[int] = None
        self.cur_minute_start_wall: Optional[float] = None
        self.cur_minute_seen: set[str] = set()

    def init_last_ts(self) -> None:
        end_ms_closed = _last_closed_minute_ms()
        for sym in self.symbols:
            last_raw = self.store.mysql_max_date(sym)
            last = _normalize_last_ts(last_raw, end_ms_closed)
            if last is not None:
                self.last_ts[sym] = int(last)

    def _on_message(self, _ws, msg: str) -> None:
        try:
            d = json.loads(msg)
        except Exception:
            return
        data = d.get("data", d)
        if not isinstance(data, dict):
            return
        if data.get("e") != "kline":
            return
        k = data.get("k", {})
        if not k or not bool(k.get("x")):
            return
        sym = str(data.get("s") or "").upper()
        if not sym:
            return
        open_ts = int(k.get("t"))
        row = (
            open_ts,
            float(k.get("o")),
            float(k.get("h")),
            float(k.get("l")),
            float(k.get("c")),
            float(k.get("v")),
        )
        with self.lock:
            last = self.last_ts.get(sym)
            if last is not None and open_ts > last + 60_000:
                self.backfill.enqueue(sym, last + 60_000, open_ts - 60_000)
            self.last_ts[sym] = int(open_ts)
            if self.cur_minute_ts != int(open_ts):
                self.cur_minute_ts = int(open_ts)
                self.cur_minute_start_wall = time.time()
                self.cur_minute_seen = set()
            self.cur_minute_seen.add(sym)
        self.store.insert_batch(sym, [row])

    def _on_error(self, _ws, err) -> None:
        print(f"[ws] error: {err}", flush=True)

    def _on_close(self, _ws, code, msg) -> None:
        print(f"[ws] closed: code={code} msg={msg}", flush=True)

    def _launch_ws(self, chunk: List[str], settings: RealtimeSettings) -> None:
        streams = [f"{s.lower()}@kline_1m" for s in chunk]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        while True:
            ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                ws.run_forever(ping_interval=int(settings.ws_ping_interval), ping_timeout=int(settings.ws_ping_timeout))
            except Exception:
                time.sleep(3)

    def start(self, settings: RealtimeSettings) -> None:
        self.init_last_ts()
        for chunk in _chunked(self.symbols, int(settings.ws_chunk)):
            th = threading.Thread(
                target=self._launch_ws,
                args=(chunk, settings),
                name=f"ws-{chunk[0]}",
                daemon=True,
            )
            th.start()

    def snapshot_progress(self) -> dict:
        with self.lock:
            ts = self.cur_minute_ts
            seen = len(self.cur_minute_seen)
            total = len(self.symbols)
        pct = (100.0 * seen / max(1, total)) if total else 0.0
        latency_sec = 0.0
        if ts is not None:
            latency_sec = max(0.0, (time.time() * 1000 - (int(ts) + 60_000)) / 1000.0)
        return {
            "minute_ts": ts,
            "seen": int(seen),
            "total": int(total),
            "pct": float(pct),
            "latency_sec": float(latency_sec),
        }


def _chunked(it: List[str], n: int) -> List[List[str]]:
    out: List[List[str]] = []
    if n <= 0:
        return [it]
    for i in range(0, len(it), n):
        out.append(it[i : i + n])
    return out


def _initial_rest_backfill(symbols: List[str], store: MySQLStore, settings: RealtimeSettings) -> None:
    backfill = RestBackfillQueue(store, settings)
    backfill.start()
    end_ms_closed = _last_closed_minute_ms()
    for sym in symbols:
        store.ensure_table(sym)
        last_raw = store.mysql_max_date(sym)
        last = _normalize_last_ts(last_raw, end_ms_closed)
        if last is None:
            start_ts = end_ms_closed - int(settings.bootstrap_days) * 24 * 60 * 60 * 1000
        else:
            start_ts = last + 60_000
        if start_ts <= end_ms_closed:
            backfill.enqueue(sym, int(start_ts), int(end_ms_closed))
    backfill.q.join()
    return


def _status_loop(ws: Optional[WsKlineIngestor], backfill: Optional[RestBackfillQueue]) -> None:
    while True:
        parts = []
        if ws is not None:
            st = ws.snapshot_progress()
            minute = st["minute_ts"]
            if minute is not None:
                minute_str = time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(minute) / 1000))
            else:
                minute_str = "--"
            parts.append(
                f"[ws] minute={minute_str} {st['seen']}/{st['total']} ({st['pct']:.1f}%) latency={st['latency_sec']:.1f}s"
            )
        if backfill is not None:
            st = backfill.snapshot_stats()
            parts.append(
                f"[rest] pending={st['remain_minutes']}m done={st['done_minutes']}m "
                f"queue={st['queue_size']} inflight={st['inflight']} eta={st['eta_sec']:.0f}s"
            )
        if parts:
            print(" | ".join(parts), flush=True)
        time.sleep(60)


def main() -> None:
    settings = RealtimeSettings(
        api_key="",
        quote_symbols_file="",
        db=MySQLConfig(host="localhost", user="root", password="2017", database="crypto"),
    )

    symbols_file = settings.quote_symbols_file or str(default_top_market_cap_path())
    symbols = load_symbols(symbols_file)
    if not symbols:
        raise RuntimeError("Nenhum simbolo valido encontrado em top_market_cap.txt")

    store = MySQLStore(settings.db)
    _initial_rest_backfill(symbols, store, settings)

    if settings.use_websocket:
        backfill = RestBackfillQueue(store, settings)
        backfill.start()
        ws = WsKlineIngestor(symbols, store, backfill)
        ws.start(settings)
        threading.Thread(target=_status_loop, args=(ws, backfill), daemon=True).start()
        while True:
            time.sleep(3600)
    else:
        # Only REST backfill loop (not recommended for realtime)
        while True:
            _initial_rest_backfill(symbols, store, settings)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
