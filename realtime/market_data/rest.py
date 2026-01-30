# -*- coding: utf-8 -*-
"""
REST Backfill - Preenchimento de lacunas de dados via API REST.
"""
from __future__ import annotations

import queue
import threading
import time
import requests
from typing import Dict, List, Set, Tuple
from realtime.market_data.mysql import MySQLStore
from realtime.bot.settings import LiveSettings


def _last_closed_minute_ms(now_ms: int | None = None) -> int:
    now_ms = int(now_ms or (time.time() * 1000))
    return (now_ms // 60_000 - 1) * 60_000


class RestBackfillQueue:
    """
    Gerencia fila de requisições REST para preencher buracos (gaps) nos dados.
    """
    def __init__(self, store: MySQLStore, settings: LiveSettings):
        self.store = store
        self.settings = settings
        self.q: "queue.Queue[tuple[str, int, int]]" = queue.Queue()
        self.inflight: Set[tuple[str, int, int]] = set()
        self.locks: Dict[str, threading.Lock] = {}
        self.threads: List[threading.Thread] = []
        self.stop_evt = threading.Event()
        self.stats_lock = threading.Lock()
        self.stats_started_at = time.time()
        self.stats_total_minutes = 0
        self.stats_done_minutes = 0
        self._warned: Set[str] = set()

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
