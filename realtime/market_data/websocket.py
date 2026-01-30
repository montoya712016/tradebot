# -*- coding: utf-8 -*-
"""
WebSocket Ingestor - Ingestão de dados de mercado via WebSocket.
"""
from __future__ import annotations

import json
import threading
import time
from typing import List, Dict, Optional, Set
import websocket
from realtime.market_data.mysql import MySQLStore
from realtime.market_data.rest import RestBackfillQueue, _last_closed_minute_ms
from realtime.bot.settings import LiveSettings


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


def _chunked(it: List[str], n: int) -> List[List[str]]:
    out: List[List[str]] = []
    if n <= 0:
        return [it]
    for i in range(0, len(it), n):
        out.append(it[i : i + n])
    return out


class WsKlineIngestor:
    """
    Gerencia conexões WebSocket para receber candles de 1m em tempo real.
    Detecta gaps e solicita backfill via REST.
    """
    def __init__(self, symbols: List[str] = None, store: MySQLStore = None, backfill: RestBackfillQueue = None, on_kline_callback: callable = None, on_kline: callable = None, on_status: callable = None, streams: List[str] = None):
        # Support default positional (symbols, store, backfill, cb)
        
        # If symbols is None but streams provided, derive symbols
        if not symbols and streams:
            self.symbols = [s.split("@")[0].upper() for s in streams]
        else:
            self.symbols = symbols or []
            
        self.store = store
        self.backfill = backfill
        self.on_kline_callback = on_kline_callback or on_kline
        self.on_status = on_status
        
        self.last_ts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.cur_minute_ts: Optional[int] = None
        self.cur_minute_start_wall: Optional[float] = None
        self.cur_minute_seen: Set[str] = set()

    def init_last_ts(self) -> None:
        if not self.store: 
            return
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
        
        # Insere no banco se disponivel
        if self.store:
            self.store.insert_batch(sym, [row])
        
        # Callback para o bot (se houver)
        if self.on_kline_callback:
            try:
                self.on_kline_callback(sym, row)
            except Exception as e:
                print(f"[ws] callback error: {e}", flush=True)

    def _on_error(self, _ws, err) -> None:
        print(f"[ws] error: {err}", flush=True)
        if self.on_status:
            self.on_status(False, 0, 0)

    def _on_close(self, _ws, code, msg) -> None:
        print(f"[ws] closed: code={code} msg={msg}", flush=True)
        if self.on_status:
            self.on_status(False, 0, 0)

    def _on_open(self, _ws) -> None:
        if self.on_status:
            self.on_status(True, 0, 0)

    def _launch_ws(self, chunk: List[str], settings: LiveSettings) -> None:
        streams = [f"{s.lower()}@kline_1m" for s in chunk]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        while True:
            ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                ws.run_forever(ping_interval=int(settings.ws_ping_interval), ping_timeout=int(settings.ws_ping_timeout))
            except Exception:
                time.sleep(3)

    def start(self, settings: LiveSettings) -> None:
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
