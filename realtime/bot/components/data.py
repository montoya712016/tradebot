from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional
import pandas as pd

from realtime.bot.settings import LiveSettings
from realtime.market_data.rolling_window import RollingWindow
from realtime.market_data.websocket import WsKlineIngestor
from realtime.market_data.rest import RestBackfillQueue, _last_closed_minute_ms
from modules.utils.time_utils import window_days_for_minutes

# Optional downloader import
try:
    from crypto.binance.download_to_mysql import (
        VALID_TIMEFRAMES,
        download_ohlc_threaded,
    )
except ImportError:
    VALID_TIMEFRAMES = ["1m"]
    download_ohlc_threaded = None

log = logging.getLogger("realtime.components.data")

class DataManager:
    """
    Manages market data: RollingWindows, Backfill, WebSocket.
    """
    def __init__(self, settings: LiveSettings, symbols: List[str]):
        self.settings = settings
        self.symbols = symbols
        self.windows: Dict[str, RollingWindow] = {}
        
        # Components
        self.backfill = RestBackfillQueue()
        self.ws_ingestor: Optional[WsKlineIngestor] = None
        
        # State
        self.ws_connected = False
        self.ws_last_ts = 0
        self.ws_msg_count = 0

    def init_windows(self):
        """Initializes rolling windows for all symbols."""
        needed_rows = int(self.settings.window_size_minutes * 1.5)
        min_rows = int(self.settings.window_size_minutes)
        
        for sym in self.symbols:
            self.windows[sym] = RollingWindow(
                symbol=sym,
                max_size=needed_rows,
                # We can load initial data here if we move _load_one logic
            )
        log.info("[data] Initialized %d windows (size=%d)", len(self.windows), needed_rows)

    def get_window(self, symbol: str) -> Optional[RollingWindow]:
        return self.windows.get(symbol)

    def start_backfill(self):
        """Starts REST backfill workers."""
        if self.settings.use_rest_backfill:
            self.backfill.start_workers(n_workers=4)
            log.info("[data] Backfill workers started")

    def stop(self):
        """Stops data threads."""
        self.backfill.stop()
        if self.ws_ingestor:
            self.ws_ingestor.stop()

    def start_websocket(self, callback_on_kline):
        """Starts WebSocket ingestor."""
        # Clean symbols for streams
        streams = [s.lower() + "@kline_1m" for s in self.symbols]
        
        self.ws_ingestor = WsKlineIngestor(
            streams=streams,
            on_kline=callback_on_kline, # Callback to main loop
            on_status=self._on_ws_status
        )
        self.ws_ingestor.start()
        log.info("[data] WebSocket started for %d symbols", len(self.symbols))

    def _on_ws_status(self, connected: bool, last_ts: int, msg_count: int):
        self.ws_connected = connected
        self.ws_last_ts = last_ts
        self.ws_msg_count = msg_count

    def fast_backfill(self):
        """ Performs massive initial download/backfill if configured. """
        if not (self.settings.use_fast_backfill and download_ohlc_threaded):
            return

        days = window_days_for_minutes(self.settings.window_size_minutes)
        log.info("[data] Iniciando downloads (INSERT DIRETO no MySQL)...")
        t0 = time.time()
        
        download_ohlc_threaded(
            symbols=self.symbols,
            days=days,
            timeframe="1m",
            concurrency=16,
            batch_size=500,
            force_update=False
        )
        log.info("[data] Todos concluídos em %.1f min", (time.time() - t0) / 60.0)
    
    def on_kline_update(self, symbol: str, row: tuple) -> RollingWindow:
        """ Updates the window for a symbol. Returns the window. """
        ts = int(row[0])
        self.ws_last_ts = max(self.ws_last_ts, ts)
        self.ws_msg_count += 1
        
        win = self.windows.get(symbol)
        if win:
            win.add_row(row)
        return win
