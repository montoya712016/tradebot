# -*- coding: utf-8 -*-
"""
Rolling Window - Janela deslizante de dados OHLC em memória.
"""
from __future__ import annotations

import threading
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class RollingWindow:
    """
    Mantém uma janela deslizante de candles OHLCV em memória.
    Thread-safe para ingestão via WebSocket e leitura pelo bot.
    """

    def __init__(self, max_rows: int, min_ready_rows: int):
        self.max_rows = int(max(1, max_rows))
        self.min_ready_rows = int(max(1, min_ready_rows))
        # (ts_ms, o, h, l, c, v)
        self.rows: List[Tuple[int, float, float, float, float, float]] = []
        self.last_ts: Optional[int] = None
        self.need_reload = False
        self.lock = threading.Lock()

    def peek_last_ts(self) -> Optional[int]:
        with self.lock:
            return self.last_ts

    def is_ready(self) -> bool:
        with self.lock:
            return len(self.rows) >= self.min_ready_rows

    def ingest(self, ts_ms: int, o: float, h: float, l: float, c: float, v: float) -> bool:
        """
        Ingere um novo candle ou atualiza o último.
        Retorna True se foi adicionado/atualizado, False se ignorado (antigo).
        """
        with self.lock:
            if self.last_ts is not None and ts_ms <= self.last_ts:
                # Atualiza candle atual (ex: WS mandando updates do minuto corrente)
                if ts_ms == self.last_ts and self.rows:
                    self.rows[-1] = (ts_ms, o, h, l, c, v)
                    return False
                return False
            
            # Detecta gap
            if self.last_ts is not None and ts_ms > self.last_ts + 60_000:
                self.need_reload = True
                
            self.rows.append((ts_ms, o, h, l, c, v))
            if len(self.rows) > self.max_rows:
                self.rows = self.rows[-self.max_rows :]
            self.last_ts = ts_ms
            return True

    def replace_from_df(self, df: pd.DataFrame) -> None:
        """Substitui todo o conteúdo da janela a partir de um DataFrame."""
        with self.lock:
            if df is None or df.empty:
                return
            rows = []
            for ts, r in df.iterrows():
                rows.append(
                    (
                        int(pd.Timestamp(ts).value // 1_000_000),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                        float(r.get("volume", 0.0)),
                    )
                )
            rows = rows[-self.max_rows :]
            self.rows = rows
            self.last_ts = rows[-1][0] if rows else None
            self.need_reload = False

    def to_frame(self) -> pd.DataFrame:
        """Converte a janela atual para DataFrame pandas."""
        with self.lock:
            if not self.rows:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            
            ts = pd.to_datetime(np.asarray([r[0] for r in self.rows], dtype=np.int64), unit="ms")
            df = pd.DataFrame(
                {
                    "open": np.asarray([r[1] for r in self.rows], dtype=np.float64),
                    "high": np.asarray([r[2] for r in self.rows], dtype=np.float64),
                    "low": np.asarray([r[3] for r in self.rows], dtype=np.float64),
                    "close": np.asarray([r[4] for r in self.rows], dtype=np.float64),
                    "volume": np.asarray([r[5] for r in self.rows], dtype=np.float64),
                },
                index=ts,
            )
            return df
