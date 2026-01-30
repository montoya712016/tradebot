# -*- coding: utf-8 -*-
"""
MySQL Storage - Configuração e persistência de dados de mercado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import mysql.connector
from mysql.connector import pooling


@dataclass
class MySQLConfig:
    host: str = "localhost"
    user: str = "root"
    password: str = "2017"
    database: str = "crypto"
    autocommit: bool = False
    pool_size: int = 8


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
