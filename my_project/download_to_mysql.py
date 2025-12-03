#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downloader de klines 1m (SPOT) super-otimizado:
- asyncio + aiohttp
- Rate limit adaptativo
- INSERT DIRETO no MySQL (sem CSV intermediário)
- Retomada automática do MAX(dates) de cada tabela
"""

import os
import math
import time
import asyncio as aio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import aiohttp
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
import mysql.connector
from mysql.connector import pooling

# ────────────── CONFIG ──────────────
API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET", "YOUR_SECRET")

DB_CFG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASS", "2017"),
    database=os.getenv("DB_NAME", "crypto"),
    autocommit=False,
)

# Pool de conexões MySQL (para inserções paralelas)
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="binance_pool",
        pool_size=16,
        **DB_CFG
    )
except Exception:
    db_pool = None

SYMBOLS_FILE = os.getenv("SYMBOLS_FILE", str((Path(__file__).resolve().parent / "top_market_cap.txt")))

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

SYMBOLS = load_symbols(SYMBOLS_FILE)

# Concorrência e RL
MAX_CONC_REQUESTS = int(os.getenv("MAX_CONC_REQUESTS", "32"))
PER_SYMBOL_CONC = int(os.getenv("PER_SYMBOL_CONC", "2"))
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "1000"))
INTERVAL = "1m"
EPOCH_TS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)

TARGET_WEIGHT_PER_MIN = int(os.getenv("TARGET_WEIGHT_PER_MIN", "5000"))
SAFETY_MARGIN = 0.90

# Batch para INSERT (pequeno para flush frequente)
INSERT_BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "500"))

# UI
REFRESH_FPS = 8
BAR_LEN = 22
PAGE_SECONDS = 2

console = Console()

# ────────────── Estado ──────────────
class SymState:
    __slots__ = ("status", "perc", "last", "done_ms", "total_ms")
    def __init__(self):
        self.status = "⏳ Aguardando"
        self.perc = 0.0
        self.last = "--:--:--"
        self.done_ms = 0
        self.total_ms = 1

stats: Dict[str, SymState] = {s: SymState() for s in SYMBOLS}
ui_dirty = aio.Event()
finished = aio.Event()

# ────────────── Tabela (UI) ──────────────
def build_table(page: int, rows: int) -> Table:
    table = Table("SÍMBOLO", "STATUS", "PROGRESSO", "HORA", box=box.ROUNDED, expand=True)
    items = list(stats.items())
    total_pages = max(1, math.ceil(len(items)/rows))
    page %= total_pages
    start, end = page*rows, page*rows + rows
    for sym, st in items[start:end]:
        filled = int(st.perc * BAR_LEN + 0.5)
        bar = "█"*filled + "─"*(BAR_LEN - filled)
        table.add_row(sym, st.status, f"[{bar}] {st.perc*100:5.1f}%", st.last)
    done = sum(1 for v in stats.values() if v.status.startswith("✅"))
    table.caption = f"Página {page+1}/{total_pages} • Concluídos {done}/{len(items)}"
    return table

async def ui_loop():
    rows = max(8, console.size.height - 6)
    page = 0
    last_rotate = time.monotonic()
    with Live(build_table(page, rows), console=console, refresh_per_second=REFRESH_FPS, screen=True) as live:
        while not finished.is_set():
            timeout = max(0.1, 1.0 / REFRESH_FPS)
            try:
                await aio.wait_for(ui_dirty.wait(), timeout=timeout)
            except aio.TimeoutError:
                pass
            need_update = False
            if ui_dirty.is_set():
                ui_dirty.clear()
                need_update = True
            now = time.monotonic()
            if now - last_rotate >= PAGE_SECONDS:
                page = (page + 1) % max(1, math.ceil(len(stats)/rows))
                last_rotate = now
                need_update = True
            if need_update:
                live.update(build_table(page, rows))

# ────────────── Rate Limit ──────────────
class WeightGovernor:
    def __init__(self, target_per_min: int, safety: float = 0.9):
        self.target = int(target_per_min * safety)
        self.window_start = time.monotonic()
        self.used = 0
        self.last_header_used = 0
        self.lock = aio.Lock()

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self.window_start >= 60.0:
            self.window_start = now
            self.used = 0
            self.last_header_used = 0

    async def before(self, expected_weight: int):
        async with self.lock:
            self._maybe_reset()
            projected = self.used + expected_weight
            if projected > self.target:
                to_wait = max(0.0, 60.0 - (time.monotonic() - self.window_start))
                if to_wait > 0:
                    await aio.sleep(to_wait)
                    self._maybe_reset()
            self.used += expected_weight

    async def after(self, resp: aiohttp.ClientResponse):
        try:
            hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if hdr and hdr.isdigit():
                self.last_header_used = int(hdr)
                self.used = max(self.used, self.last_header_used)
        except Exception:
            pass

KLINES_WEIGHT = 2
governor = WeightGovernor(TARGET_WEIGHT_PER_MIN, SAFETY_MARGIN)

# ────────────── HTTP ──────────────
class BinanceHTTP:
    BASE = "https://api.binance.com"
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.sess: Optional[aiohttp.ClientSession] = None
        self.sem_total = aio.Semaphore(MAX_CONC_REQUESTS)

    async def __aenter__(self):
        if self.sess is None:
            timeout = aiohttp.ClientTimeout(sock_connect=20, sock_read=60, total=None)
            self.sess = aiohttp.ClientSession(
                timeout=timeout,
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else None,
                raise_for_status=False,
                connector=aiohttp.TCPConnector(limit_per_host=MAX_CONC_REQUESTS, ssl=None, ttl_dns_cache=300),
            )
        return self

    async def __aexit__(self, *exc):
        if self.sess:
            await self.sess.close()

    async def klines(self, symbol: str, interval: str, start_ms: int, limit: int = 1000) -> Tuple[int, List[list], Optional[int]]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": min(1000, max(1, int(limit))),
        }
        url = f"{self.BASE}/api/v3/klines"
        async with self.sem_total:
            await governor.before(KLINES_WEIGHT)
            resp = await self.sess.get(url, params=params)
            await governor.after(resp)
            status = resp.status
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = []
            used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            try:
                used = int(used) if used is not None else None
            except Exception:
                used = None
            return status, (data if isinstance(data, list) else []), used

# ────────────── MySQL (direto) ──────────────
def get_conn():
    if db_pool:
        return db_pool.get_connection()
    return mysql.connector.connect(**DB_CFG)

def ensure_table(sym: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS `{sym}` (
            dates BIGINT PRIMARY KEY,
            open_prices DOUBLE,
            high_prices DOUBLE,
            low_prices DOUBLE,
            closing_prices DOUBLE,
            volume DOUBLE
        ) ENGINE=InnoDB;
    """)
    conn.commit()
    cur.close()
    conn.close()

def mysql_max_date(sym: str) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT MAX(dates) FROM `{sym}`")
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()

def insert_batch(sym: str, rows: List[Tuple]) -> int:
    """Insere batch via INSERT IGNORE. Retorna quantas linhas foram inseridas."""
    if not rows:
        return 0
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.executemany(
            f"INSERT IGNORE INTO `{sym}` (dates, open_prices, high_prices, low_prices, closing_prices, volume) "
            f"VALUES (%s,%s,%s,%s,%s,%s)",
            rows
        )
        conn.commit()
        return cur.rowcount
    finally:
        cur.close()
        conn.close()

# ────────────── Download ──────────────
async def download_symbol(http: BinanceHTTP, sym: str):
    st = stats[sym]
    st.status = "🔄 Baixando"
    st.last = datetime.now().strftime("%H:%M:%S")
    ui_dirty.set()

    ensure_table(sym)

    # Obtém último timestamp do banco
    now_ms = int(time.time() * 1000)
    end_ms_closed = (now_ms // 60_000 - 1) * 60_000
    
    last_raw = mysql_max_date(sym)
    if last_raw is None:
        start_ts = EPOCH_TS
    else:
        # Normaliza timestamp (segundos -> ms se necessário)
        last = int(last_raw)
        if last < 10_000_000_000:
            last *= 1000
        if last > end_ms_closed * 10:
            last //= 1000
        if last > end_ms_closed:
            last = end_ms_closed - 60_000
        start_ts = last + 60_000

    if start_ts > end_ms_closed:
        st.status = "✅ Concluído"
        st.perc = 1.0
        st.last = datetime.now().strftime("%H:%M:%S")
        ui_dirty.set()
        return

    st.total_ms = max(1, (end_ms_closed - start_ts) + 60_000)

    # Controle de progresso e batch
    sem_sym = aio.Semaphore(PER_SYMBOL_CONC)
    curr_ts = start_ts
    done_ms = 0
    batch_lock = aio.Lock()
    pending_batch: List[Tuple] = []

    async def flush_batch():
        nonlocal pending_batch
        if pending_batch:
            try:
                await aio.get_event_loop().run_in_executor(None, insert_batch, sym, pending_batch.copy())
                pending_batch.clear()
            except Exception:
                pass

    async def fetch_and_insert(ts: int):
        nonlocal done_ms, curr_ts, pending_batch
        async with sem_sym:
            status, data, _ = await http.klines(sym, INTERVAL, ts, PAGE_LIMIT)
            if status == 429:
                await aio.sleep(1.0)
                return ts
            if status >= 500:
                await aio.sleep(0.5)
                return ts
            if not data:
                return None

            # Converte para tuplas
            rows = []
            for k in data:
                rows.append((int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))

            # Adiciona ao batch e flush se necessário
            async with batch_lock:
                pending_batch.extend(rows)
                if len(pending_batch) >= INSERT_BATCH_SIZE:
                    await flush_batch()

            last_open = int(data[-1][0])
            next_ts = last_open + 60_000
            
            # Atualiza progresso
            done_ms = (last_open - start_ts) + 60_000
            st.done_ms = done_ms
            st.perc = min(1.0, done_ms / st.total_ms)
            st.last = datetime.now().strftime("%H:%M:%S")
            ui_dirty.set()
            
            return next_ts

    # Pipeline cooperativo
    pending: List[aio.Task] = []
    for _ in range(PER_SYMBOL_CONC):
        if curr_ts <= end_ms_closed:
            pending.append(aio.create_task(fetch_and_insert(curr_ts)))
        curr_ts += 60_000 * PAGE_LIMIT

    while pending:
        done_tasks, pending = await aio.wait(pending, return_when=aio.FIRST_COMPLETED)
        for d in done_tasks:
            nxt = d.result()
            if nxt and nxt <= end_ms_closed:
                pending.add(aio.create_task(fetch_and_insert(nxt)))

    # Flush final
    await flush_batch()

    st.status = "✅ Concluído"
    st.perc = 1.0
    st.last = datetime.now().strftime("%H:%M:%S")
    ui_dirty.set()

# ────────────── MAIN ──────────────
async def amain():
    if not SYMBOLS:
        console.print("[yellow]Nenhum símbolo válido encontrado.[/]")
        return
    
    console.print(f"[bold]Símbolos:[/] {len(SYMBOLS)}  •  Concurrency={MAX_CONC_REQUESTS}  •  Batch={INSERT_BATCH_SIZE}")
    console.print("[green]Iniciando downloads (INSERT DIRETO no MySQL)...[/]")

    ui_task = aio.create_task(ui_loop())
    t0 = time.time()

    async with BinanceHTTP(API_KEY) as http:
        tasks = [aio.create_task(download_symbol(http, s)) for s in SYMBOLS]
        for coro in aio.as_completed(tasks):
            try:
                await coro
            except Exception as e:
                console.print(f"[red]Erro:[/] {e}")

    finished.set()
    try:
        await aio.wait_for(ui_task, timeout=1.0)
    except aio.TimeoutError:
        ui_task.cancel()

    console.print(f"\n[bold green]Todos concluídos em {(time.time()-t0)/60:.1f} min[/]")

def main():
    try:
        aio.run(amain())
    except KeyboardInterrupt:
        finished.set()

if __name__ == "__main__":
    main()
