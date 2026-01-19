# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Downloader de klines 1m (SPOT) para MySQL.

Notas:
- parâmetros em código (sem ENV)
- NÃO versionar segredos (API key/senha). Use placeholders e preencha localmente.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio as aio
import math
import time
import sys

import aiohttp
import mysql.connector
from mysql.connector import pooling

# permitir rodar como script direto (sem PYTHONPATH)
if __package__ in (None, ""):
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "modules":
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            break

try:
    from config.symbols import default_top_market_cap_path
except Exception:
    from config.symbols import default_top_market_cap_path  # type: ignore[import]

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich import box
except Exception as e:  # pragma: no cover
    raise RuntimeError("Este pipeline requer `rich` instalado (pip install rich).") from e


@dataclass
class MySQLConfig:
    host: str = "localhost"
    user: str = "root"
    password: str = "2017"
    database: str = "crypto"
    autocommit: bool = False
    pool_size: int = 16


@dataclass
class DownloadSettings:
    # Binance
    api_key: str = ""  # preencha localmente
    quote_symbols_file: str = ""  # vazio => usa modules/top_market_cap.txt

    # HTTP / RL
    max_conc_requests: int = 32
    per_symbol_conc: int = 2
    page_limit: int = 1000
    interval: str = "1m"
    epoch_ts_ms: int = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)

    target_weight_per_min: int = 5000
    safety_margin: float = 0.90

    # INSERT
    insert_batch_size: int = 500

    # UI
    refresh_fps: int = 8
    bar_len: int = 22
    page_seconds: float = 2.0

    # DB
    db: MySQLConfig = MySQLConfig()


def _repo_root() -> Path:
    # modules/binance -> modules
    return Path(__file__).resolve().parents[3]


def _default_symbols_file() -> Path:
    # centraliza num único lugar (compat: raiz ou data/)
    return default_top_market_cap_path()


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


class SymState:
    __slots__ = ("status", "perc", "last", "done_ms", "total_ms")

    def __init__(self):
        self.status = "Aguardando"
        self.perc = 0.0
        self.last = "--:--:--"
        self.done_ms = 0
        self.total_ms = 1


def build_table(stats: Dict[str, SymState], settings: DownloadSettings, page: int, rows: int) -> Table:
    table = Table("SÍMBOLO", "STATUS", "PROGRESSO", "HORA", box=box.ROUNDED, expand=True)
    items = list(stats.items())
    total_pages = max(1, math.ceil(len(items) / rows))
    page %= total_pages
    start, end = page * rows, page * rows + rows
    for sym, st in items[start:end]:
        filled = int(st.perc * settings.bar_len + 0.5)
        bar = "█" * filled + "─" * (settings.bar_len - filled)
        table.add_row(sym, st.status, f"[{bar}] {st.perc*100:5.1f}%", st.last)
    done = sum(1 for v in stats.values() if v.status.startswith("OK"))
    table.caption = f"Página {page+1}/{total_pages} • Concluídos {done}/{len(items)}"
    return table


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


class BinanceHTTP:
    BASE = "https://api.binance.com"

    def __init__(self, api_key: str, *, max_conc: int, governor: WeightGovernor):
        self.api_key = api_key
        self.sess: Optional[aiohttp.ClientSession] = None
        self.sem_total = aio.Semaphore(int(max_conc))
        self.governor = governor

    async def __aenter__(self):
        if self.sess is None:
            timeout = aiohttp.ClientTimeout(sock_connect=20, sock_read=60, total=None)
            self.sess = aiohttp.ClientSession(
                timeout=timeout,
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else None,
                raise_for_status=False,
                connector=aiohttp.TCPConnector(limit_per_host=int(self.sem_total._value), ssl=None, ttl_dns_cache=300),
            )
        return self

    async def __aexit__(self, *exc):
        if self.sess:
            await self.sess.close()

    async def klines(self, symbol: str, interval: str, start_ms: int, limit: int = 1000) -> Tuple[int, List[list], Optional[int]]:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": min(1000, max(1, int(limit)))}
        url = f"{self.BASE}/api/v3/klines"
        async with self.sem_total:
            await self.governor.before(2)
            resp = await self.sess.get(url, params=params)  # type: ignore[union-attr]
            await self.governor.after(resp)
            status = resp.status
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = []
            used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            try:
                used_i = int(used) if used is not None else None
            except Exception:
                used_i = None
            return status, (data if isinstance(data, list) else []), used_i


class MySQLStore:
    def __init__(self, cfg: MySQLConfig):
        self.cfg = cfg
        try:
            self.pool = pooling.MySQLConnectionPool(pool_name="binance_pool", pool_size=int(cfg.pool_size), **cfg.__dict__)
        except Exception:
            self.pool = None

    def get_conn(self):
        if self.pool:
            return self.pool.get_connection()
        return mysql.connector.connect(**self.cfg.__dict__)

    def ensure_table(self, sym: str):
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


async def download_symbol(
    http: BinanceHTTP,
    store: MySQLStore,
    stats: Dict[str, SymState],
    ui_dirty: aio.Event,
    sym: str,
    *,
    settings: DownloadSettings,
) -> None:
    st = stats[sym]
    st.status = "BAIXANDO"
    st.last = datetime.now().strftime("%H:%M:%S")
    ui_dirty.set()

    store.ensure_table(sym)

    now_ms = int(time.time() * 1000)
    end_ms_closed = (now_ms // 60_000 - 1) * 60_000

    last_raw = store.mysql_max_date(sym)
    if last_raw is None:
        start_ts = int(settings.epoch_ts_ms)
    else:
        last = int(last_raw)
        if last < 10_000_000_000:
            last *= 1000
        if last > end_ms_closed * 10:
            last //= 1000
        if last > end_ms_closed:
            last = end_ms_closed - 60_000
        start_ts = last + 60_000

    if start_ts > end_ms_closed:
        st.status = "OK"
        st.perc = 1.0
        st.last = datetime.now().strftime("%H:%M:%S")
        ui_dirty.set()
        return

    st.total_ms = max(1, (end_ms_closed - start_ts) + 60_000)

    sem_sym = aio.Semaphore(int(settings.per_symbol_conc))
    batch_lock = aio.Lock()
    pending_batch: List[Tuple] = []

    async def flush_batch():
        nonlocal pending_batch
        if pending_batch:
            batch_copy = pending_batch.copy()
            pending_batch.clear()
            await aio.get_event_loop().run_in_executor(None, store.insert_batch, sym, batch_copy)

    async def fetch_and_insert(ts: int) -> Optional[int]:
        nonlocal pending_batch
        async with sem_sym:
            status, data, _ = await http.klines(sym, settings.interval, ts, settings.page_limit)
            if status == 429:
                await aio.sleep(1.0)
                return ts
            if status >= 500:
                await aio.sleep(0.5)
                return ts
            if not data:
                return None

            rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in data]
            async with batch_lock:
                pending_batch.extend(rows)
                if len(pending_batch) >= int(settings.insert_batch_size):
                    await flush_batch()

            last_open = int(data[-1][0])
            next_ts = last_open + 60_000

            done_ms = (last_open - start_ts) + 60_000
            st.done_ms = done_ms
            st.perc = min(1.0, done_ms / st.total_ms)
            st.last = datetime.now().strftime("%H:%M:%S")
            ui_dirty.set()

            return next_ts

    # pipeline cooperativo (set desde o início para evitar confusão list/set)
    pending: set[aio.Task] = set()
    curr_ts = start_ts
    for _ in range(int(settings.per_symbol_conc)):
        if curr_ts <= end_ms_closed:
            pending.add(aio.create_task(fetch_and_insert(curr_ts)))
        curr_ts += 60_000 * int(settings.page_limit)

    while pending:
        done_tasks, pending = await aio.wait(pending, return_when=aio.FIRST_COMPLETED)
        for d in done_tasks:
            nxt = d.result()
            if nxt and nxt <= end_ms_closed:
                pending.add(aio.create_task(fetch_and_insert(int(nxt))))

    await flush_batch()

    st.status = "OK"
    st.perc = 1.0
    st.last = datetime.now().strftime("%H:%M:%S")
    ui_dirty.set()


async def ui_loop(
    console: Console,
    stats: Dict[str, SymState],
    *,
    settings: DownloadSettings,
    ui_dirty: aio.Event,
    finished: aio.Event,
) -> None:
    rows = max(8, console.size.height - 6)
    page = 0
    last_rotate = time.monotonic()
    with Live(build_table(stats, settings, page, rows), console=console, refresh_per_second=int(settings.refresh_fps), screen=True) as live:
        while not finished.is_set():
            timeout = max(0.1, 1.0 / max(1, int(settings.refresh_fps)))
            try:
                await aio.wait_for(ui_dirty.wait(), timeout=timeout)
            except aio.TimeoutError:
                pass
            need_update = False
            if ui_dirty.is_set():
                ui_dirty.clear()
                need_update = True
            now = time.monotonic()
            if now - last_rotate >= float(settings.page_seconds):
                page = (page + 1) % max(1, math.ceil(len(stats) / rows))
                last_rotate = now
                need_update = True
            if need_update:
                live.update(build_table(stats, settings, page, rows))


async def amain(settings: DownloadSettings) -> None:
    console = Console()
    symbols_file = settings.quote_symbols_file or str(_default_symbols_file())
    symbols = load_symbols(symbols_file)
    if not symbols:
        console.print("[yellow]Nenhum símbolo válido encontrado.[/]")
        return

    if not settings.api_key:
        console.print("[red]API key vazia. Preencha DownloadSettings.api_key antes de rodar.[/]")
        return

    console.print(f"[bold]Símbolos:[/] {len(symbols)}  •  Concurrency={settings.max_conc_requests}  •  Batch={settings.insert_batch_size}")
    console.print("[green]Iniciando downloads (INSERT DIRETO no MySQL)...[/]")

    stats: Dict[str, SymState] = {s: SymState() for s in symbols}
    ui_dirty = aio.Event()
    finished = aio.Event()

    governor = WeightGovernor(int(settings.target_weight_per_min), float(settings.safety_margin))
    store = MySQLStore(settings.db)

    ui_task = aio.create_task(ui_loop(console, stats, settings=settings, ui_dirty=ui_dirty, finished=finished))
    t0 = time.time()

    async with BinanceHTTP(settings.api_key, max_conc=int(settings.max_conc_requests), governor=governor) as http:
        tasks = [aio.create_task(download_symbol(http, store, stats, ui_dirty, s, settings=settings)) for s in symbols]
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


def run(settings: DownloadSettings | None = None) -> None:
    settings = settings or DownloadSettings()
    aio.run(amain(settings))


def main() -> None:
    # Edite aqui (sem ENV):
    settings = DownloadSettings(
        api_key="",
        quote_symbols_file="",  # "" => usa top_market_cap.txt
        db=MySQLConfig(host="localhost", user="root", password="2017", database="crypto"),
    )
    run(settings)


if __name__ == "__main__":
    main()

