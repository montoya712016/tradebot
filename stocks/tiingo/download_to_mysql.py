# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Downloader de candles 1m (Tiingo IEX) para MySQL.

Notas:
- Usa TIINGO_API_KEY do env ou modules/config/secrets.py
- Salva OHLCV em tabelas por símbolo (dates = epoch ms)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio as aio
import math
import time
import sys
import os
import json

import aiohttp
import mysql.connector
from mysql.connector import pooling

# permitir rodar como script direto (ajusta sys.path para repo root/modules)
def _add_repo_paths() -> None:
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if p.name.lower() == "tradebot":
            root = p
            break
    if root:
        for cand in (root, root / "modules"):
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)

if __package__ in (None, ""):
    _add_repo_paths()

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich import box
except Exception as e:  # pragma: no cover
    raise RuntimeError("Este pipeline requer `rich` instalado (pip install rich).") from e


TIINGO_BASE = "https://api.tiingo.com"


@dataclass
class MySQLConfig:
    host: str = "localhost"
    user: str = "root"
    password: str = "2017"
    database: str = "stocks_us"
    autocommit: bool = False
    pool_size: int = 16


@dataclass
class DownloadSettings:
    # Tiingo
    symbols_file: str = "data/generated/tiingo_universe_seed.txt"
    resample_freq: str = "1min"
    columns: str = "date,open,high,low,close,volume"
    years_back: int = 5
    chunk_days: int = 7
    chunk_days_min: int = 3
    chunk_days_max: int = 90
    chunk_days_cache_path: str = "D:/MySQL/tiingo_chunk_days.json"

    # HTTP / RL
    max_conc_requests: int = 6
    per_symbol_conc: int = 1
    sleep_between_requests: float = 0.0
    rate_limit_sleep: float = 30.0

    # INSERT
    insert_batch_size: int = 500

    # UI
    refresh_fps: int = 8
    bar_len: int = 22
    page_seconds: float = 2.0

    # Logs
    log_path: str = "D:/MySQL/tiingo_download.log"

    # DB
    db: MySQLConfig = field(default_factory=MySQLConfig)
    create_db_if_missing: bool = True


def _load_tiingo_key() -> str:
    key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if key:
        return key
    try:
        from config import secrets as sec  # type: ignore

        key = str(getattr(sec, "TIINGO_API_KEY", "") or "").strip()
        return key
    except Exception:
        return ""


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name.lower() == "tradebot":
            return p
    return Path.cwd()


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(_repo_root() / path)


def load_symbols(path: str) -> List[str]:
    try:
        lines = Path(_resolve_path(path)).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out, seen = [], set()
    for ln in lines:
        s = ln.strip().upper()
        if not s or s.startswith("#"):
            continue
        s = s.replace(".", "-")
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


class SymState:
    __slots__ = ("status", "perc", "last", "done", "total")

    def __init__(self):
        self.status = "Aguardando"
        self.perc = 0.0
        self.last = "--:--:--"
        self.done = 0
        self.total = 1


def _estimate_trading_minutes(start_d: date, end_d: date) -> int:
    # conta dias úteis (ignora feriados) * 390 minutos
    if end_d < start_d:
        return 0
    days = 0
    d = start_d
    while d <= end_d:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    return int(days * 390)


def build_table(stats: Dict[str, SymState], settings: DownloadSettings, page: int, rows: int) -> Table:
    table = Table("SÍMBOLO", "STATUS", "PROGRESSO", "HORA", box=box.ROUNDED, expand=True)
    items = list(stats.items())
    total_pages = max(1, math.ceil(len(items) / rows))
    page %= total_pages
    start, end = page * rows, page * rows + rows
    for sym, st in items[start:end]:
        filled = int(st.perc * settings.bar_len + 0.5)
        bar = "█" * filled + "░" * (settings.bar_len - filled)
        table.add_row(sym, st.status, f"[{bar}] {st.perc*100:5.1f}%", st.last)
    done = sum(1 for v in stats.values() if v.status.startswith("OK"))
    table.caption = f"Página {page+1}/{total_pages} • Concluídos {done}/{len(items)}"
    return table


class TiingoHTTP:
    def __init__(self, api_key: str, *, max_conc: int):
        self.api_key = api_key
        self.sess: Optional[aiohttp.ClientSession] = None
        self.sem_total = aio.Semaphore(int(max_conc))

    async def __aenter__(self):
        if self.sess is None:
            timeout = aiohttp.ClientTimeout(sock_connect=20, sock_read=60, total=None)
            self.sess = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Authorization": f"Token {self.api_key}"},
                raise_for_status=False,
                connector=aiohttp.TCPConnector(limit_per_host=int(self.sem_total._value), ssl=None, ttl_dns_cache=300),
            )
        return self

    async def __aexit__(self, *exc):
        if self.sess:
            await self.sess.close()

    async def prices(self, symbol: str, start: str, end: str, *, resample: str, columns: str) -> Tuple[int, List[dict]]:
        params = {
            "startDate": start,
            "endDate": end,
            "resampleFreq": resample,
            "columns": columns,
        }
        url = f"{TIINGO_BASE}/iex/{symbol}/prices"
        async with self.sem_total:
            resp = await self.sess.get(url, params=params)  # type: ignore[union-attr]
            status = resp.status
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = []
            return status, (data if isinstance(data, list) else [])


class MySQLStore:
    def __init__(self, cfg: MySQLConfig):
        self.cfg = cfg
        try:
            self.pool = pooling.MySQLConnectionPool(pool_name="tiingo_pool", pool_size=int(cfg.pool_size), **cfg.__dict__)
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


def _parse_ts_ms(s: str) -> Optional[int]:
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _log_line(path: str, msg: str) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


def _load_chunk_cache(path: str) -> Optional[int]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        v = int(data.get("chunk_days", 0))
        return v if v > 0 else None
    except Exception:
        return None


def _save_chunk_cache(path: str, chunk_days: int) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"chunk_days": int(chunk_days)}), encoding="utf-8")
    except Exception:
        pass


def ensure_database(cfg: MySQLConfig, *, log_path: str) -> None:
    if not cfg.database:
        return
    try:
        base_cfg = {**cfg.__dict__}
        base_cfg.pop("database", None)
        conn = mysql.connector.connect(**base_cfg)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg.database}`")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        _log_line(log_path, f"[ERR] ensure_database: {type(exc).__name__}: {exc}")
        raise


async def download_symbol(
    http: TiingoHTTP,
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

    end_d = date.today()
    base_start = end_d - timedelta(days=int(settings.years_back) * 365)
    start_d = base_start
    total_minutes = _estimate_trading_minutes(base_start, end_d)
    st.total = max(1, total_minutes)

    last_raw = store.mysql_max_date(sym)
    if last_raw:
        last_ms = int(last_raw)
        if last_ms < 10_000_000_000:
            last_ms *= 1000
        last_dt = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc)
        # Se já temos dados até (ou além) de hoje, evita requisições desnecessárias
        if last_dt.date() >= end_d:
            st.status = "OK"
            st.perc = 1.0
            st.last = datetime.now().strftime("%H:%M:%S")
            ui_dirty.set()
            _log_line(settings.log_path, f"[SKIP] {sym} já atualizado até {last_dt.date().isoformat()}")
            return
        if last_dt.date() >= base_start:
            covered = _estimate_trading_minutes(base_start, min(last_dt.date(), end_d))
            st.done = min(st.total, covered)
            st.perc = min(1.0, st.done / st.total)
            st.last = datetime.now().strftime("%H:%M:%S")
            ui_dirty.set()
        # começa no dia seguinte ao último candle para não repetir range
        start_d = max(start_d, last_dt.date() + timedelta(days=1))

    cached_chunk = _load_chunk_cache(settings.chunk_days_cache_path)
    if cached_chunk:
        chunk = cached_chunk
    else:
        chunk = max(1, int(settings.chunk_days))
    chunk = max(int(settings.chunk_days_min), min(int(settings.chunk_days_max), int(chunk)))

    pending_batch: List[Tuple] = []
    batch_lock = aio.Lock()

    async def flush_batch():
        nonlocal pending_batch
        if pending_batch:
            batch_copy = pending_batch.copy()
            pending_batch.clear()
            await aio.get_event_loop().run_in_executor(None, store.insert_batch, sym, batch_copy)

    async def fetch_with_adapt(d0: date, d1: date) -> Tuple[int, List[dict], int]:
        """
        Tenta baixar um intervalo com chunk adaptativo.
        Reduz o chunk em caso de erro/timeout e grava o melhor chunk encontrado.
        """
        nonlocal chunk
        while True:
            d2 = min(end_d, d0 + timedelta(days=chunk))
            try:
                status, data = await http.prices(
                    sym,
                    d0.isoformat(),
                    d2.isoformat(),
                    resample=settings.resample_freq,
                    columns=settings.columns,
                )
            except Exception:
                status, data = 0, []

            if status in (200, 204):
                _save_chunk_cache(settings.chunk_days_cache_path, chunk)
                return status, data, chunk

            # se bateu limite ou erro, reduz o chunk e tenta novamente
            if chunk > int(settings.chunk_days_min):
                chunk = max(int(settings.chunk_days_min), chunk // 2)
                _log_line(settings.log_path, f"[CHUNK] {sym} reduzindo para {chunk}d (status={status})")
                await aio.sleep(float(settings.sleep_between_requests))
                continue

            return status, data, chunk

    d = start_d
    while d <= end_d:
        status, data, used_chunk = await fetch_with_adapt(d, end_d)
        if status == 429:
            now_str = datetime.now().strftime("%H:%M:%S")
            for v in stats.values():
                if v.status.startswith("OK"):
                    continue
                v.status = "AGUARDANDO"
                v.last = now_str
            ui_dirty.set()
            await aio.sleep(float(settings.rate_limit_sleep))
            st.status = "BAIXANDO"
            st.last = datetime.now().strftime("%H:%M:%S")
            ui_dirty.set()
            continue
        if status >= 500:
            await aio.sleep(0.5)
            continue

        rows = []
        for r in data or []:
            ts_ms = _parse_ts_ms(str(r.get("date") or ""))
            if ts_ms is None:
                continue
            rows.append(
                (
                    ts_ms,
                    float(r.get("open") or 0.0),
                    float(r.get("high") or 0.0),
                    float(r.get("low") or 0.0),
                    float(r.get("close") or 0.0),
                    float(r.get("volume") or 0.0),
                )
            )

        async with batch_lock:
            pending_batch.extend(rows)
            if len(pending_batch) >= int(settings.insert_batch_size):
                await flush_batch()

        # progresso (aprox)
        st.done = min(st.total, st.done + 390 * used_chunk)
        st.perc = min(1.0, st.done / st.total)
        st.last = datetime.now().strftime("%H:%M:%S")
        ui_dirty.set()

        if settings.sleep_between_requests:
            await aio.sleep(float(settings.sleep_between_requests))

        d = d + timedelta(days=used_chunk + 1)

    await flush_batch()

    st.status = "OK"
    st.perc = 1.0
    st.last = datetime.now().strftime("%H:%M:%S")
    ui_dirty.set()
    _log_line(settings.log_path, f"[OK] {sym}")


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
    symbols = load_symbols(settings.symbols_file)
    if not symbols:
        console.print("[yellow]Nenhum símbolo válido encontrado.[/]")
        return

    api_key = _load_tiingo_key()
    if not api_key:
        console.print("[red]TIINGO_API_KEY ausente (env ou modules/config/secrets.py).[/]")
        return

    console.print(
        f"[bold]Símbolos:[/] {len(symbols)}  •  Concurrency={settings.max_conc_requests}  •  Batch={settings.insert_batch_size}"
    )
    console.print("[green]Iniciando downloads (INSERT DIRETO no MySQL)...[/]")

    stats: Dict[str, SymState] = {s: SymState() for s in symbols}
    ui_dirty = aio.Event()
    finished = aio.Event()

    if settings.create_db_if_missing:
        try:
            ensure_database(settings.db, log_path=settings.log_path)
        except Exception as exc:
            console.print(f"[red]Erro criando database:[/] {exc}")
            return

    store = MySQLStore(settings.db)

    ui_task = aio.create_task(ui_loop(console, stats, settings=settings, ui_dirty=ui_dirty, finished=finished))
    t0 = time.time()

    async with TiingoHTTP(api_key, max_conc=int(settings.max_conc_requests)) as http:
        tasks = [aio.create_task(download_symbol(http, store, stats, ui_dirty, s, settings=settings)) for s in symbols]
        for coro in aio.as_completed(tasks):
            try:
                await coro
            except Exception as e:
                console.print(f"[red]Erro:[/] {e}")
                _log_line(settings.log_path, f"[ERR] {e}")

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
    # Edite aqui se quiser ajustes rápidos
    settings = DownloadSettings(
        symbols_file=str(_resolve_path("data/generated/tiingo_universe_seed.txt")),
        db=MySQLConfig(host="localhost", user="root", password="2017", database="stocks_us"),
    )
    run(settings)


if __name__ == "__main__":
    main()
