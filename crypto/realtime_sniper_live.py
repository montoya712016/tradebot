# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Realtime Sniper decision bot (stage 0):
- Collects 1m klines (REST bootstrap + WebSocket live)
- Keeps a rolling OHLC window in memory
- Computes features for the latest closed candle
- Runs the latest model to emit BUY decisions
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import requests
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Optional, Tuple
import sys
import os


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

import numpy as np
import pandas as pd
import websocket
import xgboost as xgb
from typing import TYPE_CHECKING

# logging simples
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("realtime")
# silencia verbosidade do mysql.connector
logging.getLogger("mysql").setLevel(logging.WARNING)
logging.getLogger("mysql.connector").setLevel(logging.WARNING)
logging.getLogger("mysql.connector.plugins").setLevel(logging.WARNING)

from crypto.realtime_bot import (
    MySQLConfig,
    MySQLStore,
    RestBackfillQueue,
    _last_closed_minute_ms,
    _normalize_last_ts,
    load_symbols,
)
try:
    from modules.config.symbols import default_top_market_cap_path
except Exception:
    from config.symbols import default_top_market_cap_path  # type: ignore[import]

try:
    from modules.utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore
except Exception:
    try:
        from utils.pushover_notify import load_default as _pushover_load_default, send_pushover as _pushover_send  # type: ignore
    except Exception:  # pragma: no cover
        _pushover_load_default = None
        _pushover_send = None

# Imports do core (executando com sys.path já ajustado)
from modules.backtest.sniper_simulator import _apply_calibration
from modules.backtest.sniper_walkforward import PeriodModel, load_period_models
from modules.prepare_features.prepare_features import FEATURE_KEYS, build_flags, run as pf_run
from modules.prepare_features.data import load_ohlc_1m_series
from modules.prepare_features import pf_config
from modules.trade_contract import DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
from modules.config.symbols import load_top_market_cap_symbols

# Downloader rápido (opcional)
try:
    from crypto.binance.download_to_mysql import (
        DownloadSettings as DLSettings,
        MySQLConfig as DLMySQLConfig,
        run as dl_run,
    )
except Exception:
    DLSettings = None  # type: ignore
    DLMySQLConfig = None  # type: ignore
    dl_run = None  # type: ignore

# secrets (ngrok, etc.)
try:
    from modules.config import secrets as _secrets  # type: ignore
except Exception:
    try:
        from config import secrets as _secrets  # type: ignore
    except Exception:
        _secrets = None  # type: ignore

# Estado para dashboard (para limpar mocks)
try:
    from modules.realtime.dashboard_state import DashboardState, AccountSummary  # type: ignore
except Exception:
    try:
        from realtime.dashboard_state import DashboardState, AccountSummary  # type: ignore
    except Exception:
        DashboardState = None  # type: ignore
        AccountSummary = None  # type: ignore

# Ngrok monolith helpers (domínio custom, basic auth)
try:
    from modules.realtime.realtime_dashboard_ngrok_monolith import NgrokConfig as NgrokCfg, NgrokManager as NgrokMgr  # type: ignore
except Exception:
    try:
        from realtime.realtime_dashboard_ngrok_monolith import NgrokConfig as NgrokCfg, NgrokManager as NgrokMgr  # type: ignore
    except Exception:
        NgrokCfg = None  # type: ignore
        NgrokMgr = None  # type: ignore


def _max_pf_window_minutes() -> int:
    values: List[int] = []
    for name in dir(pf_config):
        if not name.endswith("_MIN") and not name.endswith("_WINDOWS"):
            continue
        val = getattr(pf_config, name)
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, (list, tuple)) and item:
                    values.append(int(max(item)))
                else:
                    values.append(int(item))
    return max(values) if values else 1440


def _window_days_for_minutes(minutes: int) -> int:
    return int(max(1, int(np.ceil(minutes / 1440.0)) + 1))


def _feature_window_minutes() -> int:
    # quantidade de minutos necessária para calcular todas as features
    return _max_pf_window_minutes()


def _make_feature_flags() -> Dict[str, bool]:
    return build_flags(enable=FEATURE_KEYS, label=False)


def _safe_last_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    try:
        return df.iloc[-1]
    except Exception:
        return None


@dataclass
class LiveSettings:
    run_dir: str = "D:/astra/models_sniper/crypto/wf_002"
    symbols_file: str = ""
    quote_symbols_fallback: str = ""
    bootstrap_days: int = 7
    window_minutes: int = 0  # 0 => usa a janela mínima necessária pelas features
    min_ready_rows: int = 0  # 0 => usa a janela mínima das features
    min_market_cap_usd: float = 50_000_000.0
    use_danger_filter: bool = False
    # alocação / sizing (reflete sweep_003_20260123_102023)
    max_positions: int = 27
    total_exposure: float = 1.0
    max_trade_exposure: float = 0.07
    min_trade_exposure: float = 0.04
    exit_confirm_bars: int = 1
    tau_entry_override: float = 0.775

    ws_chunk: int = 100
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10

    rest_timeout_sec: float = 20.0
    rest_retry_sleep_sec: float = 0.5
    gap_workers: int = 2
    interval: str = "1m"
    page_limit: int = 1000

    db: MySQLConfig = field(default_factory=MySQLConfig)
    # trading
    trade_mode: str = "paper"  # "live" | "paper"
    trade_notional_usd: float = 100.0
    paper_start_equity: float = 10_000.0
    trade_fee_rate: float = 0.001  # 0.1% por trade (binance spot)
    # dashboard embedding
    start_dashboard: bool = True
    dashboard_port: int = 5055
    start_ngrok: bool = True  # tenta subir túnel ngrok como no monolito
    ngrok_authtoken_env: str = "NGROK_AUTHTOKEN"
    pushover_on: bool = True
    # backfill rápido (usa downloader async)
    use_fast_backfill: bool = True
    fast_max_conc_requests: int = 16
    fast_per_symbol_conc: int = 2
    fast_page_limit: int = 1000
    fast_target_weight: int = 5000
    fast_safety: float = 0.90
    status_every_sec: int = 10
    dashboard_push_every_sec: int = 1
    score_workers: int = 32
    symbols_limit: int = 0  # 0 => todos
    ngrok_domain: str = ""  # opcional (pago)
    ngrok_basic_user: str = ""
    ngrok_basic_pass: str = ""
    init_workers: int = 8
    prime_push_top: int = 500


@dataclass
class ModelBundle:
    model: xgb.Booster
    feature_cols: List[str]
    calib: dict
    tau_entry: float
    predictor: str = "cpu_predictor"


class PaperExecutor:
    """
    Executor de simulação: não envia ordens reais, apenas atualiza PnL virtual.
    """

    def __init__(self, cash_usd: float, fee_rate: float = 0.0):
        self.cash = float(cash_usd)
        self.fee_rate = float(fee_rate)
        self.positions: Dict[str, Dict[str, float]] = {}  # sym -> {qty, avg}
        self.trades: List[dict] = []

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        qty = float(notional_usd) / float(price)
        fee = float(notional_usd) * self.fee_rate
        total_cost = float(notional_usd) + fee
        if self.cash < total_cost:
            # Sem caixa: não faz nada
            return {"ok": False, "reason": "no_cash"}
        self.cash -= total_cost
        pos = self.positions.get(symbol, {"qty": 0.0, "avg": 0.0})
        old_qty, old_avg = float(pos["qty"]), float(pos["avg"])
        new_qty = old_qty + qty
        new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty if new_qty > 0 else price
        self.positions[symbol] = {"qty": new_qty, "avg": new_avg}
        tr = {
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "price": price,
            "notional": notional_usd,
            "fee": fee,
        }
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def close(self, symbol: str, price: float) -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"ok": False, "reason": "no_position"}
        qty = float(pos.get("qty", 0.0))
        avg = float(pos.get("avg", 0.0))
        gross_proceeds = qty * price
        fee = gross_proceeds * self.fee_rate
        pnl = gross_proceeds - fee - (qty * avg)
        self.cash += gross_proceeds - fee
        self.positions.pop(symbol, None)
        tr = {
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": price,
            "notional": gross_proceeds,
            "fee": fee,
            "pnl": pnl,
        }
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions and abs(float(self.positions[symbol].get("qty", 0.0))) > 0

    def short(self, symbol: str, price: float, notional_usd: float) -> dict:
        qty = float(notional_usd) / float(price)
        self.cash += float(notional_usd)
        pos = self.positions.get(symbol, {"qty": 0.0, "avg": 0.0})
        old_qty, old_avg = float(pos["qty"]), float(pos["avg"])
        new_qty = old_qty - qty
        # Para short, avg fica no preço da venda
        self.positions[symbol] = {"qty": new_qty, "avg": price}
        tr = {"symbol": symbol, "side": "SELL", "qty": qty, "price": price, "notional": notional_usd}
        self.trades.append(tr)
        return {"ok": True, "trade": tr}

    def equity(self, price_map: Dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            px = float(price_map.get(sym, 0.0) or 0.0)
            eq += float(pos["qty"]) * px
        return eq


class LiveExecutor:
    """
    Executor real (Binance). Depende de credenciais via env.
    """

    def __init__(self, notify: bool = False):
        try:
            from crypto.trading_client import BinanceTrader  # lazy import
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"BinanceTrader indisponível: {e}")
        self.trader = BinanceTrader()
        self.notify = bool(notify)

    def buy(self, symbol: str, price: float, notional_usd: float) -> dict:
        res = self.trader.market_buy_usdt(symbol, float(notional_usd), margin=False, leverage=1.0, notify=self.notify)
        return {"ok": True, "trade": res.__dict__}

    def short(self, symbol: str, price: float, notional_usd: float) -> dict:
        res = self.trader.market_short_usdt(symbol, float(notional_usd), notify=self.notify)
        return {"ok": True, "trade": res.__dict__}


class RollingWindow:
    def __init__(self, max_rows: int, min_ready_rows: int):
        self.max_rows = int(max(1, max_rows))
        self.min_ready_rows = int(max(1, min_ready_rows))
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
        with self.lock:
            if self.last_ts is not None and ts_ms <= self.last_ts:
                if ts_ms == self.last_ts and self.rows:
                    self.rows[-1] = (ts_ms, o, h, l, c, v)
                    return False
                return False
            if self.last_ts is not None and ts_ms > self.last_ts + 60_000:
                self.need_reload = True
            self.rows.append((ts_ms, o, h, l, c, v))
            if len(self.rows) > self.max_rows:
                self.rows = self.rows[-self.max_rows :]
            self.last_ts = ts_ms
            return True

    def replace_from_df(self, df: pd.DataFrame) -> None:
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


class LiveDecisionBot:
    def __init__(self, settings: LiveSettings):
        self.settings = settings
        self.feature_flags = _make_feature_flags()
        self.model_bundle = self._load_model_bundle()
        self.store = MySQLStore(settings.db)
        self.backfill = RestBackfillQueue(self.store, settings)
        self._backfill_started = False
        self.symbols = self._load_symbols()
        self.windows: Dict[str, RollingWindow] = {}
        self.decisions: Dict[str, int] = {}
        self.symbol_scores: Dict[str, dict] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.stop_evt = threading.Event()
        self.start_equity_usd: Optional[float] = None
        self._init_executor()
        try:
            self.start_equity_usd = float(getattr(self.executor, "cash", 0.0))
        except Exception:
            self.start_equity_usd = float(self.settings.paper_start_equity)
        self.dashboard_proc: Optional[threading.Thread] = None
        self.dash_url: Optional[str] = None
        self._pushover_warned = False
        self.trades: List[dict] = []
        self.open_positions: Dict[str, dict] = {}
        self.equity_history: List[dict] = []
        self.equity_history_max: int = 5000
        self.last_dashboard_push = 0.0
        self.ws_msg_count = 0
        self.ws_last_ts: Optional[int] = None
        self._score_fail_syms: set[str] = set()
        self._score_logged_syms: set[str] = set()
        self._init_scores_done = False
        self._score_refresh_thread: Optional[threading.Thread] = None
        self._score_keepalive_thread: Optional[threading.Thread] = None
        self._short_backfill_requested: set[str] = set()
        # cache de features por símbolo (evita recalcular para o mesmo ts)
        self._feat_cache: Dict[str, dict] = {}
        self._log_params()
        self._load_persisted_state()

    # -------------------- Persistência simples (paper) -------------------- #
    def _load_persisted_state(self) -> None:
        """
        Recarrega trades/equity/posições de um snapshot salvo para modo paper.
        """
        try:
            path = Path("data") / "state_live.json"
            if not path.exists():
                return
            raw = json.loads(path.read_text(encoding="utf-8"))
            eq_hist = raw.get("equity_history") or []
            trades = raw.get("trades") or []
            open_pos = raw.get("open_positions") or {}
            paper_state = raw.get("paper") or {}
            if eq_hist:
                self.equity_history = eq_hist[-self.equity_history_max :]
                if self.start_equity_usd is None:
                    try:
                        self.start_equity_usd = float(self.equity_history[0].get("equity_usd", 0.0))
                    except Exception:
                        pass
            if trades:
                self.trades = trades[-200:]
            if open_pos:
                self.open_positions = open_pos
            if isinstance(self.executor, PaperExecutor) and paper_state:
                cash = paper_state.get("cash")
                positions = paper_state.get("positions") or {}
                if cash is not None:
                    self.executor.cash = float(cash)
                # garante floats
                fixed = {}
                for sym, st in positions.items():
                    try:
                        fixed[sym] = {"qty": float(st.get("qty", 0.0)), "avg": float(st.get("avg", 0.0))}
                    except Exception:
                        continue
                if fixed:
                    self.executor.positions = fixed
        except Exception:
            log.warning("[persist] falha ao carregar state_live.json", exc_info=True)

    def _persist_state(self) -> None:
        """
        Persiste trades/equity/posições em disco (modo paper).
        """
        try:
            snapshot = {
                "equity_history": self.equity_history[-self.equity_history_max :],
                "trades": self.trades[-200:],
                "open_positions": self.open_positions,
                "paper": {},
            }
            if isinstance(self.executor, PaperExecutor):
                snapshot["paper"] = {
                    "cash": float(getattr(self.executor, "cash", 0.0)),
                    "positions": self.executor.positions,
                }
            path = Path("data") / "state_live.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            log.warning("[persist] falha ao salvar state_live.json", exc_info=True)

    def _load_window(self, sym: str, max_rows: int, min_rows: int) -> Optional[RollingWindow]:
        try:
            self.store.ensure_table(sym)
            limit = max_rows + 5
            conn = self.store.get_conn()
            cur = conn.cursor()
            cur.execute(
                f"SELECT dates, open_prices, high_prices, low_prices, closing_prices, volume "
                f"FROM `{sym}` ORDER BY dates DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            rows = list(reversed(rows))
            win = RollingWindow(max_rows=max_rows, min_ready_rows=min_rows)
            for r in rows:
                try:
                    win.ingest(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                except Exception:
                    continue
            return win
        except Exception as e:
            log.warning("[init][%s] falha ao carregar janela: %s: %s", sym, type(e).__name__, e)
            return None

    def _backfill_short_history(self, symbols: List[str], window_minutes: int, max_rows: int, min_rows: int) -> None:
        if not symbols:
            return
        self._start_backfill_workers()
        end_ms = _last_closed_minute_ms()
        # usa apenas a janela necessÃ¡ria para as features
        feature_minutes = _feature_window_minutes()
        backfill_minutes = max(feature_minutes, min_rows, window_minutes)
        start_ms = max(0, int(end_ms - int(backfill_minutes) * 60_000))
        # evita repetir enfileiramento para os mesmos sÃ­mbolos
        symbols = [s for s in symbols if s not in self._short_backfill_requested]
        if not symbols:
            return
        self._short_backfill_requested.update(symbols)
        log.info(
            "[init] backfill para janelas curtas: %s simbolos (range=%s .. %s)",
            len(symbols),
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(start_ms / 1000)),
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(end_ms / 1000)),
        )
        for sym in symbols:
            self.backfill.enqueue(sym, start_ms, end_ms)
        self.backfill.q.join()
        reloaded = 0
        for sym in symbols:
            win = self._load_window(sym, max_rows, min_rows)
            if win is not None:
                self.windows[sym] = win
                reloaded += 1
        ready = sum(1 for s in self.symbols if self.windows.get(s) and self.windows[s].is_ready())
        log.info("[init] backfill concluido: recarregadas=%s janelas prontas=%s/%s", reloaded, ready, len(self.symbols))

    def _upsert_symbol_score(self, sym: str, res: Optional[dict]) -> None:
        """
        Atualiza o dicionário de scores mesmo sem sinal de compra, usando o close atual.
        """
        if res is None:
            win = self.windows.get(sym)
            if not win or not win.is_ready():
                return
            ts = win.peek_last_ts()
            df = win.to_frame()
            if df.empty or ts is None:
                return
            price = float(df["close"].iloc[-1])
            res = {"symbol": sym, "p_entry": 0.0, "price": price, "ts_ms": int(ts)}
        try:
            self.symbol_scores[sym] = {
                "symbol": sym,
                "score": float(res.get("p_entry", 0.0)),
                "price": float(res.get("price", 0.0)),
                "ts_ms": int(res.get("ts_ms", 0) or 0),
                "range_1h_pct": float(res.get("range_1h_pct", 0.0)),
                "range_24h_pct": float(res.get("range_24h_pct", 0.0)),
            }
        except Exception:
            pass

    def _init_executor(self) -> None:
        mode = str(self.settings.trade_mode or "paper").lower()
        if mode == "live":
            self.executor = LiveExecutor(notify=bool(self.settings.pushover_on))
        else:
            self.executor = PaperExecutor(self.settings.paper_start_equity, fee_rate=float(self.settings.trade_fee_rate))
            log.info("[paper] start equity=%s fee=%.4f", self.settings.paper_start_equity, float(self.settings.trade_fee_rate))

    def _load_symbols(self) -> List[str]:
        if self.settings.symbols_file:
            syms = load_symbols(self.settings.symbols_file)
        else:
            fallback = self.settings.quote_symbols_fallback or str(default_top_market_cap_path())
            min_cap = float(self.settings.min_market_cap_usd or 0.0)
            syms = load_top_market_cap_symbols(
                path=fallback,
                min_cap=min_cap if min_cap > 0 else None,
            )
            log.info("[symbols] carregados por market_cap: %s (min_cap=%.0f)", len(syms), min_cap)
        if not syms:
            raise RuntimeError("no symbols found for live bot")
        if int(self.settings.symbols_limit) > 0:
            syms = syms[: int(self.settings.symbols_limit)]
            log.info("[symbols] limit=%s -> usando %s símbolos", self.settings.symbols_limit, len(syms))
        return syms

    def _load_model_bundle(self) -> ModelBundle:
        run_dir = Path(self.settings.run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise RuntimeError(f"run_dir não encontrado: {run_dir}")
        log.info("[model] carregando modelos de %s", run_dir)
        periods = load_period_models(run_dir)
        if not periods:
            raise RuntimeError("no period models found in run_dir")
        # preferir o período mais recente (0d) se existir
        pm: PeriodModel = next((p for p in periods if getattr(p, "period_days", 1e9) == 0), periods[0])
        log.info("[model] periodo selecionado: %sd (train_end=%s)", pm.period_days, pm.train_end_utc)
        feat_cols = list(pm.entry_cols_map.get("mid", pm.entry_cols))
        calib = dict(pm.entry_calib_map.get("mid", pm.entry_calib))
        tau_entry = float(self.settings.tau_entry_override or 0.0)
        if tau_entry <= 0.0:
            tau_entry = float(pm.tau_entry_map.get("mid", pm.tau_entry))
        booster = pm.entry_models.get("mid", pm.entry_model)
        predictor = self._enable_gpu_predictor(booster, feat_cols)
        return ModelBundle(
            model=booster,
            feature_cols=feat_cols,
            calib=calib,
            tau_entry=tau_entry,
            predictor=predictor,
        )

    def _enable_gpu_predictor(self, booster: xgb.Booster, feat_cols: List[str]) -> str:
        """
        Tenta usar GPU (CUDA) para inferência. Se falhar, cai para CPU.
        """
        try:
            booster.set_param({"predictor": "gpu_predictor"})
            dummy = xgb.DMatrix(np.zeros((1, len(feat_cols)), dtype=np.float32))
            booster.predict(dummy)
            log.info("[model] gpu_predictor habilitado (CUDA ok)")
            return "gpu_predictor"
        except Exception as e:
            log.info("[model] gpu_predictor indisponível (%s), usando cpu_predictor", type(e).__name__)
            try:
                booster.set_param({"predictor": "cpu_predictor"})
            except Exception:
                pass
            return "cpu_predictor"

    def _log_params(self) -> None:
        exit_span = exit_ema_span_from_window(DEFAULT_TRADE_CONTRACT, 60)
        exit_offset = float(getattr(DEFAULT_TRADE_CONTRACT, "exit_ema_init_offset_pct", 0.0) or 0.0)
        log.info(
            "[params] run_dir=%s tau_entry=%.3f exit_span=%s exit_offset=%.4f predictor=%s",
            self.settings.run_dir,
            self.model_bundle.tau_entry,
            exit_span,
            exit_offset,
            getattr(self.model_bundle, "predictor", "cpu_predictor"),
        )
        log.info(
            "[params] exposure max_pos=%s total=%.2f max_trade=%.3f min_trade=%.3f exit_confirm_bars=%s",
            self.settings.max_positions,
            self.settings.total_exposure,
            self.settings.max_trade_exposure,
            self.settings.min_trade_exposure,
            self.settings.exit_confirm_bars,
        )
        log.info(
            "[params] market_cap_min=%.0f trade_mode=%s paper_start=%.2f fee=%.4f",
            self.settings.min_market_cap_usd,
            self.settings.trade_mode,
            self.settings.paper_start_equity,
            self.settings.trade_fee_rate,
        )


    def _init_windows(self) -> None:
        log.info("[init] preparando janelas e carregando OHLC do MySQL...")
        feature_minutes = _feature_window_minutes()
        window_minutes_cfg = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else feature_minutes
        window_minutes = max(window_minutes_cfg, feature_minutes)
        max_rows = int(window_minutes + 10)
        min_rows = int(self.settings.min_ready_rows) if self.settings.min_ready_rows > 0 else int(feature_minutes + 5)
        days = _window_days_for_minutes(window_minutes)
        log.info("[init] window_minutes=%s max_rows=%s min_rows=%s days=%s", window_minutes, max_rows, min_rows, days)

        total = len(self.symbols)
        workers = max(1, int(self.settings.init_workers or 4))

        def _load_one(sym: str) -> Tuple[str, Optional[RollingWindow]]:
            try:
                win = self._load_window(sym, max_rows, min_rows)
                return sym, win
            except Exception as e:
                log.warning("[init][%s] falha ao carregar janela: %s: %s", sym, type(e).__name__, e)
                return sym, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_load_one, sym): sym for sym in self.symbols}
            done = 0
            for fut in as_completed(futures):
                done += 1
                sym, win = fut.result()
                if win is not None:
                    self.windows[sym] = win
                if done % 50 == 0 or done == total:
                    log.info("[init] janelas carregadas: %s/%s", done, total)

        # fallback sequencial para quem falhou por pool exausto
        missing = [s for s in self.symbols if s not in self.windows]
        if missing:
            log.info("[init] recarregando faltantes sequencialmente (%s)", len(missing))
            for sym in missing:
                win = self._load_window(sym, max_rows, min_rows)
                if win is not None:
                    self.windows[sym] = win

        short = [s for s, win in self.windows.items() if win is not None and len(win.rows) < min_rows]
        if self.windows:
            min_len = min(len(w.rows) for w in self.windows.values() if w is not None)
            log.info("[init] estatisticas de janela: min_rows=%s min_len=%s short=%s", min_rows, min_len, len(short))
        if short:
            self._backfill_short_history(short, window_minutes, max_rows, min_rows)

        log.info("[init] janelas prontas (%s símbolos, %s ok)", total, len(self.windows))

    def _prime_scores(self) -> None:
        if self._init_scores_done:
            return
        ready = sum(1 for s in self.symbols if self.windows.get(s) and self.windows[s].is_ready())
        log.info("[score] janelas prontas para scoring: %s/%s", ready, len(self.symbols))
        # 1) Preenche rapidamente todos os símbolos com score=0 e último preço (para listar no dashboard)
        for sym in self.symbols:
            win = self.windows.get(sym)
            if win and win.is_ready():
                self._upsert_symbol_score(sym, None)
        self._push_dashboard_state()

        # 2) Inicia thread em background para calcular scores reais do modelo
        def _refresh():
            log.info("[score] calculando scores iniciais (modelo) para %s símbolos...", len(self.symbols))
            started = time.time()
            ok = 0
            workers = max(
                4,
                int(
                    self.settings.score_workers
                    or self.settings.init_workers
                    or (os.cpu_count() or 4)
                ),
            )

            def _work(sym: str) -> Optional[dict]:
                try:
                    return self._score_symbol(sym)
                except Exception as e:
                    if sym not in self._score_fail_syms:
                        self._score_fail_syms.add(sym)
                        log.warning("[score][prime][%s] erro: %s: %s", sym, type(e).__name__, e)
                return None

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_work, sym): sym for sym in self.symbols}
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    res = fut.result()
                    self._upsert_symbol_score(futures[fut], res)
                    if res:
                        ok += 1
                    if done % 100 == 0 or done == len(self.symbols):
                        log.info("[score] iniciais (modelo): %s/%s (ok=%s)", done, len(self.symbols), ok)
            log.info("[score] iniciais (modelo) concluídos em %.1fs (ok=%s)", time.time() - started, ok)
            self._push_dashboard_state()

        self._score_refresh_thread = threading.Thread(target=_refresh, name="score-prime", daemon=True)
        self._score_refresh_thread.start()
        self._init_scores_done = True

    def _bootstrap_rest(self) -> None:
        if bool(self.settings.use_fast_backfill) and dl_run is not None and DLSettings is not None and DLMySQLConfig is not None:
            try:
                self._fast_backfill()
                return
            except Exception as e:
                print(f"[bootstrap][warn] fast_backfill falhou, caindo para REST: {type(e).__name__}: {e}", flush=True)
        self._start_backfill_workers()
        end_ms_closed = _last_closed_minute_ms()
        tasks = 0
        for sym in self.symbols:
            self.store.ensure_table(sym)
            last_raw = self.store.mysql_max_date(sym)
            last = _normalize_last_ts(last_raw, end_ms_closed)
            if last is None:
                start_ts = end_ms_closed - int(self.settings.bootstrap_days) * 24 * 60 * 60 * 1000
            else:
                start_ts = int(last) + 60_000
            if start_ts <= end_ms_closed:
                self.backfill.enqueue(sym, int(start_ts), int(end_ms_closed))
                tasks += 1
        if tasks == 0:
            log.info("[bootstrap] sem gaps para backfill REST.")
            return
        log.info(
            "[bootstrap] REST backfill enfileirado: syms=%s até %s",
            tasks,
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(end_ms_closed / 1000)),
        )
        last_done = -1
        stalled_since = None
        while True:
            st = self.backfill.snapshot_stats()
            total = max(1, st["total_minutes"])
            done = min(total, st["done_minutes"])
            pct = 100.0 * done / total if total else 0.0
            eta = st.get("eta_sec", 0.0)
            log.info(
                "[bootstrap][rest] %s/%s min (%5.1f%%) queue=%s inflight=%s eta=%ss",
                done,
                total,
                pct,
                st["queue_size"],
                st["inflight"],
                int(eta),
            )
            if done >= total:
                break
            if done == last_done and st["queue_size"] == 0 and st["inflight"] == 0:
                stalled_since = stalled_since or time.time()
                if time.time() - stalled_since > 30:
                    log.warning(
                        "[bootstrap][warn] sem progresso (queue e inflight zerados). Verifique conexão/DB/API Binance. "
                        "Possíveis causas: sem internet, binance bloqueando, erro de certificado, falha no MySQL."
                    )
                    break
            else:
                stalled_since = None
            last_done = done
            time.sleep(2.0)
        log.info("[bootstrap] REST backfill concluído.")

    def _fast_backfill(self) -> None:
        """
        Usa o downloader async (download_to_mysql) para preencher os gaps rapidamente.
        """
        # gera arquivo temporário com os símbolos atuais
        if dl_run is None or DLSettings is None or DLMySQLConfig is None:
            raise RuntimeError("fast_backfill indisponível (dl_run ausente)")
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tf:
            for s in self.symbols:
                tf.write(f"{s}\n")
            symbols_file = tf.name
        # API key opcional (para endpoints públicos pode ficar vazia)
        try:
            from modules.config import secrets as _secrets  # type: ignore
        except Exception:
            try:
                from config import secrets as _secrets  # type: ignore
            except Exception:
                _secrets = None
        api_key = str(getattr(_secrets, "BINANCE_API_KEY", "") or "")
        settings = DLSettings(
            api_key=api_key,
            quote_symbols_file=symbols_file,
            max_conc_requests=int(self.settings.fast_max_conc_requests),
            per_symbol_conc=int(self.settings.fast_per_symbol_conc),
            page_limit=int(self.settings.fast_page_limit),
            interval=str(self.settings.interval),
            target_weight_per_min=int(self.settings.fast_target_weight),
            safety_margin=float(self.settings.fast_safety),
            db=DLMySQLConfig(
                host=self.settings.db.host,
                user=self.settings.db.user,
                password=self.settings.db.password,
                database=self.settings.db.database,
                pool_size=int(self.settings.db.pool_size),
                autocommit=bool(self.settings.db.autocommit),
            ),
        )
        log.info("[bootstrap] fast_backfill (async downloader) iniciado...")
        dl_run(settings)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return pf_run(
            df.copy(),
            flags=self.feature_flags,
            plot=False,
            trade_contract=DEFAULT_TRADE_CONTRACT,
            verbose_features=False,
        )

    def _build_features_latest(self, symbol: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calcula features apenas para o último candle (ou reaproveita cache se o ts não mudou),
        usando apenas a janela mínima necessária.
        """
        ts_ms = int(df.index[-1].value // 1_000_000)
        cached = self._feat_cache.get(symbol)
        if cached and int(cached.get("ts", -1)) == ts_ms:
            row = cached.get("row")
            if isinstance(row, pd.Series):
                return row

        feat_rows = int(self.settings.min_ready_rows) if self.settings.min_ready_rows > 0 else int(_feature_window_minutes() + 5)
        tail_rows = min(len(df), max(feat_rows, 600))
        df_tail = df.tail(tail_rows)
        try:
            df_feat = self._build_features(df_tail)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
            last = _safe_last_row(df_feat)
            if last is not None:
                self._feat_cache[symbol] = {"ts": ts_ms, "row": last}
            return last
        except Exception as e:
            if symbol not in self._score_fail_syms:
                self._score_fail_syms.add(symbol)
                log.warning("[score][feat][%s] erro ao calcular features: %s: %s", symbol, type(e).__name__, e)
            return None

    def _maybe_exit(self, symbol: str, win: RollingWindow) -> None:
        """
        Exit logic: EMA starts at entry_price with an initial offset and is
        calculated somente a partir do candle de entrada. Usa span de exit do
        contrato e nÃ£o mistura histÃ³rico anterior para evitar saÃ­da imediata.
        """
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        entry_ts = pos.get("entry_ts")
        entry_price = float(pos.get("entry_price", 0.0) or pos.get("price_at_entry", 0.0) or 0.0)
        if entry_ts is None or entry_price <= 0:
            return

        df = win.to_frame()
        if df.empty:
            return

        contract = DEFAULT_TRADE_CONTRACT
        span = int(exit_ema_span_from_window(contract, 60))
        if span <= 0:
            return
        offset_pct = float(getattr(contract, "exit_ema_init_offset_pct", 0.0) or 0.0)

        entry_dt = pd.to_datetime(entry_ts, unit="s")
        df_after = df[df.index >= entry_dt]
        if df_after.empty:
            return

        closes = df_after["close"].astype(float).tolist()
        if not closes:
            return

        alpha = 2.0 / (span + 1.0)
        ema = entry_price * (1.0 - offset_pct)  # EMA inicial com offset
        for price in closes:
            ema = alpha * price + (1.0 - alpha) * ema

        last_close = float(closes[-1])
        if last_close < ema:
            self._handle_sell(symbol, last_close)

    def _score_symbol(self, symbol: str) -> Optional[dict]:
        win = self.windows.get(symbol)
        if win is None:
            return None
        if win.need_reload:
            window_minutes = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else _max_pf_window_minutes()
            days = _window_days_for_minutes(window_minutes)
            df = load_ohlc_1m_series(symbol, int(days), remove_tail_days=0)
            win.replace_from_df(df)
        if not win.is_ready():
            if symbol not in self._score_fail_syms:
                self._score_fail_syms.add(symbol)
                log.info(
                    "[score][wait] %s janela incompleta: rows=%s min_ready=%s",
                    symbol,
                    len(win.rows),
                    win.min_ready_rows,
                )
            # tenta completar janela em background (sÃ³ dispara uma vez por sÃ­mbolo)
            if symbol not in self._short_backfill_requested:
                window_minutes = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else _max_pf_window_minutes()
                end_ms = _last_closed_minute_ms()
                start_ms = max(0, int(end_ms - int(window_minutes) * 60_000))
                self._start_backfill_workers()
                self.backfill.enqueue(symbol, start_ms, end_ms)
                self._short_backfill_requested.add(symbol)
                log.info(
                    "[score][backfill] %s enfileirado p/ completar janela (rows=%s target=%s)",
                    symbol,
                    len(win.rows),
                    win.min_ready_rows,
                )
            return None
        df = win.to_frame()
        if df.empty:
            return None
        ts_ms = int(df.index[-1].value // 1_000_000)
        if len(df) < win.min_ready_rows:
            if symbol not in self._score_fail_syms:
                self._score_fail_syms.add(symbol)
                log.info(
                    "[score][warn] %s janela curta: rows=%s min_ready=%s (max=%s)",
                    symbol,
                    len(df),
                    win.min_ready_rows,
                    win.max_rows,
                )
        last = self._build_features_latest(symbol, df)
        if last is None:
            return None

        cols = self.model_bundle.feature_cols
        row = np.zeros((1, len(cols)), dtype=np.float32)
        missing_cols = []
        for i, c in enumerate(cols):
            v = float(last.get(c, 0.0))
            if not np.isfinite(v):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            if v == 0.0 and c not in last:
                missing_cols.append(c)
            row[0, i] = v
        pe = float(self.model_bundle.model.predict(xgb.DMatrix(row), validate_features=False)[0])
        pe = float(_apply_calibration(np.asarray([pe], dtype=np.float64), self.model_bundle.calib)[0])

        # variações (faixa high-low) para 1h e 24h
        range_1h = 0.0
        range_24h = 0.0
        try:
            one_h_ago = df.index[-1] - pd.Timedelta(hours=1)
            day_ago = df.index[-1] - pd.Timedelta(hours=24)
            df_1h = df[df.index >= one_h_ago]
            df_24h = df[df.index >= day_ago]
            if not df_1h.empty:
                range_1h = float(df_1h["high"].max() - df_1h["low"].min())
            if not df_24h.empty:
                range_24h = float(df_24h["high"].max() - df_24h["low"].min())
        except Exception:
            pass

        decision = pe >= self.model_bundle.tau_entry
        price = float(last.get("close", np.nan))
        range_1h_pct = (range_1h / price * 100.0) if price > 0 else 0.0
        range_24h_pct = (range_24h / price * 100.0) if price > 0 else 0.0
        res = {
            "symbol": symbol,
            "p_entry": pe,
            "buy": bool(decision),
            "price": price,
            "ts_ms": ts_ms,
            "range_1h_pct": range_1h_pct,
            "range_24h_pct": range_24h_pct,
        }
        # log diagnóstico apenas uma vez por símbolo quando score sai 0
        if pe == 0.0 and symbol not in self._score_fail_syms:
            self._score_fail_syms.add(symbol)
            log.info(
                "[score][debug] %s pe=0.0 cols=%s missing=%s",
                symbol,
                len(cols),
                len(missing_cols),
            )
        if symbol not in self._score_logged_syms:
            self._score_logged_syms.add(symbol)
            log.info(
                "[score][trace] %s ts=%s pe=%.4f feat_cols=%s",
                symbol,
                ts_ms,
                pe,
                len(cols),
            )
        return res

    def _decision_worker(self) -> None:
        while not self.stop_evt.is_set():
            try:
                sym = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                res = self._score_symbol(sym)
                self._upsert_symbol_score(sym, res)
                if res and res.get("buy"):
                    print(
                        f"[decision] BUY {res['symbol']} p_entry={res['p_entry']:.4f}",
                        flush=True,
                    )
                    self._handle_buy(res)
            except Exception as e:
                if sym not in self._score_fail_syms:
                    self._score_fail_syms.add(sym)
                    log.warning("[score][%s] erro: %s: %s", sym, type(e).__name__, e)
            finally:
                self.queue.task_done()

    def _on_kline(self, symbol: str, row: Tuple[int, float, float, float, float, float]) -> None:
        win = self.windows.get(symbol)
        if win is None:
            return
        ts_ms, o, h, l, c, v = row
        prev_last = win.peek_last_ts()
        if prev_last is not None and ts_ms > prev_last + 60_000:
            self.backfill.enqueue(symbol, prev_last + 60_000, ts_ms - 60_000)
        changed = win.ingest(ts_ms, o, h, l, c, v)
        if not changed:
            return
        # se já existe posição, avalia exits com EMA
        if self.open_positions.get(symbol):
            try:
                self._maybe_exit(symbol, win)
            except Exception:
                pass
        last_dec_ts = self.decisions.get(symbol)
        if last_dec_ts is not None and ts_ms <= last_dec_ts:
            return
        self.decisions[symbol] = ts_ms
        self.queue.put(symbol)

    def _handle_buy(self, decision: dict) -> None:
        price = float(decision.get("price", 0.0) or 0.0)
        if price <= 0:
            return
        if self.open_positions.get(decision["symbol"]):
            log.info("[trade] %s já em posição, ignorando novo BUY", decision["symbol"])
            return
        # sizing inspirado no sweep_003_20260123_102023
        used_exposure = sum(float(pos.get("weight", 0.0)) for pos in self.open_positions.values())
        open_count = len(self.open_positions)
        if int(self.settings.max_positions) > 0 and open_count >= int(self.settings.max_positions):
            log.info("[trade] limite de posições atingido (%s)", self.settings.max_positions)
            return
        remaining = float(self.settings.total_exposure) - float(used_exposure)
        if remaining <= 1e-9:
            log.info("[trade] orçamento de exposição esgotado (%.3f)", used_exposure)
            return
        desired = float(self.settings.total_exposure) / float(max(1, open_count + 1))
        weight = float(min(float(self.settings.max_trade_exposure), remaining, desired))
        if weight < float(self.settings.min_trade_exposure):
            log.info(
                "[trade] peso %.4f abaixo do mínimo %.4f (open=%s, remaining=%.4f)",
                weight,
                self.settings.min_trade_exposure,
                open_count,
                remaining,
            )
            return
        eq, _ = self._current_equity(price_hint={decision["symbol"]: price})
        notional = float(eq * weight) if eq > 0 else float(self.settings.trade_notional_usd)
        actual_weight = float(weight)
        # garante que não tentaremos comprar acima do caixa disponível
        available_cash = float(getattr(self.executor, "cash", 0.0))
        if isinstance(self.executor, PaperExecutor):
            notional = min(notional, available_cash)
        if notional <= 0:
            log.info("[trade] notional inválido (eq=%.2f weight=%.4f cash=%.2f)", eq, weight, available_cash)
            return
        if eq > 0:
            actual_weight = min(weight, notional / max(eq, 1e-9))
        mode = str(self.settings.trade_mode or "paper").lower()
        try:
            res = self.executor.buy(decision["symbol"], price, notional)
            print(f"[{mode}] buy {decision['symbol']} notional={notional:.2f} price={price:.4f} res={res}", flush=True)
            self._notify_trade(decision["symbol"], price, notional, side="BUY")
            self._record_trade(decision["symbol"], price, notional, mode, entry_ts=time.time())
            self.open_positions[decision["symbol"]] = {
                "entry_price": price,
                "entry_ts": time.time(),
                "weight": actual_weight,
                "equity_at_entry": eq,
                "price_at_entry": price,
            }
            self._push_dashboard_state(force=True)
        except Exception as e:
            print(f"[{mode}][err] buy {decision['symbol']}: {type(e).__name__}: {e}", flush=True)

    def _handle_sell(self, symbol: str, price: float) -> None:
        price = float(price or 0.0)
        if price <= 0:
            return
        mode = str(self.settings.trade_mode or "paper").lower()
        try:
            notional = 0.0
            pnl_usd = None
            if isinstance(self.executor, PaperExecutor):
                res = self.executor.close(symbol, price)
                print(f"[{mode}] sell {symbol} price={price:.4f} res={res}", flush=True)
                entry = self.open_positions.get(symbol, {})
                qty = float(res.get("trade", {}).get("qty", 0.0) or 0.0)
                pnl_usd = res.get("trade", {}).get("pnl", None)
                notional = float(res.get("trade", {}).get("notional", qty * price))
                self._record_trade(
                    symbol,
                    price,
                    None,
                    mode,
                    entry_price=float(entry.get("entry_price", price)),
                    exit_price=price,
                    qty=qty,
                    pnl_usd=pnl_usd,
                    entry_ts=entry.get("entry_ts", None),
                    exit_ts=time.time(),
                )
                self._notify_trade(symbol, price, notional, side="SELL", pnl=pnl_usd if pnl_usd is not None else None)
            else:
                # para executor real, apenas registra saída lógica
                entry = self.open_positions.get(symbol, {})
                self._record_trade(
                    symbol,
                    price,
                    None,
                    mode,
                    entry_price=float(entry.get("entry_price", price)),
                    exit_price=price,
                    qty=0.0,
                    pnl_usd=None,
                    entry_ts=entry.get("entry_ts", None),
                    exit_ts=time.time(),
                )
                self._notify_trade(symbol, price, 0.0, side="SELL", pnl=None)
            self.open_positions.pop(symbol, None)
            self._push_dashboard_state(force=True)
        except Exception as e:
            print(f"[{mode}][err] sell {symbol}: {type(e).__name__}: {e}", flush=True)

    def _notify_trade(
        self,
        symbol: str,
        price: float,
        notional: float,
        side: str = "BUY",
        pnl: float | None = None,
    ) -> None:
        if not bool(self.settings.pushover_on):
            return
        url = self.dash_url or ""
        if _pushover_load_default is None or _pushover_send is None:
            return
        cfg = _pushover_load_default(
            user_env="PUSHOVER_USER_KEY",
            token_env="PUSHOVER_TOKEN_TRADE",
            token_name_fallback="PUSHOVER_TOKEN_TRADE",
            title="Tradebot",
            priority=0,
        )
        if cfg is None:
            if not self._pushover_warned:
                print("[pushover][warn] credenciais não encontradas (PUSHOVER_USER_KEY/PUSHOVER_TOKEN_TRADE)", flush=True)
                self._pushover_warned = True
            return
        msg = f"{side.upper()} {symbol} notional={notional:.2f} price={price:.4f}"
        if pnl is not None:
            msg += f" pnl={pnl:.2f}"
        try:
            _pushover_send(msg, cfg=cfg, url=url if url else None, url_title="Dashboard")
        except Exception:
            log.warning("[pushover] envio falhou", exc_info=True)

    def _record_trade(
        self,
        symbol: str,
        price: float,
        notional: float | None,
        mode: str,
        entry_price: float | None = None,
        exit_price: float | None = None,
        qty: float | None = None,
        pnl_usd: float | None = None,
        entry_ts: float | None = None,
        exit_ts: float | None = None,
    ) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry_ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry_ts)) if entry_ts else now_iso
        exit_ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exit_ts)) if exit_ts else (now_iso if exit_price is not None else None)
        # calcula qty/pnl se faltarem
        if qty is None and notional is not None and price > 0:
            qty = notional / max(price, 1e-9)
        if pnl_usd is None and exit_price is not None and entry_price is not None and qty is not None:
            pnl_usd = (exit_price - entry_price) * qty
        pnl_pct = None
        if exit_price is not None and entry_price is not None and entry_price > 0:
            pnl_pct = ((exit_price / entry_price) - 1.0) * 100.0
        tr = {
            "ts_utc": now_iso,
            "symbol": symbol,
            "action": "BUY" if exit_price is None else "SELL",
            "side": "BUY" if exit_price is None else "SELL",
            "qty": qty if qty is not None else (notional / max(price, 1e-9) if notional else None),
            "price": entry_price if entry_price is not None else price,
            "exit_price": exit_price,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "mode": mode,
            "entry_ts_utc": entry_ts_iso if entry_ts_iso else now_iso,
            "exit_ts_utc": exit_ts_iso,
        }
        self.trades.append(tr)
        if len(self.trades) > 200:
            self.trades = self.trades[-200:]

    def _current_equity(self, price_hint: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        price_map: Dict[str, float] = {}
        if price_hint:
            price_map.update({k: float(v) for k, v in price_hint.items()})
        for sym in self.symbols:
            if sym in price_map:
                continue
            win = self.windows.get(sym)
            if win:
                df = win.to_frame()
                if not df.empty:
                    price_map[sym] = float(df["close"].iloc[-1])
        equity = float(getattr(self.executor, "cash", 0.0))
        if isinstance(self.executor, PaperExecutor):
            for sym, st in self.executor.positions.items():
                qty = float(st.get("qty", 0.0))
                avg = float(st.get("avg", 0.0))
                px = float(price_map.get(sym, avg))
                equity += qty * px
        return equity, price_map

    def _snapshot_positions(self, price_map: Optional[Dict[str, float]] = None) -> List[dict]:
        pos = []
        if isinstance(self.executor, PaperExecutor):
            price_map = price_map or {}
            for sym, st in self.executor.positions.items():
                qty = float(st.get("qty", 0.0))
                avg = float(st.get("avg", 0.0))
                px = float(price_map.get(sym, avg))
                notional = qty * px
                pnl = (px - avg) * qty
                pnl_pct = ((px / avg) - 1.0) * 100.0 if avg > 0 else 0.0
                meta = self.open_positions.get(sym, {})
                entry_ts = meta.get("entry_ts", None)
                entry_ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry_ts)) if entry_ts else None
                pos.append(
                    {
                        "symbol": sym,
                        "side": "LONG" if qty >= 0 else "SHORT",
                        "qty": qty,
                        "entry_price": avg,
                        "mark_price": px,
                        "notional_usd": notional,
                        "pnl_usd": pnl,
                        "pnl_pct": pnl_pct,
                        "entry_ts_utc": entry_ts_iso,
                        "exit_ts_utc": None,
                    }
                )
        return pos

    def _push_dashboard_state(self, force: bool = False) -> None:
        if not bool(self.settings.start_dashboard):
            return
        # garante que o dashboard esteja rodando (caso o thread tenha morrido)
        try:
            if not getattr(self, "dashboard_proc", None) or not getattr(self.dashboard_proc, "is_alive", lambda: False)():
                self.dashboard_proc = threading.Thread(
                    target=_launch_dashboard, args=(int(self.settings.dashboard_port),), name="dash", daemon=True
                )
                self.dashboard_proc.start()
        except Exception:
            pass
        now = time.time()
        if (not force) and now - self.last_dashboard_push < max(1, int(self.settings.dashboard_push_every_sec)):
            return
        self.last_dashboard_push = now
        equity, price_map = self._current_equity()
        positions = self._snapshot_positions(price_map)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # salva histórico simples para o dashboard
        self.equity_history.append({"ts_utc": now_iso, "equity_usd": equity})
        if len(self.equity_history) > self.equity_history_max:
            self.equity_history = self.equity_history[-self.equity_history_max :]
        try:
            hist_path = Path("data") / "equity_history_live.json"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            hist_path.write_text(json.dumps(self.equity_history), encoding="utf-8")
        except Exception:
            pass
        feed_delay = None
        # anchor do feed: usa ws_last_ts, mas se faltar, usa o Ãºltimo ts das janelas carregadas
        anchor_ts = int(self.ws_last_ts) if self.ws_last_ts is not None else 0
        if not anchor_ts:
            try:
                anchor_ts = max(
                    [win.peek_last_ts() or 0 for win in self.windows.values()] or [int(time.time() * 1000)]
                )
            except Exception:
                anchor_ts = int(time.time() * 1000)
        if anchor_ts:
            feed_delay = max(0.0, time.time() - (float(anchor_ts) / 1000.0))
        ts_anchor = anchor_ts or int(time.time() * 1000)
        signals = sorted(
            [
                {
                    "symbol": s["symbol"],
                    "score": s.get("score", 0.0),
                    "price": s.get("price", 0.0),
                    "ts_ms": ts_anchor,
                    "range_1h_pct": s.get("range_1h_pct", 0.0),
                    "range_24h_pct": s.get("range_24h_pct", 0.0),
                }
                for s in self.symbol_scores.values()
                if int(s.get("ts_ms", 0) or 0) > 0
            ],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        if self.start_equity_usd is None:
            try:
                self.start_equity_usd = float(self.settings.paper_start_equity or equity)
            except Exception:
                self.start_equity_usd = float(equity)
        start_eq = float(self.start_equity_usd or equity)
        unrealized_pnl = sum(float(p.get("pnl_usd", 0.0)) for p in positions)
        total_pnl = float(equity) - start_eq
        realized_pnl = total_pnl - unrealized_pnl
        payload = {
            "summary": {
                "equity_usd": equity,
                "cash_usd": float(getattr(self.executor, "cash", 0.0)),
                "exposure_usd": sum(float(p.get("notional_usd", 0.0)) for p in positions),
                "realized_pnl_usd": realized_pnl,
                "unrealized_pnl_usd": unrealized_pnl,
                "updated_at_utc": now_iso,
            },
            "positions": positions,
            "recent_trades": list(reversed(self.trades[-50:])),
            "allocation": {p["symbol"]: float(p.get("notional_usd", 0.0)) for p in positions},
            "equity_history": list(self.equity_history),
            "meta": {
                "mode": self.settings.trade_mode,
                "feed": {
                    "last_ts_ms": self.ws_last_ts,
                    "ws_msgs": self.ws_msg_count,
                    "symbols": len(self.symbols),
                    "delay_sec": feed_delay,
                },
                "signals": signals,
            },
        }
        try:
            resp = requests.post(
                f"http://127.0.0.1:{int(self.settings.dashboard_port)}/api/update", json=payload, timeout=2.5
            )
            resp.raise_for_status()
        except Exception as e:
            if not getattr(self, "_dash_warned", False):
                print(
                    f"[dash][warn] falha ao publicar estado na porta {self.settings.dashboard_port}: {type(e).__name__}: {e}",
                    flush=True,
                )
                print(
                    "Certifique-se de que o dashboard embutido subiu (thread interno) "
                    "ou ajuste DASHBOARD_PORT para a mesma porta do ngrok.",
                    flush=True,
                )
                self._dash_warned = True
        self._persist_state()

    def _status_loop(self) -> None:
        while True:
            try:
                st = self.backfill.snapshot_stats()
                log.info(
                    "[status] rest_pending=%sm q=%s inflight=%s decisions=%s scores=%s ws_threads=%s ws_msgs=%s last_ts=%s",
                    st.get("remain_minutes", 0),
                    st.get("queue_size", 0),
                    st.get("inflight", 0),
                    len(self.trades),
                    len(self.symbol_scores),
                    len(getattr(self, "_ws_threads", [])),
                    self.ws_msg_count,
                    self.ws_last_ts,
                )
                # empurra snapshot para o dashboard (respeita throttle interno)
                self._push_dashboard_state()
            except Exception:
                log.warning("status loop error", exc_info=True)
            time.sleep(int(self.settings.status_every_sec))

    def _start_score_keepalive(self) -> None:
        if self._score_keepalive_thread and self._score_keepalive_thread.is_alive():
            return

        def _loop() -> None:
            while not self.stop_evt.is_set():
                for sym in self.symbols:
                    if self.stop_evt.is_set():
                        break
                    try:
                        res = self._score_symbol(sym)
                        if res:
                            self._upsert_symbol_score(sym, res)
                    except Exception:
                        pass
                    time.sleep(0.002)
                self._push_dashboard_state()
                for _ in range(60):
                    if self.stop_evt.is_set():
                        break
                    time.sleep(1)

        self._score_keepalive_thread = threading.Thread(target=_loop, name="score-keepalive", daemon=True)
        self._score_keepalive_thread.start()

    def _launch_ws(self, chunk: List[str]) -> None:
        streams = [f"{s.lower()}@kline_1m" for s in chunk]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        log.info("[ws] starting chunk=%s url=%s", chunk, url)
        while not self.stop_evt.is_set():
            ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                ws.run_forever(ping_interval=int(self.settings.ws_ping_interval), ping_timeout=int(self.settings.ws_ping_timeout))
            except Exception as e:
                log.warning("[ws] reconnect chunk=%s err=%s", chunk, e)
                time.sleep(3)

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
        close_ts = int(k.get("T") or (open_ts + 60_000))
        self.ws_msg_count += 1
        # usamos o close_ts para medir atraso do feed; mantemos o open_ts como chave da barra no DB
        self.ws_last_ts = close_ts
        row = (
            open_ts,
            float(k.get("o")),
            float(k.get("h")),
            float(k.get("l")),
            float(k.get("c")),
            float(k.get("v")),
        )
        try:
            self.store.insert_batch(sym, [row])
        except Exception:
            pass
        self._on_kline(sym, row)

    def _on_error(self, _ws, err) -> None:
        print(f"[ws] error: {err}", flush=True)

    def _on_close(self, _ws, code, msg) -> None:
        print(f"[ws] closed: code={code} msg={msg}", flush=True)

    def start(self) -> None:
        self._bootstrap_rest()
        self._init_windows()
        self._prime_scores()
        # Mesmo usando fast_backfill, iniciamos os workers de gaps (REST) para cobrir buracos detectados pelo WS.
        self._start_backfill_workers()
        threading.Thread(target=self._decision_worker, daemon=True).start()
        self._start_score_keepalive()
        # empurra estado inicial (inclui persistência carregada)
        self._push_dashboard_state(force=True)

        if bool(self.settings.start_dashboard):
            self.dashboard_proc = threading.Thread(
                target=_launch_dashboard, args=(int(self.settings.dashboard_port),), name="dash", daemon=True
            )
            self.dashboard_proc.start()
            if bool(self.settings.start_ngrok):
                self.dash_url = _start_ngrok(int(self.settings.dashboard_port), token_env=self.settings.ngrok_authtoken_env)
                if self.dash_url:
                    print(f"[ngrok] dashboard: {self.dash_url}", flush=True)

        if int(self.settings.status_every_sec) > 0:
            threading.Thread(target=self._status_loop, daemon=True).start()

        self._ws_threads: List[threading.Thread] = []
        for chunk in _chunked(self.symbols, int(self.settings.ws_chunk)):
            th = threading.Thread(target=self._launch_ws, args=(chunk,), name=f"ws-{chunk[0]}", daemon=True)
            th.start()
            self._ws_threads.append(th)

        while True:
            time.sleep(3600)

    def _start_backfill_workers(self) -> None:
        if self._backfill_started:
            return
        try:
            self.backfill.start()
            self._backfill_started = True
        except Exception as e:
            log.warning("[backfill] nao iniciou threads: %s: %s", type(e).__name__, e)


def _chunked(it: List[str], n: int) -> List[List[str]]:
    out: List[List[str]] = []
    if n <= 0:
        return [it]
    for i in range(0, len(it), n):
        out.append(it[i : i + n])
    return out


def _start_ngrok(port: int, *, token_env: str = "NGROK_AUTHTOKEN") -> Optional[str]:
    """
    Sobe o tunel ngrok como no monolito: tenta NgrokManager (dominio custom)
    e cai para pyngrok simples. Authtoken vem de env (WF_ > padrao) ou secrets.
    """
    # 1) Tentativa via NgrokMgr (dominio fixo + basic auth)
    if NgrokCfg is not None and NgrokMgr is not None:
        try:
            cfg = NgrokCfg()
            if os.getenv("WF_NGROK_DOMAIN"):
                cfg.domain = os.getenv("WF_NGROK_DOMAIN", "").strip()
            if os.getenv("NGROK_DOMAIN"):
                cfg.domain = os.getenv("NGROK_DOMAIN", "").strip() or cfg.domain
            if os.getenv("WF_NGROK_DIR"):
                cfg.downloads_dir = Path(os.getenv("WF_NGROK_DIR", str(cfg.downloads_dir)))
            if os.getenv("NGROK_DIR"):
                cfg.downloads_dir = Path(os.getenv("NGROK_DIR", str(cfg.downloads_dir)))
            if os.getenv("WF_NGROK_BASIC_USER"):
                cfg.username = os.getenv("WF_NGROK_BASIC_USER", cfg.username)
            if os.getenv("NGROK_BASIC_USER"):
                cfg.username = os.getenv("NGROK_BASIC_USER", cfg.username)
            if os.getenv("WF_NGROK_BASIC_PASS"):
                cfg.password = os.getenv("WF_NGROK_BASIC_PASS", cfg.password)
            if os.getenv("NGROK_BASIC_PASS"):
                cfg.password = os.getenv("NGROK_BASIC_PASS", cfg.password)
            token = (
                os.getenv("WF_NGROK_AUTHTOKEN")
                or os.getenv(token_env)
                or os.getenv("NGROK_AUTHTOKEN")
                or ""
            ).strip()
            if (not token) and _secrets is not None:
                token = str(getattr(_secrets, "NGROK_AUTHTOKEN", "") or "").strip()
            if token:
                cfg.authtoken = token
            cfg.port = int(port)
            mgr = NgrokMgr(cfg)
            mgr.start()
            if cfg.domain:
                url = f"https://{cfg.domain}"
                log.info("[ngrok] usando dominio custom: %s", url)
                return url
        except Exception as e:
            print(f"[ngrok][warn] NgrokMgr falhou, tentando pyngrok: {type(e).__name__}: {e}", flush=True)

    # 2) Fallback pyngrok (subdominio aleatorio)
    try:
        from pyngrok import ngrok  # type: ignore
    except Exception:
        print("[ngrok][warn] pyngrok não disponível; túnel não iniciado (pip install pyngrok)", flush=True)
        return None

    token = (
        os.getenv("WF_NGROK_AUTHTOKEN")
        or os.getenv(token_env)
        or os.getenv("NGROK_AUTHTOKEN")
        or ""
    ).strip()
    if (not token) and _secrets is not None:
        token = str(getattr(_secrets, "NGROK_AUTHTOKEN", "") or "").strip()
    if token:
        try:
            ngrok.set_auth_token(token)
        except Exception as e:
            print(f"[ngrok][warn] set_auth_token falhou: {e}", flush=True)
    else:
        log.info("[ngrok] sem authtoken; tentando túnel temporário via pyngrok.")
    try:
        t = ngrok.connect(int(port), bind_tls=True)
        return t.public_url
    except Exception as e:
        print(f"[ngrok][warn] falhou ao abrir túnel: {type(e).__name__}: {e}", flush=True)
        return None


def _launch_dashboard(port: int) -> None:
    """
    Lança o dashboard realtime no mesmo processo (thread separada).
    """
    try:
        from modules.realtime.dashboard_server import create_app  # type: ignore
    except Exception:
        try:
            from realtime.dashboard_server import create_app  # type: ignore
        except Exception:
            print("[dash][warn] dashboard_server não encontrado", flush=True)
            return
    try:
        from werkzeug.serving import run_simple, WSGIRequestHandler  # type: ignore
    except Exception:
        print("[dash][warn] werkzeug não disponível; dashboard não iniciado", flush=True)
        return
    # silencia logs HTTP do werkzeug
    try:
        import logging

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        WSGIRequestHandler.log = lambda *args, **kwargs: None  # type: ignore
    except Exception:
        pass
    app, store = create_app(demo=False, refresh_sec=2.0)
    # limpa estado demo se conseguirmos importar as classes
    try:
        if DashboardState is not None and AccountSummary is not None:
            empty = DashboardState(
                summary=AccountSummary(equity_usd=0.0, cash_usd=0.0, exposure_usd=0.0, realized_pnl_usd=0.0, unrealized_pnl_usd=0.0),
                positions=[],
                recent_trades=[],
                allocation={},
                meta={},
                equity_history=[],
            )
            store.set(empty)
    except Exception:
        pass
    try:
        run_simple("0.0.0.0", int(port), app, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[dash][warn] falhou ao iniciar: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    settings = LiveSettings(
        run_dir="D:/astra/models_sniper/crypto/wf_002",
        symbols_file="",
        quote_symbols_fallback="",
        bootstrap_days=7,
        window_minutes=0,
        min_ready_rows=0,
        min_market_cap_usd=50_000_000.0,
        use_danger_filter=False,
        max_positions=27,
        total_exposure=1.0,
        max_trade_exposure=0.07,
        min_trade_exposure=0.04,
        exit_confirm_bars=1,
        tau_entry_override=0.775,
        db=MySQLConfig(host="localhost", user="root", password="2017", database="crypto", pool_size=32),
        trade_mode="paper",  # "paper" ou "live"
        trade_notional_usd=100.0,
        paper_start_equity=10_000.0,
        start_dashboard=True,
        dashboard_port=5055,
        init_workers=8,
        score_workers=32,
    )
    bot = LiveDecisionBot(settings)
    bot.start()


if __name__ == "__main__":
    main()
