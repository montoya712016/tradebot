# -*- coding: utf-8 -*-
"""
Sniper Bot - Lógica principal de decisão do bot Sniper.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import psutil
import requests
import atexit
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from core.contracts import DEFAULT_TRADE_CONTRACT, exit_ema_span_from_window
from core.executors import LiveExecutor, PaperExecutor
from core.models.bundle import ModelBundle
from core.utils.notify import send_pushover

from modules.backtest.sniper_walkforward import PeriodModel, load_period_models
from modules.backtest.sniper_simulator import _apply_calibration
from modules.config.symbols import load_top_market_cap_symbols, default_top_market_cap_path
from modules.prepare_features.data import load_ohlc_1m_series
from modules.prepare_features.prepare_features import run as pf_run

from realtime.bot.settings import LiveSettings
from realtime.bot.utils import (
    feature_window_minutes,
    make_feature_flags,
    safe_last_row,
    window_days_for_minutes,
    max_pf_window_minutes,
)
from realtime.market_data.mysql import MySQLStore, MySQLConfig
from realtime.market_data.rest import RestBackfillQueue, _last_closed_minute_ms
from realtime.market_data.rolling_window import RollingWindow
from realtime.market_data.websocket import WsKlineIngestor

# Imports opcionais (downloader, dashboard state, ngrok)
try:
    from crypto.binance.download_to_mysql import (
        DownloadSettings as DLSettings,
        MySQLConfig as DLMySQLConfig,
        run as dl_run,
    )
except ImportError:
    DLSettings = None
    DLMySQLConfig = None
    dl_run = None

try:
    from realtime.dashboard_state import DashboardState, AccountSummary
except ImportError:
    try:
        from modules.realtime.dashboard_state import DashboardState, AccountSummary
    except ImportError:
        DashboardState = None
        AccountSummary = None

try:
    from realtime.realtime_dashboard_ngrok_monolith import NgrokConfig as NgrokCfg, NgrokManager as NgrokMgr
except ImportError:
    try:
        from modules.realtime.realtime_dashboard_ngrok_monolith import NgrokConfig as NgrokCfg, NgrokManager as NgrokMgr
    except ImportError:
        NgrokCfg = None
        NgrokMgr = None

try:
    from modules.config import secrets as _secrets
except ImportError:
    try:
        from config import secrets as _secrets
    except ImportError:
        _secrets = None


log = logging.getLogger("realtime")


class LiveDecisionBot:
    """
    Bot de decisão em tempo real (Sniper).
    """
    def __init__(self, settings: LiveSettings):
        self.settings = settings
        self.feature_flags = make_feature_flags()
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
        self.equity_history_max: int = 50000
        self.last_dashboard_push = 0.0
        self.ws_msg_count = 0
        self.ws_last_ts: Optional[int] = None
        self._score_fail_syms: Set[str] = set()
        self._score_logged_syms: Set[str] = set()
        self._init_scores_done = False
        self._score_refresh_thread: Optional[threading.Thread] = None
        self._score_keepalive_thread: Optional[threading.Thread] = None
        self._short_backfill_requested: Set[str] = set()
        self._feat_cache: Dict[str, dict] = {}
        self._last_score_wallclock: Dict[str, float] = {}
        self._latest_sys_metrics: dict = {}
        self._runtime_stage: str = "boot"
        self._runtime_detail: str = ""
        self._runtime_stage_ts: float = time.time()
        self._last_score_cycle_sec: Optional[float] = None
        self._last_score_proc_sec: Optional[float] = None
        self._last_score_cycle_ts: Optional[float] = None
        self._last_score_cycle_symbols: int = 0
        self._sysmon_thread: Optional[threading.Thread] = None
        self._predict_lock = threading.Lock()
        self.trade_lock = threading.RLock()
        
        # WS Ingestor (substitui threads manuais)
        self.ws_ingestor = WsKlineIngestor(
            self.symbols, 
            self.store, 
            self.backfill, 
            on_kline_callback=self._on_kline,
            on_status=self._on_ws_status
        )
        
        self._log_params()
        self._load_persisted_state()

    def _on_ws_status(self, connected: bool, last_ts: int, msg_count: int):
        # Callback from WsKlineIngestor - apenas atualiza se valores são significativos
        # _on_kline já incrementa ws_msg_count diretamente, então só atualizamos last_ts aqui
        if last_ts > 0:
            self.ws_last_ts = last_ts
        # NÃO sobrescrever ws_msg_count aqui - _on_kline cuida disso

    def _load_persisted_state(self) -> None:
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
                # Filter legacy trades: remove standalone entries where side is SELL (merged trades only)
                cleaned_trades = []
                for t in trades:
                    s = str(t.get("side") or t.get("action") or "BUY").upper()
                    if s == "SELL":
                        continue
                    cleaned_trades.append(t)
                self.trades = cleaned_trades[-200:]
            if open_pos:
                self.open_positions = open_pos
            if isinstance(self.executor, PaperExecutor) and paper_state:
                cash = paper_state.get("cash")
                positions = paper_state.get("positions") or {}
                if cash is not None:
                    self.executor.cash = float(cash)
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
        feature_minutes = feature_window_minutes()
        backfill_minutes = max(feature_minutes, min_rows, window_minutes)
        start_ms = max(0, int(end_ms - int(backfill_minutes) * 60_000))
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
            try:
                from crypto.realtime_bot import load_symbols
                syms = load_symbols(self.settings.symbols_file)
            except ImportError:
                syms = []
                if Path(self.settings.symbols_file).exists():
                    syms = [l.strip().split(":")[0] for l in Path(self.settings.symbols_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
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
        pm: PeriodModel = next((p for p in periods if getattr(p, "period_days", 1e9) == 0), periods[0])
        log.info("[model] periodo selecionado: %sd (train_end=%s)", pm.period_days, pm.train_end_utc)
        feat_cols = list(pm.entry_cols_map.get("mid", pm.entry_cols))
        calib = dict(pm.entry_calib_map.get("mid", pm.entry_calib))
        tau_entry = float(self.settings.tau_entry_override or 0.0)
        if tau_entry <= 0.0:
            tau_entry = float(pm.tau_entry_map.get("mid", pm.tau_entry))
        booster = pm.entry_models.get("mid", pm.entry_model)
        predictor = self._enable_gpu_predictor(booster, feat_cols, enable_gpu=bool(self.settings.use_gpu_predictor))
        return ModelBundle(
            model=booster,
            feature_cols=feat_cols,
            calib=calib,
            tau_entry=tau_entry,
            predictor=predictor,
        )

    def _enable_gpu_predictor(self, booster: xgb.Booster, feat_cols: List[str], *, enable_gpu: bool = True) -> str:
        if enable_gpu:
            try:
                booster.set_param({"predictor": "gpu_predictor", "device": "cuda", "nthread": 1})
                dummy = xgb.DMatrix(np.zeros((1, len(feat_cols)), dtype=np.float32))
                booster.predict(dummy)
                log.info("[model] gpu_predictor habilitado (CUDA ok)")
                return "gpu_predictor"
            except Exception as e:
                log.warning("[model] gpu_predictor indisponível (%s), caindo para CPU", type(e).__name__)
        try:
            booster.set_param({"predictor": "cpu_predictor", "nthread": max(1, os.cpu_count() or 4)})
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

    def _set_runtime_stage(self, stage: str, detail: str = "") -> None:
        self._runtime_stage = stage
        self._runtime_detail = detail or ""
        self._runtime_stage_ts = time.time()

    def _init_windows(self) -> None:
        log.info("[init] preparando janelas e carregando OHLC do MySQL...")
        feature_minutes = feature_window_minutes()
        window_minutes_cfg = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else feature_minutes
        window_minutes = max(window_minutes_cfg, feature_minutes)
        max_rows = int(window_minutes + 10)
        min_rows = int(self.settings.min_ready_rows) if self.settings.min_ready_rows > 0 else int(feature_minutes + 5)
        days = window_days_for_minutes(window_minutes)
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

    def _start_backfill_workers(self) -> None:
        if self._backfill_started:
            return
        try:
            self.backfill.start()
            self._backfill_started = True
        except Exception as e:
            log.warning("[backfill] nao iniciou threads: %s: %s", type(e).__name__, e)

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
            last = _last_closed_minute_ms(last_raw) if last_raw else None # _normalize_last_ts logic inline or imported?
            # _normalize_last_ts was not imported, let's assume _last_closed_minute_ms handles it or logic is simple
            # Actually _normalize_last_ts is in rest.py but not exported. I should check rest.py exports.
            # rest.py exports _last_closed_minute_ms.
            # Let's implement simple logic here or import it if possible.
            # For now, simple logic:
            if last_raw is None:
                start_ts = end_ms_closed - int(self.settings.bootstrap_days) * 24 * 60 * 60 * 1000
            else:
                start_ts = int(last_raw) + 60_000
                if start_ts > end_ms_closed: # sanity check
                     start_ts = end_ms_closed + 60000 # skip
            
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
                    )
                    break
            else:
                stalled_since = None
            last_done = done
            time.sleep(2.0)
        log.info("[bootstrap] REST backfill concluído.")

    def _fast_backfill(self) -> None:
        if dl_run is None or DLSettings is None or DLMySQLConfig is None:
            raise RuntimeError("fast_backfill indisponível (dl_run ausente)")
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as tf:
            for s in self.symbols:
                tf.write(f"{s}\n")
            symbols_file = tf.name
        api_key = str(getattr(_secrets, "BINANCE_API_KEY", "") or "") if _secrets else ""
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
        ts_ms = int(df.index[-1].value // 1_000_000)
        cached = self._feat_cache.get(symbol)
        if cached and int(cached.get("ts", -1)) == ts_ms:
            row = cached.get("row")
            if isinstance(row, pd.Series):
                return row

        feat_rows = int(self.settings.min_ready_rows) if self.settings.min_ready_rows > 0 else int(feature_window_minutes() + 5)
        tail_rows = min(len(df), max(feat_rows, 600))
        df_tail = df.tail(tail_rows)
        try:
            df_feat = self._build_features(df_tail)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
            last = safe_last_row(df_feat)
            if last is not None:
                self._feat_cache[symbol] = {"ts": ts_ms, "row": last}
            return last
        except Exception as e:
            if symbol not in self._score_fail_syms:
                self._score_fail_syms.add(symbol)
                log.warning("[score][feat][%s] erro ao calcular features: %s: %s", symbol, type(e).__name__, e)
            return None

    def _maybe_exit(self, symbol: str, win: RollingWindow) -> None:
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
        ema = entry_price * (1.0 - offset_pct)
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
            window_minutes = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else max_pf_window_minutes()
            days = window_days_for_minutes(window_minutes)
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
            if symbol not in self._short_backfill_requested:
                window_minutes = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else max_pf_window_minutes()
                end_ms = _last_closed_minute_ms()
                start_ms = max(0, int(end_ms - int(window_minutes) * 60_000))
                self._start_backfill_workers()
                self.backfill.enqueue(symbol, start_ms, end_ms)
                self._short_backfill_requested.add(symbol)
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
            if v == 0.0 and c not in last:
                missing_cols.append(c)
            row[0, i] = v
        with self._predict_lock:
            pe = float(self.model_bundle.model.predict(xgb.DMatrix(row), validate_features=False)[0])
            pe = float(_apply_calibration(np.asarray([pe], dtype=np.float64), self.model_bundle.calib)[0])

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
        if pe == 0.0 and symbol not in self._score_fail_syms:
            self._score_fail_syms.add(symbol)
            log.info(
                "[score][debug] %s pe=0.0 cols=%s missing=%s",
                symbol,
                len(cols),
                len(missing_cols),
            )
        # LOG SEMPRE (removido filtro de logged_syms)
        log.info(
            "[score][trace] %s ts=%s pe=%.4f feat_cols=%s",
            symbol,
            ts_ms,
            pe,
            len(cols),
        )
        self._last_score_wallclock[symbol] = time.time()
        return res

    def _decision_worker(self) -> None:
        while not self.stop_evt.is_set():
            try:
                sym = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                res = self._score_symbol(sym)
                if res is None:
                    res = self.symbol_scores.get(sym)
                if res:
                    self._upsert_symbol_score(sym, res)
                if res and res.get("buy"):
                    print(
                        f"[decision] BUY {res['symbol']} p_entry={res['p_entry']:.4f}",
                        flush=True,
                    )
                    with self.trade_lock:
                        self._handle_buy(res)
            except Exception as e:
                if sym not in self._score_fail_syms:
                    self._score_fail_syms.add(sym)
                    log.warning("[score][%s] erro: %s: %s", sym, type(e).__name__, e)
            finally:
                self.queue.task_done()

    def _on_kline(self, symbol: str, row: Tuple[int, float, float, float, float, float]) -> None:
        ts_ms, o, h, l, c, v = row
        
        # Atualiza contadores de feed (corrige latência no dashboard)
        self.ws_msg_count += 1
        self.ws_last_ts = max(self.ws_last_ts or 0, ts_ms)
        
        win = self.windows.get(symbol)
        if win is None:
            return
        prev_last = win.peek_last_ts()
        if prev_last is not None and ts_ms > prev_last + 60_000:
            self.backfill.enqueue(symbol, prev_last + 60_000, ts_ms - 60_000)
        changed = win.ingest(ts_ms, o, h, l, c, v)
        if not changed:
            return
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
        with self.trade_lock:
            price = float(decision.get("price", 0.0) or 0.0)
            if price <= 0:
                return
            if self.open_positions.get(decision["symbol"]):
                log.info("[trade] %s já em posição, ignorando novo BUY", decision["symbol"])
                return
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
        with self.trade_lock:
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
        
        # Tenta carregar pushover utils se não carregado globalmente
        # (Assumindo que imports globais já cuidaram disso ou falharam silenciosamente)
        # Vamos usar send_pushover importado de core.utils.notify
        
        cfg = None
        try:
             from core.utils.notify import load_default
             cfg = load_default(
                user_env="PUSHOVER_USER_KEY",
                token_env="PUSHOVER_TOKEN_TRADE",
                token_name_fallback="PUSHOVER_TOKEN_TRADE",
                title="Tradebot",
                priority=0,
            )
        except ImportError:
            pass

        if cfg is None:
            if not self._pushover_warned:
                print("[pushover][warn] credenciais não encontradas (PUSHOVER_USER_KEY/PUSHOVER_TOKEN_TRADE)", flush=True)
                self._pushover_warned = True
            return

        msg = f"{side.upper()} {symbol} notional={notional:.2f} price={price:.4f}"
        if pnl is not None:
            msg += f" pnl={pnl:.2f}"
        try:
            send_pushover(msg, cfg=cfg, url=url if url else None, url_title="Dashboard")
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
        exit_ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exit_ts)) if exit_ts else None
        
        if qty is None and notional is not None and price > 0:
            qty = notional / max(price, 1e-9)
        if pnl_usd is None and exit_price is not None and entry_price is not None and qty is not None:
            pnl_usd = (exit_price - entry_price) * qty
        pnl_pct = None
        if exit_price is not None and entry_price is not None and entry_price > 0:
            pnl_pct = ((exit_price / entry_price) - 1.0) * 100.0
        
        if exit_price is None:
            # ENTRADA: criar novo registro com status OPEN
            tr = {
                "ts_utc": now_iso,
                "symbol": symbol,
                "side": "BUY",
                "status": "OPEN",
                "qty": qty if qty is not None else (notional / max(price, 1e-9) if notional else None),
                "entry_price": entry_price if entry_price is not None else price,
                "price": entry_price if entry_price is not None else price,
                "exit_price": None,
                "pnl_usd": None,
                "pnl_pct": None,
                "mode": mode,
                "entry_ts_utc": entry_ts_iso,
                "exit_ts_utc": None,
            }
            self.trades.append(tr)
        else:
            # SAÍDA: atualizar registro existente ao invés de criar novo
            updated = False
            for t in reversed(self.trades):
                if t.get("symbol") == symbol and t.get("status") == "OPEN":
                    t["status"] = "CLOSED"
                    t["exit_price"] = exit_price
                    t["exit_ts_utc"] = exit_ts_iso
                    t["pnl_usd"] = pnl_usd
                    t["pnl_pct"] = pnl_pct
                    updated = True
                    break
            
            # Fallback: se não encontrou registro OPEN, criar um completo
            if not updated:
                tr = {
                    "ts_utc": now_iso,
                    "symbol": symbol,
                    "side": "BUY",
                    "status": "CLOSED",
                    "qty": qty,
                    "entry_price": entry_price if entry_price is not None else price,
                    "price": entry_price if entry_price is not None else price,
                    "exit_price": exit_price,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "mode": mode,
                    "entry_ts_utc": entry_ts_iso,
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
                    "ts_ms": s.get("ts_ms", ts_anchor),
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
        stage_ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._runtime_stage_ts))
        last_cycle_ts_utc = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_score_cycle_ts))
            if self._last_score_cycle_ts
            else None
        )
        runtime_meta = {
            "stage": self._runtime_stage,
            "detail": self._runtime_detail,
            "stage_ts_utc": stage_ts_utc,
            "last_cycle_sec": self._last_score_cycle_sec,
            "last_cycle_proc_sec": self._last_score_proc_sec,
            "last_cycle_ts_utc": last_cycle_ts_utc,
            "last_cycle_symbols": self._last_score_cycle_symbols,
        }
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
            "recent_trades": list(reversed(self.trades[-200:])),
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
                "runtime": runtime_meta,
                "system": dict(self._latest_sys_metrics) if self._latest_sys_metrics else {},
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
                self._dash_warned = True
        self._persist_state()

    def _status_loop(self) -> None:
        last_logged_ts = None
        while True:
            try:
                st = self.backfill.snapshot_stats()
                current_ts = self.ws_last_ts
                if current_ts != last_logged_ts:
                    ts_human = "-"
                    try:
                        if current_ts:
                            ts_human = datetime.fromtimestamp(float(current_ts) / 1000.0, tz=timezone.utc).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                    except Exception:
                        pass
                    log.info(
                        "[status] rest_pending=%sm q=%s inflight=%s decisions=%s scores=%s ws_msgs=%s last_utc=%s",
                        st.get("remain_minutes", 0),
                        st.get("queue_size", 0),
                        st.get("inflight", 0),
                        len(self.trades),
                        len(self.symbol_scores),
                        self.ws_msg_count,
                        ts_human,
                    )

                    last_logged_ts = current_ts
                    self._push_dashboard_state()
                else:
                    # Log mesmo se timestamp nao mudou (para indicar stall)
                    log.info(
                        "[status][STALL?] rest_pending=%sm q=%s inflight=%s decisions=%s scores=%s ws_msgs=%s last_utc=%s",
                        st.get("remain_minutes", 0),
                        self.queue.qsize(), # queue.size directly
                        st.get("inflight", 0),
                        len(self.trades),
                        len(self.symbol_scores),
                        self.ws_msg_count,
                        "-" if not current_ts else datetime.fromtimestamp(float(current_ts) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    )

            except Exception:
                log.warning("status loop error", exc_info=True)
            time.sleep(1)

    def _start_score_keepalive(self) -> None:
        if self._score_keepalive_thread and self._score_keepalive_thread.is_alive():
            return

        def _loop() -> None:
            while not self.stop_evt.is_set():
                try:
                    cycle_start = time.time()
                    ok = 0
                    proc_total = 0.0
                    self._set_runtime_stage("score_keepalive", f"{len(self.symbols)} symbols")
                    chunk = 32
                    for i in range(0, len(self.symbols), chunk):
                        if self.stop_evt.is_set():
                            break
                        syms = self.symbols[i : i + chunk]
                        try:
                            t0 = time.time()
                            batch_res = self._score_symbols_batch(syms)
                            proc_total += time.time() - t0
                            for sym, res in batch_res:
                                self._upsert_symbol_score(sym, res)
                                ok += 1
                        except Exception:
                            pass
                        time.sleep(0.01)
                    elapsed = time.time() - cycle_start
                    if proc_total <= 0.0:
                        proc_total = elapsed
                    self._last_score_cycle_sec = float(elapsed)
                    self._last_score_proc_sec = float(proc_total)
                    self._last_score_cycle_ts = time.time()
                    self._last_score_cycle_symbols = int(ok)
                    self._set_runtime_stage("idle")
                    self._push_dashboard_state()
                    # Log organizado do tempo de processamento (features + inferência)
                    if ok > 0:
                        avg_sym = proc_total / max(1, ok)
                        log.info(
                            "[perf][score_keepalive] symbols=%s proc=%.2fs cycle=%.2fs avg_sym=%.4fs",
                            ok,
                            proc_total,
                            elapsed,
                            avg_sym,
                        )
                    else:
                        log.info(
                            "[perf][score_keepalive] symbols=0 proc=%.2fs cycle=%.2fs",
                            proc_total,
                            elapsed,
                        )
                except Exception:
                    log.warning("[score][keepalive] loop error", exc_info=True)
                for _ in range(30):
                    if self.stop_evt.is_set():
                        break
                    time.sleep(1)

        self._score_keepalive_thread = threading.Thread(target=_loop, name="score-keepalive", daemon=True)
        self._score_keepalive_thread.start()

    def _system_snapshot(self, proc: Optional[psutil.Process] = None) -> dict:
        now = time.time()
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        rec: dict = {"ts": now, "ts_utc": ts_utc}
        try:
            rec["cpu_pct"] = float(psutil.cpu_percent(interval=None))
            freq = psutil.cpu_freq()
            if freq:
                rec["cpu_mhz"] = float(freq.current)
        except Exception:
            pass
        try:
            mem = psutil.virtual_memory()
            rec["mem_pct"] = float(mem.percent)
            rec["mem_used_gb"] = float(mem.used) / (1024 ** 3)
            rec["mem_total_gb"] = float(mem.total) / (1024 ** 3)
        except Exception:
            pass
        try:
            p = proc or psutil.Process()
            rec["proc_cpu_pct"] = float(p.cpu_percent(interval=None))
            rec["proc_mem_gb"] = float(p.memory_info().rss) / (1024 ** 3)
        except Exception:
            pass
        cpu_temp_max = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                vals: list[float] = []
                for arr in temps.values():
                    for t in arr:
                        cur = getattr(t, "current", None)
                        if cur is not None:
                            vals.append(float(cur))
                if vals:
                    cpu_temp_max = max(vals)
                    rec["temp_c"] = cpu_temp_max
        except Exception:
            pass
        gpu_temp = None
        try:
            import GPUtil  # type: ignore
            gpus = GPUtil.getGPUs()
            if gpus:
                try:
                    rec["gpu_pct"] = float(max(g.load for g in gpus) * 100.0)
                except Exception:
                    pass
                try:
                    temps = [g.temperature for g in gpus if g.temperature is not None]
                    if temps:
                        gpu_temp = float(max(temps))
                        rec["gpu_temp_c"] = gpu_temp
                except Exception:
                    pass
        except Exception:
            pass
        if "gpu_pct" not in rec:
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    rec["gpu_pct"] = float(util.gpu)
                    try:
                        gpu_temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                        rec["gpu_temp_c"] = gpu_temp
                    except Exception:
                        pass
                finally:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass
            except Exception:
                pass
        if gpu_temp is not None:
            if cpu_temp_max is None:
                rec["temp_c"] = gpu_temp
            else:
                rec["temp_c"] = max(cpu_temp_max, gpu_temp)
        return rec

    def _start_system_monitor(self) -> None:
        if not bool(self.settings.monitor_system):
            return
        if self._sysmon_thread and self._sysmon_thread.is_alive():
            return
        interval = max(1, int(self.settings.monitor_interval_sec or 5))
        log_path = Path(self.settings.monitor_log_path)
        proc = psutil.Process()
        proc.cpu_percent(interval=None)

        def _loop() -> None:
            while not self.stop_evt.is_set():
                try:
                    rec = self._system_snapshot(proc)
                    self._latest_sys_metrics = rec
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                except Exception:
                    pass
                time.sleep(interval)

        self._sysmon_thread = threading.Thread(target=_loop, name="sysmon", daemon=True)
        self._sysmon_thread.start()

    def _score_symbols_batch(self, symbols: List[str]) -> List[Tuple[str, dict]]:
        rows: List[np.ndarray] = []
        metas: List[tuple[str, dict]] = []
        cols = self.model_bundle.feature_cols
        for sym in symbols:
            win = self.windows.get(sym)
            if win is None or not win.is_ready():
                continue
            if win.need_reload:
                try:
                    window_minutes = int(self.settings.window_minutes) if self.settings.window_minutes > 0 else max_pf_window_minutes()
                    days = window_days_for_minutes(window_minutes)
                    df_reload = load_ohlc_1m_series(sym, int(days), remove_tail_days=0)
                    win.replace_from_df(df_reload)
                except Exception:
                    continue
            df = win.to_frame()
            if df.empty:
                continue
            last_ts = int(df.index[-1].value // 1_000_000)
            if len(df) < win.min_ready_rows:
                continue
            last = self._build_features_latest(sym, df)
            if last is None:
                continue
            price = float(last.get("close", np.nan))
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

            row = np.zeros((len(cols),), dtype=np.float32)
            for i, c in enumerate(cols):
                v = float(last.get(c, 0.0))
                if not np.isfinite(v):
                    v = 0.0
                row[i] = v
            metas.append(
                (
                    sym,
                    {
                        "price": price,
                        "ts_ms": last_ts,
                        "range_1h_pct": (range_1h / price * 100.0) if price > 0 else 0.0,
                        "range_24h_pct": (range_24h / price * 100.0) if price > 0 else 0.0,
                    },
                )
            )
            rows.append(row)

        if not rows:
            return []
        try:
            dmat = xgb.DMatrix(np.vstack(rows))
            with self._predict_lock:
                preds = self.model_bundle.model.predict(dmat, validate_features=False)
                preds = _apply_calibration(np.asarray(preds, dtype=np.float64), self.model_bundle.calib)
        except Exception as e:
            log.warning("[score][batch] falha no predict: %s: %s", type(e).__name__, e)
            return []
        results: List[Tuple[str, dict]] = []
        for (sym, meta), pe in zip(metas, preds):
            res = {
                "symbol": sym,
                "p_entry": float(pe),
                "buy": bool(float(pe) >= self.model_bundle.tau_entry),
                **meta,
            }
            results.append((sym, res))
        return results

    def _prime_scores(self) -> None:
        if self._init_scores_done:
            return
        ready = sum(1 for s in self.symbols if self.windows.get(s) and self.windows[s].is_ready())
        log.info("[score] janelas prontas para scoring: %s/%s", ready, len(self.symbols))

        def _refresh():
            self._set_runtime_stage("score_prime", f"{len(self.symbols)} symbols")
            log.info("[score] calculando scores iniciais (modelo) para %s símbolos (batch)...", len(self.symbols))
            started = time.time()
            ok = 0
            proc_total = 0.0
            chunk = max(8, int(self.settings.score_workers or 16))
            for i in range(0, len(self.symbols), chunk):
                batch = self.symbols[i : i + chunk]
                try:
                    t0 = time.time()
                    batch_res = self._score_symbols_batch(batch)
                    proc_total += time.time() - t0
                    for sym, res in batch_res:
                        self._upsert_symbol_score(sym, res)
                        ok += 1
                except Exception as e:
                    log.warning("[score][prime] batch falhou: %s: %s", type(e).__name__, e)
                if (i + chunk) % 100 == 0 or (i + chunk) >= len(self.symbols):
                    log.info("[score] iniciais (modelo): %s/%s (ok=%s)", min(i + chunk, len(self.symbols)), len(self.symbols), ok)
            elapsed = time.time() - started
            self._last_score_cycle_sec = float(elapsed)
            self._last_score_proc_sec = float(proc_total)
            self._last_score_cycle_ts = time.time()
            self._last_score_cycle_symbols = int(ok)
            log.info("[score] iniciais (modelo) concluídos em %.1fs (ok=%s)", elapsed, ok)
            self._set_runtime_stage("idle")
            self._push_dashboard_state()

        self._score_refresh_thread = threading.Thread(target=_refresh, name="score-prime", daemon=True)
        self._score_refresh_thread.start()
        self._init_scores_done = True

    def start(self) -> None:
        self._set_runtime_stage("bootstrap")
        if self._last_score_cycle_sec is None:
            self._last_score_cycle_sec = 0.0
        if self._last_score_proc_sec is None:
            self._last_score_proc_sec = 0.0
        if self._last_score_cycle_ts is None:
            self._last_score_cycle_ts = time.time()
        if self._last_score_cycle_symbols == 0:
            self._last_score_cycle_symbols = 0
        self._bootstrap_rest()
        self._set_runtime_stage("init_windows")
        self._init_windows()
        self._prime_scores()
        self._set_runtime_stage("backfill")
        self._start_backfill_workers()
        
        # Start Multiple Decision Workers
        n_workers = max(4, int(self.settings.score_workers or 8))
        log.info("[init] iniciando %s decision workers...", n_workers)
        for i in range(n_workers):
            threading.Thread(target=self._decision_worker, daemon=True, name=f"dec-{i}").start()

        self._start_score_keepalive()
        self._start_system_monitor()
        self._push_dashboard_state(force=True)
        self._set_runtime_stage("idle")

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

        # Inicia WS Ingestor
        self.ws_ingestor.start(self.settings)
        
        # Loop principal (mantém main thread viva)
        while True:
            time.sleep(3600)


def _start_ngrok(port: int, *, token_env: str = "NGROK_AUTHTOKEN") -> Optional[str]:
    if NgrokCfg is not None and NgrokMgr is not None:
        try:
            cfg = NgrokCfg()
            if os.getenv("WF_NGROK_DOMAIN"):
                cfg.domain = os.getenv("WF_NGROK_DOMAIN", "").strip()
            if os.getenv("NGROK_DOMAIN"):
                cfg.domain = os.getenv("NGROK_DOMAIN", "").strip() or cfg.domain
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
                return f"https://{cfg.domain}"
        except Exception as e:
            print(f"[ngrok][warn] NgrokMgr falhou, tentando pyngrok: {type(e).__name__}: {e}", flush=True)

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
        except Exception:
            pass
    try:
        t = ngrok.connect(int(port), bind_tls=True)
        return t.public_url
    except Exception as e:
        print(f"[ngrok][warn] falhou ao abrir túnel: {type(e).__name__}: {e}", flush=True)
        return None


def _launch_dashboard(port: int) -> None:
    try:
        from modules.realtime.dashboard_server import create_app
    except ImportError:
        try:
            from realtime.dashboard_server import create_app
        except ImportError:
            print("[dash][warn] dashboard_server não encontrado", flush=True)
            return
    try:
        from werkzeug.serving import run_simple, WSGIRequestHandler
    except ImportError:
        print("[dash][warn] werkzeug não disponível; dashboard não iniciado", flush=True)
        return
    try:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        WSGIRequestHandler.log = lambda *args, **kwargs: None
    except Exception:
        pass
    app, store = create_app(demo=False, refresh_sec=2.0)
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


def _install_fatal_logger() -> None:
    def _hook(exc_type, exc, tb):
        try:
            log.critical("[fatal] exceção não tratada: %s: %s", exc_type.__name__, exc, exc_info=(exc_type, exc, tb))
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    def _on_exit() -> None:
        try:
            log.info("[lifecycle] processo encerrando (atexit)")
        except Exception:
            pass

    sys.excepthook = _hook
    atexit.register(_on_exit)


def main() -> None:
    _install_fatal_logger()
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
        trade_mode="paper",
        trade_notional_usd=100.0,
        paper_start_equity=10_000.0,
        start_dashboard=True,
        dashboard_port=5055,
        init_workers=8,
        score_workers=16,
        score_cooldown_sec=0.0,
        monitor_system=True,
        monitor_interval_sec=1,
        monitor_log_path="data/sysmon.jsonl",
    )
    bot = LiveDecisionBot(settings)
    bot.start()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        try:
            log.critical("[fatal] exceção na thread principal: %s: %s", type(e).__name__, e, exc_info=True)
        except Exception:
            pass
        raise
