# -*- coding: utf-8 -*-
"""
Live Settings - Configurações do bot em tempo real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from realtime.market_data.mysql import MySQLConfig


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
    score_workers: int = 16
    symbols_limit: int = 0  # 0 => todos
    ngrok_domain: str = ""  # opcional (pago)
    ngrok_basic_user: str = ""
    ngrok_basic_pass: str = ""
    init_workers: int = 8
    prime_push_top: int = 500
    # throttling/scoring
    score_cooldown_sec: float = 0.0  # 0 => sem cooldown
    # monitoramento de recursos
    monitor_system: bool = True
    monitor_interval_sec: int = 1
    monitor_log_path: str = "data/sysmon.jsonl"
    # preditor
    use_gpu_predictor: bool = True
