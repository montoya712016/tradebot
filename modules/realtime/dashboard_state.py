from __future__ import annotations

import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AccountSummary:
    equity_usd: float
    cash_usd: float
    exposure_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    updated_at_utc: str = field(default_factory=_utc_iso)


@dataclass
class Position:
    symbol: str
    side: str  # "LONG" | "SHORT"
    qty: float
    entry_price: float
    mark_price: float
    notional_usd: float
    pnl_usd: float
    pnl_pct: float


@dataclass
class Trade:
    ts_utc: str
    symbol: str
    action: str  # "BUY" | "SELL" | "CLOSE" | etc.
    qty: float
    price: float
    pnl_usd: float | None = None


@dataclass
class DashboardState:
    summary: AccountSummary
    positions: list[Position] = field(default_factory=list)
    recent_trades: list[Trade] = field(default_factory=list)
    allocation: dict[str, float] = field(default_factory=dict)  # symbol -> notional_usd
    meta: dict[str, Any] = field(default_factory=dict)
    equity_history: list[dict[str, Any]] = field(default_factory=list)  # [{ts_utc, equity_usd, exposure_usd}]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generated_at_utc"] = _utc_iso()
        return d


class StateStore:
    """
    Store simples em memória, thread-safe.

    Depois a gente pluga o bot real chamando `update_from_payload()`.
    """

    def __init__(self, initial: DashboardState):
        self._lock = threading.Lock()
        self._state = initial

    def get(self) -> DashboardState:
        with self._lock:
            return self._state

    def set(self, st: DashboardState) -> None:
        with self._lock:
            self._state = st

    def update_from_payload(self, payload: dict[str, Any]) -> None:
        """
        Payload esperado (tudo opcional):
        - summary: {...}
        - positions: [...]
        - recent_trades: [...]
        - allocation: {...}
        - meta: {...}
        - equity_history: [...]  (opcional; se ausente, será alimentada com summary)
        """
        with self._lock:
            cur = self._state.to_dict()
            cur.update(payload or {})

            summary_in = cur.get("summary") or {}
            summary = AccountSummary(**summary_in)
            positions = [Position(**p) for p in (cur.get("positions") or [])]
            trades = [Trade(**t) for t in (cur.get("recent_trades") or [])]
            allocation = dict(cur.get("allocation") or {})
            meta = dict(cur.get("meta") or {})
            hist_in = list(cur.get("equity_history") or self._state.equity_history or [])
            # acrescenta ponto atual se summary disponível
            if summary.equity_usd is not None:
                hist_in.append(
                    {
                        "ts_utc": summary.updated_at_utc or _utc_iso(),
                        "equity_usd": float(summary.equity_usd),
                        "exposure_usd": float(summary.exposure_usd),
                    }
                )
            # limita tamanho para evitar crescimento ilimitado
            MAX_POINTS = 500
            if len(hist_in) > MAX_POINTS:
                hist_in = hist_in[-MAX_POINTS:]

            self._state = DashboardState(
                summary=summary,
                positions=positions,
                recent_trades=trades,
                allocation=allocation,
                meta=meta,
                equity_history=hist_in,
            )


def create_demo_state() -> DashboardState:
    positions = [
        Position(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.015,
            entry_price=41250.0,
            mark_price=41820.0,
            notional_usd=0.015 * 41820.0,
            pnl_usd=0.015 * (41820.0 - 41250.0),
            pnl_pct=(41820.0 / 41250.0 - 1.0) * 100.0,
        ),
        Position(
            symbol="ETHUSDT",
            side="LONG",
            qty=0.35,
            entry_price=2210.0,
            mark_price=2258.0,
            notional_usd=0.35 * 2258.0,
            pnl_usd=0.35 * (2258.0 - 2210.0),
            pnl_pct=(2258.0 / 2210.0 - 1.0) * 100.0,
        ),
        Position(
            symbol="SOLUSDT",
            side="LONG",
            qty=7.0,
            entry_price=96.0,
            mark_price=93.6,
            notional_usd=7.0 * 93.6,
            pnl_usd=7.0 * (93.6 - 96.0),
            pnl_pct=(93.6 / 96.0 - 1.0) * 100.0,
        ),
    ]
    exposure = sum(p.notional_usd for p in positions)
    unreal = sum(p.pnl_usd for p in positions)
    summary = AccountSummary(
        equity_usd=2500.0 + unreal,
        cash_usd=2500.0 - exposure,
        exposure_usd=exposure,
        realized_pnl_usd=35.2,
        unrealized_pnl_usd=unreal,
    )
    allocation = {p.symbol: float(p.notional_usd) for p in positions}
    trades = [
        Trade(ts_utc=_utc_iso(), symbol="BTCUSDT", action="BUY", qty=0.015, price=41250.0),
        Trade(ts_utc=_utc_iso(), symbol="ETHUSDT", action="BUY", qty=0.35, price=2210.0),
        Trade(ts_utc=_utc_iso(), symbol="SOLUSDT", action="BUY", qty=7.0, price=96.0),
    ]
    # histórico sintético
    equity_history: list[dict[str, Any]] = []
    base_eq = summary.equity_usd
    for i in range(40):
        ts = _utc_iso()
        drift = (i - 20) * 0.001
        eq = base_eq * (1.0 + drift)
        equity_history.append({"ts_utc": ts, "equity_usd": float(eq), "exposure_usd": float(exposure)})
    return DashboardState(
        summary=summary,
        positions=positions,
        recent_trades=trades,
        allocation=allocation,
        meta={"mode": "demo"},
        equity_history=equity_history,
    )


class DemoStateGenerator:
    """
    Gera variações suaves no estado, só pra visualizar 'tempo real'.
    """

    def __init__(self, store: StateStore, refresh_sec: float = 2.0):
        self.store = store
        self.refresh_sec = float(refresh_sec)
        self._evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._evt.set()

    def _loop(self) -> None:
        while not self._evt.is_set():
            time.sleep(self.refresh_sec)
            st = self.store.get()
            pos = list(st.positions)
            if not pos:
                continue

            # mexe o mark_price levemente
            new_positions: list[Position] = []
            for p in pos:
                drift = random.uniform(-0.004, 0.004)
                new_mark = max(0.0001, p.mark_price * (1.0 + drift))
                pnl = p.qty * (new_mark - p.entry_price) * (1.0 if p.side.upper() == "LONG" else -1.0)
                pnl_pct = (new_mark / p.entry_price - 1.0) * 100.0
                notional = p.qty * new_mark
                new_positions.append(
                    Position(
                        symbol=p.symbol,
                        side=p.side,
                        qty=p.qty,
                        entry_price=p.entry_price,
                        mark_price=new_mark,
                        notional_usd=notional,
                        pnl_usd=pnl,
                        pnl_pct=pnl_pct,
                    )
                )

            exposure = sum(p.notional_usd for p in new_positions)
            unreal = sum(p.pnl_usd for p in new_positions)
            summary = AccountSummary(
                equity_usd=max(0.0, 2500.0 + unreal),
                cash_usd=2500.0 - exposure,
                exposure_usd=exposure,
                realized_pnl_usd=st.summary.realized_pnl_usd,
                unrealized_pnl_usd=unreal,
            )
            allocation = {p.symbol: float(p.notional_usd) for p in new_positions}

            self.store.set(
                DashboardState(
                    summary=summary,
                    positions=new_positions,
                    recent_trades=st.recent_trades[:],
                    allocation=allocation,
                    meta=st.meta,
                )
            )
