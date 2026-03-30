from __future__ import annotations

import logging

from core.utils.notify import PushoverConfig, load_default, send_pushover


log = logging.getLogger("realtime.services.pushover")


def load_trade_pushover_config() -> PushoverConfig | None:
    return load_default(
        user_env="PUSHOVER_USER_KEY",
        token_env="PUSHOVER_TOKEN_TRADE",
        token_name_fallback="PUSHOVER_TOKEN_TRADE",
        title="Tradebot",
        priority=0,
    )


def send_trade_notification(
    *,
    symbol: str,
    side: str,
    price: float,
    notional: float,
    pnl: float | None = None,
    dashboard_url: str | None = None,
    cfg: PushoverConfig | None = None,
) -> bool:
    cfg = cfg or load_trade_pushover_config()
    if cfg is None:
        return False
    msg = f"{str(side).upper()} {symbol} notional={float(notional):.2f} price={float(price):.4f}"
    if pnl is not None:
        msg += f" pnl={float(pnl):.2f}"
    try:
        ok, _, err = send_pushover(
            msg,
            cfg=cfg,
            url=str(dashboard_url or "").strip() or None,
            url_title="Dashboard" if dashboard_url else None,
        )
        if not ok and err:
            log.warning("[pushover] envio falhou: %s", err)
        return bool(ok)
    except Exception:
        log.warning("[pushover] envio falhou", exc_info=True)
        return False
