from __future__ import annotations

"""
NotificaÃ§Ãµes via Pushover.

Uso (recomendado):
- Defina variÃ¡veis de ambiente:
  - PUSHOVER_USER_KEY
  - PUSHOVER_TOKEN_TRADE (ou PUSHOVER_TOKEN_ERROR, etc.)

Opcional (local):
- Se existir `config/secrets.py`, ele pode ser usado como fallback.
  Esse arquivo deve ficar no `.gitignore`.
"""

from dataclasses import dataclass
import json
import os
import urllib.parse
import urllib.request
from typing import Any


PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


@dataclass(frozen=True)
class PushoverConfig:
    user: str
    token: str
    title: str | None = None
    device: str | None = None
    priority: int | None = None
    sound: str | None = None


def _env(name: str) -> str | None:
    v = os.getenv(name)
    v = v.strip() if isinstance(v, str) else None
    return v or None


def load_from_env(
    *,
    user_env: str = "PUSHOVER_USER_KEY",
    token_env: str = "PUSHOVER_TOKEN",
    title: str | None = None,
    device: str | None = None,
    priority: int | None = None,
    sound: str | None = None,
) -> PushoverConfig | None:
    user = _env(user_env)
    token = _env(token_env)
    if not user or not token:
        return None
    return PushoverConfig(user=user, token=token, title=title, device=device, priority=priority, sound=sound)


def load_from_local_secrets(
    *,
    token_name: str = "PUSHOVER_TOKEN_TRADE",
    title: str | None = None,
    device: str | None = None,
    priority: int | None = None,
    sound: str | None = None,
) -> PushoverConfig | None:
    """
    Fallback local: lÃª `config/secrets.py` (se existir).
    """
    try:
        # import absoluto (funciona quando `modules/` esta no sys.path)
        from config import secrets as sec  # type: ignore
    except Exception:
        try:
            # import relativo (quando importado como pacote)
            from config import secrets as sec  # type: ignore
        except Exception:
            return None

    user = getattr(sec, "PUSHOVER_USER_KEY", None)
    token = getattr(sec, str(token_name), None)
    if not isinstance(user, str) or not user.strip():
        return None
    if not isinstance(token, str) or not token.strip():
        return None
    return PushoverConfig(user=user.strip(), token=token.strip(), title=title, device=device, priority=priority, sound=sound)


def load_default(
    *,
    user_env: str = "PUSHOVER_USER_KEY",
    token_env: str = "PUSHOVER_TOKEN_TRADE",
    token_name_fallback: str = "PUSHOVER_TOKEN_TRADE",
    title: str | None = None,
    device: str | None = None,
    priority: int | None = None,
    sound: str | None = None,
) -> PushoverConfig | None:
    """
    Ordem:
    1) ENV
    2) config/secrets.py (local)
    """
    cfg = load_from_env(user_env=user_env, token_env=token_env, title=title, device=device, priority=priority, sound=sound)
    if cfg is not None:
        return cfg
    return load_from_local_secrets(token_name=token_name_fallback, title=title, device=device, priority=priority, sound=sound)


def send_pushover(
    message: str,
    *,
    cfg: PushoverConfig | None = None,
    token: str | None = None,
    user: str | None = None,
    title: str | None = None,
    url: str | None = None,
    url_title: str | None = None,
    device: str | None = None,
    priority: int | None = None,
    sound: str | None = None,
    timeout_sec: float = 10.0,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """
    Envia mensagem via Pushover.

    Retorna: (ok, json_resposta|None, erro|None)
    """
    if cfg is not None:
        if user is None:
            user = cfg.user
        if token is None:
            token = cfg.token
        if title is None:
            title = cfg.title
        if device is None:
            device = cfg.device
        if priority is None:
            priority = cfg.priority
        if sound is None:
            sound = cfg.sound

    user = (user or "").strip()
    token = (token or "").strip()
    if not user or not token:
        return False, None, "missing_user_or_token"

    data: dict[str, Any] = {"token": token, "user": user, "message": str(message)}
    if title:
        data["title"] = str(title)
    if url:
        data["url"] = str(url)
    if url_title:
        data["url_title"] = str(url_title)
    if device:
        data["device"] = str(device)
    if priority is not None:
        data["priority"] = int(priority)
    if sound:
        data["sound"] = str(sound)

    try:
        payload = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(PUSHOVER_URL, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read()
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            obj = {"raw": raw.decode("utf-8", errors="replace")}
        return True, obj, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


