from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return ""


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


@dataclass(frozen=True)
class DashboardAuthConfig:
    enabled: bool
    session_key: str
    user_db_path: Path
    bootstrap_username: str
    bootstrap_password: str


def load_dashboard_auth_config(
    *,
    user_envs: tuple[str, ...] = ("ASTRA_DASHBOARD_USER", "DASHBOARD_USER", "NGROK_BASIC_USER"),
    pass_envs: tuple[str, ...] = ("ASTRA_DASHBOARD_PASS", "DASHBOARD_PASS", "NGROK_BASIC_PASS"),
    default_user: str = "",
    default_pass: str = "",
    disable_envs: tuple[str, ...] = ("ASTRA_DASHBOARD_DISABLE_AUTH", "DASHBOARD_DISABLE_AUTH"),
    session_key: str = "astra_dashboard_auth",
    db_envs: tuple[str, ...] = ("ASTRA_DASHBOARD_USER_DB", "DASHBOARD_USER_DB"),
    default_db_relpath: str = "local/dashboard_users.json",
) -> DashboardAuthConfig:
    disabled = False
    for name in disable_envs:
        value = str(os.getenv(name, "") or "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            disabled = True
            break

    db_raw = _first_env(db_envs) or default_db_relpath
    user_db_path = Path(db_raw)
    if not user_db_path.is_absolute():
        user_db_path = Path.cwd() / user_db_path

    return DashboardAuthConfig(
        enabled=not disabled,
        session_key=session_key,
        user_db_path=user_db_path,
        bootstrap_username=_normalize_username(_first_env(user_envs) or default_user),
        bootstrap_password=_first_env(pass_envs) or default_pass,
    )


def normalize_next_url(target: str | None, fallback: str = "/") -> str:
    target = str(target or "").strip()
    if target.startswith("/") and not target.startswith("//"):
        return target
    return fallback


def _default_store() -> dict[str, Any]:
    return {"version": 1, "users": []}


def _write_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(path.parent), suffix=".tmp") as tmp:
        json.dump(store, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_store()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return _default_store()
        users = data.get("users")
        if not isinstance(users, list):
            data["users"] = []
        return data
    except Exception:
        return _default_store()


def _find_user(store: dict[str, Any], username: str) -> dict[str, Any] | None:
    norm = _normalize_username(username)
    for user in store.get("users", []):
        if _normalize_username(user.get("username", "")) == norm:
            return user
    return None


def ensure_user_store(cfg: DashboardAuthConfig) -> Path:
    store = _load_store(cfg.user_db_path)
    changed = False
    if cfg.bootstrap_username and cfg.bootstrap_password and not _find_user(store, cfg.bootstrap_username):
        store["users"].append(
            {
                "full_name": "Bootstrap User",
                "cpf": "",
                "phone": "",
                "email": "",
                "username": cfg.bootstrap_username,
                "password_hash": generate_password_hash(cfg.bootstrap_password),
                "enabled": True,
                "is_admin": True,
                "is_owner": True,
                "created_at": _utc_now_iso(),
                "approved_at": _utc_now_iso(),
                "notes": "bootstrap account",
            }
        )
        changed = True
    if changed or not cfg.user_db_path.exists():
        _write_store(cfg.user_db_path, store)
    return cfg.user_db_path


def get_user(cfg: DashboardAuthConfig, username: str) -> dict[str, Any] | None:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    return _find_user(store, username)


def list_users(cfg: DashboardAuthConfig) -> list[dict[str, Any]]:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    rows: list[dict[str, Any]] = []
    for user in store.get("users", []):
        rows.append(
            {
                "full_name": str(user.get("full_name", "") or ""),
                "cpf": str(user.get("cpf", "") or ""),
                "phone": str(user.get("phone", "") or ""),
                "email": str(user.get("email", "") or ""),
                "username": str(user.get("username", "") or ""),
                "enabled": bool(user.get("enabled")),
                "is_admin": bool(user.get("is_admin")),
                "is_owner": bool(user.get("is_owner")),
                "created_at": user.get("created_at"),
                "approved_at": user.get("approved_at"),
                "notes": str(user.get("notes", "") or ""),
            }
        )
    rows.sort(key=lambda item: (not item["enabled"], item["username"]))
    return rows


def session_has_access(cfg: DashboardAuthConfig, username: str | None) -> bool:
    user = get_user(cfg, str(username or ""))
    if not user:
        return False
    return bool(user.get("enabled"))


def session_is_admin(cfg: DashboardAuthConfig, username: str | None) -> bool:
    user = get_user(cfg, str(username or ""))
    if not user:
        return False
    return bool(user.get("enabled")) and bool(user.get("is_admin"))


def session_is_owner(cfg: DashboardAuthConfig, username: str | None) -> bool:
    user = get_user(cfg, str(username or ""))
    if not user:
        return False
    return bool(user.get("enabled")) and bool(user.get("is_owner"))


def verify_user_login(cfg: DashboardAuthConfig, username: str, password: str) -> tuple[bool, str]:
    user = get_user(cfg, username)
    if not user:
        return False, "Usuário ou senha inválidos."
    if not bool(user.get("enabled")):
        return False, "Conta cadastrada, mas ainda sem acesso liberado."
    password_hash = str(user.get("password_hash", "") or "")
    if not password_hash or not check_password_hash(password_hash, str(password or "")):
        return False, "Usuário ou senha inválidos."
    return True, ""


def register_user(
    cfg: DashboardAuthConfig,
    *,
    full_name: str,
    cpf: str,
    phone: str,
    email: str,
    username: str,
    password: str,
) -> tuple[bool, str]:
    ensure_user_store(cfg)
    full_name = str(full_name or "").strip()
    cpf = str(cpf or "").strip()
    phone = str(phone or "").strip()
    email = _normalize_email(email)
    norm = _normalize_username(username)
    cpf_digits = "".join(ch for ch in cpf if ch.isdigit())
    phone_digits = "".join(ch for ch in phone if ch.isdigit())

    if len(full_name) < 5:
        return False, "Informe o nome completo."
    if len(cpf_digits) != 11:
        return False, "Informe um CPF com 11 dígitos."
    if len(phone_digits) < 10:
        return False, "Informe um telefone válido com DDD."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Informe um email válido."
    if len(norm) < 3:
        return False, "O usuário precisa ter pelo menos 3 caracteres."
    if len(str(password or "")) < 6:
        return False, "A senha precisa ter pelo menos 6 caracteres."

    store = _load_store(cfg.user_db_path)
    existing = _find_user(store, norm)
    if existing:
        if bool(existing.get("enabled")):
            return False, "Esse usuário já existe e já tem acesso liberado."
        return False, "Esse usuário já foi cadastrado e está aguardando liberação."

    store["users"].append(
        {
            "full_name": full_name,
            "cpf": cpf_digits,
            "phone": phone,
            "email": email,
            "username": norm,
            "password_hash": generate_password_hash(str(password)),
            "enabled": False,
            "is_admin": False,
            "is_owner": False,
            "created_at": _utc_now_iso(),
            "approved_at": None,
            "notes": "pending approval",
        }
    )
    _write_store(cfg.user_db_path, store)
    return True, "Cadastro criado. Aguarde a liberação manual do acesso."


def set_user_enabled(cfg: DashboardAuthConfig, username: str, enabled: bool) -> tuple[bool, str]:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    user = _find_user(store, username)
    if not user:
        return False, "Usuário não encontrado."
    user["enabled"] = bool(enabled)
    user["approved_at"] = _utc_now_iso() if enabled else None
    user["notes"] = "approved via admin" if enabled else "blocked via admin"
    _write_store(cfg.user_db_path, store)
    return True, "Usuário atualizado."


def set_user_admin(cfg: DashboardAuthConfig, username: str, is_admin: bool) -> tuple[bool, str]:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    user = _find_user(store, username)
    if not user:
        return False, "Usuário não encontrado."
    user["is_admin"] = bool(is_admin)
    if bool(is_admin):
        user["notes"] = "promoted to admin"
    elif not user.get("notes"):
        user["notes"] = "standard user"
    _write_store(cfg.user_db_path, store)
    return True, "Papel administrativo atualizado."


def set_user_owner(cfg: DashboardAuthConfig, username: str, is_owner: bool) -> tuple[bool, str]:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    user = _find_user(store, username)
    if not user:
        return False, "Usuário não encontrado."
    user["is_owner"] = bool(is_owner)
    if bool(is_owner):
        user["is_admin"] = True
        user["notes"] = "promoted to owner"
    elif not user.get("notes"):
        user["notes"] = "admin user"
    _write_store(cfg.user_db_path, store)
    return True, "Papel de owner atualizado."


def delete_user(cfg: DashboardAuthConfig, username: str) -> tuple[bool, str]:
    ensure_user_store(cfg)
    store = _load_store(cfg.user_db_path)
    norm = _normalize_username(username)
    users = store.get("users", [])
    new_users = [u for u in users if _normalize_username(u.get("username", "")) != norm]
    if len(new_users) == len(users):
        return False, "Usuário não encontrado."
    store["users"] = new_users
    _write_store(cfg.user_db_path, store)
    return True, "Usuário removido."
