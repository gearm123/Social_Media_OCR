"""PostgreSQL pool for ``DATABASE_URL`` (Render ``translate-chat-db``, Neon, local, etc.)."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from psycopg_pool import ConnectionPool

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def _optional_env(key: str) -> str:
    """Same pattern as ``paddle_client`` / ``billing_api``: optional string from the environment."""
    return (os.environ.get(key) or "").strip()


def normalize_database_url(url: str) -> str:
    u = url.strip()
    if u.startswith("postgres://"):
        return "postgresql://" + u[len("postgres://") :]
    return u


def raw_database_url_from_environment() -> str:
    """Non-empty URL from env or ``DATABASE_URL_FILE``, or ``''``. Never log the return value."""
    for key in ("DATABASE_URL", "POSTGRES_URL"):
        v = _optional_env(key).lstrip("\ufeff")
        if v:
            return v
    path = _optional_env("DATABASE_URL_FILE")
    if path:
        try:
            p = Path(path)
            if p.is_file():
                lines = p.read_text(encoding="utf-8").splitlines()
                u = (lines[0] if lines else "").strip().lstrip("\ufeff")
                if u:
                    return u
        except OSError:
            pass
    return ""


def get_database_url() -> str:
    raw = raw_database_url_from_environment()
    if not raw:
        raise RuntimeError(
            "DATABASE_URL is not set. On the Render web service, set DATABASE_URL to the "
            "Internal Database URL from translate-chat-db (or use POSTGRES_URL). "
            "Local: add DATABASE_URL to .env. See GET /health field database_url_configured."
        )
    return normalize_database_url(raw)


def get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                conninfo=get_database_url(),
                min_size=1,
                max_size=10,
            )
        return _pool
