"""SQLite user accounts (local file — no external DB). Thread-safe for FastAPI + job threads."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    created_at: str


class UserStore:
    def __init__(self, db_path: Path):
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def create_user(self, email: str, password_hash: str) -> UserRecord:
        uid = uuid.uuid4().hex
        created = _utc_now_iso()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (uid, email.strip().lower(), password_hash, created),
                    )
                    conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE" in str(e).upper() or "unique" in str(e).lower():
                    raise ValueError("email already registered") from e
                raise
        return UserRecord(id=uid, email=email.strip().lower(), created_at=created)

    def get_by_email(self, email: str) -> Optional[tuple[UserRecord, str]]:
        """Return (UserRecord, password_hash) or None."""
        em = email.strip().lower()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
                    (em,),
                ).fetchone()
        if not row:
            return None
        rec = UserRecord(id=row["id"], email=row["email"], created_at=row["created_at"])
        return rec, row["password_hash"]

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, email, created_at FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
        if not row:
            return None
        return UserRecord(id=row["id"], email=row["email"], created_at=row["created_at"])


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> None:
    if not email or not email.strip():
        raise ValueError("email is required")
    if not _EMAIL_RE.match(email.strip()):
        raise ValueError("invalid email format")


def validate_password(password: str) -> None:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")


def user_store_from_env(base_dir: Path) -> UserStore:
    """Default: ``<base_dir>/data/users.sqlite3``. Override with USER_DB_PATH."""
    raw = os.environ.get("USER_DB_PATH", "").strip()
    path = Path(raw) if raw else base_dir / "data" / "users.sqlite3"
    return UserStore(path)
