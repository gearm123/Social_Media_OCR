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
from typing import Any, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Password login is disabled for rows using this sentinel (social-only accounts).
OAUTH_PASSWORD_SENTINEL = "__oauth_no_password__"


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    username: Optional[str]
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

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "username" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN username TEXT COLLATE NOCASE")
        if "oauth_provider" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN oauth_provider TEXT")
        if "oauth_subject" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN oauth_subject TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username "
            "ON users(username) WHERE username IS NOT NULL AND length(trim(username)) > 0"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth "
            "ON users(oauth_provider, oauth_subject) "
            "WHERE oauth_provider IS NOT NULL AND oauth_subject IS NOT NULL"
        )
        conn.commit()

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
                self._migrate(conn)

    def _row_to_record(self, row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            id=row["id"],
            email=row["email"],
            username=row["username"] if "username" in row.keys() and row["username"] else None,
            created_at=row["created_at"],
        )

    def create_user_with_password(
        self, email: str, password_hash: str, username: str
    ) -> UserRecord:
        uid = uuid.uuid4().hex
        created = _utc_now_iso()
        un = username.strip()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO users (id, email, password_hash, created_at, username,
                            oauth_provider, oauth_subject)
                        VALUES (?, ?, ?, ?, ?, NULL, NULL)
                        """,
                        (uid, email.strip().lower(), password_hash, created, un),
                    )
                    conn.commit()
            except sqlite3.IntegrityError as e:
                msg = str(e).lower()
                if "users.email" in msg or "email" in msg:
                    raise ValueError("email already registered") from e
                if "username" in msg:
                    raise ValueError("username already taken") from e
                raise ValueError("could not create account") from e
        return UserRecord(id=uid, email=email.strip().lower(), username=un, created_at=created)

    def create_oauth_user(
        self,
        provider: str,
        subject: str,
        email: str,
        username: str,
    ) -> UserRecord:
        uid = uuid.uuid4().hex
        created = _utc_now_iso()
        prov = provider.strip().lower()
        sub = subject.strip()
        em = email.strip().lower()
        un = username.strip()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO users (id, email, password_hash, created_at, username,
                            oauth_provider, oauth_subject)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (uid, em, OAUTH_PASSWORD_SENTINEL, created, un, prov, sub),
                    )
                    conn.commit()
            except sqlite3.IntegrityError as e:
                msg = str(e).lower()
                if "email" in msg and "unique" in msg:
                    raise ValueError("email already registered") from e
                if "username" in msg:
                    raise ValueError("username already taken") from e
                if "oauth" in msg or "idx_users_oauth" in msg:
                    raise ValueError("oauth account already linked") from e
                raise ValueError("could not create account") from e
        return UserRecord(id=uid, email=em, username=un, created_at=created)

    def get_by_oauth(self, provider: str, subject: str) -> Optional[UserRecord]:
        prov = provider.strip().lower()
        sub = subject.strip()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, email, username, created_at FROM users
                    WHERE oauth_provider = ? AND oauth_subject = ?
                    """,
                    (prov, sub),
                ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_email(self, email: str) -> Optional[Tuple[UserRecord, str]]:
        em = email.strip().lower()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, email, username, created_at, password_hash FROM users WHERE email = ?
                    """,
                    (em,),
                ).fetchone()
        if not row:
            return None
        rec = UserRecord(
            id=row["id"],
            email=row["email"],
            username=row["username"] if row["username"] else None,
            created_at=row["created_at"],
        )
        return rec, row["password_hash"]

    def get_by_id(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, email, username, created_at FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)


_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,32}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(email: str) -> None:
    if not email or not email.strip():
        raise ValueError("email is required")
    if not _EMAIL_RE.match(email.strip()):
        raise ValueError("invalid email format")


def validate_password(password: str) -> None:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")


def validate_username(username: str) -> None:
    if not username or not username.strip():
        raise ValueError("username is required")
    s = username.strip()
    if not _USERNAME_RE.match(s):
        raise ValueError(
            "username must be 3–32 characters and use only letters, numbers, underscore, hyphen"
        )


def suggest_username_from_email_or_name(email: str, name: Optional[str]) -> str:
    base = ""
    if name:
        base = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())[:32].strip("_")
    if len(base) < 3:
        local = email.split("@", 1)[0]
        base = re.sub(r"[^a-zA-Z0-9_-]+", "_", local)[:32].strip("_") or "user"
    if len(base) < 3:
        base = "user"
    return base[:32]


def user_store_from_env(base_dir: Path) -> UserStore:
    """Default: ``<base_dir>/data/users.sqlite3``. Override with USER_DB_PATH."""
    raw = os.environ.get("USER_DB_PATH", "").strip()
    path = Path(raw) if raw else base_dir / "data" / "users.sqlite3"
    return UserStore(path)
