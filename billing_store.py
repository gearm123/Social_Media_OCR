"""SQLite billing entitlements (same DB file as users). Thread-safe."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

# Anonymous free tier: client generates id (e.g. UUID hex), sends as X-Guest-Billing-Id
_GUEST_KEY_RE = re.compile(r"^[a-fA-F0-9]{8,64}$")

FREE_RUNS_MAX = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def billing_db_path(base_dir: Path) -> Path:
    raw = os.environ.get("USER_DB_PATH", "").strip()
    return Path(raw) if raw else base_dir / "data" / "users.sqlite3"


def billing_enforce_enabled() -> bool:
    return os.environ.get("BILLING_ENFORCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt or not str(dt).strip():
        return None
    s = str(dt).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_guest_key(raw: Optional[str]) -> Optional[str]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if not _GUEST_KEY_RE.match(s):
        return None
    return s.lower()


def access_until_active(access_until: Optional[str]) -> bool:
    end = _parse_iso(access_until)
    if end is None:
        return False
    return end > _utc_now()


class BillingStore:
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
                    CREATE TABLE IF NOT EXISTS billing_entitlements (
                        user_id TEXT PRIMARY KEY,
                        stripe_customer_id TEXT,
                        stripe_subscription_id TEXT,
                        access_until TEXT,
                        paid_job_credits INTEGER NOT NULL DEFAULT 0,
                        free_runs_used INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS billing_webhook_events (
                        event_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_billing_customer "
                    "ON billing_entitlements(stripe_customer_id) "
                    "WHERE stripe_customer_id IS NOT NULL"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS billing_guest_entitlements (
                        guest_key TEXT PRIMARY KEY,
                        free_runs_used INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def ensure_row(self, user_id: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO billing_entitlements (
                        user_id, stripe_customer_id, stripe_subscription_id,
                        access_until, paid_job_credits, free_runs_used, updated_at
                    )
                    VALUES (?, NULL, NULL, NULL, 0, 0, ?)
                    ON CONFLICT(user_id) DO NOTHING
                    """,
                    (user_id, now),
                )
                conn.commit()

    def get_entitlements(self, user_id: str) -> dict[str, Any]:
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT user_id, stripe_customer_id, stripe_subscription_id,
                           access_until, paid_job_credits, free_runs_used, updated_at
                    FROM billing_entitlements WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        if not row:
            return {}
        d = dict(row)
        d["has_unlimited"] = access_until_active(d.get("access_until"))
        d["free_runs_remaining"] = max(0, FREE_RUNS_MAX - int(d.get("free_runs_used") or 0))
        return d

    def get_user_id_by_stripe_customer(self, customer_id: str) -> Optional[str]:
        if not customer_id:
            return None
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT user_id FROM billing_entitlements WHERE stripe_customer_id = ?",
                    (customer_id,),
                ).fetchone()
        return row["user_id"] if row else None

    def try_claim_webhook_event(self, event_id: str) -> bool:
        """Atomically claim an event. True = we own it and should process; False = duplicate."""
        now = _utc_now_iso()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO billing_webhook_events (event_id, created_at) VALUES (?, ?)",
                        (event_id, now),
                    )
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def release_webhook_event(self, event_id: str) -> None:
        """If processing fails, release so Stripe retry can be processed."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM billing_webhook_events WHERE event_id = ?",
                    (event_id,),
                )
                conn.commit()

    def set_stripe_customer(self, user_id: str, customer_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET stripe_customer_id = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (customer_id, now, user_id),
                )
                conn.commit()

    def add_job_credits(self, user_id: str, delta: int) -> None:
        if delta == 0:
            return
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET paid_job_credits = MAX(0, paid_job_credits + ?), updated_at = ?
                    WHERE user_id = ?
                    """,
                    (delta, now, user_id),
                )
                conn.commit()

    def extend_access_hours(self, user_id: str, hours: float) -> None:
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT access_until FROM billing_entitlements WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                current_end = _parse_iso(row["access_until"] if row else None)
                base = _utc_now()
                if current_end and current_end > base:
                    base = current_end
                new_end = base + timedelta(hours=hours)
                now = _utc_now_iso()
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET access_until = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (new_end.isoformat(), now, user_id),
                )
                conn.commit()

    def set_access_until_unix(self, user_id: str, unix_ts: int) -> None:
        """Stripe current_period_end is Unix seconds UTC."""
        dt = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET access_until = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (dt.isoformat(), now, user_id),
                )
                conn.commit()

    def update_subscription_fields(
        self,
        user_id: str,
        subscription_id: Optional[str],
        access_until_unix: Optional[int],
    ) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        access_iso: Optional[str] = None
        if access_until_unix is not None:
            access_iso = datetime.fromtimestamp(
                int(access_until_unix), tz=timezone.utc
            ).isoformat()
        with self._lock:
            with self._connect() as conn:
                if access_iso is not None:
                    conn.execute(
                        """
                        UPDATE billing_entitlements
                        SET stripe_subscription_id = ?,
                            access_until = ?,
                            updated_at = ?
                        WHERE user_id = ?
                        """,
                        (subscription_id, access_iso, now, user_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE billing_entitlements
                        SET stripe_subscription_id = ?, updated_at = ?
                        WHERE user_id = ?
                        """,
                        (subscription_id, now, user_id),
                    )
                conn.commit()

    def ensure_guest_row(self, guest_key: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO billing_guest_entitlements (guest_key, free_runs_used, updated_at)
                    VALUES (?, 0, ?)
                    ON CONFLICT(guest_key) DO NOTHING
                    """,
                    (guest_key, now),
                )
                conn.commit()

    def get_guest_entitlements(self, guest_key: str) -> dict[str, Any]:
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT guest_key, free_runs_used, updated_at
                    FROM billing_guest_entitlements WHERE guest_key = ?
                    """,
                    (guest_key,),
                ).fetchone()
        used = int(row["free_runs_used"] or 0) if row else 0
        return {
            "guest_key": guest_key,
            "free_runs_used": used,
            "free_runs_remaining": max(0, FREE_RUNS_MAX - used),
        }

    def guest_can_start_job(self, guest_key: str, image_count: int) -> Tuple[bool, str, str]:
        """Guests: free tier only — one image, up to FREE_RUNS_MAX successful runs."""
        if image_count < 1:
            return False, "", "no_files"
        if image_count > 1:
            return False, "", "multi_requires_plan"
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT free_runs_used FROM billing_guest_entitlements WHERE guest_key = ?",
                    (guest_key,),
                ).fetchone()
        used = int(row["free_runs_used"] or 0) if row else 0
        if used >= FREE_RUNS_MAX:
            return False, "", "free_exhausted"
        return True, "guest_free", ""

    def guest_apply_successful_job(self, guest_key: str) -> None:
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_guest_entitlements
                    SET free_runs_used = MIN(?, free_runs_used + 1), updated_at = ?
                    WHERE guest_key = ?
                    """,
                    (FREE_RUNS_MAX, now, guest_key),
                )
                conn.commit()

    def can_start_job(self, user_id: str, image_count: int) -> Tuple[bool, str, str]:
        """
        Returns (ok, consumption, error_code).
        consumption: unlimited | credit | free
        error_code: '' | free_exhausted | multi_requires_plan
        """
        if image_count < 1:
            return False, "", "no_files"

        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT access_until, paid_job_credits, free_runs_used
                    FROM billing_entitlements WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        au = row["access_until"] if row else None
        credits = int(row["paid_job_credits"] or 0) if row else 0
        free_used = int(row["free_runs_used"] or 0) if row else 0

        if access_until_active(au):
            return True, "unlimited", ""

        free_left = max(0, FREE_RUNS_MAX - free_used)

        if image_count > 1:
            if credits > 0:
                return True, "credit", ""
            return False, "", "multi_requires_plan"

        # Single image
        if free_left > 0:
            return True, "free", ""
        if credits > 0:
            return True, "credit", ""
        return False, "", "free_exhausted"

    def apply_successful_job(self, user_id: str, consumption: str) -> None:
        consumption = (consumption or "").strip().lower()
        if consumption == "unlimited" or not consumption:
            return
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                if consumption == "credit":
                    conn.execute(
                        """
                        UPDATE billing_entitlements
                        SET paid_job_credits = MAX(0, paid_job_credits - 1), updated_at = ?
                        WHERE user_id = ?
                        """,
                        (now, user_id),
                    )
                elif consumption == "free":
                    conn.execute(
                        """
                        UPDATE billing_entitlements
                        SET free_runs_used = MIN(?, free_runs_used + 1),
                            updated_at = ?
                        WHERE user_id = ?
                        """,
                        (FREE_RUNS_MAX, now, user_id),
                    )
                conn.commit()


_billing_singleton: Optional[BillingStore] = None


def get_billing_store(base_dir: Path) -> BillingStore:
    global _billing_singleton
    if _billing_singleton is None:
        _billing_singleton = BillingStore(billing_db_path(base_dir))
    return _billing_singleton
