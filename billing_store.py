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

# Signed-in users: free single-image tries (separate from anonymous guest trials).
USER_FREE_RUNS_MAX = 1
# Anonymous guests: free single-image tries before one-time purchase.
GUEST_FREE_RUNS_MAX = 1


def subscription_runs_cap() -> int:
    """Max successful jobs per calendar month while subscription is active (~$2.50 COGS at default)."""
    try:
        return max(1, int(os.environ.get("BILLING_SUBSCRIPTION_RUNS_PER_MONTH", "7").strip()))
    except ValueError:
        return 7


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
                self._migrate_entitlements(conn)
                self._migrate_guest_entitlements(conn)
                conn.commit()

    def _migrate_entitlements(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(billing_entitlements)").fetchall()}
        if "paddle_customer_id" not in cols:
            conn.execute("ALTER TABLE billing_entitlements ADD COLUMN paddle_customer_id TEXT")
        if "paddle_address_id" not in cols:
            conn.execute("ALTER TABLE billing_entitlements ADD COLUMN paddle_address_id TEXT")
        if "paddle_subscription_id" not in cols:
            conn.execute("ALTER TABLE billing_entitlements ADD COLUMN paddle_subscription_id TEXT")
        conn.execute(
            """
            UPDATE billing_entitlements SET paddle_customer_id = stripe_customer_id
            WHERE paddle_customer_id IS NULL AND stripe_customer_id IS NOT NULL
            """
        )
        conn.execute(
            """
            UPDATE billing_entitlements SET paddle_subscription_id = stripe_subscription_id
            WHERE paddle_subscription_id IS NULL AND stripe_subscription_id IS NOT NULL
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_billing_paddle_customer "
            "ON billing_entitlements(paddle_customer_id) "
            "WHERE paddle_customer_id IS NOT NULL"
        )
        if "sub_quota_month" not in cols:
            conn.execute(
                "ALTER TABLE billing_entitlements ADD COLUMN sub_quota_month TEXT"
            )
        if "sub_quota_used" not in cols:
            conn.execute(
                "ALTER TABLE billing_entitlements ADD COLUMN sub_quota_used INTEGER NOT NULL DEFAULT 0"
            )

    def _migrate_guest_entitlements(self, conn: sqlite3.Connection) -> None:
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(billing_guest_entitlements)").fetchall()
        }
        if "paid_job_credits" not in cols:
            conn.execute(
                "ALTER TABLE billing_guest_entitlements ADD COLUMN paid_job_credits INTEGER NOT NULL DEFAULT 0"
            )
        if "paddle_customer_id" not in cols:
            conn.execute(
                "ALTER TABLE billing_guest_entitlements ADD COLUMN paddle_customer_id TEXT"
            )
        if "paddle_address_id" not in cols:
            conn.execute(
                "ALTER TABLE billing_guest_entitlements ADD COLUMN paddle_address_id TEXT"
            )

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
                           paddle_customer_id, paddle_address_id, paddle_subscription_id,
                           access_until, paid_job_credits, free_runs_used, updated_at,
                           sub_quota_month, sub_quota_used
                    FROM billing_entitlements WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        if not row:
            return {}
        d = dict(row)
        d["paddle_customer_id"] = d.get("paddle_customer_id") or d.get("stripe_customer_id")
        d["paddle_address_id"] = d.get("paddle_address_id")
        d["paddle_subscription_id"] = d.get("paddle_subscription_id") or d.get("stripe_subscription_id")
        au = d.get("access_until")
        cap = subscription_runs_cap()
        ym = _utc_now().strftime("%Y-%m")
        sm = d.get("sub_quota_month")
        su = int(d.get("sub_quota_used") or 0)
        sub_active = access_until_active(au)
        eff_used = 0 if (not sm or sm != ym) else su
        rem = max(0, cap - eff_used) if sub_active else 0
        d["has_unlimited"] = False
        d["subscription_active"] = sub_active
        d["subscription_runs_cap"] = cap
        d["subscription_runs_used_this_month"] = eff_used
        d["subscription_runs_remaining"] = rem if sub_active else 0
        d["subscription_quota_month"] = ym if sub_active else None
        d["free_runs_remaining"] = max(
            0, USER_FREE_RUNS_MAX - int(d.get("free_runs_used") or 0)
        )
        return d

    def get_user_id_by_paddle_customer(self, customer_id: str) -> Optional[str]:
        if not customer_id:
            return None
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT user_id FROM billing_entitlements
                    WHERE paddle_customer_id = ? OR stripe_customer_id = ?
                    """,
                    (customer_id, customer_id),
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
        """If processing fails, release so the payment provider can retry safely."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM billing_webhook_events WHERE event_id = ?",
                    (event_id,),
                )
                conn.commit()

    def set_paddle_customer(self, user_id: str, customer_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET paddle_customer_id = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (customer_id, now, user_id),
                )
                conn.commit()

    def set_paddle_address(self, user_id: str, address_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET paddle_address_id = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (address_id, now, user_id),
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
                        SET paddle_subscription_id = ?,
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
                        SET paddle_subscription_id = ?, updated_at = ?
                        WHERE user_id = ?
                        """,
                        (subscription_id, now, user_id),
                    )
                conn.commit()

    def set_subscription_access_iso(
        self,
        user_id: str,
        subscription_id: Optional[str],
        access_until_iso: str,
    ) -> None:
        """Paddle subscription period end as RFC 3339 / ISO 8601."""
        now = _utc_now_iso()
        self.ensure_row(user_id)
        s = access_until_iso.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        normalized = dt.astimezone(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_entitlements
                    SET paddle_subscription_id = COALESCE(?, paddle_subscription_id),
                        access_until = ?,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (subscription_id, normalized, now, user_id),
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
                    SELECT guest_key, free_runs_used, paid_job_credits,
                           paddle_customer_id, paddle_address_id, updated_at
                    FROM billing_guest_entitlements WHERE guest_key = ?
                    """,
                    (guest_key,),
                ).fetchone()
        used = int(row["free_runs_used"] or 0) if row else 0
        credits = int(row["paid_job_credits"] or 0) if row else 0
        return {
            "guest_key": guest_key,
            "free_runs_used": used,
            "free_runs_remaining": max(0, GUEST_FREE_RUNS_MAX - used),
            "paid_job_credits": credits,
            "paddle_customer_id": (row["paddle_customer_id"] if row else None) or None,
            "paddle_address_id": (row["paddle_address_id"] if row else None) or None,
        }

    def guest_can_start_job(self, guest_key: str, image_count: int) -> Tuple[bool, str, str]:
        """Guests: paid credits (multi-image ok) or one free single-image run per guest key."""
        if image_count < 1:
            return False, "", "no_files"
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT free_runs_used, paid_job_credits
                    FROM billing_guest_entitlements WHERE guest_key = ?
                    """,
                    (guest_key,),
                ).fetchone()
        used = int(row["free_runs_used"] or 0) if row else 0
        credits = int(row["paid_job_credits"] or 0) if row else 0
        if credits > 0:
            return True, "guest_credit", ""
        if image_count > 1:
            return False, "", "multi_requires_plan"
        if used >= GUEST_FREE_RUNS_MAX:
            return False, "", "free_exhausted"
        return True, "guest_free", ""

    def guest_add_job_credits(self, guest_key: str, delta: int) -> None:
        if delta == 0:
            return
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_guest_entitlements
                    SET paid_job_credits = MAX(0, paid_job_credits + ?), updated_at = ?
                    WHERE guest_key = ?
                    """,
                    (delta, now, guest_key),
                )
                conn.commit()

    def set_guest_paddle_customer(self, guest_key: str, customer_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_guest_entitlements
                    SET paddle_customer_id = ?, updated_at = ?
                    WHERE guest_key = ?
                    """,
                    (customer_id, now, guest_key),
                )
                conn.commit()

    def set_guest_paddle_address(self, guest_key: str, address_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE billing_guest_entitlements
                    SET paddle_address_id = ?, updated_at = ?
                    WHERE guest_key = ?
                    """,
                    (address_id, now, guest_key),
                )
                conn.commit()

    def guest_apply_successful_job(self, guest_key: str, consumption: str) -> None:
        consumption = (consumption or "").strip().lower()
        if consumption not in ("guest_free", "guest_credit"):
            return
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            with self._connect() as conn:
                if consumption == "guest_credit":
                    conn.execute(
                        """
                        UPDATE billing_guest_entitlements
                        SET paid_job_credits = MAX(0, paid_job_credits - 1), updated_at = ?
                        WHERE guest_key = ?
                        """,
                        (now, guest_key),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE billing_guest_entitlements
                        SET free_runs_used = MIN(?, free_runs_used + 1), updated_at = ?
                        WHERE guest_key = ?
                        """,
                        (GUEST_FREE_RUNS_MAX, now, guest_key),
                    )
                conn.commit()

    def can_start_job(self, user_id: str, image_count: int) -> Tuple[bool, str, str]:
        """
        Returns (ok, consumption, error_code).
        consumption: sub_quota | credit | free | guest_free (guest handled elsewhere)
        error_code: '' | free_exhausted | multi_requires_plan | quota_exhausted
        """
        if image_count < 1:
            return False, "", "no_files"

        self.ensure_row(user_id)
        now_iso = _utc_now_iso()
        cap = subscription_runs_cap()
        ym = _utc_now().strftime("%Y-%m")

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT access_until, paid_job_credits, free_runs_used,
                           sub_quota_month, sub_quota_used
                    FROM billing_entitlements WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
                if not row:
                    return False, "", "free_exhausted"
                au = row["access_until"]
                credits = int(row["paid_job_credits"] or 0)
                free_used = int(row["free_runs_used"] or 0)
                sm = row["sub_quota_month"]
                used = int(row["sub_quota_used"] or 0)

                if access_until_active(au):
                    if sm != ym:
                        conn.execute(
                            """
                            UPDATE billing_entitlements
                            SET sub_quota_month = ?, sub_quota_used = 0, updated_at = ?
                            WHERE user_id = ?
                            """,
                            (ym, now_iso, user_id),
                        )
                        used = 0
                    if used >= cap:
                        conn.commit()
                        return False, "", "quota_exhausted"
                    conn.commit()
                    return True, "sub_quota", ""

                free_left = max(0, USER_FREE_RUNS_MAX - free_used)

                if image_count > 1:
                    if credits > 0:
                        conn.commit()
                        return True, "credit", ""
                    conn.commit()
                    return False, "", "multi_requires_plan"

                if free_left > 0:
                    conn.commit()
                    return True, "free", ""
                if credits > 0:
                    conn.commit()
                    return True, "credit", ""
                conn.commit()
                return False, "", "free_exhausted"

    def apply_successful_job(self, user_id: str, consumption: str) -> None:
        consumption = (consumption or "").strip().lower()
        if consumption in ("unlimited", "") or not consumption:
            return
        now = _utc_now_iso()
        ym = _utc_now().strftime("%Y-%m")
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
                        (USER_FREE_RUNS_MAX, now, user_id),
                    )
                elif consumption == "sub_quota":
                    row = conn.execute(
                        """
                        SELECT sub_quota_month, sub_quota_used
                        FROM billing_entitlements WHERE user_id = ?
                        """,
                        (user_id,),
                    ).fetchone()
                    sm = row["sub_quota_month"] if row else None
                    used = int(row["sub_quota_used"] or 0) if row else 0
                    if sm != ym:
                        conn.execute(
                            """
                            UPDATE billing_entitlements
                            SET sub_quota_month = ?, sub_quota_used = 1, updated_at = ?
                            WHERE user_id = ?
                            """,
                            (ym, now, user_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE billing_entitlements
                            SET sub_quota_used = sub_quota_used + 1, updated_at = ?
                            WHERE user_id = ?
                            """,
                            (now, user_id),
                        )
                conn.commit()


_billing_singleton: Optional[BillingStore] = None


def get_billing_store(base_dir: Path) -> BillingStore:
    global _billing_singleton
    if _billing_singleton is None:
        _billing_singleton = BillingStore(billing_db_path(base_dir))
    return _billing_singleton
