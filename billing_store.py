"""Billing entitlements in PostgreSQL (same ``DATABASE_URL`` as users). Thread-safe."""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row

from activity_log import actor_fields, write_activity
from db_postgres import get_pool

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


def billing_enforce_enabled() -> bool:
    return os.environ.get("BILLING_ENFORCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _exempt_csv_set(env_key: str) -> set[str]:
    raw = os.environ.get(env_key, "")
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def billing_exempt_user(email: str, username: Optional[str] = None) -> bool:
    """True when email or username is listed in BILLING_EXEMPT_* (comma-separated, case-insensitive)."""
    emails = _exempt_csv_set("BILLING_EXEMPT_EMAILS")
    names = _exempt_csv_set("BILLING_EXEMPT_USERNAMES")
    e = (email or "").strip().lower()
    if e and e in emails:
        return True
    u = (username or "").strip().lower()
    return bool(u and u in names)


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
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS billing_entitlements (
                            user_id TEXT PRIMARY KEY,
                            stripe_customer_id TEXT,
                            stripe_subscription_id TEXT,
                            access_until TEXT,
                            paid_job_credits INTEGER NOT NULL DEFAULT 0,
                            free_runs_used INTEGER NOT NULL DEFAULT 0,
                            updated_at TEXT NOT NULL,
                            paddle_customer_id TEXT,
                            paddle_address_id TEXT,
                            paddle_subscription_id TEXT,
                            sub_quota_month TEXT,
                            sub_quota_used INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS billing_webhook_events (
                            event_id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS billing_one_time_txn_credits (
                            transaction_id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_billing_customer
                        ON billing_entitlements(stripe_customer_id)
                        WHERE stripe_customer_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS billing_guest_entitlements (
                            guest_key TEXT PRIMARY KEY,
                            free_runs_used INTEGER NOT NULL DEFAULT 0,
                            updated_at TEXT NOT NULL,
                            paid_job_credits INTEGER NOT NULL DEFAULT 0,
                            paddle_customer_id TEXT,
                            paddle_address_id TEXT
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_billing_paddle_customer
                        ON billing_entitlements(paddle_customer_id)
                        WHERE paddle_customer_id IS NOT NULL
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_guest_paddle_customer
                        ON billing_guest_entitlements(paddle_customer_id)
                        WHERE paddle_customer_id IS NOT NULL
                        """
                    )
                conn.commit()

    def try_claim_one_time_txn_credit(self, transaction_id: Optional[str]) -> bool:
        """Return True if we should apply one-time credits for this Paddle transaction (first event only)."""
        if not transaction_id or not str(transaction_id).strip():
            return True
        tid = str(transaction_id).strip()
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO billing_one_time_txn_credits (transaction_id, created_at)
                            VALUES (%s, %s)
                            """,
                            (tid, now),
                        )
                    conn.commit()
                    return True
                except UniqueViolation:
                    conn.rollback()
                    return False

    def get_guest_key_by_paddle_customer_id(self, customer_id: Optional[str]) -> Optional[str]:
        if not customer_id or not str(customer_id).strip():
            return None
        cid = str(customer_id).strip()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT guest_key FROM billing_guest_entitlements
                        WHERE paddle_customer_id = %s
                        LIMIT 1
                        """,
                        (cid,),
                    )
                    row = cur.fetchone()
        if not row:
            return None
        gk = row["guest_key"]
        return str(gk).strip() or None

    def ensure_row(self, user_id: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO billing_entitlements (
                            user_id, stripe_customer_id, stripe_subscription_id,
                            access_until, paid_job_credits, free_runs_used, updated_at,
                            sub_quota_used
                        )
                        VALUES (%s, NULL, NULL, NULL, 0, 0, %s, 0)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        (user_id, now),
                    )
                conn.commit()

    def get_entitlements(self, user_id: str) -> dict[str, Any]:
        self.ensure_row(user_id)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT user_id, stripe_customer_id, stripe_subscription_id,
                               paddle_customer_id, paddle_address_id, paddle_subscription_id,
                               access_until, paid_job_credits, free_runs_used, updated_at,
                               sub_quota_month, sub_quota_used
                        FROM billing_entitlements WHERE user_id = %s
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
        if not row:
            return {}
        d = dict(row)
        d["paddle_customer_id"] = d.get("paddle_customer_id") or d.get("stripe_customer_id")
        d["paddle_address_id"] = d.get("paddle_address_id")
        d["paddle_subscription_id"] = d.get("paddle_subscription_id") or d.get(
            "stripe_subscription_id"
        )
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT user_id FROM billing_entitlements
                        WHERE paddle_customer_id = %s OR stripe_customer_id = %s
                        """,
                        (customer_id, customer_id),
                    )
                    row = cur.fetchone()
        return row["user_id"] if row else None

    def try_claim_webhook_event(self, event_id: str) -> bool:
        """Atomically claim an event. True = we own it and should process; False = duplicate."""
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO billing_webhook_events (event_id, created_at) VALUES (%s, %s)",
                            (event_id, now),
                        )
                    conn.commit()
                    return True
                except UniqueViolation:
                    conn.rollback()
                    return False

    def release_webhook_event(self, event_id: str) -> None:
        """If processing fails, release so the payment provider can retry safely."""
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM billing_webhook_events WHERE event_id = %s",
                        (event_id,),
                    )
                conn.commit()

    def set_paddle_customer(self, user_id: str, customer_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET paddle_customer_id = %s, updated_at = %s
                        WHERE user_id = %s
                        """,
                        (customer_id, now, user_id),
                    )
                conn.commit()

    def set_paddle_address(self, user_id: str, address_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_row(user_id)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET paddle_address_id = %s, updated_at = %s
                        WHERE user_id = %s
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET paid_job_credits = GREATEST(0, paid_job_credits + %s), updated_at = %s
                        WHERE user_id = %s
                        """,
                        (delta, now, user_id),
                    )
                conn.commit()

    def extend_access_hours(self, user_id: str, hours: float) -> None:
        self.ensure_row(user_id)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT access_until FROM billing_entitlements WHERE user_id = %s",
                        (user_id,),
                    )
                    row = cur.fetchone()
                current_end = _parse_iso(row["access_until"] if row else None)
                base = _utc_now()
                if current_end and current_end > base:
                    base = current_end
                new_end = base + timedelta(hours=hours)
                now = _utc_now_iso()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET access_until = %s, updated_at = %s
                        WHERE user_id = %s
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET access_until = %s, updated_at = %s
                        WHERE user_id = %s
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    if access_iso is not None:
                        cur.execute(
                            """
                            UPDATE billing_entitlements
                            SET paddle_subscription_id = %s,
                                access_until = %s,
                                updated_at = %s
                            WHERE user_id = %s
                            """,
                            (subscription_id, access_iso, now, user_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE billing_entitlements
                            SET paddle_subscription_id = %s, updated_at = %s
                            WHERE user_id = %s
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_entitlements
                        SET paddle_subscription_id = COALESCE(%s, paddle_subscription_id),
                            access_until = %s,
                            updated_at = %s
                        WHERE user_id = %s
                        """,
                        (subscription_id, normalized, now, user_id),
                    )
                conn.commit()

    def ensure_guest_row(self, guest_key: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO billing_guest_entitlements (guest_key, free_runs_used, updated_at)
                        VALUES (%s, 0, %s)
                        ON CONFLICT (guest_key) DO NOTHING
                        """,
                        (guest_key, now),
                    )
                conn.commit()

    def get_guest_entitlements(self, guest_key: str) -> dict[str, Any]:
        self.ensure_guest_row(guest_key)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT guest_key, free_runs_used, paid_job_credits,
                               paddle_customer_id, paddle_address_id, updated_at
                        FROM billing_guest_entitlements WHERE guest_key = %s
                        """,
                        (guest_key,),
                    )
                    row = cur.fetchone()
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT free_runs_used, paid_job_credits
                        FROM billing_guest_entitlements WHERE guest_key = %s
                        """,
                        (guest_key,),
                    )
                    row = cur.fetchone()
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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_guest_entitlements
                        SET paid_job_credits = GREATEST(0, paid_job_credits + %s), updated_at = %s
                        WHERE guest_key = %s
                        """,
                        (delta, now, guest_key),
                    )
                conn.commit()

    def set_guest_paddle_customer(self, guest_key: str, customer_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_guest_entitlements
                        SET paddle_customer_id = %s, updated_at = %s
                        WHERE guest_key = %s
                        """,
                        (customer_id, now, guest_key),
                    )
                conn.commit()

    def set_guest_paddle_address(self, guest_key: str, address_id: str) -> None:
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE billing_guest_entitlements
                        SET paddle_address_id = %s, updated_at = %s
                        WHERE guest_key = %s
                        """,
                        (address_id, now, guest_key),
                    )
                conn.commit()

    def guest_apply_successful_job(
        self, guest_key: str, consumption: str, job_id: Optional[str] = None
    ) -> None:
        consumption = (consumption or "").strip().lower()
        if consumption not in ("guest_free", "guest_credit"):
            return
        now = _utc_now_iso()
        self.ensure_guest_row(guest_key)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    if consumption == "guest_credit":
                        cur.execute(
                            """
                            UPDATE billing_guest_entitlements
                            SET paid_job_credits = GREATEST(0, paid_job_credits - 1), updated_at = %s
                            WHERE guest_key = %s
                            """,
                            (now, guest_key),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE billing_guest_entitlements
                            SET free_runs_used = LEAST(%s, free_runs_used + 1), updated_at = %s
                            WHERE guest_key = %s
                            """,
                            (GUEST_FREE_RUNS_MAX, now, guest_key),
                        )
                conn.commit()
        if consumption == "guest_free":
            write_activity(
                "free_trial_consumed",
                job_id=job_id,
                free_trial_type=consumption,
                **actor_fields(guest_key=guest_key),
            )

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
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT access_until, paid_job_credits, free_runs_used,
                               sub_quota_month, sub_quota_used
                        FROM billing_entitlements WHERE user_id = %s
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        conn.commit()
                        return False, "", "free_exhausted"
                    au = row["access_until"]
                    credits = int(row["paid_job_credits"] or 0)
                    free_used = int(row["free_runs_used"] or 0)
                    sm = row["sub_quota_month"]
                    used = int(row["sub_quota_used"] or 0)

                    if access_until_active(au):
                        if sm != ym:
                            cur.execute(
                                """
                                UPDATE billing_entitlements
                                SET sub_quota_month = %s, sub_quota_used = 0, updated_at = %s
                                WHERE user_id = %s
                                """,
                                (ym, now_iso, user_id),
                            )
                            used = 0
                        if used < cap:
                            conn.commit()
                            return True, "sub_quota", ""
                        if credits > 0:
                            conn.commit()
                            return True, "credit", ""
                        conn.commit()
                        return False, "", "quota_exhausted"

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

    def apply_successful_job(
        self, user_id: str, consumption: str, job_id: Optional[str] = None
    ) -> None:
        consumption = (consumption or "").strip().lower()
        if consumption in ("unlimited", "") or not consumption:
            return
        now = _utc_now_iso()
        ym = _utc_now().strftime("%Y-%m")
        self.ensure_row(user_id)
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if consumption == "credit":
                        cur.execute(
                            """
                            UPDATE billing_entitlements
                            SET paid_job_credits = GREATEST(0, paid_job_credits - 1), updated_at = %s
                            WHERE user_id = %s
                            """,
                            (now, user_id),
                        )
                    elif consumption == "free":
                        cur.execute(
                            """
                            UPDATE billing_entitlements
                            SET free_runs_used = LEAST(%s, free_runs_used + 1),
                                updated_at = %s
                            WHERE user_id = %s
                            """,
                            (USER_FREE_RUNS_MAX, now, user_id),
                        )
                    elif consumption == "sub_quota":
                        cur.execute(
                            """
                            SELECT sub_quota_month, sub_quota_used
                            FROM billing_entitlements WHERE user_id = %s
                            """,
                            (user_id,),
                        )
                        row = cur.fetchone()
                        sm = row["sub_quota_month"] if row else None
                        used = int(row["sub_quota_used"] or 0) if row else 0
                        if sm != ym:
                            cur.execute(
                                """
                                UPDATE billing_entitlements
                                SET sub_quota_month = %s, sub_quota_used = 1, updated_at = %s
                                WHERE user_id = %s
                                """,
                                (ym, now, user_id),
                            )
                        else:
                            cur.execute(
                                """
                                UPDATE billing_entitlements
                                SET sub_quota_used = sub_quota_used + 1, updated_at = %s
                                WHERE user_id = %s
                                """,
                                (now, user_id),
                            )
                conn.commit()
        if consumption == "free":
            write_activity(
                "free_trial_consumed",
                job_id=job_id,
                free_trial_type=consumption,
                **actor_fields(user_id=user_id),
            )


_billing_singleton: Optional[BillingStore] = None


def get_billing_store(base_dir: Path) -> BillingStore:
    global _billing_singleton
    if _billing_singleton is None:
        _ = base_dir
        _billing_singleton = BillingStore()
    return _billing_singleton
