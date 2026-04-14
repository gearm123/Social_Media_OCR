"""Live backend usage summary stored separately in PostgreSQL."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg.rows import dict_row

from db_postgres import get_pool, raw_database_url_from_environment

_log = logging.getLogger("translate_chat.usage_report")

_REPORT_KEY = "global"
_singleton: Optional["UsageReportStore"] = None
_singleton_lock = threading.Lock()
_missing_db_logged = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _log_missing_db_once() -> None:
    global _missing_db_logged
    with _singleton_lock:
        if _missing_db_logged:
            return
        _missing_db_logged = True
    _log.info("usage report skipped: DATABASE_URL not configured")


class UsageReportStore:
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
                        CREATE TABLE IF NOT EXISTS backend_usage_report (
                            report_key TEXT PRIMARY KEY,
                            algorithm_runs_total BIGINT NOT NULL DEFAULT 0,
                            free_trial_attempts_total BIGINT NOT NULL DEFAULT 0,
                            users_total BIGINT NOT NULL DEFAULT 0,
                            users_signed_up_today BIGINT NOT NULL DEFAULT 0,
                            signup_day TEXT,
                            last_algorithm_run_at TEXT,
                            updated_at TEXT NOT NULL
                        )
                        """
                    )
                conn.commit()

    def _table_exists(self, cur, table_name: str) -> bool:
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        return bool(row and row[0])

    def _seed_counts(self, cur) -> dict[str, int]:
        counts = {
            "algorithm_runs_total": 0,
            "free_trial_attempts_total": 0,
            "users_total": 0,
            "users_signed_up_today": 0,
        }
        today = _utc_today()
        if self._table_exists(cur, "backend_activity_event_stats"):
            cur.execute(
                """
                SELECT count
                FROM backend_activity_event_stats
                WHERE monitor_key = %s AND event = %s
                """,
                ("backend", "job_created"),
            )
            row = cur.fetchone()
            counts["algorithm_runs_total"] = int((row[0] if row else 0) or 0)
        if self._table_exists(cur, "billing_entitlements"):
            cur.execute("SELECT COALESCE(SUM(free_runs_used), 0) FROM billing_entitlements")
            row = cur.fetchone()
            counts["free_trial_attempts_total"] += int((row[0] if row else 0) or 0)
        if self._table_exists(cur, "billing_guest_entitlements"):
            cur.execute("SELECT COALESCE(SUM(free_runs_used), 0) FROM billing_guest_entitlements")
            row = cur.fetchone()
            counts["free_trial_attempts_total"] += int((row[0] if row else 0) or 0)
        if self._table_exists(cur, "users"):
            cur.execute("SELECT COUNT(*) FROM users")
            row = cur.fetchone()
            counts["users_total"] = int((row[0] if row else 0) or 0)
            cur.execute(
                """
                SELECT COUNT(*)
                FROM users
                WHERE DATE(timezone('utc', created_at::timestamptz)) =
                      DATE(timezone('utc', now()))
                """
            )
            row = cur.fetchone()
            counts["users_signed_up_today"] = int((row[0] if row else 0) or 0)
        counts["signup_day"] = today
        return counts

    def _ensure_row(self, cur) -> None:
        cur.execute(
            "SELECT report_key FROM backend_usage_report WHERE report_key = %s",
            (_REPORT_KEY,),
        )
        if cur.fetchone():
            return
        now = _utc_now_iso()
        counts = self._seed_counts(cur)
        cur.execute(
            """
            INSERT INTO backend_usage_report (
                report_key,
                algorithm_runs_total,
                free_trial_attempts_total,
                users_total,
                users_signed_up_today,
                signup_day,
                last_algorithm_run_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NULL, %s)
            ON CONFLICT (report_key) DO NOTHING
            """,
            (
                _REPORT_KEY,
                counts["algorithm_runs_total"],
                counts["free_trial_attempts_total"],
                counts["users_total"],
                counts["users_signed_up_today"],
                counts["signup_day"],
                now,
            ),
        )

    def _refresh_user_counts(self, cur) -> None:
        now = _utc_now_iso()
        today = _utc_today()
        users_total = 0
        users_today = 0
        if self._table_exists(cur, "users"):
            cur.execute("SELECT COUNT(*) FROM users")
            row = cur.fetchone()
            users_total = int((row[0] if row else 0) or 0)
            cur.execute(
                """
                SELECT COUNT(*)
                FROM users
                WHERE DATE(timezone('utc', created_at::timestamptz)) =
                      DATE(timezone('utc', now()))
                """
            )
            row = cur.fetchone()
            users_today = int((row[0] if row else 0) or 0)
        cur.execute(
            """
            UPDATE backend_usage_report
            SET users_total = %s,
                users_signed_up_today = %s,
                signup_day = %s,
                updated_at = %s
            WHERE report_key = %s
            """,
            (users_total, users_today, today, now, _REPORT_KEY),
        )

    def increment_algorithm_runs(self) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    self._ensure_row(cur)
                    cur.execute(
                        """
                        UPDATE backend_usage_report
                        SET algorithm_runs_total = algorithm_runs_total + 1,
                            last_algorithm_run_at = %s,
                            updated_at = %s
                        WHERE report_key = %s
                        """,
                        (now, now, _REPORT_KEY),
                    )
                conn.commit()

    def increment_free_trial_attempts(self) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    self._ensure_row(cur)
                    cur.execute(
                        """
                        UPDATE backend_usage_report
                        SET free_trial_attempts_total = free_trial_attempts_total + 1,
                            updated_at = %s
                        WHERE report_key = %s
                        """,
                        (now, _REPORT_KEY),
                    )
                conn.commit()

    def refresh_user_totals(self) -> None:
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    self._ensure_row(cur)
                    self._refresh_user_counts(cur)
                conn.commit()

    def read_report(self) -> dict[str, Any]:
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    self._ensure_row(cur)
                    self._refresh_user_counts(cur)
                    cur.execute(
                        """
                        SELECT report_key, algorithm_runs_total, free_trial_attempts_total,
                               users_total, users_signed_up_today, signup_day,
                               last_algorithm_run_at, updated_at
                        FROM backend_usage_report
                        WHERE report_key = %s
                        """,
                        (_REPORT_KEY,),
                    )
                    row = cur.fetchone()
                conn.commit()
        if not row:
            return {}
        return {
            "algorithm_runs_total": int(row["algorithm_runs_total"] or 0),
            "free_trial_attempts_total": int(row["free_trial_attempts_total"] or 0),
            "users_total": int(row["users_total"] or 0),
            "users_signed_up_today": int(row["users_signed_up_today"] or 0),
            "signup_day": row.get("signup_day"),
            "last_algorithm_run_at": row.get("last_algorithm_run_at"),
            "updated_at": row.get("updated_at"),
            "storage": "postgres",
        }


def get_usage_report_store() -> Optional[UsageReportStore]:
    global _singleton
    if not raw_database_url_from_environment():
        _log_missing_db_once()
        return None
    with _singleton_lock:
        if _singleton is None:
            _singleton = UsageReportStore()
        return _singleton


def note_algorithm_run() -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.increment_algorithm_runs()
    except Exception:
        _log.exception("usage report update failed for algorithm run")
        return False
    return True


def note_free_trial_attempt() -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.increment_free_trial_attempts()
    except Exception:
        _log.exception("usage report update failed for free trial attempt")
        return False
    return True


def note_user_signup() -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.refresh_user_totals()
    except Exception:
        _log.exception("usage report update failed for user signup")
        return False
    return True


def read_usage_report() -> Optional[dict[str, Any]]:
    store = get_usage_report_store()
    if store is None:
        return None
    try:
        return store.read_report()
    except Exception:
        _log.exception("usage report read failed")
        return None
