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


def _log_report_snapshot(event: str, report: Optional[dict[str, Any]]) -> None:
    if not isinstance(report, dict):
        _log.info("usage report %s", event)
        return
    _log.info(
        (
            "usage report %s: runs_total=%s completed=%s failed=%s free_trials=%s "
            "users_total=%s users_today=%s updated_at=%s"
        ),
        event,
        int(report.get("algorithm_runs_total") or 0),
        int(report.get("algorithm_runs_completed_total") or 0),
        int(report.get("algorithm_runs_failed_total") or 0),
        int(report.get("free_trial_attempts_total") or 0),
        int(report.get("users_total") or 0),
        int(report.get("users_signed_up_today") or 0),
        report.get("updated_at"),
    )


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
                            algorithm_runs_completed_total BIGINT NOT NULL DEFAULT 0,
                            algorithm_runs_failed_total BIGINT NOT NULL DEFAULT 0,
                            free_trial_attempts_total BIGINT NOT NULL DEFAULT 0,
                            users_total BIGINT NOT NULL DEFAULT 0,
                            users_signed_up_today BIGINT NOT NULL DEFAULT 0,
                            signup_day TEXT,
                            last_algorithm_run_at TEXT,
                            pass1_timeout_failures_total BIGINT NOT NULL DEFAULT 0,
                            pass2_success_try1_total BIGINT NOT NULL DEFAULT 0,
                            pass2_success_try2_total BIGINT NOT NULL DEFAULT 0,
                            pass3_success_try1_total BIGINT NOT NULL DEFAULT 0,
                            pass3_success_try2_total BIGINT NOT NULL DEFAULT 0,
                            updated_at TEXT NOT NULL
                        )
                        """
                    )
                    for col_def in (
                        "algorithm_runs_completed_total BIGINT NOT NULL DEFAULT 0",
                        "algorithm_runs_failed_total BIGINT NOT NULL DEFAULT 0",
                        "pass1_timeout_failures_total BIGINT NOT NULL DEFAULT 0",
                        "pass2_success_try1_total BIGINT NOT NULL DEFAULT 0",
                        "pass2_success_try2_total BIGINT NOT NULL DEFAULT 0",
                        "pass3_success_try1_total BIGINT NOT NULL DEFAULT 0",
                        "pass3_success_try2_total BIGINT NOT NULL DEFAULT 0",
                    ):
                        cur.execute(
                            f"ALTER TABLE backend_usage_report ADD COLUMN IF NOT EXISTS {col_def}"
                        )
                conn.commit()

    def _table_exists(self, cur, table_name: str) -> bool:
        cur.execute(
            "SELECT to_regclass(%s) AS regclass_name",
            (f"public.{table_name}",),
        )
        row = cur.fetchone()
        if not row:
            return False
        if isinstance(row, dict):
            return bool(row.get("regclass_name"))
        return bool(row[0])

    def _query_scalar(self, cur, sql: str, params: tuple = ()) -> int:
        cur.execute(sql, params)
        row = cur.fetchone()
        if not row:
            return 0
        if isinstance(row, dict):
            return int((next(iter(row.values())) if row else 0) or 0)
        return int((row[0] if row else 0) or 0)

    def _seed_counts(self, cur) -> dict[str, Any]:
        counts: dict[str, Any] = {
            "algorithm_runs_total": 0,
            "algorithm_runs_completed_total": 0,
            "algorithm_runs_failed_total": 0,
            "free_trial_attempts_total": 0,
            "users_total": 0,
            "users_signed_up_today": 0,
            "pass1_timeout_failures_total": 0,
            "pass2_success_try1_total": 0,
            "pass2_success_try2_total": 0,
            "pass3_success_try1_total": 0,
            "pass3_success_try2_total": 0,
            "signup_day": _utc_today(),
        }
        if self._table_exists(cur, "backend_activity_event_stats"):
            counts["algorithm_runs_total"] = self._query_scalar(
                cur,
                """
                SELECT count
                FROM backend_activity_event_stats
                WHERE monitor_key = %s AND event = %s
                """,
                ("backend", "job_created"),
            )
            counts["algorithm_runs_completed_total"] = self._query_scalar(
                cur,
                """
                SELECT count
                FROM backend_activity_event_stats
                WHERE monitor_key = %s AND event = %s
                """,
                ("backend", "job_completed"),
            )
            counts["algorithm_runs_failed_total"] = self._query_scalar(
                cur,
                """
                SELECT count
                FROM backend_activity_event_stats
                WHERE monitor_key = %s AND event = %s
                """,
                ("backend", "job_failed"),
            )
        if self._table_exists(cur, "billing_entitlements"):
            counts["free_trial_attempts_total"] += self._query_scalar(
                cur, "SELECT COALESCE(SUM(free_runs_used), 0) FROM billing_entitlements"
            )
        if self._table_exists(cur, "billing_guest_entitlements"):
            counts["free_trial_attempts_total"] += self._query_scalar(
                cur,
                "SELECT COALESCE(SUM(free_runs_used), 0) FROM billing_guest_entitlements",
            )
        if self._table_exists(cur, "users"):
            counts["users_total"] = self._query_scalar(cur, "SELECT COUNT(*) FROM users")
            counts["users_signed_up_today"] = self._query_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM users
                WHERE DATE(timezone('utc', created_at::timestamptz)) =
                      DATE(timezone('utc', now()))
                """,
            )
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
                algorithm_runs_completed_total,
                algorithm_runs_failed_total,
                free_trial_attempts_total,
                users_total,
                users_signed_up_today,
                signup_day,
                last_algorithm_run_at,
                pass1_timeout_failures_total,
                pass2_success_try1_total,
                pass2_success_try2_total,
                pass3_success_try1_total,
                pass3_success_try2_total,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (report_key) DO NOTHING
            """,
            (
                _REPORT_KEY,
                counts["algorithm_runs_total"],
                counts["algorithm_runs_completed_total"],
                counts["algorithm_runs_failed_total"],
                counts["free_trial_attempts_total"],
                counts["users_total"],
                counts["users_signed_up_today"],
                counts["signup_day"],
                counts["pass1_timeout_failures_total"],
                counts["pass2_success_try1_total"],
                counts["pass2_success_try2_total"],
                counts["pass3_success_try1_total"],
                counts["pass3_success_try2_total"],
                now,
            ),
        )

    def _refresh_user_counts(self, cur) -> None:
        now = _utc_now_iso()
        today = _utc_today()
        users_total = 0
        users_today = 0
        if self._table_exists(cur, "users"):
            users_total = self._query_scalar(cur, "SELECT COUNT(*) FROM users")
            users_today = self._query_scalar(
                cur,
                """
                SELECT COUNT(*)
                FROM users
                WHERE DATE(timezone('utc', created_at::timestamptz)) =
                      DATE(timezone('utc', now()))
                """,
            )
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

    def _increment_pass_success(self, cur, pass_name: str, attempt: Optional[int]) -> None:
        if pass_name not in ("pass2", "pass3"):
            return
        if attempt not in (1, 2):
            return
        now = _utc_now_iso()
        col = f"{pass_name}_success_try{attempt}_total"
        cur.execute(
            f"""
            UPDATE backend_usage_report
            SET {col} = {col} + 1,
                updated_at = %s
            WHERE report_key = %s
            """,
            (now, _REPORT_KEY),
        )

    def record_run_started(self) -> None:
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

    def record_run_completed(self, pass_outcomes: Optional[dict[str, Any]] = None) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    self._ensure_row(cur)
                    cur.execute(
                        """
                        UPDATE backend_usage_report
                        SET algorithm_runs_completed_total = algorithm_runs_completed_total + 1,
                            updated_at = %s
                        WHERE report_key = %s
                        """,
                        (now, _REPORT_KEY),
                    )
                    po = pass_outcomes or {}
                    for pass_name in ("pass2", "pass3"):
                        meta = po.get(pass_name) if isinstance(po, dict) else None
                        if not isinstance(meta, dict) or not meta.get("applied"):
                            continue
                        self._increment_pass_success(
                            cur, pass_name, meta.get("successful_attempt")
                        )
                conn.commit()

    def record_run_failed(self, pass_outcomes: Optional[dict[str, Any]] = None) -> None:
        now = _utc_now_iso()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    self._ensure_row(cur)
                    cur.execute(
                        """
                        UPDATE backend_usage_report
                        SET algorithm_runs_failed_total = algorithm_runs_failed_total + 1,
                            updated_at = %s
                        WHERE report_key = %s
                        """,
                        (now, _REPORT_KEY),
                    )
                    pass1 = (pass_outcomes or {}).get("pass1") if isinstance(pass_outcomes, dict) else None
                    if isinstance(pass1, dict) and pass1.get("status") == "timeout_exhausted":
                        cur.execute(
                            """
                            UPDATE backend_usage_report
                            SET pass1_timeout_failures_total = pass1_timeout_failures_total + 1,
                                updated_at = %s
                            WHERE report_key = %s
                            """,
                            (now, _REPORT_KEY),
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
                        SELECT report_key,
                               algorithm_runs_total,
                               algorithm_runs_completed_total,
                               algorithm_runs_failed_total,
                               free_trial_attempts_total,
                               users_total,
                               users_signed_up_today,
                               signup_day,
                               last_algorithm_run_at,
                               pass1_timeout_failures_total,
                               pass2_success_try1_total,
                               pass2_success_try2_total,
                               pass3_success_try1_total,
                               pass3_success_try2_total,
                               updated_at
                        FROM backend_usage_report
                        WHERE report_key = %s
                        """,
                        (_REPORT_KEY,),
                    )
                    row = cur.fetchone()
                conn.commit()
        if not row:
            return {}
        p2_try1 = int(row["pass2_success_try1_total"] or 0)
        p2_try2 = int(row["pass2_success_try2_total"] or 0)
        p3_try1 = int(row["pass3_success_try1_total"] or 0)
        p3_try2 = int(row["pass3_success_try2_total"] or 0)
        return {
            "algorithm_runs_total": int(row["algorithm_runs_total"] or 0),
            "algorithm_runs_completed_total": int(row["algorithm_runs_completed_total"] or 0),
            "algorithm_runs_failed_total": int(row["algorithm_runs_failed_total"] or 0),
            "free_trial_attempts_total": int(row["free_trial_attempts_total"] or 0),
            "users_total": int(row["users_total"] or 0),
            "users_signed_up_today": int(row["users_signed_up_today"] or 0),
            "signup_day": row.get("signup_day"),
            "last_algorithm_run_at": row.get("last_algorithm_run_at"),
            "pass1_timeout_failures_total": int(row["pass1_timeout_failures_total"] or 0),
            "pass2_success_try1_total": p2_try1,
            "pass2_success_try2_total": p2_try2,
            "pass2_meaningful_runs_total": p2_try1 + p2_try2,
            "pass3_success_try1_total": p3_try1,
            "pass3_success_try2_total": p3_try2,
            "pass3_meaningful_runs_total": p3_try1 + p3_try2,
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


def note_algorithm_started() -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.record_run_started()
        _log_report_snapshot("updated after algorithm start", store.read_report())
    except Exception:
        _log.exception("usage report update failed for algorithm start")
        return False
    return True


def note_algorithm_completed(pass_outcomes: Optional[dict[str, Any]] = None) -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.record_run_completed(pass_outcomes)
        _log_report_snapshot("updated after algorithm completion", store.read_report())
    except Exception:
        _log.exception("usage report update failed for algorithm completion")
        return False
    return True


def note_algorithm_failed(pass_outcomes: Optional[dict[str, Any]] = None) -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.record_run_failed(pass_outcomes)
        _log_report_snapshot("updated after algorithm failure", store.read_report())
    except Exception:
        _log.exception("usage report update failed for algorithm failure")
        return False
    return True


def note_free_trial_attempt() -> bool:
    store = get_usage_report_store()
    if store is None:
        return False
    try:
        store.increment_free_trial_attempts()
        _log_report_snapshot("updated after free trial attempt", store.read_report())
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
        _log_report_snapshot("updated after user signup", store.read_report())
    except Exception:
        _log.exception("usage report update failed for user signup")
        return False
    return True


def read_usage_report() -> Optional[dict[str, Any]]:
    store = get_usage_report_store()
    if store is None:
        return None
    try:
        report = store.read_report()
        _log_report_snapshot("read", report)
        return report
    except Exception:
        _log.exception("usage report read failed")
        return None
