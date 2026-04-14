"""Tests for the live backend usage report."""

import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import usage_report
import web_app


class _FakeUsageStore:
    def __init__(self):
        self.algorithm_starts = 0
        self.algorithm_completions = []
        self.algorithm_failures = []
        self.free_trial_attempts = 0
        self.user_refreshes = 0
        self.report = {
            "algorithm_runs_total": 7,
            "algorithm_runs_completed_total": 5,
            "algorithm_runs_failed_total": 2,
            "free_trial_attempts_total": 3,
            "users_total": 12,
            "users_signed_up_today": 2,
            "signup_day": "2026-04-14",
            "last_algorithm_run_at": "2026-04-14T12:00:00+00:00",
            "pass1_timeout_failures_total": 1,
            "pass2_success_try1_total": 3,
            "pass2_success_try2_total": 1,
            "pass2_meaningful_runs_total": 4,
            "pass3_success_try1_total": 2,
            "pass3_success_try2_total": 0,
            "pass3_meaningful_runs_total": 2,
            "updated_at": "2026-04-14T12:05:00+00:00",
            "storage": "postgres",
        }

    def record_run_started(self):
        self.algorithm_starts += 1

    def record_run_completed(self, pass_outcomes=None):
        self.algorithm_completions.append(pass_outcomes)

    def record_run_failed(self, pass_outcomes=None):
        self.algorithm_failures.append(pass_outcomes)

    def increment_free_trial_attempts(self):
        self.free_trial_attempts += 1

    def refresh_user_totals(self):
        self.user_refreshes += 1

    def read_report(self):
        return dict(self.report)


class TestUsageReport(unittest.TestCase):
    def test_table_exists_accepts_dict_row_shape(self):
        class _FakeCursor:
            def execute(self, *_args, **_kwargs):
                return None

            def fetchone(self):
                return {"regclass_name": "users"}

        store = usage_report.UsageReportStore.__new__(usage_report.UsageReportStore)
        self.assertTrue(store._table_exists(_FakeCursor(), "users"))

    def test_query_scalar_accepts_dict_row_shape(self):
        class _FakeCursor:
            def execute(self, *_args, **_kwargs):
                return None

            def fetchone(self):
                return {"count": 7}

        store = usage_report.UsageReportStore.__new__(usage_report.UsageReportStore)
        self.assertEqual(store._query_scalar(_FakeCursor(), "SELECT COUNT(*) FROM users"), 7)

    def test_note_updates_use_store_when_available(self):
        fake_store = _FakeUsageStore()
        pass_outcomes = {
            "pass2": {"applied": True, "successful_attempt": 2},
            "pass3": {"applied": False, "successful_attempt": None},
        }
        with patch.object(usage_report, "get_usage_report_store", return_value=fake_store):
            self.assertTrue(usage_report.note_algorithm_started())
            self.assertTrue(usage_report.note_algorithm_completed(pass_outcomes))
            self.assertTrue(usage_report.note_algorithm_failed({"pass1": {"status": "timeout_exhausted"}}))
            self.assertTrue(usage_report.note_free_trial_attempt())
            self.assertTrue(usage_report.note_user_signup())
            self.assertEqual(usage_report.read_usage_report(), fake_store.report)

        self.assertEqual(fake_store.algorithm_starts, 1)
        self.assertEqual(fake_store.algorithm_completions, [pass_outcomes])
        self.assertEqual(
            fake_store.algorithm_failures,
            [{"pass1": {"status": "timeout_exhausted"}}],
        )
        self.assertEqual(fake_store.free_trial_attempts, 1)
        self.assertEqual(fake_store.user_refreshes, 1)

    def test_note_updates_skip_cleanly_without_db(self):
        with patch.object(usage_report, "get_usage_report_store", return_value=None):
            self.assertFalse(usage_report.note_algorithm_started())
            self.assertFalse(usage_report.note_algorithm_completed())
            self.assertFalse(usage_report.note_algorithm_failed())
            self.assertFalse(usage_report.note_free_trial_attempt())
            self.assertFalse(usage_report.note_user_signup())
            self.assertIsNone(usage_report.read_usage_report())

    def test_monitor_usage_endpoint_returns_usage_payload(self):
        report = {
            "algorithm_runs_total": 19,
            "algorithm_runs_completed_total": 17,
            "algorithm_runs_failed_total": 2,
            "free_trial_attempts_total": 5,
            "users_total": 44,
            "users_signed_up_today": 6,
            "signup_day": "2026-04-14",
            "last_algorithm_run_at": "2026-04-14T13:00:00+00:00",
            "pass1_timeout_failures_total": 1,
            "pass2_success_try1_total": 9,
            "pass2_success_try2_total": 3,
            "pass2_meaningful_runs_total": 12,
            "pass3_success_try1_total": 6,
            "pass3_success_try2_total": 0,
            "pass3_meaningful_runs_total": 6,
            "updated_at": "2026-04-14T13:01:00+00:00",
            "storage": "postgres",
        }
        with patch.dict(os.environ, {"MONITOR_READ_TOKEN": "secret"}, clear=False):
            with patch.object(web_app, "read_usage_report", return_value=report):
                client = TestClient(web_app.app)
                resp = client.get(
                    "/monitor/usage",
                    headers={"X-Monitor-Token": "secret"},
                )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["database_configured"])
        self.assertEqual(body["usage"], report)


if __name__ == "__main__":
    unittest.main()
