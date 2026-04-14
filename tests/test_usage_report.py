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
        self.algorithm_runs = 0
        self.free_trial_attempts = 0
        self.user_refreshes = 0
        self.report = {
            "algorithm_runs_total": 7,
            "free_trial_attempts_total": 3,
            "users_total": 12,
            "users_signed_up_today": 2,
            "signup_day": "2026-04-14",
            "last_algorithm_run_at": "2026-04-14T12:00:00+00:00",
            "updated_at": "2026-04-14T12:05:00+00:00",
            "storage": "postgres",
        }

    def increment_algorithm_runs(self):
        self.algorithm_runs += 1

    def increment_free_trial_attempts(self):
        self.free_trial_attempts += 1

    def refresh_user_totals(self):
        self.user_refreshes += 1

    def read_report(self):
        return dict(self.report)


class TestUsageReport(unittest.TestCase):
    def test_note_updates_use_store_when_available(self):
        fake_store = _FakeUsageStore()
        with patch.object(usage_report, "get_usage_report_store", return_value=fake_store):
            self.assertTrue(usage_report.note_algorithm_run())
            self.assertTrue(usage_report.note_free_trial_attempt())
            self.assertTrue(usage_report.note_user_signup())
            self.assertEqual(usage_report.read_usage_report(), fake_store.report)

        self.assertEqual(fake_store.algorithm_runs, 1)
        self.assertEqual(fake_store.free_trial_attempts, 1)
        self.assertEqual(fake_store.user_refreshes, 1)

    def test_note_updates_skip_cleanly_without_db(self):
        with patch.object(usage_report, "get_usage_report_store", return_value=None):
            self.assertFalse(usage_report.note_algorithm_run())
            self.assertFalse(usage_report.note_free_trial_attempt())
            self.assertFalse(usage_report.note_user_signup())
            self.assertIsNone(usage_report.read_usage_report())

    def test_monitor_usage_endpoint_returns_usage_payload(self):
        report = {
            "algorithm_runs_total": 19,
            "free_trial_attempts_total": 5,
            "users_total": 44,
            "users_signed_up_today": 6,
            "signup_day": "2026-04-14",
            "last_algorithm_run_at": "2026-04-14T13:00:00+00:00",
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
