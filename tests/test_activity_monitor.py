"""Tests for the PostgreSQL-backed activity monitor plumbing."""

import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import activity_log
import web_app


class _FakeActivityStore:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def recent_events(self, limit=100, event=None):
        rows = list(reversed(self.records))
        if event:
            rows = [row for row in rows if row.get("event") == event]
        return rows[:limit]

    def summary(self, event=None):
        rows = self.recent_events(limit=500, event=event)
        return {
            "total_events": len(rows),
            "counts_by_event": {event: len(rows)} if event and rows else {},
            "latest_by_event": ({event: rows[0]["ts"]} if event and rows else {}),
            "latest_event_at": (rows[0]["ts"] if rows else None),
            "latest_event": (rows[0] if rows else None),
            "storage": "postgres",
        }


class TestActivityMonitor(unittest.TestCase):
    def test_write_activity_uses_store_and_sanitizes_payload(self):
        fake_store = _FakeActivityStore()
        with patch.object(activity_log, "get_activity_store", return_value=fake_store):
            ok = activity_log.write_activity(
                "job_completed",
                job_id="job_123",
                notes="x" * 400,
                nested={"status": "done"},
            )

        self.assertTrue(ok)
        self.assertEqual(len(fake_store.records), 1)
        record = fake_store.records[0]
        self.assertEqual(record["event"], "job_completed")
        self.assertEqual(record["job_id"], "job_123")
        self.assertTrue(record["notes"].endswith("..."))
        self.assertEqual(record["nested"]["status"], "done")

    def test_monitor_endpoint_returns_db_summary_shape(self):
        events = [
            {
                "ts": "2026-04-14T12:00:00+00:00",
                "event": "job_completed",
                "job_id": "job_123",
            }
        ]
        summary = {
            "total_events": 12,
            "counts_by_event": {"job_completed": 9, "job_failed": 3},
            "latest_by_event": {"job_completed": "2026-04-14T12:00:00+00:00"},
            "latest_event_at": "2026-04-14T12:00:00+00:00",
            "latest_event": events[0],
            "storage": "postgres",
        }
        with patch.dict(os.environ, {"MONITOR_READ_TOKEN": "secret"}, clear=False):
            with patch.object(web_app, "read_recent_activity", return_value=events):
                with patch.object(web_app, "read_activity_summary", return_value=summary):
                    client = TestClient(web_app.app)
                    resp = client.get(
                        "/monitor/activity?limit=5",
                        headers={"X-Monitor-Token": "secret"},
                    )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["storage"], "postgres")
        self.assertEqual(body["events"], events)
        self.assertEqual(body["summary"], summary)


if __name__ == "__main__":
    unittest.main()
