"""Regression tests for Gemini transient HTTP retry behavior."""

import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import requests

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ocr_translate


@contextmanager
def _noop_wait(*_args, **_kwargs):
    yield


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = int(status_code)
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} Server Error")
            err.response = self
            raise err

    def json(self):
        return self._payload


class TestGeminiRetryBehavior(unittest.TestCase):
    def test_transient_status_retry_delay_defaults_are_spread_out(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_TRANSIENT_HTTP_RETRY_BASE_DELAY_SEC": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_MAX_DELAY_SEC": "",
            },
            clear=False,
        ):
            self.assertEqual(ocr_translate._gemini_transient_status_retry_delay_sec(0), 4.0)
            self.assertEqual(ocr_translate._gemini_transient_status_retry_delay_sec(1), 8.0)
            self.assertEqual(ocr_translate._gemini_transient_status_retry_delay_sec(2), 16.0)

    def test_pass1_advances_to_next_attempt_after_transient_status_exhaustion(self):
        payload = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "ok"}]},
                }
            ],
            "usageMetadata": {},
        }
        responses = [_FakeResponse(503) for _ in range(4)] + [_FakeResponse(200, payload)]
        retry_events = []

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_BASE_DELAY_SEC": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_MAX_DELAY_SEC": "",
            },
            clear=False,
        ), patch.object(
            ocr_translate, "_gemini_discover_if_needed", return_value=True
        ), patch.object(
            ocr_translate, "_gemini_pipeline_http_wait", side_effect=lambda *_a, **_k: _noop_wait()
        ), patch.object(
            ocr_translate, "_gemini_build_generation_config", return_value={"maxOutputTokens": 1024}
        ), patch.object(
            ocr_translate, "_compact_verbose_logs", return_value=False
        ), patch.object(
            ocr_translate, "_pipeline_verbose", return_value=False
        ), patch.object(
            ocr_translate, "_notify_gemini_retry", side_effect=retry_events.append
        ), patch.object(
            ocr_translate, "_gemini_api_key", "test-key"
        ), patch.object(
            ocr_translate, "_gemini_active_model", ("gemini-2.5-pro", "v1beta")
        ), patch(
            "requests.post", side_effect=responses
        ) as post_mock, patch(
            "time.sleep"
        ) as sleep_mock:
            result = ocr_translate._gemini_generate("hello", timeout=66, pass_num=1)

        self.assertEqual(result.text, "ok")
        self.assertEqual(post_mock.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [4.0, 8.0, 16.0])
        self.assertTrue(any(event.get("clear_retry_label") for event in retry_events))


if __name__ == "__main__":
    unittest.main()
