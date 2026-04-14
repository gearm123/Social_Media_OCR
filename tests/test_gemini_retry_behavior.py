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
    def setUp(self):
        ocr_translate.reset_gemini_pass_outcomes()

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
        self.assertEqual(result.meta["successful_attempt"], 2)
        self.assertEqual(result.meta["status"], "success")
        self.assertEqual(post_mock.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [4.0, 8.0, 16.0])
        self.assertTrue(any(event.get("clear_retry_label") for event in retry_events))
        retry_eta_values = [
            event.get("eta_extra_sec")
            for event in retry_events
            if not event.get("clear_retry_label")
        ]
        self.assertTrue(retry_eta_values)
        self.assertTrue(all(value is not None and value >= 0 for value in retry_eta_values))
        self.assertTrue(
            all("added_eta_sec" not in event for event in retry_events if not event.get("clear_retry_label"))
        )
        self.assertEqual(
            ocr_translate.get_gemini_pass_outcomes().get(1, {}).get("successful_attempt"),
            2,
        )

    def test_pass3_retries_once_after_timeout(self):
        payload = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "ok-pass3"}]},
                }
            ],
            "usageMetadata": {},
        }
        responses = [requests.exceptions.Timeout(), _FakeResponse(200, payload)]

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
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
            ocr_translate, "_notify_gemini_retry"
        ), patch.object(
            ocr_translate, "_gemini_api_key", "test-key"
        ), patch.object(
            ocr_translate, "_gemini_active_model", ("gemini-2.5-pro", "v1beta")
        ), patch(
            "requests.post", side_effect=responses
        ) as post_mock:
            result = ocr_translate._gemini_generate("hello", timeout=66, pass_num=3)

        self.assertEqual(result.text, "ok-pass3")
        self.assertEqual(result.meta["successful_attempt"], 2)
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(
            ocr_translate.get_gemini_pass_outcomes().get(3, {}).get("successful_attempt"),
            2,
        )

    def test_pass1_logs_overload_trigger_after_transient_status_exhaustion(self):
        responses = [_FakeResponse(503) for _ in range(12)]

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
            ocr_translate, "_notify_gemini_retry"
        ), patch.object(
            ocr_translate, "_log_servers_overloaded_trigger"
        ) as overload_log, patch.object(
            ocr_translate, "_gemini_api_key", "test-key"
        ), patch.object(
            ocr_translate, "_gemini_active_model", ("gemini-2.5-pro", "v1beta")
        ), patch(
            "requests.post", side_effect=responses
        ), patch(
            "time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "SERVERS_OVERLOADED"):
                ocr_translate._gemini_generate("hello", timeout=66, pass_num=1)

        overload_log.assert_called_once_with(
            pass_num=1,
            reason="transient_http_status_exhausted",
            max_tries=3,
            transient_status_retry_count=3,
            timed_out_attempts=0,
            final_http_status=503,
        )

    def test_pass1_logs_overload_trigger_after_timeout_exhaustion(self):
        responses = [requests.exceptions.Timeout(), requests.exceptions.Timeout(), requests.exceptions.Timeout()]

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
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
            ocr_translate, "_notify_gemini_retry"
        ), patch.object(
            ocr_translate, "_log_servers_overloaded_trigger"
        ) as overload_log, patch.object(
            ocr_translate, "_gemini_api_key", "test-key"
        ), patch.object(
            ocr_translate, "_gemini_active_model", ("gemini-2.5-pro", "v1beta")
        ), patch(
            "requests.post", side_effect=responses
        ):
            with self.assertRaises(requests.exceptions.Timeout):
                ocr_translate._gemini_generate("hello", timeout=66, pass_num=1)

        overload_log.assert_called_once_with(
            pass_num=1,
            reason="timeout_exhausted",
            max_tries=3,
            transient_status_retry_count=0,
            timed_out_attempts=3,
            final_http_status=None,
            final_attempt_timeout_sec=40,
        )


if __name__ == "__main__":
    unittest.main()
