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
        ocr_translate._gemini_active_model = None
        ocr_translate._gemini_model_candidates = None

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
        # Pass 1 reset patience 2: two 503s restart the same outer attempt (12s wait each), then 200.
        responses = [_FakeResponse(503), _FakeResponse(503), _FakeResponse(200, payload)]
        retry_events = []

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_BASE_DELAY_SEC": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_MAX_DELAY_SEC": "",
                # One model: transient exhaustion advances to the next timeout attempt (no model switch).
                "GEMINI_PASS1_MODEL_CHAIN": "gemini-2.5-pro",
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
        self.assertEqual(result.meta["successful_attempt"], 1)
        self.assertEqual(result.meta["status"], "success")
        self.assertEqual(post_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [12.0, 12.0])
        self.assertTrue(any(event.get("clear_retry_label") for event in retry_events))
        self.assertEqual(
            ocr_translate.get_gemini_pass_outcomes().get(1, {}).get("successful_attempt"),
            1,
        )

    def test_pass3_single_http_attempt_no_timeout_retry(self):
        """Pass 3 uses one HTTP attempt; timeout does not trigger a second try."""
        responses = [requests.exceptions.Timeout()]

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
            with self.assertRaises(requests.exceptions.Timeout):
                ocr_translate._gemini_generate("hello", timeout=66, pass_num=3)

        self.assertEqual(post_mock.call_count, 1)

    def test_pass3_http_503_skips_without_retry_or_model_fallback(self):
        responses = [_FakeResponse(503)]

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRIES": "",
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
            with self.assertRaisesRegex(RuntimeError, r"Gemini Pass 3 skipped after HTTP 503"):
                ocr_translate._gemini_generate("hello", timeout=66, pass_num=3)

        self.assertEqual(post_mock.call_count, 1)

    def test_pass1_fails_over_to_fallback_model_after_transient_status_exhaustion(self):
        payload = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "ok-via-flash"}]},
                }
            ],
            "usageMetadata": {},
        }
        # Exhaust reset patience on Pro (2 restarts = 3 POSTs), then Flash succeeds.
        responses = [_FakeResponse(503), _FakeResponse(503), _FakeResponse(503), _FakeResponse(200, payload)]
        post_urls = []

        def _fake_post(url, json=None, timeout=None):
            post_urls.append(url)
            resp = responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

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
            ocr_translate, "_gemini_api_key", "test-key"
        ), patch.object(
            ocr_translate,
            "_gemini_active_model",
            ("gemini-2.5-pro", "v1beta"),
        ), patch.object(
            ocr_translate,
            "_gemini_pass1_fixed_model_chain",
            return_value=[
                ("gemini-2.5-pro", "v1beta"),
                ("gemini-2.5-flash", "v1beta"),
            ],
        ), patch(
            "requests.post", side_effect=_fake_post
        ), patch(
            "time.sleep"
        ) as sleep_mock:
            result = ocr_translate._gemini_generate("hello", timeout=66, pass_num=1)

        self.assertEqual(result.text, "ok-via-flash")
        self.assertEqual(result.meta["successful_attempt"], 1)
        self.assertEqual(result.meta["model"], "gemini-2.5-flash")
        self.assertEqual(result.meta["model_failovers"], 1)
        self.assertEqual(len(post_urls), 4)
        self.assertTrue(all("gemini-2.5-pro" in url for url in post_urls[:3]))
        self.assertIn("gemini-2.5-flash", post_urls[3])
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [12.0, 12.0])

    def test_pass1_logs_overload_trigger_after_transient_status_exhaustion(self):
        # No 503 resets: each outer attempt gets one POST; 3×503 exhausts Pass 1.
        responses = [_FakeResponse(503) for _ in range(3)]

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRIES": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_BASE_DELAY_SEC": "",
                "GEMINI_TRANSIENT_HTTP_RETRY_MAX_DELAY_SEC": "",
                "GEMINI_PASS1_MODEL_CHAIN": "gemini-2.5-pro",
                "GEMINI_PASS1_RESET_PATIENCE": "0",
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
            transient_status_retry_count=0,
            timed_out_attempts=0,
            final_http_status=503,
        )

    def test_pass1_logs_overload_trigger_after_timeout_exhaustion(self):
        responses = [requests.exceptions.Timeout(), requests.exceptions.Timeout(), requests.exceptions.Timeout()]

        with patch.dict(
            os.environ,
            {
                "GEMINI_HTTP_RETRIES": "",
                "GEMINI_PASS1_MODEL_CHAIN": "gemini-2.5-pro",
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


class TestGeminiModelDiscovery(unittest.TestCase):
    def test_tts_models_excluded_from_pipeline_discovery(self):
        ex = ocr_translate._gemini_exclude_from_pipeline_discovery
        self.assertTrue(ex("gemini-2.5-pro-preview-tts"))
        self.assertFalse(ex("gemini-2.5-pro-preview-05-06"))
        self.assertFalse(ex("gemini-2.5-flash"))


if __name__ == "__main__":
    unittest.main()
