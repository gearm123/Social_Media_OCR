"""Tests for pipeline retry/failure log summaries."""

import os
import sys
import unittest

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main


class TestPipelineLogging(unittest.TestCase):
    def test_format_pass_attempt_log_summarizes_non_ideal_attempts(self):
        line = main._format_pass_attempt_log(
            "pass2",
            {
                "status": "success",
                "successful_attempt": 2,
                "max_tries": 2,
                "timed_out_attempts": 1,
                "transient_status_retry_count": 0,
                "model": "gemini-2.5-flash",
                "model_failovers": 1,
            },
        )

        self.assertIn("[pipeline] pass2 attempts summary:", line)
        self.assertIn("status=success", line)
        self.assertIn("attempt=2/2", line)
        self.assertIn("timed_out_attempts=1", line)
        self.assertIn("model=gemini-2.5-flash", line)
        self.assertIn("model_failovers=1", line)

    def test_format_pass_attempt_log_skips_clean_first_try_success(self):
        line = main._format_pass_attempt_log(
            "pass3",
            {
                "status": "success",
                "successful_attempt": 1,
                "max_tries": 2,
                "timed_out_attempts": 0,
                "transient_status_retry_count": 0,
                "model_failovers": 0,
            },
        )

        self.assertEqual(line, "")


if __name__ == "__main__":
    unittest.main()
