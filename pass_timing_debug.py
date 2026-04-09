"""
Rolling statistics for Gemini HTTP passes (Pass 1–3), written when ``PIPELINE_VERBOSE=1`` (``--verbose``).

Human-readable logs (only three pass blocks) live under ``timing_debug/pass_timing_debug.txt`` or
``timing_debug/pass_timing_debug_hurry_up.txt``. Each block includes rolling probabilities, including
``all_http_attempts_timed_out_probability``: share of runs where the pass ended after HTTP read timeouts
on the final attempt (for two-try passes, both tries timed out). Full Welford state is kept in sidecar
JSON files (``pass_timing_debug_state.json`` / ``pass_timing_debug_hurry_up_state.json``) so averages stay
correct across runs. Delete both the ``.txt`` and matching ``*_state.json`` to reset a mode completely.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

TIMING_DEBUG_DIR = Path(__file__).resolve().parent / "timing_debug"
# Standard runs (``PIPELINE_HURRY_UP`` unset / false).
LOG_PATH = TIMING_DEBUG_DIR / "pass_timing_debug.txt"
LOG_PATH_HURRY_UP = TIMING_DEBUG_DIR / "pass_timing_debug_hurry_up.txt"
STATE_PATH = TIMING_DEBUG_DIR / "pass_timing_debug_state.json"
STATE_PATH_HURRY_UP = TIMING_DEBUG_DIR / "pass_timing_debug_hurry_up_state.json"


def _active_log_path() -> Path:
    """Separate rolling logs for ``--hurry-up`` vs default timeouts/thinking."""
    hu = os.environ.get("PIPELINE_HURRY_UP", "").strip().lower() in ("1", "true", "yes")
    return LOG_PATH_HURRY_UP if hu else LOG_PATH


def _active_state_path() -> Path:
    hu = os.environ.get("PIPELINE_HURRY_UP", "").strip().lower() in ("1", "true", "yes")
    return STATE_PATH_HURRY_UP if hu else STATE_PATH


# Only these passes get a block (main vision pipeline).
TRACKED_PASSES = (1, 2, 3)


def _welford_update(state: Dict[str, Any], x: float) -> None:
    """Update running mean and sum of squared deviations (for sample variance)."""
    n = int(state.get("n", 0)) + 1
    state["n"] = n
    mean = float(state.get("mean", 0.0))
    delta = x - mean
    mean += delta / n
    state["mean"] = mean
    state["M2"] = float(state.get("M2", 0.0)) + delta * (x - mean)


def _empty_pass_block() -> Dict[str, Any]:
    return {
        "runs": 0,
        "retry_fallback_count": 0,
        "failed_all_attempts_count": 0,
        "exhausted_on_timeout_count": 0,
        "try1": {"n": 0, "mean": 0.0, "M2": 0.0},
        "try2": {"n": 0, "mean": 0.0, "M2": 0.0},
        "overall": {"n": 0, "mean": 0.0, "M2": 0.0},
        "overall_no_timeout": {"n": 0, "mean": 0.0, "M2": 0.0},
    }


def _normalize_pass_block(raw: Any) -> Dict[str, Any]:
    block = raw if isinstance(raw, dict) else {}
    out = _empty_pass_block()
    out["runs"] = int(block.get("runs", 0))
    out["retry_fallback_count"] = int(block.get("retry_fallback_count", 0))
    out["failed_all_attempts_count"] = int(block.get("failed_all_attempts_count", 0))
    out["exhausted_on_timeout_count"] = int(block.get("exhausted_on_timeout_count", 0))
    for sub in ("try1", "try2", "overall", "overall_no_timeout"):
        st = block.get(sub)
        if isinstance(st, dict) and "M2" in st and "mean" in st:
            out[sub] = {
                "n": int(st.get("n", 0)),
                "mean": float(st.get("mean", 0.0)),
                "M2": float(st.get("M2", 0.0)),
            }
    return out


def _normalize_root(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    return {
        "pass_1": _normalize_pass_block(data.get("pass_1")),
        "pass_2": _normalize_pass_block(data.get("pass_2")),
        "pass_3": _normalize_pass_block(data.get("pass_3")),
    }


def _load_root_from_legacy_txt(log_path: Path) -> Optional[Dict[str, Any]]:
    """Load state embedded in an older ``.txt`` that ended with a JSON object."""
    if not log_path.is_file():
        return None
    try:
        raw = log_path.read_text(encoding="utf-8")
        start = raw.find("{")
        if start < 0:
            return None
        return _normalize_root(json.loads(raw[start:]))
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _load_root() -> Dict[str, Any]:
    state_path = _active_state_path()
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return _normalize_root(data)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    legacy = _load_root_from_legacy_txt(_active_log_path())
    if legacy is not None:
        return legacy
    return {"pass_1": _empty_pass_block(), "pass_2": _empty_pass_block(), "pass_3": _empty_pass_block()}


def _public_log_lines(root: Dict[str, Any]) -> List[str]:
    """Short human summary only (three pass blocks); no JSON, no variance."""
    lines: List[str] = []
    for key in ("pass_1", "pass_2", "pass_3"):
        block = root.get(key) or _empty_pass_block()
        runs = int(block.get("runs", 0))
        rfc = int(block.get("retry_fallback_count", 0))
        fac = int(block.get("failed_all_attempts_count", 0))
        eto = int(block.get("exhausted_on_timeout_count", 0))
        p_fb = (rfc / runs) if runs else 0.0
        p_fail = (fac / runs) if runs else 0.0
        p_all_timeouts = (eto / runs) if runs else 0.0
        lines.append(f"# --- {key.upper().replace('_', ' ')} ---")
        lines.append(f"# runs_total: {runs}")
        lines.append(f"# try_1_fallback_probability: {p_fb:.6f}")
        lines.append(f"# try_2_fail_probability: {p_fail:.6f}")
        lines.append(
            f"# all_http_attempts_timed_out_probability: {p_all_timeouts:.6f}  "
            f"(runs where every HTTP attempt for this pass hit read timeout; {eto}/{runs})"
        )
        for label in ("try1", "try2", "overall"):
            st = block.get(label) or {}
            n = int(st.get("n", 0))
            avg = float(st.get("mean", 0.0)) if n else 0.0
            if label == "overall":
                lines.append(f"# overall run average seconds = {avg:.3f}")
            else:
                lines.append(f"# {label}: current average seconds = {avg:.3f}")
        st_nt = block.get("overall_no_timeout") or {}
        n_nt = int(st_nt.get("n", 0))
        avg_nt = float(st_nt.get("mean", 0.0)) if n_nt else 0.0
        lines.append(f"# overall_no_timeout: current average seconds = {avg_nt:.3f}")
        lines.append("#")
    return lines


def _save_root(root: Dict[str, Any]) -> None:
    TIMING_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    root = _normalize_root(root)
    state_path = _active_state_path()
    log_path = _active_log_path()
    state_path.write_text(json.dumps(root, indent=2, sort_keys=False), encoding="utf-8")
    log_path.write_text("\n".join(_public_log_lines(root)) + "\n", encoding="utf-8")


@dataclass
class GeminiPassHttpTiming:
    """Collects one ``_gemini_generate`` HTTP attempt sequence (single pass)."""

    pass_num: int
    max_tries: int
    t_pass_wall: float = field(default_factory=lambda: __import__("time").time())
    # Recorded contribution per attempt index for running averages (timeout → penalty = configured cap).
    try_contrib: List[Optional[float]] = field(default_factory=list)
    second_attempt_started: bool = False
    exhausted_on_timeout: bool = False
    pass_failed: bool = False
    any_timeout: bool = False

    def _ensure(self, attempt_i: int) -> None:
        while len(self.try_contrib) <= attempt_i:
            self.try_contrib.append(None)

    def record_timeout(self, attempt_i: int, attempt_timeout_sec: float) -> None:
        self.any_timeout = True
        self._ensure(attempt_i)
        self.try_contrib[attempt_i] = float(attempt_timeout_sec)

    def record_success(self, attempt_i: int, wall_elapsed_sec: float) -> None:
        self._ensure(attempt_i)
        self.try_contrib[attempt_i] = float(max(0.0, wall_elapsed_sec))

    def mark_second_attempt_started(self) -> None:
        self.second_attempt_started = True

    def overall_wall_sec(self) -> float:
        import time

        return max(0.0, time.time() - self.t_pass_wall)


def record_gemini_pass_http_timing(timing: GeminiPassHttpTiming) -> None:
    """Merge one pass's outcome into the active timing log (standard vs hurry_up)."""
    pn = timing.pass_num
    if pn is None or int(pn) not in TRACKED_PASSES:
        return

    root = _load_root()
    key = f"pass_{int(pn)}"
    block = root.setdefault(key, _empty_pass_block())
    block["runs"] = int(block.get("runs", 0)) + 1

    if timing.second_attempt_started:
        block["retry_fallback_count"] = int(block.get("retry_fallback_count", 0)) + 1

    if timing.exhausted_on_timeout or timing.pass_failed:
        block["failed_all_attempts_count"] = int(block.get("failed_all_attempts_count", 0)) + 1

    if timing.exhausted_on_timeout:
        block["exhausted_on_timeout_count"] = int(block.get("exhausted_on_timeout_count", 0)) + 1

    tc = timing.try_contrib
    if len(tc) >= 1 and tc[0] is not None:
        _welford_update(block["try1"], float(tc[0]))

    if len(tc) >= 2 and tc[1] is not None:
        _welford_update(block["try2"], float(tc[1]))

    _welford_update(block["overall"], float(timing.overall_wall_sec()))

    if not timing.any_timeout:
        _welford_update(block["overall_no_timeout"], float(timing.overall_wall_sec()))

    _save_root(root)
