import os
import json
import cv2
import math
import time
import sys
import re
import threading
from contextlib import contextmanager
from difflib import SequenceMatcher
import unicodedata
from typing import NamedTuple, Optional, Any, Dict, List, Set, Tuple
import numpy as np
from config import (
    DEBUG_DIR,
    GOOGLE_VISION_API_KEY,
    OUTPUT_DIR,
    ocr_engine,
    translator,
    translation_cache,
)
from pass_timing_debug import GeminiPassHttpTiming, record_gemini_pass_http_timing

# --------------------------------------------------
# SOURCE LANGUAGE  (auto-detected from Vision OCR)
# --------------------------------------------------
# Populated after the first OCR call from ocr_and_translate_full_image.
_source_lang_code = "auto"
_source_lang_name = "the source language"
_gemini_retry_notifier = threading.local()
_gemini_pass_outcomes = threading.local()


def _set_source_language(code: str, name: str):
    global _source_lang_code, _source_lang_name
    _source_lang_code = code
    _source_lang_name = name


def set_gemini_retry_notifier(cb) -> None:
    _gemini_retry_notifier.cb = cb


def clear_gemini_retry_notifier() -> None:
    if hasattr(_gemini_retry_notifier, "cb"):
        delattr(_gemini_retry_notifier, "cb")


def _notify_gemini_retry(payload: dict) -> None:
    cb = getattr(_gemini_retry_notifier, "cb", None)
    if cb is None:
        return
    try:
        cb(payload)
    except Exception:
        pass


def reset_gemini_pass_outcomes() -> None:
    _gemini_pass_outcomes.data = {}


def get_gemini_pass_outcomes() -> Dict[int, Dict[str, Any]]:
    raw = getattr(_gemini_pass_outcomes, "data", {}) or {}
    out: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            pk = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            out[pk] = dict(value)
    return out


def _record_gemini_pass_outcome(pass_num: Optional[int], **payload: Any) -> None:
    if pass_num is None:
        return
    try:
        pn = int(pass_num)
    except (TypeError, ValueError):
        return
    if pn not in (1, 2, 3):
        return
    data = get_gemini_pass_outcomes()
    payload["pass_num"] = pn
    data[pn] = payload
    _gemini_pass_outcomes.data = data


def _pipeline_verbose() -> bool:
    return os.environ.get("PIPELINE_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _gemini_pass_timing_enabled_for_pass(pass_num: Optional[int]) -> bool:
    """Whether this ``_gemini_generate`` call writes rolling timing stats.

    Log file is ``timing_debug/pass_timing_debug.txt`` (or ``..._hurry_up.txt`` when hurry-up);
    rolling state is in ``pass_timing_debug_state.json`` (or ``..._hurry_up_state.json``).
    During ``run_pipeline_job``, ``PIPELINE_TIMING_DIFFICULTY`` limits which pass blocks update
    (1→pass 1 only, 2→passes 1–2, 3→passes 1–3).
    """
    if pass_num is None or int(pass_num) not in (1, 2, 3):
        return False
    raw = os.environ.get("PIPELINE_TIMING_DIFFICULTY", "").strip()
    if not raw:
        return True
    try:
        d = int(raw)
    except ValueError:
        return True
    d = max(1, min(3, d))
    return int(pass_num) <= d


def _pipeline_hurry_up() -> bool:
    """Set for the duration of ``run_pipeline_job`` when ``--hurry-up`` / ``PIPELINE_HURRY_UP=1``."""
    return os.environ.get("PIPELINE_HURRY_UP", "").strip().lower() in ("1", "true", "yes")


def _compact_verbose_logs() -> bool:
    """Shorter ``--verbose`` console layout (default when PIPELINE_VERBOSE is on).

    Set ``PIPELINE_VERBOSE_LAYOUT=classic`` (or ``full`` / ``old``) to restore the previous
    long Gemini / phase lines.
    """
    if not _pipeline_verbose():
        return False
    v = os.environ.get("PIPELINE_VERBOSE_LAYOUT", "").strip().lower()
    return v not in ("classic", "full", "old")


def _pv(*args, **kwargs) -> None:
    """Gemini/OCR detail logs — enable with PIPELINE_VERBOSE=1."""
    if _pipeline_verbose():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def gemini_pass_timeout_sec(pass_num: int) -> int:
    """HTTP client read timeout (seconds) for each Gemini ``generateContent`` attempt.

    Resolution order:

    1. ``GEMINI_PASS{n}_TIMEOUT_SEC`` for the pass number of that HTTP call when set and valid.
    2. Else ``GEMINI_REQUEST_TIMEOUT_SEC`` when set and valid.
    3. Else pass **1** defaults to **66** s; passes **2–3** to **70** s; any other pass number defaults
       to **50** s (each pass may retry on timeout where enabled; see ``GEMINI_HTTP_RETRIES``).

    Minimum 30 seconds.
    """
    n = int(pass_num)
    specific = os.environ.get(f"GEMINI_PASS{n}_TIMEOUT_SEC", "").strip()
    if specific:
        try:
            return max(30, int(specific))
        except ValueError:
            pass
    legacy = os.environ.get("GEMINI_REQUEST_TIMEOUT_SEC", "").strip()
    if legacy:
        try:
            return max(30, int(legacy))
        except ValueError:
            pass
    if n == 1:
        default = 66
    elif n in (2, 3):
        default = 70
    else:
        default = 50
    return max(30, default)


def _gemini_http_extra_retries_on_timeout() -> int:
    """Extra ``generateContent`` attempts after :class:`requests.exceptions.Timeout`.

    Default **1** → **2** attempts for most passes. Pass **1** uses **3** attempts when this env var is
    unset (see ``_gemini_http_max_tries_for_pass``). Set ``GEMINI_HTTP_RETRIES=0`` to disable extras.
    Capped at 5 extra retries.
    """
    v = os.environ.get("GEMINI_HTTP_RETRIES", "").strip()
    if not v:
        return 1
    if v.lower() in ("0", "false", "no", "off"):
        return 0
    try:
        return max(0, min(int(v), 5))
    except ValueError:
        return 1


def _gemini_http_max_tries_for_pass(pass_num: Optional[int]) -> int:
    """Pass **2**: **2** HTTP attempts (timeout retry). Pass **3**: **2** attempts.

    Pass **1** (when ``GEMINI_HTTP_RETRIES`` is unset): **3** attempts; else ``1 + GEMINI_HTTP_RETRIES``.
    Other passes: ``1 + GEMINI_HTTP_RETRIES`` (default **2** attempts).
    """
    if pass_num is not None and int(pass_num) == 2:
        return 2
    if pass_num is not None and int(pass_num) == 3:
        return 2
    if pass_num is not None and int(pass_num) == 1:
        if not os.environ.get("GEMINI_HTTP_RETRIES", "").strip():
            return 3
    return 1 + _gemini_http_extra_retries_on_timeout()


def _gemini_http_extra_retries_on_transient_status() -> int:
    """Extra retries for transient upstream Gemini HTTP responses like 503."""
    raw = os.environ.get("GEMINI_TRANSIENT_HTTP_RETRIES", "").strip()
    if not raw:
        return 3
    if raw.lower() in ("0", "false", "no", "off"):
        return 0
    try:
        return max(0, min(int(raw), 3))
    except ValueError:
        return 3


def _is_transient_gemini_http_status(status_code: int) -> bool:
    return int(status_code) in (429, 500, 502, 503, 504)


def _gemini_transient_status_retry_delay_sec(retry_i: int) -> float:
    """Delay between retries for transient Gemini HTTP statuses.

    Defaults to a slower 4s / 8s / 16s backoff so short upstream overload windows do not burn through
    all transient retries immediately. Both the base and cap are env-overridable.
    """
    raw_base = os.environ.get("GEMINI_TRANSIENT_HTTP_RETRY_BASE_DELAY_SEC", "").strip()
    raw_cap = os.environ.get("GEMINI_TRANSIENT_HTTP_RETRY_MAX_DELAY_SEC", "").strip()
    try:
        base = float(raw_base) if raw_base else 4.0
    except ValueError:
        base = 4.0
    try:
        cap = float(raw_cap) if raw_cap else 30.0
    except ValueError:
        cap = 30.0
    base = max(0.5, base)
    cap = max(base, cap)
    return min(cap, base * (2 ** max(0, int(retry_i))))


def _redact_gemini_api_key(text: str) -> str:
    return re.sub(r"(key=)[^&\s]+", r"\1[REDACTED]", str(text or ""))


def _gemini_attempt_timeout_sec(
    pass_num: Optional[int],
    attempt_i: int,
    base_timeout: float,
) -> int:
    """Per-attempt HTTP read timeout.

    Pass **1**: first try *base_timeout* (from ``gemini_pass_timeout_sec(1)``, default **66** s);
    second try **80** s; third try **40** s. Pass **2** (normal): **36** s then **22** s. Pass **3** (normal):
    **36** s then **24** s.

    ``--hurry-up``: Pass **1** **35** s, then **45** s, then **27** s; Pass **2** **18** s then **10** s;
    Pass **3** **16** s then **10** s.
    """
    if _pipeline_hurry_up() and pass_num is not None:
        pn = int(pass_num)
        if pn == 1:
            if attempt_i == 0:
                return 35
            if attempt_i == 1:
                return 45
            return 27
        if pn == 2:
            return 18 if attempt_i == 0 else 10
        if pn == 3:
            return 16 if attempt_i == 0 else 10
    base = int(max(30, round(float(base_timeout))))
    if pass_num is None:
        return base
    pn = int(pass_num)
    if pn == 1 and attempt_i > 0:
        if attempt_i == 1:
            return max(30, 80)
        return max(30, 40)
    if pn == 2:
        if attempt_i == 0:
            return 36
        return 22
    if pn == 3:
        return 36 if attempt_i == 0 else 24
    return base


# Gemini 2.5 Pro: API allows ``thinkingBudget`` only in **128…32768** (cannot disable with 0).
_GEMINI_25_PRO_MIN_THINKING_BUDGET = 128
# Pass 1 only: second HTTP attempt (after timeout on first). **80** s read timeout in
# ``_gemini_attempt_timeout_sec``; **1920** thinking budget here (not Pass 2).
_GEMINI_PASS1_RETRY_THINKING_BUDGET = 1920
# Pass 1 third HTTP attempt: **40** s read timeout; **960** thinking budget.
_GEMINI_PASS1_RETRY3_THINKING_BUDGET = 960
# Pass 2 second HTTP attempt (take-your-time timeout retry): **1920** thinking budget.
_GEMINI_PASS2_RETRY_THINKING_BUDGET = 1920
# Pass 3 first attempt (take-your-time): **1920** thinking budget.
_GEMINI_PASS3_TRY1_THINKING_BUDGET = 1920
# Pass 4+ or *pass_num* unset: first-attempt ``thinkingBudget`` for **Gemini 2.5**.
_GEMINI_FIRST_TRY_THINKING_BUDGET = 3840
# ``--hurry-up`` / ``PIPELINE_HURRY_UP=1`` — ground truth (timeouts in ``_gemini_attempt_timeout_sec``):
#   Pass 1: 35s/1920, 45s/960, 27s/960  |  Pass 2: 18s/960, 10s/480  |  Pass 3: 16s/960
# See ``_gemini_build_generation_config`` for how budgets attach to each attempt.
_HURRY_UP_PASS1_TRY1_THINKING = 1920
_HURRY_UP_PASS1_TRY2_THINKING = 960
_HURRY_UP_PASS2_TRY1_THINKING = 960
_HURRY_UP_PASS2_TRY2_THINKING = 480
_HURRY_UP_PASS3_TRY1_THINKING = 960


def _gemini_build_generation_config(
    model: str,
    pass_num: Optional[int],
    temperature: Optional[float],
    max_out: int,
    *,
    use_json_output: bool,
    minimal_thinking: bool,
    attempt_i: int = 0,
) -> Dict[str, Any]:
    """Build ``generationConfig`` for ``generateContent`` (**Gemini 2.5**).

    Pipeline contract:

    - **Pass 1**, first HTTP attempt: omit ``thinkingConfig`` (no explicit thinking budget).
    - **Pass 1**, second HTTP attempt (timeout retry only): ``thinkingBudget`` =
      ``_GEMINI_PASS1_RETRY_THINKING_BUDGET`` (**1920**), clamped to the model range.
    - **Pass 1**, third HTTP attempt: ``thinkingBudget`` = ``_GEMINI_PASS1_RETRY3_THINKING_BUDGET`` (**960**).
    - **Pass 2** (normal): first HTTP attempt — omit ``thinkingConfig``; second (timeout retry) —
      ``thinkingBudget`` **1920** (``_GEMINI_PASS2_RETRY_THINKING_BUDGET``).
    - **Pass 2** (``--hurry-up``): fixed budgets per attempt (see constants below).
    - **Pass 3** (normal): ``thinkingBudget`` **1920** (``_GEMINI_PASS3_TRY1_THINKING_BUDGET``).
    - **Pass 3** (``--hurry-up``): ``_HURRY_UP_PASS3_TRY1_THINKING`` (**960**).
    - **Pass 4+** or *pass_num* unset (e.g. status bar): first attempt **3840** (clamped); HTTP
      timeout retry → **Pro** **128** / **Flash** ``0``.

    HTTP timeouts are enforced in ``_gemini_attempt_timeout_sec`` / ``gemini_pass_timeout_sec``,
    not in this dict.
    """
    temp = 0.0 if temperature is None else max(0.0, min(1.0, float(temperature)))
    gen_cfg: Dict[str, Any] = {
        "maxOutputTokens": max(1024, min(int(max_out), 65536)),
        "temperature": temp,
        "topP": 1.0,
    }
    m = (model or "").lower()
    is_25 = "2.5" in m
    is_25_pro = is_25 and ("pro" in m and "flash" not in m)

    if is_25:
        pn = int(pass_num) if pass_num is not None else None
        if _pipeline_hurry_up() and pn in (1, 2, 3):
            if pn == 1:
                _tb = (
                    _HURRY_UP_PASS1_TRY1_THINKING
                    if attempt_i == 0
                    else _HURRY_UP_PASS1_TRY2_THINKING
                )
            elif pn == 2:
                _tb = _HURRY_UP_PASS2_TRY2_THINKING if minimal_thinking else _HURRY_UP_PASS2_TRY1_THINKING
            else:
                _tb = _HURRY_UP_PASS3_TRY1_THINKING
            if is_25_pro:
                _tb = max(_GEMINI_25_PRO_MIN_THINKING_BUDGET, min(int(_tb), 32768))
            else:
                _tb = max(0, min(int(_tb), 24576))
            gen_cfg["thinkingConfig"] = {"thinkingBudget": _tb}
        elif pn == 2 and not _pipeline_hurry_up():
            if minimal_thinking:
                _tb = _GEMINI_PASS2_RETRY_THINKING_BUDGET
                if is_25_pro:
                    _tb = max(_GEMINI_25_PRO_MIN_THINKING_BUDGET, min(int(_tb), 32768))
                else:
                    _tb = max(0, min(int(_tb), 24576))
                gen_cfg["thinkingConfig"] = {"thinkingBudget": _tb}
            elif not _compact_verbose_logs():
                _pv(
                    "[GEMINI] Pass 2: thinkingConfig omitted "
                    "(no explicit thinking budget).",
                )
        elif pn == 3 and not _pipeline_hurry_up():
            _tb = _GEMINI_PASS3_TRY1_THINKING_BUDGET
            if is_25_pro:
                _tb = max(_GEMINI_25_PRO_MIN_THINKING_BUDGET, min(int(_tb), 32768))
            else:
                _tb = max(0, min(int(_tb), 24576))
            gen_cfg["thinkingConfig"] = {"thinkingBudget": _tb}
        elif pn == 1:
            if minimal_thinking:
                _tb = (
                    _GEMINI_PASS1_RETRY_THINKING_BUDGET
                    if attempt_i <= 1
                    else _GEMINI_PASS1_RETRY3_THINKING_BUDGET
                )
                if is_25_pro:
                    _tb = max(_GEMINI_25_PRO_MIN_THINKING_BUDGET, min(int(_tb), 32768))
                else:
                    _tb = max(0, min(int(_tb), 24576))
                gen_cfg["thinkingConfig"] = {"thinkingBudget": _tb}
            elif not _compact_verbose_logs():
                _pv(
                    "[GEMINI] Pass 1: thinkingConfig omitted "
                    "(no explicit thinking budget).",
                )
        elif minimal_thinking:
            if is_25_pro:
                gen_cfg["thinkingConfig"] = {
                    "thinkingBudget": _GEMINI_25_PRO_MIN_THINKING_BUDGET,
                }
            else:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        else:
            if pn == 2:
                if not _compact_verbose_logs():
                    _pv(
                        "[GEMINI] Pass 2 first try: thinkingConfig omitted "
                        "(no explicit thinking budget).",
                    )
            else:
                _tb = _GEMINI_FIRST_TRY_THINKING_BUDGET
                if is_25_pro:
                    _tb = max(_GEMINI_25_PRO_MIN_THINKING_BUDGET, min(int(_tb), 32768))
                else:
                    _tb = max(0, min(int(_tb), 24576))
                gen_cfg["thinkingConfig"] = {"thinkingBudget": _tb}

    if use_json_output:
        gen_cfg["responseMimeType"] = "application/json"
    return gen_cfg


def collect_vision_ocr_hints(page_images):
    """Run Google Vision DOCUMENT_TEXT_DETECTION on each page crop; return one string.

    Used as an **auxiliary** transcript for Gemini Pass 1 (names, numbers, Thai spellings).
    Returns ``""`` if Vision is not configured or all pages fail.
    """
    if not GOOGLE_VISION_API_KEY:
        return ""
    parts = []
    for i, img in enumerate(page_images or []):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        try:
            txt = ocr_engine.document_plain_text(img)
        except Exception as e:
            _pv(f"[OCR HINT] Page {i + 1} Vision failed: {e}")
            continue
        if txt:
            parts.append(
                f"--- Page {i + 1} (same order as single-page image {i + 1} before the stitch) ---\n"
                f"{txt}\n"
            )
    return "\n".join(parts).strip()


def _vision_role_from_bbox(bbox, img_w: int) -> str:
    """Map paragraph bbox to Gemini role using horizontal center (Messenger-style)."""
    x1, y1, x2, y2 = bbox
    w = float(max(img_w, 1))
    cx = (x1 + x2) / 2.0 / w
    left_max = float(os.environ.get("GEMINI_OCR_LEFT_MAX_CX", "0.46"))
    right_min = float(os.environ.get("GEMINI_OCR_RIGHT_MIN_CX", "0.54"))
    if cx <= left_max:
        return "contact"
    if cx >= right_min:
        return "user"
    return "system"


def collect_vision_ocr_structured_hints(page_images):
    """Per-page Vision paragraphs with **contact|user|system** from bbox center.

    One indexed line per paragraph, top-to-bottom. Tuned for left=incoming / right=outgoing.
    Returns ``""`` if Vision unavailable or all pages fail.

    .. note::
        The main Gemini pipeline prefers :func:`collect_vision_ocr_stitch_high_confidence_hints`
        (word-level, confidence-filtered, mapped to the stitched canvas).
    """
    if not GOOGLE_VISION_API_KEY:
        return ""
    parts = []
    global_idx = 0
    for pi, img in enumerate(page_images or []):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        try:
            paras = ocr_engine.document_paragraph_boxes(img)
        except Exception as e:
            _pv(f"[OCR STRUCT] Page {pi + 1} Vision failed: {e}")
            continue
        _h, w = img.shape[:2]
        parts.append(f"\n### Page {pi + 1} (image width {w}px)\n")
        for p in paras:
            role = _vision_role_from_bbox(p["bbox"], w)
            t = (p["text"] or "").replace("\n", " ").strip()
            if not t:
                continue
            parts.append(f"[{global_idx}] [{role}] {t}\n")
            global_idx += 1
    return "".join(parts).strip()


def collect_vision_ocr_stitch_high_confidence_hints(
    page_images,
    page_ranges,
    combined_h: int,
    combined_w: int,
    num_stitch_bands: int = 1,
):
    """Per-page Vision **word** boxes, filtered by confidence, mapped to stitched-image coordinates.

    Each screenshot is OCR'd separately. Only tokens with score ≥ ``GEMINI_OCR_HINT_MIN_CONF``
    (default ``0.82``) are listed. Positions use the **same vertical layout** as the stitched
    image: ``y_norm`` is center-of-word / stitched height (0–1), so they stay valid if the
    stitch is uniformly scaled for the API.

    *num_stitch_bands* must match how many vertical bands Gemini will receive (1 = one full
    stitch; >1 = equal-height slices). Each line then includes ``stitch_band=b/K`` so the model
    knows **which input image** contains that token while **y_norm** stays on the **unified**
    scroll (seamless across bands).

    Per-word ``[contact|user|system]`` tags are **omitted by default** — Vision's horizontal
    bucketing mis-tags URLs, banners, and wrapped lines. Set ``GEMINI_OCR_HINT_ROLES=1`` to
    restore them (legacy / debugging).

    Returns ``""`` if Vision is unavailable or all pages fail.
    """
    if not GOOGLE_VISION_API_KEY:
        return ""
    include_spk_roles = os.environ.get("GEMINI_OCR_HINT_ROLES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    min_conf = float(os.environ.get("GEMINI_OCR_HINT_MIN_CONF", "0.90"))
    min_conf = max(0.90, min(1.0, min_conf))

    pages = list(page_images or [])
    ranges = list(page_ranges or [])
    if len(pages) != len(ranges):
        n = min(len(pages), len(ranges))
        _pv(
            f"[OCR STITCH] page_images ({len(pages)}) vs page_ranges ({len(ranges)}) "
            f"— using first {n} pairs"
        )
        pages = pages[:n]
        ranges = ranges[:n]

    ch = max(int(combined_h), 1)
    cw = max(int(combined_w), 1)
    K = max(1, int(num_stitch_bands))
    rows = []

    for pi, (img, pr) in enumerate(zip(pages, ranges)):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        start_y = int(pr[0])
        ih, iw = img.shape[:2]
        if iw <= 0 or ih <= 0:
            continue
        try:
            detections = ocr_engine.predict(img)
        except Exception as e:
            _pv(f"[OCR STITCH] Page {pi + 1} Vision failed: {e}")
            continue

        for det in detections:
            text = (det.get("text") or "").strip()
            if not text:
                continue
            conf = float(det.get("score") or 0.0)
            if conf < min_conf:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            role = (
                _vision_role_from_bbox((x1, y1, x2, y2), iw)
                if include_spk_roles
                else None
            )
            gcy = start_y + (y1 + y2) * 0.5
            gcx = (x1 + x2) * 0.5
            y_norm = gcy / ch
            x_norm = gcx / cw
            b0 = min(K - 1, max(0, int(y_norm * K)))
            if y_norm >= 1.0:
                b0 = K - 1
            rows.append(
                {
                    "pi": pi,
                    "y1": start_y + y1,
                    "x1": x1,
                    "role": role,
                    "text": text,
                    "conf": conf,
                    "y_norm": y_norm,
                    "x_norm": x_norm,
                    "stitch_band": b0 + 1,
                }
            )

    if not rows:
        return ""

    rows.sort(key=lambda r: (r["y1"], r["x1"]))
    band_help = ""
    if K > 1:
        band_help = (
            f"The stitch is sent as **{K}** equal-height vertical **bands** (same scroll). "
            f"**stitch_band=b** = look in **Gemini input image b** (1=top … {K}=bottom). "
            f"That band covers **y_norm** in **[ (b-1)/{K}, b/{K} )** on the **one** virtual canvas — "
            f"use **y_norm** to place the token in the correct **message** (bubble) along the thread; "
            f"use **stitch_band** only to open the right crop.\n\n"
        )
    parts = [
        "\n### Stitched layout (reference)\n",
        band_help,
        f"Virtual stitched canvas: height={ch}px width={cw}px. "
        f"**y_norm** / **x_norm** = word center as fraction of that full canvas (not per-band). "
        f"Only tokens with Vision confidence ≥ {min_conf:.2f} are listed. "
        + (
            "**No** `contact`/`user` tags on lines — use **bubble side in images** for speaker.\n\n"
            if not include_spk_roles
            else "**Bracket roles** from Vision geometry are **unreliable** — prefer bubble pixels for JSON `role`.\n\n"
        ),
    ]
    for i, r in enumerate(rows):
        # JSON-quote token text so special characters cannot break the prompt
        tjson = json.dumps(r["text"], ensure_ascii=False)
        sb = f"stitch_band={r['stitch_band']}/{K} " if K > 1 else ""
        spk = f"[{r['role']}] " if include_spk_roles and r.get("role") else ""
        parts.append(
            f"[{i}] {sb}y_norm={r['y_norm']:.4f} x_norm={r['x_norm']:.4f} "
            f"{spk}conf={r['conf']:.3f} text={tjson} "
            f"(source page {r['pi'] + 1})\n"
        )

    return "".join(parts).strip()


def format_craft_bands_for_gemini_prompt(combined_h: int, craft_objects_sorted) -> str:
    """One markdown line per CRAFT row: message_index and approximate y_norm span on the stitch."""
    if not craft_objects_sorted:
        return ""
    ch = max(int(combined_h), 1)
    lines = []
    for i, obj in enumerate(craft_objects_sorted):
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        y0 = float(bbox[1]) / ch
        y1 = float(bbox[3]) / ch
        lines.append(
            f"- **message_index {i}** (top→bottom): **y_norm** ≈ {y0:.4f}–{y1:.4f}\n"
        )
    return "".join(lines)


def _craft_message_index_for_gcy(gcy: float, craft_objects_sorted) -> int:
    """Map a combined-canvas Y (pixels) to CRAFT row index."""
    if not craft_objects_sorted:
        return 0
    for i, obj in enumerate(craft_objects_sorted):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if bbox[1] <= gcy <= bbox[3]:
            return i
    best_i = 0
    best_d = 1e18
    for i, obj in enumerate(craft_objects_sorted):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        cy = (bbox[1] + bbox[3]) * 0.5
        d = abs(float(gcy) - cy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def collect_vision_ocr_stitch_by_message_index(
    combined_img,
    combined_h: int,
    combined_w: int,
    num_stitch_bands: int,
    craft_objects_sorted,
):
    """Run Vision **once** on the **full stitched** image; bucket words under CRAFT ``message_index``.

    Word center *(x, y)* is in **combined canvas** coordinates — same space as CRAFT boxes from
    :func:`run_craft_and_group_on_combined`. Assignment: vertical overlap / nearest CRAFT row.

    Pass 2 uses this only as **text** context; **speakers** come from Gemini Pass 1.
    """
    if not GOOGLE_VISION_API_KEY or not craft_objects_sorted:
        return ""
    if combined_img is None or getattr(combined_img, "size", 0) == 0:
        return ""

    min_conf = float(os.environ.get("GEMINI_OCR_HINT_MIN_CONF", "0.82"))
    min_conf = max(0.0, min(1.0, min_conf))
    include_spk_roles = os.environ.get("GEMINI_OCR_HINT_ROLES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    ih, iw = combined_img.shape[:2]
    if iw <= 0 or ih <= 0:
        return ""

    ch = max(int(combined_h), 1)
    cw = max(int(combined_w), 1)
    K = max(1, int(num_stitch_bands))
    buckets: Dict[int, list] = {}

    try:
        detections = ocr_engine.predict(combined_img)
    except Exception as e:
        _pv(f"[OCR BY-MSG] Vision on stitched image failed: {e}")
        return ""

    for det in detections:
        text = (det.get("text") or "").strip()
        if not text:
            continue
        conf = float(det.get("score") or 0.0)
        if conf < min_conf:
            continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        role = (
            _vision_role_from_bbox((x1, y1, x2, y2), iw)
            if include_spk_roles
            else None
        )
        gcy = (y1 + y2) * 0.5
        gcx = (x1 + x2) * 0.5
        y_norm = gcy / ch
        x_norm = gcx / cw
        b0 = min(K - 1, max(0, int(y_norm * K)))
        if y_norm >= 1.0:
            b0 = K - 1
        midx = _craft_message_index_for_gcy(gcy, craft_objects_sorted)
        buckets.setdefault(midx, []).append(
            {
                "y_norm": y_norm,
                "x_norm": x_norm,
                "conf": conf,
                "text": text,
                "role": role,
                "stitch_band": b0 + 1,
            }
        )

    if not buckets:
        return ""

    parts = [
        "OCR run on the **full stitched** image (same canvas as CRAFT). "
        "Only **high-confidence** tokens (`conf >= 0.90`) are kept. "
        "Tokens are grouped by **message_index** (CRAFT row). **Not** used for speaker — Pass 1 decides `role`.\n\n",
    ]
    for midx in sorted(buckets.keys()):
        items = sorted(buckets[midx], key=lambda z: (z["y_norm"], z["x_norm"]))
        parts.append(f"### message_index {midx}\n")
        for j, it in enumerate(items):
            tjson = json.dumps(it["text"], ensure_ascii=False)
            sb = f"stitch_band={it['stitch_band']}/{K} " if K > 1 else ""
            spk = f"[{it['role']}] " if include_spk_roles and it.get("role") else ""
            parts.append(
                f"  [{j}] {sb}y_norm={it['y_norm']:.4f} x_norm={it['x_norm']:.4f} "
                f"{spk}conf={it['conf']:.3f} text={tjson}\n"
            )
        parts.append("\n")

    return "".join(parts).strip()


def collect_vision_ocr_crop_by_message_index(crop_images):
    """Run Vision on ordered message crops; each crop index is already `message_index`."""
    if not GOOGLE_VISION_API_KEY or not crop_images:
        return ""

    min_conf = float(os.environ.get("GEMINI_OCR_HINT_MIN_CONF", "0.90"))
    min_conf = max(0.0, min(1.0, min_conf))
    parts = [
        "OCR run on the ordered per-message crops. "
        "Each block already matches the same `message_index` as the crop list. "
        "Only high-confidence tokens are kept.\n\n"
    ]
    kept_any = False

    for midx, crop in enumerate(crop_images):
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        ih, iw = crop.shape[:2]
        if ih <= 0 or iw <= 0:
            continue
        try:
            detections = ocr_engine.predict(crop)
        except Exception as e:
            _pv(f"[OCR BY-MSG] Vision on crop {midx} failed: {e}")
            continue

        items = []
        for det in detections:
            text = (det.get("text") or "").strip()
            if not text:
                continue
            conf = float(det.get("score") or 0.0)
            if conf < min_conf:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            items.append(
                {
                    "y_norm": ((y1 + y2) * 0.5) / float(ih),
                    "x_norm": ((x1 + x2) * 0.5) / float(iw),
                    "conf": conf,
                    "text": text,
                }
            )
        if not items:
            continue
        kept_any = True
        items = sorted(items, key=lambda z: (z["y_norm"], z["x_norm"]))
        parts.append(f"### message_index {midx}\n")
        for j, it in enumerate(items):
            tjson = json.dumps(it["text"], ensure_ascii=False)
            parts.append(
                f"  [{j}] y_norm={it['y_norm']:.4f} x_norm={it['x_norm']:.4f} "
                f"conf={it['conf']:.3f} text={tjson}\n"
            )
        parts.append("\n")

    if not kept_any:
        return ""
    return "".join(parts).strip()


def collect_vision_ocr_page_by_message_index(page_images, page_ranges, craft_objects_sorted):
    """Run Vision on the cleaned page images and bucket tokens by CRAFT message_index."""
    if not GOOGLE_VISION_API_KEY or not page_images or not page_ranges or not craft_objects_sorted:
        return ""

    min_conf = float(os.environ.get("GEMINI_OCR_HINT_MIN_CONF", "0.90"))
    min_conf = max(0.0, min(1.0, min_conf))
    page_map = {int(pr[2]): pr for pr in page_ranges}
    page_rows: Dict[int, list] = {}
    for midx, obj in enumerate(craft_objects_sorted):
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        page_index = int(obj.get("page_index", 0))
        pr = page_map.get(page_index)
        if pr is None:
            continue
        start_y = float(pr[0])
        page_rows.setdefault(page_index, []).append(
            {
                "message_index": midx,
                "y0": float(bbox[1]) - start_y,
                "y1": float(bbox[3]) - start_y,
            }
        )
    buckets: Dict[int, list] = {}

    for page_idx, page_img in enumerate(page_images):
        if page_img is None or getattr(page_img, "size", 0) == 0:
            continue
        pr = page_map.get(page_idx)
        if pr is None:
            continue
        start_y = float(pr[0])
        ih, iw = page_img.shape[:2]
        if ih <= 0 or iw <= 0:
            continue
        try:
            detections = ocr_engine.predict(page_img)
        except Exception as e:
            _pv(f"[OCR BY-MSG] Vision on page {page_idx} failed: {e}")
            continue

        rows_here = page_rows.get(page_idx, [])
        row_pad = 18.0

        for det in detections:
            text = (det.get("text") or "").strip()
            if not text:
                continue
            conf = float(det.get("score") or 0.0)
            if conf < min_conf:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            lcy = (y1 + y2) * 0.5
            candidate_rows = [
                r for r in rows_here
                if (r["y0"] - row_pad) <= lcy <= (r["y1"] + row_pad)
            ]
            if not candidate_rows:
                continue
            best_row = min(
                candidate_rows,
                key=lambda r: abs(lcy - ((r["y0"] + r["y1"]) * 0.5)),
            )
            gcx = (x1 + x2) * 0.5
            midx = int(best_row["message_index"])
            buckets.setdefault(midx, []).append(
                {
                    "y_norm": lcy / float(ih),
                    "x_norm": gcx / float(iw),
                    "conf": conf,
                    "text": text,
                    "page": page_idx + 1,
                }
            )

    if not buckets:
        return ""

    parts = [
        "OCR run on the cleaned page images. "
        "Tokens are assigned to `message_index` using the same CRAFT row map. "
        "Only high-confidence tokens are kept.\n\n"
    ]
    for midx in sorted(buckets.keys()):
        items = sorted(buckets[midx], key=lambda z: (z["page"], z["y_norm"], z["x_norm"]))
        parts.append(f"### message_index {midx}\n")
        for j, it in enumerate(items):
            tjson = json.dumps(it["text"], ensure_ascii=False)
            parts.append(
                f"  [{j}] page={it['page']} y_norm={it['y_norm']:.4f} "
                f"x_norm={it['x_norm']:.4f} conf={it['conf']:.3f} text={tjson}\n"
            )
        parts.append("\n")

    return "".join(parts).strip()


def _is_non_latin(text: str) -> bool:
    """Return True if *text* contains characters outside the Latin/ASCII range.

    Used as a language-agnostic replacement for the old _is_thai() guard:
    if Gemini returns source-language text instead of English, the captured
    text will contain non-Latin characters and we know to look elsewhere for
    the English translation.
    """
    return any(ord(c) > 0x024F for c in text if not c.isspace())


def _prefer_english_surface(text: str, fallback: str = "") -> str:
    """Keep English-only renderable text when Gemini returns mixed-script output."""
    t = (text or "").strip()
    fb = (fallback or "").strip()
    if not t:
        return fb
    if not _is_non_latin(t):
        return t

    # Common Gemini pattern: "<source> (English translation)"
    paren_hits = re.findall(r"\(([^()]+)\)", t)
    for cand in reversed(paren_hits):
        c = cand.strip()
        if c and not _is_non_latin(c):
            return c

    # Fallback: use the English-only baseline if available.
    if fb and not _is_non_latin(fb):
        return fb
    return t

try:
    from pythainlp.util import normalize as thai_normalize
    from pythainlp.tokenize import word_tokenize
    from pythainlp.spell import correct as thai_spell_correct
except ImportError:
    thai_normalize = None
    word_tokenize = None
    thai_spell_correct = None

# --------------------------------------------------
# TRANSLATION (UNCHANGED)
# --------------------------------------------------

def translate_to_en(text):
    """Translate *text* from its detected language to English via Google Translate."""
    if not text or not text.strip():
        return ""
    if text in translation_cache:
        return translation_cache[text]
    try:
        translated = translator.translate(text)
        translation_cache[text] = translated
        return translated
    except Exception:
        return ""


# Backward-compatible alias
translate_th_to_en = translate_to_en


def _translate_or_preserve_text(text):
    if not text or not text.strip():
        return ""
    if _looks_like_link_text(text):
        return text
    return translate_th_to_en(text)


_BLOCK_SEP = "\n<<<MSG>>>\n"
_gemini_api_key = None
_gemini_active_model = None   # (model_name, api_version) tuple once found

# Preferred model name substrings in priority order
_GEMINI_PREFER = ["gemini-2.5", "gemini-2.0", "gemini-1.5"]


def _gemini_discover_model(api_key):
    """Query ListModels to find the best generateContent-capable model."""
    import requests as _req
    for ver in ("v1beta", "v1"):
        try:
            r = _req.get(
                f"https://generativelanguage.googleapis.com/{ver}/models?key={api_key}",
                timeout=15,
            )
            if r.status_code != 200:
                continue
            models = r.json().get("models", [])
            # Keep only models that support generateContent
            capable = [
                m["name"].replace("models/", "")
                for m in models
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and "name" in m
            ]
            if not capable:
                continue
            # Pick highest-priority model, preferring Pro over Flash.
            for pref in _GEMINI_PREFER:
                for name in capable:
                    if pref in name and "pro" in name:
                        return name, ver
                for name in capable:
                    if pref in name and "flash" in name:
                        return name, ver
            # Fallback: first capable model
            return capable[0], ver
        except Exception as e:
            _pv(f"[GEMINI] ListModels error ({ver}): {e}")
    return None, None


class GeminiApiResult(NamedTuple):
    """Result of a ``generateContent`` call (text + full parsed JSON for debugging)."""

    text: str
    response_json: Optional[Dict[str, Any]]
    meta: Optional[Dict[str, Any]] = None


def _gemini_discover_if_needed() -> bool:
    """Load API key from env and pick a model. Returns False if Gemini is unavailable."""
    global _gemini_api_key, _gemini_active_model

    if _gemini_api_key is None:
        _gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
    if not _gemini_api_key:
        return False
    if _gemini_active_model is None:
        if not _compact_verbose_logs():
            _pv("[GEMINI] Discovering available models...")
        model, ver = _gemini_discover_model(_gemini_api_key)
        if model:
            _gemini_active_model = (model, ver)
            if _compact_verbose_logs():
                print("**************", flush=True)
            _pv(f"[GEMINI] Using model: {model} (API {ver})")
        else:
            _pv("[GEMINI] No generateContent-capable model found — refinement disabled")
            _gemini_api_key = ""
            return False
    return True


def _gemini_candidate_finish_reason(api_payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not api_payload:
        return None
    cands = api_payload.get("candidates") or []
    if not cands:
        return None
    return cands[0].get("finishReason")


def _gemini_pass_summary_enabled() -> bool:
    """One-line [gemini] pass summary; on if GEMINI_PASS_SUMMARY=1 or (unset and PIPELINE_VERBOSE).

    Compact verbose layout disables this unless GEMINI_PASS_SUMMARY is explicitly on.
    """
    v = os.environ.get("GEMINI_PASS_SUMMARY", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    if _compact_verbose_logs():
        return False
    return _pipeline_verbose()


def _gemini_log_pass_summary(
    pass_num: Optional[int],
    wall_sec: float,
    thinking_budget_cap: Any,
    thoughts_used: Any,
    total_tokens: Any,
    finish_reason: Any,
    out_chars: int,
    note: str,
) -> None:
    if pass_num is None or not _gemini_pass_summary_enabled():
        return
    th = thoughts_used if thoughts_used is not None else "n/a"
    tt = total_tokens if total_tokens is not None else "n/a"
    fr = finish_reason if finish_reason is not None else "n/a"
    cap = thinking_budget_cap if thinking_budget_cap is not None else "n/a"
    extra = f" note={note}" if note else ""
    print(
        f"[gemini] pass {pass_num}: {wall_sec:.1f}s wall "
        f"thinkingBudget_cap={cap} thoughts_tokens={th} total_tokens={tt} "
        f"finish={fr} json_chars={out_chars}{extra}",
        flush=True,
    )


def _gemini_wait_ui_enabled() -> bool:
    """TTY spinner during Gemini HTTP wait. On if GEMINI_WAIT_UI=1 or (unset and PIPELINE_VERBOSE)."""
    if not sys.stdout.isatty():
        return False
    v = os.environ.get("GEMINI_WAIT_UI", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _pipeline_verbose()


@contextmanager
def _gemini_pipeline_http_wait(pass_num: Optional[int], http_timeout: int):
    """While blocking on generateContent: one status line, then TTY live progress (spinner bar)."""
    if pass_num is None:
        yield
        return

    t0 = time.time()
    _show_http_wait_log = _pipeline_verbose()
    if _show_http_wait_log:
        if _compact_verbose_logs():
            if pass_num == 2:
                print(
                    f"[pipeline] Pass {pass_num} — awaiting Gemini HTTP response "
                    f"HTTP read max {http_timeout}s.",
                    flush=True,
                )
        else:
            print(
                f"[pipeline] Pass {pass_num} — awaiting Gemini HTTP response "
                f"(request in flight; model thinking/generating until the API returns). "
                f"HTTP read max {http_timeout}s.",
                flush=True,
            )

    stop = threading.Event()
    last_len_holder = [96]

    def _spinner() -> None:
        if not _gemini_wait_ui_enabled():
            return
        spin = "|/-\\"
        width = 22
        block = 6
        i = 0
        while not stop.wait(0.12):
            elapsed = time.time() - t0
            off = i % (width - block + 1)
            inner = "".join("=" if off <= j < off + block else "·" for j in range(width))
            bar = f"[{inner}]"
            sp = spin[i % len(spin)]
            msg = (
                f"[pipeline] Pass {pass_num} — live wait  {elapsed:5.1f}s  {bar} {sp}  "
                f"(HTTP max {http_timeout}s)"
            )
            pad = max(0, last_len_holder[0] - len(msg))
            sys.stdout.write("\r" + msg + " " * pad)
            sys.stdout.flush()
            last_len_holder[0] = max(last_len_holder[0], len(msg))
            i += 1

    workers = []
    if _gemini_wait_ui_enabled():
        ws = threading.Thread(target=_spinner, daemon=True)
        ws.start()
        workers.append(ws)

    try:
        yield
    except BaseException:
        elapsed = time.time() - t0
        if _show_http_wait_log:
            print(
                f"[pipeline] Pass {pass_num} — wait ended after {elapsed:.1f}s "
                f"(timeout, HTTP error, or parse failure — see messages above).",
                flush=True,
            )
        raise
    else:
        elapsed = time.time() - t0
        if _show_http_wait_log and not _compact_verbose_logs():
            print(
                f"[pipeline] Pass {pass_num} — Gemini HTTP response finished in {elapsed:.1f}s "
                f"(status OK, body read and JSON parsed).",
                flush=True,
            )
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=3.0)
        if _gemini_wait_ui_enabled():
            clear = max(last_len_holder[0], 100)
            sys.stdout.write("\r" + " " * clear + "\r")
            sys.stdout.flush()


def _gemini_generate(
    prompt,
    image_b64=None,
    image_b64_list=None,
    timeout=50,
    max_output_tokens_override: Optional[int] = None,
    pass_num: Optional[int] = None,
    temperature: Optional[float] = None,
):
    """Call Gemini REST API directly (text-only or multimodal).

    Pass *image_b64* (single JPEG base64) or *image_b64_list* (several JPEGs in order)
    before the text prompt.

    *pass_num* drives thinking budget, HTTP timeout, and retry count.

    **Gemini 2.5** (passes **1–3**): pass **1** — no ``thinkingConfig`` on first try (Pro: historical
    omit; **66** s default HTTP read timeout); second try **1920** thinking budget, **80** s; third try
    **960** thinking budget, **40** s. Pass **2** — **36** s / **22** s per attempt; first try omits
    ``thinkingConfig``, second try **1920** thinking budget. Pass **3** — **36** s then **24** s, with retry
    using minimal thinking on the second attempt.
    With ``PIPELINE_HURRY_UP`` / ``--hurry-up``, passes **1–3** use fixed ``thinkingBudget`` and shorter
    timeouts: pass **1** **35**/**45**/**27** s with **1920**/**960**/**960** thinking; pass **2** retries once
    (**18** s / **10** s; **960** / **480** thinking); pass **3** **16** s / **10** s, **960** then retry-minimal thinking.
    Pass **4+** / unset: **3840** first try; retry **Pro** **128** / **Flash** ``0``.
    See ``_gemini_build_generation_config`` and ``_gemini_attempt_timeout_sec``.

    Returns ``GeminiApiResult(text, response_json)``, or ``None`` if the API key / model
    is not configured. *response_json* is the full API object (for empty/blocked replies).
    """
    global _gemini_api_key, _gemini_active_model
    import requests as _req

    if not _gemini_discover_if_needed():
        return None

    model, ver = _gemini_active_model
    url = (
        f"https://generativelanguage.googleapis.com/{ver}/models/"
        f"{model}:generateContent?key={_gemini_api_key}"
    )

    parts = []
    if image_b64_list:
        for b64 in image_b64_list:
            if b64:
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
    elif image_b64:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_b64}})
    parts.append({"text": prompt})

    if max_output_tokens_override is not None:
        max_out = int(max_output_tokens_override)
    else:
        # Default 32768 keeps total generation (thinking + visible) closer to historical runtime;
        # set GEMINI_MAX_OUTPUT_TOKENS=65536 for very long transcripts if you hit MAX_TOKENS.
        max_out = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "32768"))
    _jm = os.environ.get("GEMINI_JSON_OUTPUT", "0").strip().lower()
    use_json_output = _jm not in ("0", "false", "no", "off")
    _rel = os.environ.get("GEMINI_SAFETY_RELAXED", "1").strip().lower()
    use_relaxed_safety = _rel not in ("0", "false", "no", "off")

    image_count = len(image_b64_list or ([] if not image_b64 else [image_b64]))
    prompt_chars = len(prompt or "")
    _cp = _compact_verbose_logs()
    t_pass_wall = time.time()
    recv_lat = 0.0
    http_code = 0
    data: Dict[str, Any] = {}
    _tb_disp: Any = None
    max_tries = _gemini_http_max_tries_for_pass(pass_num)
    http_timing: Optional[GeminiPassHttpTiming] = None
    if _pipeline_verbose() and _gemini_pass_timing_enabled_for_pass(pass_num):
        http_timing = GeminiPassHttpTiming(int(pass_num), max_tries)
        http_timing.t_pass_wall = t_pass_wall
    _http_timing_recorded = False

    def _flush_http_timing() -> None:
        nonlocal _http_timing_recorded
        if http_timing and not _http_timing_recorded:
            record_gemini_pass_http_timing(http_timing)
            _http_timing_recorded = True

    def _retry_eta_elapsed_sec(started_at: float) -> float:
        """Return the wasted time so far for the current pass.

        The frontend treats ``eta_extra_sec`` as an absolute extra duration, so report the real
        elapsed wall time for the pass that is being retried instead of a synthetic timeout/backoff
        estimate.
        """
        return round(max(0.0, time.time() - float(started_at)), 1)

    timed_out_attempts = 0

    for attempt_i in range(max_tries):
        if (
            _cp
            and pass_num is not None
            and int(pass_num) == 1
            and attempt_i == 0
        ):
            print("**********PASS1**********", flush=True)
            print("", flush=True)
        if http_timing and attempt_i == 1:
            http_timing.mark_second_attempt_started()
        minimal_thinking = attempt_i > 0
        attempt_timeout = _gemini_attempt_timeout_sec(pass_num, attempt_i, timeout)
        gen_cfg = _gemini_build_generation_config(
            model,
            pass_num,
            temperature,
            max_out,
            use_json_output=use_json_output,
            minimal_thinking=minimal_thinking,
            attempt_i=attempt_i,
        )
        _tc = gen_cfg.get("thinkingConfig") or {}
        _tb_disp_attempt = _tc.get("thinkingBudget")
        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": gen_cfg,
        }
        if use_relaxed_safety:
            payload["safetySettings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
        if not (_cp and pass_num == 3):
            _retry_tag = " minimal_thinking=1" if minimal_thinking else ""
            _pv(
                f"[GEMINI] generateContent POST: model={model} pass={pass_num!r} "
                f"images_in_payload={image_count} prompt_chars={prompt_chars} "
                f"max_tokens={gen_cfg['maxOutputTokens']} thinkingBudget={_tb_disp_attempt!r} "
                f"timeout={attempt_timeout}s attempt={attempt_i + 1}/{max_tries}{_retry_tag}",
            )
        transient_status_retry_i = 0
        attempt_succeeded = False
        timed_out = False
        exhausted_transient_status_code: Optional[int] = None
        exhausted_transient_error_text = ""
        while True:
            try:
                t_attempt_wall = time.time()
                with _gemini_pipeline_http_wait(pass_num, int(attempt_timeout)):
                    t_http = time.time()
                    r = _req.post(url, json=payload, timeout=attempt_timeout)
                    recv_lat = time.time() - t_http
                    http_code = int(r.status_code)
                    if not _cp:
                        _pv(
                            f"[GEMINI] HTTP response received in {recv_lat:.1f}s with status {http_code}",
                        )
                    r.raise_for_status()
                    t_json = time.time()
                    data = r.json()
                    if not _cp:
                        _pv(
                            f"[GEMINI] Response JSON parsed in {time.time()-t_json:.1f}s",
                        )
                if http_timing:
                    http_timing.record_success(attempt_i, time.time() - t_attempt_wall)
                _tb_disp = _tb_disp_attempt
                if transient_status_retry_i > 0:
                    _notify_gemini_retry({"clear_retry_label": True})
                attempt_succeeded = True
                break
            except _req.exceptions.Timeout:
                timed_out = True
                timed_out_attempts += 1
                break
            except _req.exceptions.HTTPError as e:
                status_code = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                err_text = _redact_gemini_api_key(str(e))
                max_status_retries = _gemini_http_extra_retries_on_transient_status()
                if _is_transient_gemini_http_status(status_code) and transient_status_retry_i < max_status_retries:
                    retry_delay = _gemini_transient_status_retry_delay_sec(transient_status_retry_i)
                    eta_extra_sec = _retry_eta_elapsed_sec(t_pass_wall)
                    retry_label = (
                        f"Pass {pass_num} — Gemini {status_code} retrying current attempt…"
                        if pass_num is not None
                        else f"Gemini {status_code} retrying current attempt…"
                    )
                    _notify_gemini_retry(
                        {
                            "label": retry_label,
                            "eta_extra_sec": eta_extra_sec,
                            "reset_phase_started_at": True,
                            "pass_num": int(pass_num) if pass_num is not None else None,
                            "attempt_i": int(attempt_i),
                            "attempt_timeout_sec": int(attempt_timeout),
                            "status_code": status_code,
                        }
                    )
                    print(
                        f"[GEMINI] HTTP {status_code} on pass={pass_num!r} "
                        f"(attempt {attempt_i + 1}/{max_tries}, retry {transient_status_retry_i + 1}/{max_status_retries}) "
                        f"— retrying same attempt in {retry_delay:.1f}s…",
                        flush=True,
                    )
                    transient_status_retry_i += 1
                    time.sleep(retry_delay)
                    continue
                if _is_transient_gemini_http_status(status_code):
                    exhausted_transient_status_code = status_code
                    exhausted_transient_error_text = err_text
                    break
                if http_timing:
                    http_timing.pass_failed = True
                    _flush_http_timing()
                _record_gemini_pass_outcome(
                    pass_num,
                    status="failed",
                    successful_attempt=None,
                    max_tries=max_tries,
                    timed_out_attempts=timed_out_attempts,
                    transient_status_retry_count=transient_status_retry_i,
                    final_http_status=status_code,
                )
                if pass_num is not None and _gemini_pass_summary_enabled():
                    print(f"[gemini] pass {pass_num} stopped: HTTP {status_code}: {err_text}", flush=True)
                raise RuntimeError(err_text)
            except _req.exceptions.RequestException as e:
                err_text = _redact_gemini_api_key(str(e))
                if http_timing:
                    http_timing.pass_failed = True
                    _flush_http_timing()
                _record_gemini_pass_outcome(
                    pass_num,
                    status="failed",
                    successful_attempt=None,
                    max_tries=max_tries,
                    timed_out_attempts=timed_out_attempts,
                    transient_status_retry_count=transient_status_retry_i,
                    final_http_status=http_code or None,
                )
                if pass_num is not None and _gemini_pass_summary_enabled():
                    print(f"[gemini] pass {pass_num} stopped: request error: {err_text}", flush=True)
                raise RuntimeError(err_text)
        if attempt_succeeded:
            break
        if exhausted_transient_status_code is not None:
            status_code = int(exhausted_transient_status_code)
            if attempt_i < max_tries - 1:
                _notify_gemini_retry({"clear_retry_label": True})
                _next_to = _gemini_attempt_timeout_sec(pass_num, attempt_i + 1, timeout)
                print(
                    f"[GEMINI] HTTP {status_code} kept failing on pass={pass_num!r} "
                    f"(attempt {attempt_i + 1}/{max_tries}) — advancing to next attempt "
                    f"with timeout={_next_to}s.",
                    flush=True,
                )
                continue
            if http_timing:
                http_timing.pass_failed = True
                _flush_http_timing()
            _record_gemini_pass_outcome(
                pass_num,
                status="failed",
                successful_attempt=None,
                max_tries=max_tries,
                timed_out_attempts=timed_out_attempts,
                transient_status_retry_count=transient_status_retry_i,
                final_http_status=status_code,
            )
            if pass_num is not None and _gemini_pass_summary_enabled():
                print(
                    f"[gemini] pass {pass_num} stopped: HTTP {status_code} after transient retries: "
                    f"{exhausted_transient_error_text}",
                    flush=True,
                )
            if pass_num is not None and int(pass_num) == 1:
                raise RuntimeError(
                    "SERVERS_OVERLOADED: Gemini Pass 1 stayed unavailable after all attempts and transient retries."
                )
            raise RuntimeError(exhausted_transient_error_text or f"HTTP {status_code}")
        if timed_out:
            if http_timing:
                http_timing.record_timeout(attempt_i, float(attempt_timeout))
            if attempt_i < max_tries - 1:
                _mlow = (model or "").lower()
                _retry_hint = ""
                if "2.5" in _mlow:
                    if pass_num is not None and int(pass_num) == 1:
                        _next_to = _gemini_attempt_timeout_sec(pass_num, attempt_i + 1, timeout)
                        if _pipeline_hurry_up():
                            _next_tb = _HURRY_UP_PASS1_TRY2_THINKING
                        else:
                            _na = attempt_i + 1
                            _next_tb = (
                                _GEMINI_PASS1_RETRY_THINKING_BUDGET
                                if _na == 1
                                else _GEMINI_PASS1_RETRY3_THINKING_BUDGET
                            )
                        _retry_hint = (
                            f" (next attempt: {_next_to}s timeout, "
                            f"thinkingBudget={_next_tb})"
                        )
                    elif pass_num is not None and int(pass_num) == 2:
                        _next_to = _gemini_attempt_timeout_sec(pass_num, attempt_i + 1, timeout)
                        _next_tb = (
                            _HURRY_UP_PASS2_TRY2_THINKING
                            if _pipeline_hurry_up()
                            else _GEMINI_PASS2_RETRY_THINKING_BUDGET
                        )
                        _retry_hint = (
                            f" (next attempt: {_next_to}s timeout, "
                            f"thinkingBudget={_next_tb})"
                        )
                    else:
                        _is_pro = "pro" in _mlow and "flash" not in _mlow
                        _retry_hint = (
                            f" (next attempt: thinkingBudget="
                            f"{_GEMINI_25_PRO_MIN_THINKING_BUDGET} for 2.5 Pro)"
                            if _is_pro
                            else " (next attempt: thinkingBudget=0)"
                        )
                print(
                    f"[GEMINI] HTTP timeout after {attempt_timeout}s "
                    f"(attempt {attempt_i + 1}/{max_tries}, pass={pass_num!r}) — retrying…{_retry_hint}",
                    flush=True,
                )
                continue
            if pass_num is not None and _gemini_pass_summary_enabled():
                _final_to = _gemini_attempt_timeout_sec(pass_num, max_tries - 1, timeout)
                if int(pass_num) in (2, 3):
                    print(
                        f"[gemini] pass {pass_num} stopped: HTTP timeout after {_final_to}s "
                        f"({max_tries} attempt(s); set GEMINI_PASS{pass_num}_TIMEOUT_SEC or "
                        f"GEMINI_REQUEST_TIMEOUT_SEC)",
                        flush=True,
                    )
                else:
                    print(
                        f"[gemini] pass {pass_num} stopped: HTTP timeout after {_final_to}s "
                        f"({max_tries} attempt(s); set GEMINI_PASS{pass_num}_TIMEOUT_SEC, "
                        f"GEMINI_REQUEST_TIMEOUT_SEC, or GEMINI_HTTP_RETRIES)",
                    flush=True,
                )
            if http_timing:
                http_timing.exhausted_on_timeout = True
                _flush_http_timing()
            _record_gemini_pass_outcome(
                pass_num,
                status="timeout_exhausted",
                successful_attempt=None,
                max_tries=max_tries,
                timed_out_attempts=timed_out_attempts,
                transient_status_retry_count=transient_status_retry_i,
                final_http_status=http_code or None,
            )
            raise _req.exceptions.Timeout()

    _flush_http_timing()
    result_meta = {
        "status": "success",
        "successful_attempt": int(attempt_i) + 1,
        "max_tries": max_tries,
        "timed_out_attempts": timed_out_attempts,
        "transient_status_retry_count": transient_status_retry_i,
        "final_http_status": http_code or None,
    }
    _record_gemini_pass_outcome(pass_num, **result_meta)

    if _cp and pass_num is not None:
        wall = time.time() - t_pass_wall
        pn = int(pass_num)
        if pn == 3:
            print(f"[GEMINI]  HTTP response received in {recv_lat:.1f}s", flush=True)
            print(
                f"[pipeline] Pass {pn} — Gemini HTTP response finished in {wall:.1f}s ",
                flush=True,
            )
            print("********Pass3 finished*********", flush=True)
        else:
            print(
                f"[GEMINI] HTTP response received in {recv_lat:.1f}s with status {http_code}",
                flush=True,
            )
            if pn == 1:
                print(
                    f"[pipeline] Pass 1 — Gemini HTTP response finished in {wall:.1f}s",
                    flush=True,
                )
                print("", flush=True)
                print("", flush=True)
            elif pn == 2:
                print(
                    f"[pipeline] Pass {pn} — Gemini HTTP response finished in {wall:.1f}s ",
                    flush=True,
                )
                print("***********Pass 2 finished**************", flush=True)
            else:
                print(
                    f"[pipeline] Pass {pn} — Gemini HTTP response finished in {wall:.1f}s",
                    flush=True,
                )

    pf = data.get("promptFeedback")
    if pf:
        _pv(f"[GEMINI] promptFeedback: {pf}")

    candidates = data.get("candidates") or []
    usage = data.get("usageMetadata") or {}
    thoughts_tokens = usage.get("thoughtsTokenCount")
    total_tokens = usage.get("totalTokenCount")

    if not candidates:
        print(
            "[GEMINI] No candidates in API response — often prompt blocked or request too large. "
            "See result/gemini_debug.txt (=== GEMINI API ===)."
        )
        _gemini_log_pass_summary(
            pass_num=pass_num,
            wall_sec=time.time() - t_pass_wall,
            thinking_budget_cap=_tb_disp,
            thoughts_used=thoughts_tokens,
            total_tokens=total_tokens,
            finish_reason=None,
            out_chars=0,
            note="no_candidates",
        )
        return GeminiApiResult("", data, result_meta)

    c0 = candidates[0]
    fr = c0.get("finishReason")
    if not _compact_verbose_logs():
        _pv(
            f"[GEMINI] Candidate summary: finishReason={fr or 'unknown'} "
            f"totalTokens={total_tokens if total_tokens is not None else 'n/a'} "
            f"thoughtsTokens={thoughts_tokens if thoughts_tokens is not None else 'n/a'}",
        )
    if fr and fr not in ("STOP", "MAX_TOKENS", "FINISH_REASON_STOP", "FINISH_REASON_MAX_TOKENS"):
        _pv(f"[GEMINI] finishReason: {fr}")

    content = c0.get("content") or {}
    resp_parts = content.get("parts") or []
    text = "".join(
        p.get("text", "") for p in resp_parts if isinstance(p, dict)
    )
    sr = c0.get("safetyRatings")
    if sr and not text.strip():
        _pv(f"[GEMINI] safetyRatings: {sr}")
    if not text.strip():
        print(
            "[GEMINI] Empty model text — check finishReason / promptFeedback in "
            "result/gemini_debug.txt (=== GEMINI API ===)."
        )
    _gemini_log_pass_summary(
        pass_num=pass_num,
        wall_sec=time.time() - t_pass_wall,
        thinking_budget_cap=_tb_disp,
        thoughts_used=thoughts_tokens,
        total_tokens=total_tokens,
        finish_reason=fr,
        out_chars=len(text),
        note="",
    )
    return GeminiApiResult(text, data, result_meta)


def _resize_image_for_gemini_vision(img, max_long_edge=16384, min_short_edge=800):
    """Resize *img* so Gemini can read bubble text on very tall combined screenshots.

    A naive ``max(h,w) <= 1600`` scale makes a 1300×20000px stitch into ~104×1600 —
    unreadable.  We cap the *long* edge but ensure the *short* edge stays >= *min_short_edge*
    when physically possible; if still impossible, prefer the long-edge cap.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    h, w = img.shape[:2]
    long_m = max(h, w)
    short_m = min(h, w)
    if long_m <= 0 or short_m <= 0:
        return img

    scale = min(1.0, max_long_edge / long_m)
    if short_m * scale < min_short_edge:
        scale = min_short_edge / short_m
    if long_m * scale > max_long_edge:
        scale = max_long_edge / long_m

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _prepare_stitch_for_gemini(img):
    """Resize **only** the stitched image for Gemini: never upscale; avoid crushing width.

    Returns ``(out_bgr, user_note, error_message)``. *error_message* is set when the band is
    too extreme to resize safely; the caller may try more vertical slices via
    :func:`_prepare_stitch_slices_for_gemini`.

    Environment:
      GEMINI_STITCH_MAX_LONG_EDGE — max(longest side) after resize (default ``16384``).
      GEMINI_STITCH_ABORT_LONG_EDGE — if native longest side exceeds this, abort (default ``120000``).
      GEMINI_STITCH_ABORT_MIN_SHORT — if the shorter side after **downscale-only** would be below
        this (in px), abort as unreadable (default ``280``).
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img, "", ""

    h, w = img.shape[:2]
    long_m = max(h, w)
    short_m = min(h, w)
    if long_m <= 0 or short_m <= 0:
        return img, "", ""

    abort_native = int(os.environ.get("GEMINI_STITCH_ABORT_LONG_EDGE", "120000"))
    if long_m > abort_native:
        return (
            None,
            "",
            (
                f"concatenated image too long ({w}×{h}px, long edge {long_m}px > "
                f"{abort_native}); use fewer screenshots or increase GEMINI_STITCH_ABORT_LONG_EDGE"
            ),
        )

    max_long = int(os.environ.get("GEMINI_STITCH_MAX_LONG_EDGE", "16384"))
    min_short_floor = int(os.environ.get("GEMINI_STITCH_ABORT_MIN_SHORT", "280"))

    # Downscale only (never enlarge tall stitches — that blows API limits).
    scale = min(1.0, float(max_long) / float(long_m))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_short = min(new_w, new_h)

    if new_short < min_short_floor:
        return (
            None,
            "",
            (
                f"concatenated image too long to resize safely: after fitting long edge to "
                f"{max_long}px the short side would be {new_short}px (< {min_short_floor}). "
                f"Native size {w}×{h}px — try automatic stitch splitting (more vertical bands)"
            ),
        )

    if new_w == w and new_h == h:
        return img, "", ""

    interp = cv2.INTER_AREA
    out = cv2.resize(img, (new_w, new_h), interpolation=interp)
    note = (
        f"Stitched image was **uniformly downscaled** for the API from {w}×{h}px to "
        f"{new_w}×{new_h}px (long edge cap {max_long}px). Layout and y_norm hints still match "
        f"the original stack."
    )
    return out, note, ""


def _split_combined_vertical_equal(img, n: int):
    """Split *img* into *n* contiguous horizontal strips (equal height bands, remainder to lower indices)."""
    if img is None or getattr(img, "size", 0) == 0 or n < 1:
        return []
    if n == 1:
        return [img]
    h = int(img.shape[0])
    out = []
    y0 = 0
    for i in range(n):
        y1 = (i + 1) * h // n
        if y1 > y0:
            out.append(img[y0:y1, :].copy())
        y0 = y1
    return out if out else [img]


def _prepare_stitch_slices_for_gemini(combined_img):
    """Prepare the stitched canvas for Gemini as one or more vertical slices.

    Tries **one** full-height image first (with downscale-only policy). If that fails, splits the
    same canvas into **2, 3, …** equal-height bands until each band passes
    :func:`_prepare_stitch_for_gemini`, up to ``GEMINI_STITCH_MAX_SLICES`` (default **5**).

    Returns ``(prepared_bgr_list, human_note, n_slices, error_str)``.
    """
    if combined_img is None or getattr(combined_img, "size", 0) == 0:
        return [], "", 0, ""

    max_slices = int(os.environ.get("GEMINI_STITCH_MAX_SLICES", "5"))
    max_slices = max(1, min(20, max_slices))  # sanity cap

    last_err = ""
    for n in range(1, max_slices + 1):
        bands = _split_combined_vertical_equal(combined_img, n)
        if len(bands) != n:
            last_err = f"stitch split internal error (expected {n} bands)"
            continue
        prepared = []
        slice_notes = []
        ok = True
        for bi, band in enumerate(bands):
            rc, note, err = _prepare_stitch_for_gemini(band)
            if err:
                ok = False
                last_err = err
                break
            prepared.append(rc)
            if note:
                slice_notes.append(f"Band {bi + 1}/{n}: {note}")
        if ok and prepared:
            if n == 1:
                meta = slice_notes[0] if slice_notes else ""
            else:
                h0 = combined_img.shape[0]
                meta = (
                    f"Full stitch ({combined_img.shape[1]}×{h0}px) was sent as **{n} equal-height** "
                    f"vertical bands (contiguous, top→bottom). **Band 1** includes the status bar; "
                    f"later bands are lower in the same conversation.\n"
                )
                if slice_notes:
                    meta += "\n".join(slice_notes)
            _pv(f"[GEMINI] Stitch prepared as {n} vertical band(s)")
            return prepared, meta.strip(), n, ""

    return (
        [],
        "",
        0,
        last_err
        or (
            f"concatenated image could not be prepared for Gemini even after splitting into "
            f"{max_slices} equal-height bands"
        ),
    )


prepare_stitch_bands_for_gemini = _prepare_stitch_slices_for_gemini


def _jpeg_b64_from_bgr(img, quality=90):
    import base64 as _b64

    if img is None or getattr(img, "size", 0) == 0:
        return ""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return _b64.b64encode(buf.tobytes()).decode("utf-8")


def _canonicalize_gemini_role(role_raw: str) -> str:
    """Map model-specific labels to exactly one of: contact | user | system.

    **contact** = other person / incoming / LEFT grey bubble.
    **user** = phone owner / outgoing / RIGHT colored bubble.
    Do NOT treat pronouns like \"you\" / \"them\" as roles — those were mis-mapped before.
    """
    r = (role_raw or "").strip().lower()
    if r in (
        "receiver",
        "incoming",
        "other_party",
        "left",
        "person_a",
        "other",
        "contact",
        "person1",
    ):
        return "contact"
    if r in ("sender", "outgoing", "self", "right", "person_b", "user", "person2"):
        return "user"
    if r in ("system", "timestamp", "center", "ui", "meta", "notice"):
        return "system"
    return r


def _looks_like_real_system_row(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    tl = t.lower()

    blocked_fragments = (
        "gift.truemoney.com",
        "campaign/?",
        "https://",
        "http://",
        "การหลอกลวง",
        "ขอให้คุณ",
        "โอนเงินหรือบอกรหัส",
        "ใช้งานเมื่อ",
    )
    if any(fragment in t or fragment in tl for fragment in blocked_fragments):
        return False

    if re.search(r"\b\d{1,2}[:.]\d{2}\b", t):
        return True

    allowed_fragments = (
        "โทรอีกครั้ง",
        "โทรกลับ",
        "การโทร",
        "ด้วยเสียง",
        "วิดีโอคอล",
        "สายที่ไม่ได้รับ",
        "ไม่รับสาย",
        "missed call",
        "audio call",
        "video call",
        "called",
        "call again",
    )
    if any(fragment in t or fragment in tl for fragment in allowed_fragments):
        return True

    if re.search(r"(จ\.|อ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.)\s*เวลา\s*\d{1,2}[:.]\d{2}", t):
        return True

    return False


def _filter_pass1_messages(messages):
    out = []
    for m in list(messages or []):
        role = _canonicalize_gemini_role(m.get("role") or "")
        text_src = (m.get("text_src") or m.get("text_en") or "").strip()
        if not text_src:
            continue
        if role == "system" and not _looks_like_real_system_row(text_src):
            continue
        mm = dict(m)
        mm["role"] = role
        out.append(mm)
    return out


def _write_gemini_prompt_file(filename: str, label: str, prompt: str):
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{label}\n")
            f.write("=" * 60 + "\n\n")
            f.write(prompt)
        _pv(f"[GEMINI] Prompt file → {path}")
        if _compact_verbose_logs() and filename == "gemini_prompt_pass1.txt":
            print("**************", flush=True)
    except Exception as exc:
        _pv(f"[GEMINI] Could not write prompt file {filename}: {exc}")


def _extract_json_object(text: str) -> Optional[str]:
    """If the model adds prose around JSON, pull the first top-level ``{...}`` object."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_gemini_full_vision_json(raw: str):
    """Parse JSON from Gemini; return (contact_name, messages_list, ambiguity_ledger).

    *ambiguity_ledger* is a list of dicts from pass 1, e.g.
    ``[{"id": "REF1", "message_index": 2, "note": "..."}]`` — may be empty.
    """
    if not raw or not raw.strip():
        return "", [], []
    text = raw.strip().lstrip("\ufeff")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        blob = _extract_json_object(text)
        if not blob:
            raise
        data = json.loads(blob)
    name = (data.get("contact_name") or data.get("contact") or "").strip()
    msgs = data.get("messages") or data.get("conversation") or []
    ledger = data.get("ambiguity_ledger") or data.get("ambiguities") or []
    if not isinstance(msgs, list):
        return name, [], []
    if not isinstance(ledger, list):
        ledger = []
    normalized = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        raw_role = (m.get("role") or m.get("speaker") or "").strip().lower()
        role = _canonicalize_gemini_role(raw_role)
        if role not in ("contact", "user", "system"):
            _pv(
                f"[GEMINI] Unknown message role {raw_role!r} → treating as 'system' "
                f"(check gemini_debug.txt)"
            )
            role = "system"
        crop_text_src = (m.get("crop_text_src") or "").strip()
        text_src = (m.get("text_src") or m.get("source_text") or crop_text_src or "").strip()
        text_en_debug = (
            m.get("text_en_debug")
            or m.get("debug_translation")
            or m.get("text_en")
            or m.get("english")
            or ""
        ).strip()
        te = (
            m.get("text_en")
            or text_src
            or m.get("text")
            or m.get("english")
            or ""
        ).strip()
        if not (te or text_src or text_en_debug):
            continue
        ev_ts = m.get("event_timestamp")
        if ev_ts is None:
            ev_ts = m.get("timestamp")
        event_timestamp = (str(ev_ts).strip() if ev_ts is not None else "") or ""
        normalized.append(
            {
                "role": role,
                "text_en": te or text_src or text_en_debug,
                "text_src": text_src,
                "crop_text_src": crop_text_src,
                "text_en_debug": text_en_debug,
                "legibility": (m.get("legibility") or "").strip(),
                "note": (m.get("note") or "").strip(),
                "event_timestamp": event_timestamp,
            }
        )
    return name, normalized, ledger


def _pass1_has_suspicious_repetition(messages) -> bool:
    """Detect degenerate Pass 1 outputs with repeated non-link content."""
    msgs = list(messages or [])
    if len(msgs) < 8:
        return False

    counts = {}
    prev = None
    streak = 0
    max_streak = 0

    for m in msgs:
        t = (m.get("text_src") or m.get("text_en") or "").strip()
        if not t or _looks_like_link_text(t) or re.fullmatch(r"[\d\s:./-]+", t):
            prev = None
            streak = 0
            continue
        counts[t] = counts.get(t, 0) + 1
        if t == prev:
            streak += 1
        else:
            prev = t
            streak = 1
        max_streak = max(max_streak, streak)

    repeated_max = max(counts.values(), default=0)
    return max_streak >= 4 or repeated_max >= max(5, len(msgs) // 3)


_REF_PLACEHOLDER_RE = re.compile(r"⟦REF\d+⟧")


def _conversation_has_ref_placeholders(messages):
    return any(
        _REF_PLACEHOLDER_RE.search((m.get("text_en") or ""))
        for m in (messages or [])
    )


def _gemini_resolve_referent_placeholders(
    contact_name: str,
    messages: list,
    ambiguity_ledger: list,
    timeout: int,
):
    """Pass 2: text-only — translate the source transcript into English."""
    if not messages:
        return messages, contact_name

    payload = {
        "contact_name": contact_name,
        "ambiguity_ledger": ambiguity_ledger or [],
        "messages": [
            {
                "message_index": i,
                "role": m["role"],
                "text_src": m["text_en"],
            }
            for i, m in enumerate(messages)
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    expected_n = len(messages)

    prompt = f"""You are **PASS 2 of 2**: translate an already-transcribed mobile chat into English.

## Input JSON
- `messages[i].message_index == i` is the canonical row index from the stitched chat.
- `text_src` is the current source-language transcription for that exact row.
- `role` is already fixed from bubble side and must never change.

## Your task
- Translate each `text_src` into natural, faithful English as `text_en`.
- Keep the same meaning, order, speaker, timestamps, money, places, and system notices.
- Read the whole thread top→bottom so short replies make conversational sense.
- If a source line is ambiguous, choose the most likely faithful English from the local chat context.
- Do **not** invent extra content that is not supported by the source transcript.

## Rules
- Keep **message count** and **message_index** alignment identical.
- For each output row, copy `role` exactly from the input row with the same `message_index`.
- Translate **only** the text content. Do not merge, split, reorder, or delete messages.
- `text_en` must be English. `contact_name` may remain as-is.

## Output
Return **only** valid JSON (no markdown):
{{"contact_name":"<string>","messages":[{{"role":"contact|user|system","text_en":"<english translation>"}}]}}

INPUT JSON:
{payload_json}"""

    _write_gemini_prompt_file(
        "gemini_prompt_pass2.txt",
        "Pass 2 (final translation)",
        prompt,
    )

    try:
        gres = _gemini_generate(prompt, timeout=timeout, pass_num=2)
        raw = gres.text if gres else ""
    except Exception as e:
        print(f"[GEMINI] Pass 2 translation failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}")
        return messages, contact_name

    if not raw or not raw.strip():
        print("[GEMINI] Pass 2 empty response — keeping source transcript")
        _append_gemini_debug_pass2(prompt, raw or "")
        return messages, contact_name

    try:
        name2, msgs2, _ledger = _parse_gemini_full_vision_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 2 JSON parse failed: {e} — keeping source transcript")
        _append_gemini_debug_pass2(prompt, raw)
        return messages, contact_name

    if not msgs2 or len(msgs2) != len(messages):
        _pv(
            f"[GEMINI] Pass 2 message count mismatch ({len(msgs2)} vs {len(messages)}) — keeping source transcript"
        )
        _append_gemini_debug_pass2(prompt, raw)
        return messages, contact_name

    # If pass 2 changed any role or reordered semantics, do not merge — wrong line ↔ side pairing.
    for i, m in enumerate(msgs2):
        r2 = _canonicalize_gemini_role(m.get("role") or "")
        if r2 not in ("contact", "user", "system"):
            r2 = "system"
        if r2 != messages[i]["role"]:
            _pv(
                f"[GEMINI] Pass 2 role mismatch at index {i}: pass1={messages[i]['role']!r} "
                f"pass2={m.get('role')!r} — discarding entire pass 2, keeping pass 1"
            )
            _append_gemini_debug_pass2(prompt, raw + "\n\n[MERGE ABORTED: role drift]\n")
            return messages, contact_name

    merged = []
    for i, m in enumerate(msgs2):
        merged.append(
            {"role": messages[i]["role"], "text_en": m.get("text_en", messages[i]["text_en"])}
        )
    final_name = (name2 or contact_name or "").strip() or contact_name
    _append_gemini_debug_pass2(prompt, raw)
    if _conversation_has_ref_placeholders(merged):
        _pv("[GEMINI] Pass 2 left some ⟦REF⟧ tokens — check gemini_debug.txt")
    else:
        _pv("[GEMINI] Pass 2 applied (final English translation).")
    return merged, final_name


def _append_gemini_debug_ocr_refine(prompt: str, response: str):
    path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("PASS 1b — OCR-GUIDED TRANSCRIPT REWRITE (text-only)\n")
            f.write("=" * 60 + "\n\n")
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(response or "(empty)\n")
    except Exception as exc:
        _pv(f"[GEMINI] Could not append OCR-refine debug: {exc}")


def _append_gemini_debug_crop_refine(prompt: str, response: str, batch_label: str):
    path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"PASS 2 — CROP REFINE ({batch_label})\n")
            f.write("=" * 60 + "\n\n")
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(response or "(empty)\n")
    except Exception as exc:
        _pv(f"[GEMINI] Could not append crop-refine debug: {exc}")


def _gemini_crop_refine_pass(
    contact_name: str,
    messages: list,
    crop_images: list,
    timeout: int,
):
    """Pass 2: transcribe ordered message crops in the original language."""
    if not messages or not crop_images:
        return [], contact_name, {
            "enabled": bool(crop_images),
            "skipped": True,
            "reason": "missing_messages_or_crops",
            "input_messages": len(messages or []),
            "input_crops": len(crop_images or []),
        }

    batch_size = max(1, int(os.environ.get("GEMINI_CROP_REFINE_BATCH_SIZE", "5")))
    crop_msgs = [dict(m) for m in messages]
    batch_debug = []
    usable_n = min(len(messages), len(crop_images))

    for start in range(0, usable_n, batch_size):
        end = min(usable_n, start + batch_size)
        batch_msgs = messages[start:end]
        batch_imgs = crop_images[start:end]
        batch_payload = {
            "contact_name": contact_name,
            "messages": [
                {
                    "message_index": start + i,
                    "role": m["role"],
                    "pass1_text_src": m.get("text_src") or m.get("text_en") or "",
                }
                for i, m in enumerate(batch_msgs)
            ],
        }
        payload_json = json.dumps(batch_payload, ensure_ascii=False, indent=2)
        prompt = f"""You are PASS 2 of 3.

You receive a small ordered batch of cropped Facebook/Messenger message images.
Image 1 corresponds to `message_index {start}`, image 2 to `message_index {start + 1}`, and so on.

Task for each message:
1. Read the cropped image by itself.
2. Transcribe the visible message in the original language as `crop_text_src`.
3. Use the provided `pass1_text_src` only as a fallback anchor when the crop is noisy or incomplete.
4. Keep the same ordering and roles.

Rules:
- Keep exactly {end - start} messages.
- Keep the same `message_index` values.
- Keep `role` unchanged.
- Do not merge, split, reorder, add, or remove messages.
- Do not translate to English in this pass.

Input JSON:
{payload_json}

Output JSON only:
{{"messages":[{{"message_index":<int>,"role":"contact|user|system","crop_text_src":"<source-language crop reading>","legibility":"clear|partial|noise","note":"<short note>"}}]}}"""

        batch_label = f"indices {start}-{end - 1}"

        img_b64_list = []
        for img in batch_imgs:
            if img is None or getattr(img, "size", 0) == 0:
                continue
            b64 = _jpeg_b64_from_bgr(img, quality=90)
            if b64:
                img_b64_list.append(b64)

        try:
            gres = _gemini_generate(
                prompt, image_b64_list=img_b64_list, timeout=timeout, pass_num=2
            )
            raw = (gres.text if gres else "") or ""
        except Exception as e:
            _append_gemini_debug_crop_refine(prompt, f"ERROR: {e}", batch_label)
            batch_debug.append(
                {"start": start, "end": end - 1, "ok": False, "reason": f"request_error: {e}"}
            )
            continue

        if not raw.strip():
            _append_gemini_debug_crop_refine(prompt, raw, batch_label)
            batch_debug.append(
                {"start": start, "end": end - 1, "ok": False, "reason": "empty_response"}
            )
            continue

        try:
            _name2, msgs2, _ledger2 = _parse_gemini_full_vision_json(raw)
        except json.JSONDecodeError as e:
            _append_gemini_debug_crop_refine(prompt, raw, batch_label)
            batch_debug.append(
                {"start": start, "end": end - 1, "ok": False, "reason": f"json_error: {e}"}
            )
            continue

        if len(msgs2) != len(batch_msgs):
            _append_gemini_debug_crop_refine(prompt, raw, batch_label)
            batch_debug.append(
                {
                    "start": start,
                    "end": end - 1,
                    "ok": False,
                    "reason": f"count_mismatch:{len(msgs2)}",
                }
            )
            continue

        role_drift = False
        for i, m in enumerate(msgs2):
            if _canonicalize_gemini_role(m.get("role") or "") != batch_msgs[i]["role"]:
                role_drift = True
                break
        if role_drift:
            _append_gemini_debug_crop_refine(prompt, raw + "\n[ABORT: role drift]\n", batch_label)
            batch_debug.append(
                {"start": start, "end": end - 1, "ok": False, "reason": "role_drift"}
            )
            continue

        for i, m in enumerate(msgs2):
            crop_msgs[start + i] = {
                "role": batch_msgs[i]["role"],
                "text_en": (batch_msgs[i].get("text_src") or batch_msgs[i].get("text_en") or "").strip(),
                "text_src": (batch_msgs[i].get("text_src") or batch_msgs[i].get("text_en") or "").strip(),
                "crop_text_src": (m.get("crop_text_src") or m.get("text_src") or "").strip(),
                "legibility": (m.get("legibility") or "").strip(),
                "note": (m.get("note") or "").strip(),
            }
        _append_gemini_debug_crop_refine(prompt, raw, batch_label)
        batch_debug.append({"start": start, "end": end - 1, "ok": True, "reason": ""})

    return crop_msgs, contact_name, {
        "enabled": True,
        "skipped": False,
        "batch_size": batch_size,
        "input_messages": len(messages),
        "input_crops": len(crop_images),
        "usable_pairs": usable_n,
        "output_messages": len(crop_msgs),
        "batches": batch_debug,
    }


def _gemini_ocr_context_refine_pass(
    contact_name: str,
    pass1_messages: list,
    crop_messages: list,
    ambiguity_ledger: list,
    ocr_pass2_text: str,
    timeout: int,
    context_images=None,
    prompt_label: str = "Pass 3 (OCR-guided final translation)",
):
    """Pass 3: fuse Pass 1 + Pass 2 + OCR hints and return final English.

    Returns ``(messages, contact_name, ambiguity_ledger)`` — on failure returns inputs unchanged.
    """
    if not pass1_messages:
        return pass1_messages, contact_name, ambiguity_ledger or []

    payload = {
        "contact_name": contact_name,
        "ambiguity_ledger": ambiguity_ledger or [],
        "messages": [
            {
                "message_index": i,
                "role": m.get("role"),
                "pass1_text_src": (m.get("text_src") or m.get("text_en") or "").strip(),
                "pass2_crop_text_src": (
                    crop_messages[i].get("crop_text_src")
                    if i < len(crop_messages or [])
                    else ""
                ) or "",
                "pass2_legibility": (
                    crop_messages[i].get("legibility")
                    if i < len(crop_messages or [])
                    else ""
                ) or "",
                "pass2_note": (
                    crop_messages[i].get("note")
                    if i < len(crop_messages or [])
                    else ""
                ) or "",
            }
            for i, m in enumerate(pass1_messages)
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    expected_n = len(pass1_messages)
    max_c = int(os.environ.get("GEMINI_OCR_HINTS_MAX_CHARS", "120000"))
    ot = (ocr_pass2_text or "").strip()
    if len(ot) > max_c:
        ot = ot[:max_c] + "\n\n[... OCR block truncated ...]"

    ocr_section = (
        "## High-confidence OCR on the cleaned page images, by **message_index**\n"
        "Each block lists only high-confidence Vision words assigned to that same message row. "
        "These are **source-language fragments** and **soft hints** only.\n\n"
        f"{ot}\n\n"
        if ot
        else "## OCR hints\nNo OCR hints were available for this run. Fuse from Pass 1 and Pass 2 alone.\n\n"
    )

    prompt = f"""You are **PASS 3 of 3**: finalize a mobile chat conversation per message.

## Shared index system (critical)
- The input transcript uses **`message_index` = array index = top-to-bottom conversation order**.
- The Pass 2 crop transcript and OCR hint blocks use that **same** `message_index`.
- Therefore all inputs for `message_index k` refer to the same message row.
- The input contains exactly **{expected_n}** messages, so the output must also contain exactly **{expected_n}** messages.

## Speaker policy (non-negotiable)
- `role` was fixed in Pass 1 from bubble side in the stitched image.
- You **must not** change, swap, or “correct” any `role`.

## Input JSON
{payload_json}

{ocr_section}## Your task
- Process the conversation **per message object**.
- For each `message_index`, reconstruct the best final source reading by considering:
  - `pass1_text_src`
  - `pass2_crop_text_src`
  - `pass2_legibility`
  - OCR hints for the same `message_index`
- Treat OCR as soft context only.
- If `pass2_legibility` is `noise` or `partial`, prefer Pass 1 unless the crop still clearly adds a small detail.
- If the crop clearly reads the message better, prefer the crop.
- Then translate the chosen final reading to English as `text_en`.
- Read the whole thread so short replies still make conversational sense.

## Rules (critical)
- Return **only** valid JSON (no markdown fences): `{{"contact_name":"...","messages":[{{"message_index":0,"role":"...","text_src":"...","text_en":"..."}}]}}`
- **Same** `messages.length` as input. **Same** `message_index` mapping. **Same** `role` per index.
- Use OCR as **general context**: these words were **probably** in that message. Recreate the message with that added information in mind, then translate it.
- Do **not** paste OCR fragments blindly if they make the line worse.
- Ignore obvious artifacts, call banners, broken URLs, duplicated UI labels, and visual debris unless they are clearly the actual message content.
- `text_en` must be English.
- `text_src` must be the chosen final source-language reading for that message.
- **Do not** add, drop, merge, or split messages.

## Output
{{"contact_name":"<string>","messages":[{{"message_index":0,"role":"contact|user|system","text_src":"<final source reading>","text_en":"<final English line>"}}]}}"""

    _write_gemini_prompt_file("gemini_prompt_pass3.txt", prompt_label, prompt)

    img_b64_list = []
    for img in list(context_images or []):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        b64 = _jpeg_b64_from_bgr(img, quality=90)
        if b64:
            img_b64_list.append(b64)

    try:
        gres = _gemini_generate(
            prompt, image_b64_list=img_b64_list, timeout=timeout, pass_num=3
        )
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 3 failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}")
        return pass1_messages, contact_name, ambiguity_ledger or []

    if not raw.strip():
        print("[GEMINI] Pass 3 empty — keeping source transcript")
        _append_gemini_debug_pass2(prompt, raw)
        return pass1_messages, contact_name, ambiguity_ledger or []

    try:
        name2, msgs2, ledger2 = _parse_gemini_full_vision_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 3 JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw)
        return pass1_messages, contact_name, ambiguity_ledger or []

    if not msgs2 or len(msgs2) != len(pass1_messages):
        _pv(
            f"[GEMINI] Pass 3 message count mismatch ({len(msgs2)} vs {len(pass1_messages)}) — "
            "keeping source transcript"
        )
        _append_gemini_debug_pass2(prompt, raw)
        return pass1_messages, contact_name, ambiguity_ledger or []

    role_drift_indices = []
    for i, m in enumerate(msgs2):
        r2 = _canonicalize_gemini_role(m.get("role") or "")
        if r2 not in ("contact", "user", "system"):
            r2 = "system"
        if r2 != pass1_messages[i]["role"]:
            role_drift_indices.append(i)

    if role_drift_indices:
        _pv(
            f"[GEMINI] Pass 3 role drift at indices {role_drift_indices[:8]} "
            f"(keeping Pass 1 roles, using Pass 3 text)"
        )
        _append_gemini_debug_pass2(
            prompt,
            raw + f"\n[ROLE DRIFT IGNORED: {role_drift_indices}]\n",
        )

    merged = [
        {
            "role": pass1_messages[i]["role"],
            "text_src": (
                msgs2[i].get("text_src")
                or pass1_messages[i].get("text_src")
                or pass1_messages[i].get("text_en")
                or ""
            ).strip(),
            "text_en": "",
        }
        for i in range(len(pass1_messages))
    ]
    for i, mm in enumerate(merged):
        src = (mm.get("text_src") or "").strip()
        en_raw = (
            msgs2[i].get("text_en")
            or pass1_messages[i].get("text_en")
            or ""
        ).strip()
        mm["text_en"] = _prefer_english_surface(en_raw, translate_to_en(src) or src)
    final_name = (name2 or contact_name or "").strip() or contact_name
    ledger_out = ledger2 if isinstance(ledger2, list) else (ambiguity_ledger or [])
    if not role_drift_indices:
        _append_gemini_debug_pass2(prompt, raw)
    return merged, final_name, ledger_out


def _pass2_norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip()).casefold()


def _rect_xy_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return dx, dy


class _UnionFind:
    def __init__(self, n: int):
        self._p = list(range(n))

    def find(self, i: int) -> int:
        p = self._p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i: int, j: int) -> None:
        pi, pj = self.find(i), self.find(j)
        if pi != pj:
            self._p[pj] = pi


def _pass2_word_center_dist(a: dict, b: dict) -> float:
    dx = float(a["cx"]) - float(b["cx"])
    dy = float(a["cy"]) - float(b["cy"])
    return math.hypot(dx, dy)


def _pass2_cluster_max_center_pair_dist(words: List[dict]) -> float:
    n = len(words)
    if n < 2:
        return 0.0
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            m = max(m, _pass2_word_center_dist(words[i], words[j]))
    return m


def _split_pass2_ocr_cluster_by_diameter(
    word_boxes: List[dict],
    max_diam_px: float,
) -> List[List[dict]]:
    """Bisect clusters whose bbox-center pairwise L2 exceeds *max_diam_px* (breaks union-find chains)."""
    n = len(word_boxes)
    if n <= 1:
        return [word_boxes]
    if _pass2_cluster_max_center_pair_dist(word_boxes) <= max_diam_px:
        return [word_boxes]

    best_dd = -1.0
    ia, ib = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            dd = _pass2_word_center_dist(word_boxes[i], word_boxes[j])
            if dd > best_dd:
                best_dd, ia, ib = dd, i, j

    sa, sb = word_boxes[ia], word_boxes[ib]
    ga, gb = [], []
    for w in word_boxes:
        da = _pass2_word_center_dist(w, sa)
        db = _pass2_word_center_dist(w, sb)
        (ga if da <= db else gb).append(w)

    if not ga or not gb or len(ga) == n or len(gb) == n:
        ordered = sorted(word_boxes, key=lambda w: (float(w["cy"]), float(w["cx"])))
        mid = max(1, n // 2)
        ga, gb = ordered[:mid], ordered[mid:]

    out: List[List[dict]] = []
    for g in (ga, gb):
        out.extend(_split_pass2_ocr_cluster_by_diameter(g, max_diam_px))
    return out


def _cluster_pass2_ocr_words(
    word_boxes: List[dict],
    iw: int,
    ih: int,
) -> List[List[dict]]:
    """Group OCR words into spatial clusters (message-bubble candidates).

    Each cluster is an unordered list of OCR word dicts (``text``, bbox, ``cx``/``cy``, …); the
    only purpose of a group is which detection belongs together. Any joined string built later is
    for Pass 1 resemblance only, not a semantic transcript.

    **Step 1 — proximity graph (union–find):** For each pair of word bboxes, compute axis-aligned
    *separating gaps* ``dx, dy`` (0 when projections overlap on that axis). Merge into the same
    component iff **both** ``dx <= max_gx`` and ``dy <= max_gy`` (see env fractions × page size + ε).
    Every input word therefore lies in **exactly one** raw component (singletons allowed).

    **Step 2 — optional L² diameter cap:** Union–find is *single-linkage*, so components can grow
    elongated chains. If ``PASS2_OCR_CLUSTER_MAX_DIAM_FRAC`` > 0, split any component whose
    maximum **center-to-center** Euclidean distance exceeds
    ``frac * hypot(iw, ih)`` until each piece satisfies the cap.

    **Single image only:** Callers pass words from **one** page image; clusters never span images.

    Callers should pass **every** OCR word that should participate in spatial grouping; confidence
    filtering for Pass 2 hints happens **after** clustering in :func:`build_pass2_per_message_ocr_hints`.
    """
    n = len(word_boxes)
    if n == 0:
        return []
    # Merge when bbox gaps ≤ these distances: fraction × page W/H (resolution-native). Defaults are
    # relaxed so long URLs, wrapped lines, and multi-line bubbles merge; tune down if adjacent
    # bubbles chain together. ε = small fraction of min(w,h) for tiny crops only.
    gx_frac = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_X_FRAC", "0.068"))
    gy_frac = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_Y_FRAC", "0.078"))
    _eps = 0.003 * min(float(iw), float(ih))
    max_gx = max(_eps, gx_frac * float(iw))
    max_gy = max(_eps, gy_frac * float(ih))

    uf = _UnionFind(n)
    rects = [
        (
            float(w["x1"]),
            float(w["y1"]),
            float(w["x2"]),
            float(w["y2"]),
        )
        for w in word_boxes
    ]
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = _rect_xy_gap(rects[i], rects[j])
            if dx <= max_gx and dy <= max_gy:
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    clusters: List[List[dict]] = []
    for _root, idxs in groups.items():
        members = [word_boxes[k] for k in idxs]
        members.sort(key=lambda w: (float(w["cy"]), float(w["cx"])))
        clusters.append(members)

    clusters.sort(key=lambda cl: (min(float(w["cy"]) for w in cl), min(float(w["cx"]) for w in cl)))

    # Caps how far word centers can spread in one cluster (breaks chains across adjacent bubbles).
    max_frac = float(os.environ.get("PASS2_OCR_CLUSTER_MAX_DIAM_FRAC", "0.26"))
    if max_frac > 0:
        diag = math.hypot(float(iw), float(ih))
        max_diam_px = max(0.004 * diag, max_frac * diag)
        split_out: List[List[dict]] = []
        for cl in clusters:
            split_out.extend(_split_pass2_ocr_cluster_by_diameter(cl, max_diam_px))
        clusters = split_out
        clusters.sort(
            key=lambda c: (min(float(w["cy"]) for w in c), min(float(w["cx"]) for w in c))
        )
    return clusters


def _pass2_ocr_overlap_resolve_enabled() -> bool:
    v = os.environ.get("PASS2_OCR_CLUSTER_RESOLVE_OVERLAP", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _pass2_union_bbox_float(words: List[dict]) -> Tuple[float, float, float, float]:
    return (
        min(float(w["x1"]) for w in words),
        min(float(w["y1"]) for w in words),
        max(float(w["x2"]) for w in words),
        max(float(w["y2"]) for w in words),
    )


def _pass2_xyxy_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Area of axis-aligned intersection; 0 if disjoint or degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def _pass2_bbox_iou(
    ui: Tuple[float, float, float, float],
    uj: Tuple[float, float, float, float],
) -> float:
    inter = _pass2_xyxy_intersection_area(ui, uj)
    if inter <= 0:
        return 0.0
    ai = max(0.0, ui[2] - ui[0]) * max(0.0, ui[3] - ui[1])
    aj = max(0.0, uj[2] - uj[0]) * max(0.0, uj[3] - uj[1])
    u = ai + aj - inter
    return inter / u if u > 0 else 0.0


def _pass2_xyxy_intersection_positive(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    min_side: float = 2.0,
) -> Optional[Tuple[float, float, float, float]]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return (x1, y1, x2, y2)


def _pass2_word_bbox_intersects_rect(w: dict, rect: Tuple[float, float, float, float]) -> bool:
    wx1, wy1, wx2, wy2 = float(w["x1"]), float(w["y1"]), float(w["x2"]), float(w["y2"])
    rx1, ry1, rx2, ry2 = rect
    ix1 = max(wx1, rx1)
    iy1 = max(wy1, ry1)
    ix2 = min(wx2, rx2)
    iy2 = min(wy2, ry2)
    return ix2 > ix1 + 1e-6 and iy2 > iy1 + 1e-6


def _pass2_word_group_centroid(words: List[dict]) -> Optional[Tuple[float, float]]:
    if not words:
        return None
    sx = sum(float(w["cx"]) for w in words)
    sy = sum(float(w["cy"]) for w in words)
    n = float(len(words))
    return (sx / n, sy / n)


def _pass2_l2_centers(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pass2_dist_remainder_to_C(
    remainder: List[dict],
    original: List[dict],
    cent_C: Tuple[float, float],
) -> float:
    """L2 from remainder centroid to C; if remainder is empty, use *original* cluster centroid."""
    if remainder:
        c = _pass2_word_group_centroid(remainder)
        assert c is not None
        return _pass2_l2_centers(c, cent_C)
    c0 = _pass2_word_group_centroid(original)
    if c0 is None:
        return float("inf")
    return _pass2_l2_centers(c0, cent_C)


def _pass2_overlap_assign_mode() -> str:
    """``PASS2_OCR_CLUSTER_OVERLAP_ASSIGN``: ``nearest`` (default) or ``centroid`` (legacy)."""
    v = os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_ASSIGN", "nearest").strip().lower()
    if v in ("centroid", "global", "legacy"):
        return "centroid"
    return "nearest"


def _pass2_min_dist_word_to_any(w: dict, others: List[dict]) -> float:
    wx, wy = float(w["cx"]), float(w["cy"])
    return min(
        _pass2_l2_centers((wx, wy), (float(r["cx"]), float(r["cy"]))) for r in others
    )


def _pass2_union_width(words: List[dict]) -> float:
    if not words:
        return 0.0
    u = _pass2_union_bbox_float(words)
    return max(0.0, u[2] - u[0])


def _pass2_mean_cx(words: List[dict]) -> float:
    if not words:
        return 0.0
    return sum(float(w["cx"]) for w in words) / float(len(words))


def _pass2_remainder_layout_kind(words: List[dict], iw: int) -> str:
    """Rough chat layout: *column* (sidebar / stacked), *row* (bubble / horizontal run), *neutral*.

    Uses union bbox aspect plus **fraction of page width** so multi-line bubbles (tall unions that
    still span much of the screen) count as *row*, not *neutral*.
    """
    if not words:
        return "neutral"
    u = _pass2_union_bbox_float(words)
    w = max(0.0, u[2] - u[0])
    h = max(0.0, u[3] - u[1])
    if w < 1.0 or h < 1.0:
        return "neutral"
    iw = max(int(iw), 1)
    col_thr = float(os.environ.get("PASS2_OCR_CLUSTER_COLUMN_ASPECT", "1.75"))
    row_thr = float(os.environ.get("PASS2_OCR_CLUSTER_ROW_ASPECT", "1.15"))
    col_max_w = float(os.environ.get("PASS2_OCR_CLUSTER_COLUMN_MAX_WIDTH_FRAC", "0.34")) * float(iw)
    row_min_w = float(os.environ.get("PASS2_OCR_CLUSTER_ROW_MIN_WIDTH_FRAC", "0.20")) * float(iw)
    if h >= col_thr * w:
        return "column"
    if w >= row_thr * h:
        return "row"
    if w >= row_min_w and len(words) >= 2:
        return "row"
    if w <= col_max_w and h >= 1.12 * w:
        return "column"
    return "neutral"


def _pass2_overlap_assign_score(
    c_words: List[dict],
    remainder: List[dict],
    original: List[dict],
    cent_c: Tuple[float, float],
) -> float:
    """Lower is better: how strongly disputed words *C* attach to one cluster's remainder.

    * ``nearest`` — mean L2 from each word in *C* to its nearest word in *remainder* (local).
      Falls back to centroid-vs-C when *remainder* is empty.
    * ``centroid`` — L2 from *remainder* centroid to centroid(C); empty *remainder* uses *original*.
    """
    if _pass2_overlap_assign_mode() == "centroid":
        return _pass2_dist_remainder_to_C(remainder, original, cent_c)
    if remainder:
        return sum(_pass2_min_dist_word_to_any(w, remainder) for w in c_words) / float(
            len(c_words)
        )
    return _pass2_dist_remainder_to_C(remainder, original, cent_c)


def _pass2_overlap_apply_lateral_bias(
    dist_a: float,
    dist_b: float,
    c_words: List[dict],
    a_prime: List[dict],
    b_prime: List[dict],
    ai: List[dict],
    bj: List[dict],
    iw: int,
) -> Tuple[float, float]:
    """Chat layout: when disputed text sits right of the screen, penalize the more-left cluster.

    Uses ``PASS2_OCR_CLUSTER_OVERLAP_LATERAL_BIAS`` (default 0.24; 0 = off) and
    ``PASS2_OCR_CLUSTER_OVERLAP_RIGHT_FRAC`` (fraction of image width).
    """
    bias = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_LATERAL_BIAS", "0.24"))
    if bias <= 0:
        return dist_a, dist_b
    thr = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_RIGHT_FRAC", "0.28")) * float(iw)
    cc = _pass2_word_group_centroid(c_words)
    if cc is None or cc[0] <= thr:
        return dist_a, dist_b
    mxa = _pass2_mean_cx(a_prime if a_prime else ai)
    mxb = _pass2_mean_cx(b_prime if b_prime else bj)
    sep = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_LATERAL_SEP_FRAC", "0.045")) * float(iw)
    if abs(mxa - mxb) < sep:
        return dist_a, dist_b
    if mxa < mxb:
        return dist_a * (1.0 + bias), dist_b
    return dist_a, dist_b * (1.0 + bias)


def _pass2_overlap_apply_column_row_bias(
    dist_a: float,
    dist_b: float,
    a_prime: List[dict],
    b_prime: List[dict],
    ai: List[dict],
    bj: List[dict],
    iw: int,
) -> Tuple[float, float]:
    """Penalize *column* vs *row*, and *non-row* vs *row* (bubble wins over neutral/column tails).

    Disabled when ``PASS2_OCR_CLUSTER_OVERLAP_COLUMN_PENALTY`` and
    ``PASS2_OCR_CLUSTER_OVERLAP_NONROW_VS_ROW_PENALTY`` are both 0.
    """
    pen = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_COLUMN_PENALTY", "0.48"))
    pen_nr = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_NONROW_VS_ROW_PENALTY", "0.40"))
    wa = a_prime if a_prime else ai
    wb = b_prime if b_prime else bj
    ka = _pass2_remainder_layout_kind(wa, iw)
    kb = _pass2_remainder_layout_kind(wb, iw)
    if pen > 0:
        if ka == "column" and kb == "row":
            return dist_a * (1.0 + pen), dist_b
        if kb == "column" and ka == "row":
            return dist_a, dist_b * (1.0 + pen)
    if pen_nr > 0:
        if ka != "row" and kb == "row":
            return dist_a * (1.0 + pen_nr), dist_b
        if kb != "row" and ka == "row":
            return dist_a, dist_b * (1.0 + pen_nr)
    return dist_a, dist_b


def _pass2_overlap_apply_width_ratio_bias(
    dist_a: float,
    dist_b: float,
    wa: float,
    wb: float,
) -> Tuple[float, float]:
    """When one remainder is much wider than the other, penalize the narrow side's score.

    Catches neutral+neutral aspect cases where one cluster is still clearly a horizontal run.
    """
    ratio = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_WIDTH_RATIO", "2.0"))
    pen = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_WIDTH_RATIO_PENALTY", "0.32"))
    min_px = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_WIDTH_MIN_PX", "12"))
    if pen <= 0 or ratio <= 1.0:
        return dist_a, dist_b
    mn = min(wa, wb)
    mx = max(wa, wb)
    if mn < min_px or mx < mn * ratio:
        return dist_a, dist_b
    if wa + 1e-6 < wb:
        return dist_a * (1.0 + pen), dist_b
    if wb + 1e-6 < wa:
        return dist_a, dist_b * (1.0 + pen)
    return dist_a, dist_b


def _pass2_try_resolve_one_cluster_bbox_overlap(clusters: List[List[dict]], iw: int) -> bool:
    """If some pair of cluster union bboxes overlap, peel intersecting words into C and reattach C
    to the side with lower score (NN/centroid + column/row + width-ratio + lateral + width ties).

    Returns True if *clusters* was mutated (caller should remove empty lists).
    """
    n = len(clusters)
    for i in range(n):
        for j in range(i + 1, n):
            ai, bj = clusters[i], clusters[j]
            if not ai or not bj:
                continue
            ui = _pass2_union_bbox_float(ai)
            uj = _pass2_union_bbox_float(bj)
            rect = _pass2_xyxy_intersection_positive(ui, uj)
            if rect is None:
                continue

            c_ids: Set[int] = set()
            c_words: List[dict] = []
            for w in ai:
                if _pass2_word_bbox_intersects_rect(w, rect):
                    c_words.append(w)
                    c_ids.add(id(w))
            for w in bj:
                if id(w) in c_ids:
                    continue
                if _pass2_word_bbox_intersects_rect(w, rect):
                    c_words.append(w)
                    c_ids.add(id(w))

            if not c_words:
                continue

            a_prime = [w for w in ai if id(w) not in c_ids]
            b_prime = [w for w in bj if id(w) not in c_ids]

            cent_c = _pass2_word_group_centroid(c_words)
            if cent_c is None:
                continue

            wa = _pass2_union_width(a_prime) if a_prime else _pass2_union_width(ai)
            wb = _pass2_union_width(b_prime) if b_prime else _pass2_union_width(bj)

            dist_a = _pass2_overlap_assign_score(c_words, a_prime, ai, cent_c)
            dist_b = _pass2_overlap_assign_score(c_words, b_prime, bj, cent_c)
            dist_a, dist_b = _pass2_overlap_apply_column_row_bias(
                dist_a, dist_b, a_prime, b_prime, ai, bj, iw
            )
            dist_a, dist_b = _pass2_overlap_apply_width_ratio_bias(dist_a, dist_b, wa, wb)
            dist_a, dist_b = _pass2_overlap_apply_lateral_bias(
                dist_a, dist_b, c_words, a_prime, b_prime, ai, bj, iw
            )
            eps_px = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_WIDTH_EPS_PX", "8"))
            rel_tol = float(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_REL_TIE_FRAC", "0.035"))
            denom = max(dist_a, dist_b, 1e-6)
            near_tie = abs(dist_a - dist_b) <= rel_tol * denom

            def _assign_c_to_a() -> None:
                clusters[i] = a_prime + c_words
                clusters[j] = b_prime

            def _assign_c_to_b() -> None:
                clusters[i] = a_prime
                clusters[j] = b_prime + c_words

            if near_tie:
                if wa > wb + eps_px:
                    _assign_c_to_a()
                elif wb > wa + eps_px:
                    _assign_c_to_b()
                elif dist_a < dist_b - 1e-9:
                    _assign_c_to_a()
                elif dist_b < dist_a - 1e-9:
                    _assign_c_to_b()
                else:
                    _assign_c_to_a()
            else:
                if dist_a < dist_b - 1e-9:
                    _assign_c_to_a()
                elif dist_b < dist_a - 1e-9:
                    _assign_c_to_b()
                else:
                    _assign_c_to_a()

            return True
    return False


def _pass2_try_rejoin_one_cluster_pair(clusters: List[List[dict]], iw: int, ih: int) -> bool:
    """Merge two clusters if union bboxes overlap strongly and merged center spread is below cap.

    Recovers one bubble split by diameter when ``PASS2_OCR_CLUSTER_REJOIN_IOU_MIN`` > 0.
    """
    iou_min = float(os.environ.get("PASS2_OCR_CLUSTER_REJOIN_IOU_MIN", "0"))
    if iou_min <= 0:
        return False
    max_frac = float(os.environ.get("PASS2_OCR_CLUSTER_REJOIN_MAX_DIAM_FRAC", "0.44"))
    diag = math.hypot(float(iw), float(ih))
    max_diam = max(0.004 * diag, max_frac * diag)

    best_iou = -1.0
    best_lo, best_hi = -1, -1
    n = len(clusters)
    for ii in range(n):
        for jj in range(ii + 1, n):
            ai, bj = clusters[ii], clusters[jj]
            if not ai or not bj:
                continue
            ui, uj = _pass2_union_bbox_float(ai), _pass2_union_bbox_float(bj)
            iou = _pass2_bbox_iou(ui, uj)
            if iou < iou_min or iou <= best_iou:
                continue
            merged = ai + bj
            if _pass2_cluster_max_center_pair_dist(merged) > max_diam:
                continue
            best_iou, best_lo, best_hi = iou, ii, jj

    if best_lo < 0:
        return False
    lo, hi = best_lo, best_hi
    merged = clusters[lo] + clusters[hi]
    merged.sort(key=lambda w: (float(w["cy"]), float(w["cx"])))
    clusters.pop(hi)
    clusters.pop(lo)
    clusters.append(merged)
    return True


def _pass2_resolve_overlapping_ocr_clusters(
    clusters: List[List[dict]],
    iw: int,
    ih: int,
) -> List[List[dict]]:
    """Post-pass: split words in union-bbox overlap between two clusters and assign them to one side.

    Scoring: ``PASS2_OCR_CLUSTER_OVERLAP_ASSIGN=nearest`` (default) uses mean distance from each
    disputed word to the nearest remaining word in that cluster; ``centroid`` uses remainder
    centroid vs centroid(C). Then: column-vs-row union-bbox penalty (tall sidebar vs wide bubble),
    width-ratio penalty when one remainder is much wider, lateral bias when *C* is right of
    ``PASS2_OCR_CLUSTER_OVERLAP_RIGHT_FRAC``, and near-ties prefer the wider remainder.

    Optional rejoin: ``PASS2_OCR_CLUSTER_REJOIN_IOU_MIN`` > 0 merges oversplit pairs whose IoU and
    merged center diameter pass thresholds.

    Overlap peeling is skipped when ``PASS2_OCR_CLUSTER_RESOLVE_OVERLAP=0``. IoU rejoin still runs
    when ``PASS2_OCR_CLUSTER_REJOIN_IOU_MIN`` > 0 so diameter-oversplit bubbles can merge.
    """
    out = [c for c in clusters if c]
    if not out:
        return out

    if _pass2_ocr_overlap_resolve_enabled():
        max_iter = max(1, int(os.environ.get("PASS2_OCR_CLUSTER_OVERLAP_RESOLVE_MAX_ITERS", "64")))
        for _ in range(max_iter):
            if not _pass2_try_resolve_one_cluster_bbox_overlap(out, iw):
                break
            out = [c for c in out if c]

    rj_iters = max(1, int(os.environ.get("PASS2_OCR_CLUSTER_REJOIN_MAX_ITERS", "32")))
    for _ in range(rj_iters):
        if not _pass2_try_rejoin_one_cluster_pair(out, iw, ih):
            break
        out = [c for c in out if c]

    out.sort(
        key=lambda cl: (min(float(w["cy"]) for w in cl), min(float(w["cx"]) for w in cl))
    )
    return out


def _pass2_token_jaccard(a: str, b: str) -> float:
    ta = {t for t in _pass2_norm_text(a).split() if t}
    tb = {t for t in _pass2_norm_text(b).split() if t}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _pass2_char_jaccard(a: str, b: str) -> float:
    """Helps Thai / no-space scripts where word token Jaccard is weak."""
    sa = {c for c in _pass2_norm_text(a) if not c.isspace()}
    sb = {c for c in _pass2_norm_text(b) if not c.isspace()}
    if len(sa) < 2 or len(sb) < 2:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pass2_text_resemblance_score(cluster_text: str, pass1_text: str) -> float:
    """Similarity in [0,1] between clustered OCR text and a Pass 1 line."""
    p1 = _pass2_norm_text(pass1_text)
    oc = _pass2_norm_text(cluster_text)
    if not p1 or not oc:
        return 0.0
    r = SequenceMatcher(None, oc, p1).ratio()
    j = _pass2_token_jaccard(cluster_text, pass1_text)
    cj = _pass2_char_jaccard(cluster_text, pass1_text)
    # Blend character overlap with sequence score so OCR clusters still match Pass 1 when wording differs slightly.
    blended = 0.55 * max(r, j) + 0.45 * cj
    base = max(r, j, blended)
    if oc in p1 or p1 in oc:
        return max(base, 0.88)
    return min(1.0, base)


def _pass2_token_overlap_f1(cluster_text: str, pass1_text: str) -> float:
    """Token F1 in [0,1] between OCR cluster text and Pass 1 line (helps short bubbles)."""
    ta = {t for t in _pass2_norm_text(cluster_text).split() if t}
    tb = {t for t in _pass2_norm_text(pass1_text).split() if t}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    prec = inter / len(ta)
    rec = inter / len(tb)
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _pass2_message_spatial_y_norm(m: dict) -> Optional[float]:
    """Pass 1 row vertical hint in [0,1] if present; else None (skip spatial term)."""
    for key in ("y_norm", "y_norm_center", "craft_y_norm", "center_y_norm"):
        v = m.get(key)
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            return float(v)
    return None


def _pass2_message_spatial_page_index(m: dict) -> Optional[int]:
    for key in ("page_index", "stitch_page", "image_index", "band_index"):
        v = m.get(key)
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v >= 0:
            return int(v)
    return None


def _pass2_cluster_page_y_norm(cl: List[dict]) -> Tuple[Optional[int], float]:
    """Primary page_index and mean y_norm (cy/page_h) for cluster *cl*."""
    if not cl:
        return None, 0.5
    pis = [int(w.get("page_index", 0) or 0) for w in cl]
    pi = max(set(pis), key=pis.count)
    same = [w for w in cl if int(w.get("page_index", 0) or 0) == pi]
    if not same:
        same = cl
    yns = []
    for w in same:
        ph = float(w.get("page_h") or 0.0)
        if ph > 1e-6:
            yns.append(float(w["cy"]) / ph)
    if not yns:
        return pi, 0.5
    return pi, sum(yns) / float(len(yns))


def _pass2_spatial_alignment_score(cl: List[dict], msg: dict) -> Optional[float]:
    """[0,1] alignment from Pass 1 y_norm (when present) to cluster vertical position; else None."""
    my = _pass2_message_spatial_y_norm(msg)
    if my is None:
        return None
    cpi, cy = _pass2_cluster_page_y_norm(cl)
    mpi = _pass2_message_spatial_page_index(msg)
    if mpi is not None and cpi is not None and mpi != cpi:
        return 0.15
    span = float(os.environ.get("PASS2_OCR_MATCH_SPATIAL_Y_SPAN", "4.0"))
    span = max(0.5, span)
    d = abs(cy - my)
    return max(0.0, 1.0 - min(1.0, d * span))


def _pass2_softmax_matrix_cols(
    logits: List[List[float]], temperature: float
) -> List[List[float]]:
    """``logits[g][m]`` → ``probs[g][m]`` with softmax over *g* for each column *m*."""
    g = len(logits)
    if g == 0:
        return []
    n = len(logits[0])
    temp = max(1e-6, float(temperature))
    out: List[List[float]] = [[0.0] * n for _ in range(g)]
    for m in range(n):
        col = [logits[gi][m] for gi in range(g)]
        mx = max(col)
        exps = [math.exp((c - mx) / temp) for c in col]
        s = sum(exps)
        if s <= 0:
            continue
        for gi in range(g):
            out[gi][m] = exps[gi] / s
    return out


def _match_ocr_clusters_to_pass1(
    clusters: List[List[dict]],
    pass1_messages: list,
) -> Tuple[Dict[int, int], Dict[int, float], List[dict], dict]:
    """Map Pass 1 rows to OCR clusters with a **one-to-one** partial matching.

    Clusters *g* and messages *m* get raw scores ``S[g,m]``; logits softmax over *g* yield ``P[g|m]``.
    Edges with ``P[g|m] ≥ PASS2_OCR_MATCH_MIN_CONFIDENCE`` are candidates.  Assignments are chosen by
    **greedy max-probability matching**: sort candidate edges by ``P`` descending and take an edge
    iff that message and that cluster are still free.  So each cluster is linked to **at most one**
    message and each message to **at most one** cluster (partial bijection).  A message ends up with
    no cluster if every ``P[g|m]`` is below the threshold, or if stronger edges consumed all suitable
    clusters first.  Unmatched rows keep Pass 1 text (no OCR hints / ``ocr_match_verified`` false).

    ``PASS2_OCR_MATCH_MIN_RAW_SCORE`` is diagnostic only (see ``match_summary``).

    Returns ``(message_to_cluster, message_to_confidence, cluster_debug_list, match_summary)``.
    """
    n = len(pass1_messages)
    g = len(clusters)
    w_text = float(os.environ.get("PASS2_OCR_MATCH_W_TEXT", "0.52"))
    w_ov = float(os.environ.get("PASS2_OCR_MATCH_W_OVERLAP", "0.38"))
    w_sp = float(os.environ.get("PASS2_OCR_MATCH_W_SPATIAL", "0.10"))
    w_text = max(0.0, w_text)
    w_ov = max(0.0, w_ov)
    w_sp = max(0.0, w_sp)
    logit_scale = float(os.environ.get("PASS2_OCR_MATCH_LOGIT_SCALE", "14.0"))
    logit_scale = max(0.1, logit_scale)
    softmax_temp = float(os.environ.get("PASS2_OCR_MATCH_SOFTMAX_TEMP", "1.0"))
    softmax_temp = max(1e-6, softmax_temp)
    min_raw = float(os.environ.get("PASS2_OCR_MATCH_MIN_RAW_SCORE", "0.04"))
    min_raw = max(0.0, min(1.0, min_raw))
    conf_threshold = float(os.environ.get("PASS2_OCR_MATCH_MIN_CONFIDENCE", "0.22"))
    conf_threshold = max(0.0, min(1.0, conf_threshold))

    def _empty_summary() -> dict:
        return {
            "pass1_message_count": n,
            "ocr_cluster_count": g,
            "matched_pairs": 0,
            "messages_without_group": n,
            "messages_would_verify": 0,
            "groups_without_message": g,
            "messages_below_match_confidence": 0,
            "groups_rejected_low_match_score": 0,
            "groups_rejected_ambiguous": 0,
            "groups_eligible_but_unassigned": 0,
            "messages_unmatched_cluster_competition": 0,
        }

    dbg_clusters: List[dict] = []
    if n == 0 or g == 0:
        return {}, {}, dbg_clusters, _empty_summary()

    pass1_lines = [
        (pass1_messages[i].get("text_src") or pass1_messages[i].get("text_en") or "").strip()
        for i in range(n)
    ]
    cluster_texts: List[str] = []
    for cl in clusters:
        parts = [(w.get("text") or "").strip() for w in cl]
        parts = [p for p in parts if p]
        cluster_texts.append(" ".join(parts))

    any_msg_spatial = any(_pass2_message_spatial_y_norm(pass1_messages[i]) is not None for i in range(n))

    # S[g][m] raw combined scores
    scores: List[List[float]] = []
    for gi in range(g):
        row: List[float] = []
        ct = cluster_texts[gi]
        cl = clusters[gi]
        for mi in range(n):
            line = pass1_lines[mi]
            if not line:
                row.append(0.0)
                continue
            text_s = _pass2_text_resemblance_score(ct, line)
            ov_s = _pass2_token_overlap_f1(ct, line)
            sp_s = _pass2_spatial_alignment_score(cl, pass1_messages[mi])
            if sp_s is not None:
                wsum = w_text + w_ov + w_sp
                if wsum <= 0:
                    combined = 0.0
                else:
                    combined = (w_text * text_s + w_ov * ov_s + w_sp * sp_s) / wsum
            else:
                wsum2 = w_text + w_ov
                if wsum2 <= 0:
                    combined = 0.0
                else:
                    combined = (w_text * text_s + w_ov * ov_s) / wsum2
            row.append(max(0.0, min(1.0, combined)))
        scores.append(row)

    logits: List[List[float]] = [[logit_scale * scores[gi][mi] for mi in range(n)] for gi in range(g)]
    probs = _pass2_softmax_matrix_cols(logits, softmax_temp)

    msg_to_cl: Dict[int, int] = {}
    msg_to_sc: Dict[int, float] = {}
    pairs: List[Tuple[float, int, int]] = []
    for mi in range(n):
        if not pass1_lines[mi]:
            continue
        for gi in range(g):
            p = float(probs[gi][mi])
            if p >= conf_threshold:
                pairs.append((p, mi, gi))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    matched_mi: Set[int] = set()
    matched_gi: Set[int] = set()
    for p, mi, gi in pairs:
        if mi in matched_mi or gi in matched_gi:
            continue
        msg_to_cl[mi] = gi
        msg_to_sc[mi] = p
        matched_mi.add(mi)
        matched_gi.add(gi)

    would_verify = len(msg_to_cl)

    for gi in range(g):
        pcol = [round(probs[gi][mi], 5) for mi in range(n)] if gi < len(probs) else []
        scol = [round(scores[gi][mi], 5) for mi in range(n)]
        best_m = max(range(n), key=lambda mi: scores[gi][mi]) if n else 0
        s1 = scores[gi][best_m] if n else 0.0
        row_pairs = sorted([(mi, scores[gi][mi]) for mi in range(n)], key=lambda t: -t[1])
        s2 = row_pairs[1][1] if n > 1 else 0.0
        dbg_clusters.append(
            {
                "cluster_index": gi,
                "cluster_text": cluster_texts[gi],
                "word_count": len(clusters[gi]),
                "best_message_index": best_m,
                "best_raw_score": round(s1, 4),
                "second_best_raw_score": round(s2, 4) if n > 1 else None,
                "raw_score_by_message": scol,
                "prob_given_message": pcol,
                "rejected_ambiguous": False,
                "rejected_low_score": False,
            }
        )

    assigned_gi = set(msg_to_cl.values())
    unmatched_msg = n - len(msg_to_cl)
    below_conf = sum(
        1
        for mi in range(n)
        if pass1_lines[mi]
        and max(probs[gi][mi] for gi in range(g)) < conf_threshold
    )
    matched_msg_indices = set(msg_to_cl.keys())
    competition_unmatched = sum(
        1
        for mi in range(n)
        if pass1_lines[mi]
        and mi not in matched_msg_indices
        and max(probs[gi][mi] for gi in range(g)) >= conf_threshold
    )
    low_raw = sum(
        1
        for mi in range(n)
        if pass1_lines[mi] and max(scores[gi][mi] for gi in range(g)) < min_raw
    )
    summary = {
        "algorithm": "greedy_one_to_one_softmax_threshold",
        "pass1_message_count": n,
        "ocr_cluster_count": g,
        "matched_pairs": len(msg_to_cl),
        "messages_without_group": unmatched_msg,
        "messages_would_verify": would_verify,
        "groups_without_message": g - len(assigned_gi),
        "messages_below_match_confidence": below_conf,
        "messages_unmatched_cluster_competition": competition_unmatched,
        "groups_rejected_low_match_score": low_raw,
        "groups_rejected_ambiguous": 0,
        "groups_eligible_but_unassigned": 0,
        "spatial_hint_used": bool(any_msg_spatial),
        "thresholds_applied": {
            "min_raw_score": min_raw,
            "min_confidence_for_assignment": conf_threshold,
            "min_confidence_for_hints": conf_threshold,
            "logit_scale": logit_scale,
            "softmax_temperature": softmax_temp,
            "weights_text_overlap_spatial": [w_text, w_ov, w_sp],
        },
    }
    return msg_to_cl, msg_to_sc, dbg_clusters, summary


def _print_pass2_ocr_match_summary(
    match_summary: dict,
    ocr_word_count: int,
    min_ocr_conf: float,
) -> None:
    """Always-on summary: OCR cluster ↔ Pass 1 row matches (soft hints only; no reordering)."""
    n = int(match_summary.get("pass1_message_count") or 0)
    g = int(match_summary.get("ocr_cluster_count") or 0)
    m = int(match_summary.get("matched_pairs") or 0)
    mw = int(match_summary.get("messages_without_group") or 0)
    gu = int(match_summary.get("groups_without_message") or 0)
    lo = int(match_summary.get("groups_rejected_low_match_score") or 0)
    bc = int(match_summary.get("messages_below_match_confidence") or 0)
    cu = int(match_summary.get("messages_unmatched_cluster_competition") or 0)
    am = int(match_summary.get("groups_rejected_ambiguous") or 0)
    lg = int(match_summary.get("groups_eligible_but_unassigned") or 0)
    wv = int(match_summary.get("messages_would_verify") or 0)
    tty = getattr(sys.stdout, "isatty", lambda: False)()
    b, off = ("\033[1m", "\033[0m") if tty else ("", "")

    if _compact_verbose_logs():
        print("", flush=True)
        print("*******Pass 2 — OCR clusters vs Pass 1 messages******", flush=True)
        print(f"  Total Pass 1 messages:     {n}", flush=True)
        print(f"  Total OCR groups:          {g}", flush=True)
        print(f"  Unmatched groups:          {gu}", flush=True)
        print(f"  Unmatched messages:        {mw}", flush=True)
        print("", flush=True)
        return

    print(f"{b}Pass 2 — OCR clusters vs Pass 1 messages{off}", flush=True)
    print(f"{b}  Total Pass 1 messages:     {n}{off}", flush=True)
    print(f"{b}  Total OCR groups:          {g}{off}", flush=True)
    print(f"{b}  Unmatched groups:          {gu}{off}", flush=True)
    print(f"{b}  Unmatched messages:        {mw}{off}", flush=True)
    print(
        f"  {wv} row(s): match-confidence OK → Pass 2 OCR hint lists (tokens ≥ {min_ocr_conf:.2f}).",
        flush=True,
    )
    print(
        f"  {max(0, n - wv)} row(s): no OCR hints → Pass 2 keeps Pass 1 `text_src` for those indices.",
        flush=True,
    )
    print(
        f"  Match model: softmax P(cluster|message); assign only if P(top cluster) ≥ threshold; "
        f"assigned rows={m}. "
        f"Below threshold≈{bc} row(s); unmatched despite threshold (competition)≈{cu}; "
        f"diagnostic max-raw below floor≈{lo}.",
        flush=True,
    )
    _pv(
        f"[PASS2 OCR] tuning: ocr_words>={min_ocr_conf:.2f}: {ocr_word_count}  "
        f"matched_pairs={m}  thresholds={match_summary.get('thresholds_applied')}"
    )
    if g > n:
        print(
            f"  Note: {g} OCR groups vs {n} Pass 1 rows — often multi-line bubbles split by "
            f"gap/diameter rules, or extra UI text. Set PASS2_OCR_LOG_CLUSTERS=1 for per-group "
            f"words (also writes result/pass2_ocr_cluster_groups.txt). Tune "
            f"PASS2_OCR_CLUSTER_GAP_Y_FRAC / PASS2_OCR_CLUSTER_MAX_DIAM_FRAC if needed.",
            flush=True,
        )


def _pass2_ocr_should_log_cluster_detail() -> bool:
    v = os.environ.get("PASS2_OCR_LOG_CLUSTERS", "").strip().lower()
    return v in ("1", "true", "yes", "on") or _pipeline_verbose()


def _pass2_ocr_should_emit_cluster_debug_images() -> bool:
    """Write ``pass2_ocr_clusters_page_*.png`` when Pass 2 OCR clustering runs.

    - ``PASS2_OCR_CLUSTER_DEBUG_IMAGES=0`` — never write overlay images.
    - ``PASS2_OCR_CLUSTER_DEBUG_IMAGES=1`` — always write (when clusters exist and pages are valid).
    - Unset — same as detailed cluster log: on if ``PIPELINE_VERBOSE`` or ``PASS2_OCR_LOG_CLUSTERS``.
    """
    v = os.environ.get("PASS2_OCR_CLUSTER_DEBUG_IMAGES", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return _pass2_ocr_should_log_cluster_detail()


def _pass2_should_write_hint_audit_file() -> bool:
    """Write ``pass2_ocr_hint_audit.txt`` under :data:`OUTPUT_DIR` unless
    ``PASS2_OCR_HINT_AUDIT_FILE=0`` / ``false`` / ``off``.
    """
    v = os.environ.get("PASS2_OCR_HINT_AUDIT_FILE", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _pass2_cluster_text_map(cluster_dbg: List[dict]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for d in cluster_dbg or []:
        gi = d.get("cluster_index")
        if gi is None:
            continue
        out[int(gi)] = (d.get("cluster_text") or "").strip()
    return out


def _pass2_write_ocr_hints_audit_file(
    pass1_messages: list,
    per_index: List[dict],
    cluster_dbg: List[dict],
    min_ocr_conf: float,
    match_min_confidence: float,
) -> None:
    """Append-free overwrite: Pass 1 line vs matched OCR cluster and high-conf tokens (same as former console audit)."""
    if not _pass2_should_write_hint_audit_file():
        return
    n = len(pass1_messages or [])
    if n == 0:
        return
    ct_map = _pass2_cluster_text_map(cluster_dbg)
    bar = "=" * 72
    lines: List[str] = [
        bar,
        " Pass 2 — OCR hint audit (Pass 1 text_src vs matched cluster & tokens → Gemini)",
        f" OCR tokens listed below require score≥{min_ocr_conf:.2f}; "
        f"row must pass match-confidence≥{match_min_confidence:.2f} for a hint list.",
        bar,
    ]
    for row in per_index:
        mi = int(row.get("message_index", -1))
        if mi < 0 or mi >= n:
            continue
        m = pass1_messages[mi]
        role = m.get("role") or ""
        src = (m.get("text_src") or m.get("text_en") or "").strip()
        verified = bool(row.get("ocr_match_verified"))
        gi = row.get("matched_cluster_index")
        mc = row.get("match_confidence")
        tokens = row.get("tokens") or []
        ctext = ct_map.get(int(gi), "") if gi is not None else ""
        lines.append(f"\n--- message_index={mi}  role={role}  ocr_match_verified={verified} ---")
        lines.append(f"pass1_text_src: {json.dumps(src, ensure_ascii=False)}")
        if gi is not None:
            lines.append(
                f"matched_cluster_index={gi}  match_confidence={mc}  "
                f"cluster_text (full OCR group): {json.dumps(ctext, ensure_ascii=False)}"
            )
        else:
            lines.append("matched_cluster_index=None (no cluster above raw-score floor)")
        if verified and not tokens:
            lines.append(
                "[note] verified=True but hint token list is empty — every word in the cluster "
                f"was below OCR conf {min_ocr_conf:.2f}; Pass 2 sees no list for this index."
            )
        elif tokens:
            parts = []
            for t in tokens:
                tx = t.get("text", "")
                cf = t.get("conf", "")
                parts.append(f"{json.dumps(tx, ensure_ascii=False)} ({cf})")
            lines.append("hint_tokens → Gemini: " + " | ".join(parts))
        elif not verified:
            wh = row.get("withheld_from_prompt")
            lines.append(
                f"no hint list (withheld_from_prompt={wh}); Pass 2 must keep pass1_text_src verbatim."
            )
    lines.extend(
        [
            "",
            bar,
            " Pass 2 user prompt is written to: gemini_prompt_pass2.txt (and gemini_prompt_pass2_retry.txt on retry).",
            bar,
            "",
        ]
    )
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "pass2_ocr_hint_audit.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print(f"[PASS2 OCR] Wrote hint audit → {path}", flush=True)
    except OSError as e:
        _pv(f"[PASS2 OCR] Could not write pass2_ocr_hint_audit.txt: {e}")


def _pass2_ocr_cluster_union_bbox(words: List[dict]) -> Optional[Tuple[int, int, int, int]]:
    """Axis-aligned union of word bboxes; integers clamped for OpenCV drawing."""
    if not words:
        return None
    x1 = min(float(w["x1"]) for w in words)
    y1 = min(float(w["y1"]) for w in words)
    x2 = max(float(w["x2"]) for w in words)
    y2 = max(float(w["y2"]) for w in words)
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


_PASS2_OCR_DEBUG_COLORS_BGR: List[Tuple[int, int, int]] = [
    (40, 180, 255),
    (180, 105, 255),
    (0, 200, 100),
    (255, 150, 80),
    (230, 80, 180),
    (60, 220, 230),
    (180, 180, 50),
    (130, 90, 230),
]


def _pass2_ocr_bgr_for_cluster(global_index: int) -> Tuple[int, int, int]:
    return _PASS2_OCR_DEBUG_COLORS_BGR[global_index % len(_PASS2_OCR_DEBUG_COLORS_BGR)]


def _pass2_ocr_draw_cluster_debug_legend(vis: np.ndarray) -> None:
    """Bottom banner on debug PNGs: active gap/diameter fractions (matches env defaults)."""
    h, w = vis.shape[:2]
    if h < 16 or w < 64:
        return
    gx = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_X_FRAC", "0.068"))
    gy = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_Y_FRAC", "0.078"))
    diam = float(os.environ.get("PASS2_OCR_CLUSTER_MAX_DIAM_FRAC", "0.26"))
    pad = 8
    line = (
        f"PASS2 OCR cluster  gx_frac={gx:.3f}  gy_frac={gy:.3f}  max_diam_frac={diam:.3f}  "
        f"(merge if gap<=frac*W/H; split if center_L2>frac*diag)"
    )
    fs = max(0.38, min(0.72, float(w) / 1200.0))
    thick = 1
    max_tw = max(32, w - 2 * pad)
    for _ in range(12):
        (tw, th), _bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        if tw <= max_tw:
            break
        fs = max(0.22, fs * 0.88)
    (_tw, th), _bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    bar_h = min(h, th + pad * 2)
    y0 = h - bar_h
    cv2.rectangle(vis, (0, y0), (w, h), (28, 28, 28), -1)
    tx, ty = pad, h - pad
    cv2.putText(
        vis,
        line,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        line,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (235, 235, 235),
        1,
        lineType=cv2.LINE_AA,
    )


def _pass2_ocr_emit_cluster_debug_images(
    page_images,
    clusters: List[List[dict]],
) -> List[str]:
    """One PNG per input page: union bbox of each spatial cluster (all grouped OCR words), BGR overlay.

    Filenames: ``pass2_ocr_clusters_page_01.png``, … under :data:`OUTPUT_DIR`.
    Cluster labels ``g0``, ``g1``, … match indices in ``pass2_ocr_cluster_groups.txt``.
    """
    if not clusters or not _pass2_ocr_should_emit_cluster_debug_images():
        return []

    pages = list(page_images or [])
    if not pages:
        return []

    written: List[str] = []
    for page_idx, page_img in enumerate(pages):
        if page_img is None or getattr(page_img, "size", 0) == 0:
            continue
        h, w = page_img.shape[:2]
        if page_img.ndim == 2:
            vis = cv2.cvtColor(page_img, cv2.COLOR_GRAY2BGR)
        elif page_img.shape[2] == 4:
            vis = cv2.cvtColor(page_img, cv2.COLOR_BGRA2BGR)
        else:
            vis = page_img.copy()

        overlay = vis.copy()
        draw_specs: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int], str]] = []

        for gi, cl in enumerate(clusters):
            if not cl or int(cl[0].get("page_index", -1)) != page_idx:
                continue
            box = _pass2_ocr_cluster_union_bbox(cl)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            color = _pass2_ocr_bgr_for_cluster(gi)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            draw_specs.append((box, color, f"g{gi}"))

        if draw_specs:
            cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)
        for (x1, y1, x2, y2), color, label in draw_specs:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tx = x1
            ty = y1 - 6
            if ty < 12:
                ty = y2 + 18
            if ty >= h:
                ty = h - 2
            cv2.putText(
                vis,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (32, 32, 32),
                3,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        _pass2_ocr_draw_cluster_debug_legend(vis)

        fname = f"pass2_ocr_clusters_page_{page_idx + 1:02d}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            if not cv2.imwrite(out_path, vis):
                _pv(f"[PASS2 OCR] imwrite failed for {out_path}")
                continue
            written.append(out_path)
        except OSError as exc:
            _pv(f"[PASS2 OCR] could not write {out_path}: {exc}")

    if written:
        print(
            f"[PASS2 OCR] Wrote {len(written)} cluster overlay PNG(s) → {OUTPUT_DIR}",
            flush=True,
        )
    return written


def _pass2_ocr_emit_filtered_cluster_report(
    clusters: List[List[dict]],
    min_conf: float,
) -> None:
    """Log each OCR cluster after grouping (all words in the group). Console + file when enabled.

    Every line lists one OCR token; ``[hint]`` marks conf ≥ ``GEMINI_PASS2_OCR_MIN_CONF`` (those
    are the only tokens copied into Pass 2 hint lists).

    - ``PASS2_OCR_LOG_CLUSTERS=1``: print full report and write ``pass2_ocr_cluster_groups.txt``.
    - ``PIPELINE_VERBOSE=1`` only: write file and a short ``_pv`` pointer (avoids huge TTY spam).
    """
    if not clusters or not _pass2_ocr_should_log_cluster_detail():
        return

    gx = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_X_FRAC", "0.068"))
    gy = float(os.environ.get("PASS2_OCR_CLUSTER_GAP_Y_FRAC", "0.078"))
    diam = float(os.environ.get("PASS2_OCR_CLUSTER_MAX_DIAM_FRAC", "0.26"))
    lines: List[str] = [
        "=== Pass 2 OCR clusters (all grouped words; hint filter applied only to prompt lists) ===",
        f"GEMINI_PASS2_OCR_MIN_CONF (hint filter) = {min_conf:.4f}",
        f"union–find: PASS2_OCR_CLUSTER_GAP_X_FRAC={gx}  PASS2_OCR_CLUSTER_GAP_Y_FRAC={gy}",
        f"diameter split: PASS2_OCR_CLUSTER_MAX_DIAM_FRAC={diam}  (0 disables split)",
        f"total_clusters={len(clusters)}",
        "",
    ]
    for ci, cl in enumerate(clusters):
        if not cl:
            continue
        pi = int(cl[0].get("page_index", 0))
        confs = [float(w.get("conf") or 0.0) for w in cl]
        cmin, cmax = (min(confs), max(confs)) if confs else (0.0, 0.0)
        parts = [(w.get("text") or "").strip() for w in cl]
        joined = " ".join(p for p in parts if p)
        ys = [float(w.get("cy", 0.0)) for w in cl]
        xs = [float(w.get("cx", 0.0)) for w in cl]
        y0, y1 = (min(ys), max(ys)) if ys else (0.0, 0.0)
        x0, x1 = (min(xs), max(xs)) if xs else (0.0, 0.0)
        lines.append(
            f"--- cluster {ci}  page_index={pi}  words={len(cl)}  "
            f"conf min={cmin:.3f} max={cmax:.3f}  "
            f"cx≈{x0:.0f}–{x1:.0f} cy≈{y0:.0f}–{y1:.0f} ---"
        )
        lines.append(f"    joined: {joined[:500]}{'…' if len(joined) > 500 else ''}")
        for wi, w in enumerate(cl):
            t = (w.get("text") or "").strip()
            cf = float(w.get("conf") or 0.0)
            tag = "hint" if cf >= min_conf else "group_only"
            lines.append(
                f"    [{wi}] {json.dumps(t, ensure_ascii=False)}  conf={cf:.4f}  [{tag}]"
            )
        lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    path = os.path.join(OUTPUT_DIR, "pass2_ocr_cluster_groups.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
    except OSError as exc:
        _pv(f"[PASS2 OCR] could not write {path}: {exc}")

    loud = os.environ.get("PASS2_OCR_LOG_CLUSTERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if loud:
        print("", flush=True)
        print("*******Pass 2 — OCR cluster groups (all words; [hint] = in prompt)******", flush=True)
        print(body, end="", flush=True)
        print(f"[PASS2 OCR] Full cluster listing → {path}", flush=True)
    else:
        _pv(f"[PASS2 OCR] Wrote {len(clusters)} cluster(s) detail → {path}")


def build_pass2_per_message_ocr_hints(
    page_images,
    pass1_messages: list,
) -> Tuple[str, List[bool], dict]:
    """Per-page OCR → spatial clusters → match clusters to **existing** Pass 1 row indices.

    Clustering runs **separately on each input image**; a group never mixes words from two images.
    **Every** non-empty OCR token with a valid bbox is included in the grouping graph so low- and
    high-confidence words jointly define cluster shape. After that, only tokens with
    conf ≥ ``GEMINI_PASS2_OCR_MIN_CONF`` (default 0.92) are listed as Pass 2 hints.

    Does **not** reorder or re-index messages.

    Returns ``(hints_markdown, ocr_match_verified_per_message, debug_dict)``.
    """
    n = len(pass1_messages or [])
    min_conf = float(os.environ.get("GEMINI_PASS2_OCR_MIN_CONF", "0.92"))
    min_conf = max(0.0, min(1.0, min_conf))
    match_min_confidence = float(os.environ.get("PASS2_OCR_MATCH_MIN_CONFIDENCE", "0.22"))
    match_min_confidence = max(0.0, min(1.0, match_min_confidence))

    empty_verified = [False] * n
    base_debug: dict = {
        "min_confidence": min_conf,
        "ocr_source": "per_page_images",
        "pass1_message_count": n,
        "matching": {
            "min_raw_score": float(os.environ.get("PASS2_OCR_MATCH_MIN_RAW_SCORE", "0.04")),
            "min_confidence_for_hints": match_min_confidence,
            "logit_scale": float(os.environ.get("PASS2_OCR_MATCH_LOGIT_SCALE", "14.0")),
            "softmax_temperature": float(os.environ.get("PASS2_OCR_MATCH_SOFTMAX_TEMP", "1.0")),
            "weights_text_overlap_spatial": [
                float(os.environ.get("PASS2_OCR_MATCH_W_TEXT", "0.52")),
                float(os.environ.get("PASS2_OCR_MATCH_W_OVERLAP", "0.38")),
                float(os.environ.get("PASS2_OCR_MATCH_W_SPATIAL", "0.10")),
            ],
            "cluster_gap_x_frac": float(os.environ.get("PASS2_OCR_CLUSTER_GAP_X_FRAC", "0.068")),
            "cluster_gap_y_frac": float(os.environ.get("PASS2_OCR_CLUSTER_GAP_Y_FRAC", "0.078")),
            "cluster_max_diam_frac": float(os.environ.get("PASS2_OCR_CLUSTER_MAX_DIAM_FRAC", "0.26")),
            "cluster_overlap_resolve": _pass2_ocr_overlap_resolve_enabled(),
            "cluster_overlap_assign": _pass2_overlap_assign_mode(),
        },
        "ocr_clusters": [],
        "per_index": [],
        "cluster_debug_image_paths": [],
    }

    if n == 0:
        return "", [], {**base_debug, "reason": "no_pass1_messages"}

    pages = list(page_images or [])
    if not pages or all(p is None or getattr(p, "size", 0) == 0 for p in pages):
        st = {
            "pass1_message_count": n,
            "ocr_cluster_count": 0,
            "matched_pairs": 0,
            "messages_without_group": n,
            "messages_would_verify": 0,
            "groups_without_message": 0,
            "messages_below_match_confidence": 0,
            "messages_unmatched_cluster_competition": 0,
            "groups_rejected_low_match_score": 0,
            "groups_rejected_ambiguous": 0,
            "groups_eligible_but_unassigned": 0,
            "note": "no_page_images",
        }
        base_debug["match_summary"] = st
        _print_pass2_ocr_match_summary(st, 0, min_conf)
        return "", empty_verified, {**base_debug, "reason": "no_page_images"}

    clusters_full: List[List[dict]] = []
    words_kept_ge_min_conf = 0

    for page_idx, page_img in enumerate(pages):
        if page_img is None or getattr(page_img, "size", 0) == 0:
            continue
        ph, pw = page_img.shape[:2]
        ph = max(int(ph), 1)
        pw = max(int(pw), 1)
        try:
            detections = ocr_engine.predict(page_img)
        except Exception as e:
            _pv(f"[PASS2 OCR] predict on page {page_idx} failed: {e}")
            continue

        word_boxes: List[dict] = []
        for det in detections or []:
            text = (det.get("text") or "").strip()
            if not text:
                continue
            conf = float(det.get("score") or 0.0)
            bbox = det.get("bbox") or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            word_boxes.append(
                {
                    "text": text,
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) * 0.5,
                    "cy": (y1 + y2) * 0.5,
                    "page_index": page_idx,
                    "page_h": float(ph),
                    "page_w": float(pw),
                }
            )

        if not word_boxes:
            continue

        # One OCR + cluster pass per page: words from different images are never in the same group.
        raw_clusters = _cluster_pass2_ocr_words(word_boxes, pw, ph)
        raw_clusters = _pass2_resolve_overlapping_ocr_clusters(raw_clusters, pw, ph)
        for cl in raw_clusters:
            if not cl:
                continue
            clusters_full.append(cl)
            words_kept_ge_min_conf += sum(1 for w in cl if float(w["conf"]) >= min_conf)

    if not clusters_full:
        st = {
            "pass1_message_count": n,
            "ocr_cluster_count": 0,
            "matched_pairs": 0,
            "messages_without_group": n,
            "messages_would_verify": 0,
            "groups_without_message": 0,
            "messages_below_match_confidence": 0,
            "messages_unmatched_cluster_competition": 0,
            "groups_rejected_low_match_score": 0,
            "groups_rejected_ambiguous": 0,
            "groups_eligible_but_unassigned": 0,
            "note": "no_clusters_after_per_page_ocr",
        }
        base_debug["match_summary"] = st
        _print_pass2_ocr_match_summary(st, 0, min_conf)
        return "", empty_verified, {**base_debug, "reason": "no_clusters_after_per_page_ocr"}

    clusters = clusters_full
    _pass2_ocr_emit_filtered_cluster_report(clusters, min_conf)
    dbg_img_paths = _pass2_ocr_emit_cluster_debug_images(pages, clusters)
    if dbg_img_paths:
        base_debug["cluster_debug_image_paths"] = dbg_img_paths
    msg_to_cl, msg_to_sc, cluster_dbg, match_summary = _match_ocr_clusters_to_pass1(
        clusters, pass1_messages
    )
    match_summary = dict(match_summary)
    match_summary["ocr_word_count_above_conf"] = words_kept_ge_min_conf
    base_debug["match_summary"] = match_summary
    _print_pass2_ocr_match_summary(match_summary, words_kept_ge_min_conf, min_conf)

    # Injective: each cluster index appears at most once in msg_to_cl values.
    cl_to_msg = {cl_i: mi for mi, cl_i in msg_to_cl.items()}
    for item in cluster_dbg:
        gi = item.get("cluster_index")
        if gi is not None:
            item["assigned_message_index"] = cl_to_msg.get(gi)
            if 0 <= gi < len(clusters) and clusters[gi]:
                item["page_index"] = int(clusters[gi][0].get("page_index", 0))
    base_debug["ocr_clusters"] = cluster_dbg

    buckets: List[List[dict]] = [[] for _ in range(n)]
    verified: List[bool] = [False] * n
    for mi in range(n):
        gi = msg_to_cl.get(mi)
        sc = msg_to_sc.get(mi, 0.0)
        use_hints = (
            gi is not None
            and 0 <= gi < len(clusters)
            and float(sc) >= match_min_confidence
        )
        if use_hints:
            verified[mi] = True
            for w in clusters[gi]:
                if float(w["conf"]) < min_conf:
                    continue
                ph_f = float(w["page_h"])
                pw_f = float(w["page_w"])
                buckets[mi].append(
                    {
                        "text": w["text"],
                        "conf": float(w["conf"]),
                        "page_index": int(w.get("page_index", 0)),
                        "y_norm": round(float(w["cy"]) / ph_f, 5),
                        "x_norm": round(float(w["cx"]) / pw_f, 5),
                    }
                )
            buckets[mi].sort(key=lambda z: (z["page_index"], z["y_norm"], z["x_norm"]))
        base_debug["per_index"].append(
            {
                "message_index": mi,
                "matched_cluster_index": gi,
                "match_confidence": round(sc, 4) if gi is not None else None,
                "match_score": round(sc, 4) if gi is not None else None,
                "ocr_match_verified": verified[mi],
                "withheld_from_prompt": not verified[mi],
                "ocr_token_count": len(buckets[mi]),
                "tokens": [
                    {
                        "text": b["text"],
                        "conf": round(float(b["conf"]), 4),
                        "page_index": b.get("page_index"),
                    }
                    for b in buckets[mi]
                ],
            }
        )

    parts = [
        "OCR word candidates per `message_index` (from the screenshot). Only some indices have a list; "
        "each bullet is one token whose vision confidence met the hint floor for this run.\n\n",
        f"Hint confidence floor: ≥ {min_conf:.2f}.\n\n",
    ]

    any_verified = False
    for i in range(n):
        if not (verified[i] and buckets[i]):
            continue
        any_verified = True
        parts.append(f"### message_index {i}\n")
        for b in buckets[i]:
            tjson = json.dumps(b["text"], ensure_ascii=False)
            parts.append(f"  - {tjson}\n")
        parts.append("\n")

    if not any_verified:
        parts.append("_No hint lists for any index._\n")

    _pass2_write_ocr_hints_audit_file(
        pass1_messages,
        base_debug.get("per_index") or [],
        base_debug.get("ocr_clusters") or [],
        min_conf,
        match_min_confidence,
    )

    return "".join(parts).strip() + "\n", verified, base_debug


def _gemini_ocr_hints_refine_pass(
    contact_name: str,
    pass1_messages: list,
    ocr_hints_text: str,
    timeout: int,
    ocr_match_verified: Optional[List[bool]] = None,
):
    """Pass 2: optional per-index OCR hint lists + short rules in the prompt.

    Merge step forces ``text_src`` back to Pass 1 for any index with
    ``ocr_match_verified`` false, regardless of model output.
    """
    def _pass2_meta(
        applied: bool, gres: Optional[GeminiApiResult] = None, reason: str = ""
    ) -> Dict[str, Any]:
        gmeta = (getattr(gres, "meta", None) or {}) if gres is not None else {}
        return {
            "applied": bool(applied),
            "successful_attempt": gmeta.get("successful_attempt"),
            "status": gmeta.get("status"),
            "reason": reason or None,
        }

    if not pass1_messages:
        return pass1_messages, contact_name, _pass2_meta(False, reason="missing_pass1_messages")

    expected_n = len(pass1_messages)
    if ocr_match_verified is not None:
        if len(ocr_match_verified) != expected_n:
            ocr_match_verified = None
        elif not any(ocr_match_verified):
            _pv("[GEMINI] Pass 2 skipped — no Pass 1 row matched an OCR cluster above confidence / ambiguity thresholds")
            return [dict(m) for m in pass1_messages], contact_name, _pass2_meta(
                False, reason="no_verified_ocr_matches"
            )

    payload_messages = []
    for i, m in enumerate(pass1_messages):
        row = {
            "message_index": i,
            "role": m.get("role"),
            "pass1_text_src": (m.get("text_src") or m.get("text_en") or "").strip(),
        }
        if ocr_match_verified is not None:
            row["ocr_match_verified"] = bool(ocr_match_verified[i])
        payload_messages.append(row)

    payload = {"contact_name": contact_name, "messages": payload_messages}
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    hints_full = (ocr_hints_text or "").strip()
    max_hint_chars = int(os.environ.get("GEMINI_PASS2_OCR_MAX_CHARS", "24000"))
    retry_hint_chars = int(os.environ.get("GEMINI_PASS2_OCR_RETRY_MAX_CHARS", "12000"))

    def _truncate_hints(text: str, limit: int) -> str:
        text = (text or "").strip()
        if not text or len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n\n[... OCR hints truncated ...]"

    def _build_prompt(hints_text: str, retry_note: str = "") -> str:
        hf = float(os.environ.get("GEMINI_PASS2_OCR_MIN_CONF", "0.92"))
        hf = max(0.0, min(1.0, hf))
        if ocr_match_verified is not None:
            return f"""You are PASS 2 of 3.

{payload_json}

Below, some `message_index` sections list **OCR word candidates** for that row (tokens with vision confidence ≥ {hf:.2f}).

{hints_text if hints_text else "(no OCR candidate lists)"}

Meaning of each list:
- We **estimate these words have high probability of belonging inside that message’s true on-screen text** (the pipeline can mis-associate a cluster, so this is not guaranteed).
- The lists are **guidance only**: you are **not** forced to use any word, and nothing must appear verbatim.
- When a list is present and `ocr_match_verified` is true, **consider rewriting** `text_src` if Pass 1 likely disagrees with the screenshot; you may still leave `pass1_text_src` unchanged when it already matches the screen.

Rules:
- Output exactly {expected_n} messages: same order and `role` as Pass 1.
- If `ocr_match_verified` is false **or** that index has no OCR candidate list below: set `text_src` exactly to `pass1_text_src` (unchanged).
- If `ocr_match_verified` is true **and** a list exists for that index: you may rewrite `text_src` in the source language following the meaning above.
- Do not add or remove messages or translate to English. Do not use candidates from another index’s list.
{retry_note}

Output JSON only:
{{"contact_name":"<string>","messages":[{{"message_index":0,"role":"contact|user|system","text_src":"<string>"}}]}}"""

        return f"""You are PASS 2 of 3.

{payload_json}

Some `message_index` sections list **OCR word candidates** (vision confidence ≥ {hf:.2f}); others have no list.

{hints_text if hints_text else "(no OCR candidate lists)"}

Meaning of each list:
- We **estimate these words have high probability of belonging in that message’s true on-screen text** (matching can be wrong).
- **Guidance only**—not mandatory, not required verbatim.
- **Consider rewriting** `text_src` when the candidates suggest Pass 1 may be off; keep Pass 1 when it already fits.

Rules:
- Exactly {expected_n} messages; same order and `role` as Pass 1.
- No list for an index: keep `text_src` the same as Pass 1.
- List present: you may rewrite in the source language as above.
- Do not add/remove messages or translate to English. Do not use another index’s candidates.
{retry_note}

Output JSON only:
{{"contact_name":"<string>","messages":[{{"message_index":0,"role":"contact|user|system","text_src":"<string>"}}]}}"""

    hints = _truncate_hints(hints_full, max_hint_chars)
    prompt = _build_prompt(hints)
    _write_gemini_prompt_file("gemini_prompt_pass2.txt", "Pass 2 (OCR-guided rewrite)", prompt)
    p2_temp = max(
        0.0,
        min(0.35, float(os.environ.get("GEMINI_PASS2_TEMPERATURE", "0.12"))),
    )
    try:
        gres = _gemini_generate(prompt, timeout=timeout, pass_num=2, temperature=p2_temp)
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 2 failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}", "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name, _pass2_meta(False, reason="request_failed")

    if not raw.strip():
        _pv("[GEMINI] Pass 2 empty — retrying once with shorter OCR hints")
        retry_hints = _truncate_hints(hints_full, retry_hint_chars)
        retry_prompt = _build_prompt(
            retry_hints,
            retry_note=(
                "\n- Reminder: false or no list → verbatim Pass 1; list + true → OCR candidates are probabilistic guidance—consider a rewrite if Pass 1 disagrees with the screen."
                if ocr_match_verified is not None
                else "\n- If candidates are weak or irrelevant, keep Pass 1 wording."
            ),
        )
        _write_gemini_prompt_file(
            "gemini_prompt_pass2_retry.txt",
            "Pass 2 (OCR-guided rewrite, retry)",
            retry_prompt,
        )
        try:
            gres = _gemini_generate(
                retry_prompt, timeout=timeout, pass_num=2, temperature=p2_temp
            )
            raw = (gres.text if gres else "") or ""
            prompt = retry_prompt
        except Exception as e:
            print(f"[GEMINI] Pass 2 retry failed: {e}")
            _append_gemini_debug_pass2(retry_prompt, f"ERROR: {e}", "PASS 2 — OCR-guided rewrite retry")
            return pass1_messages, contact_name, _pass2_meta(False, reason="retry_request_failed")
        if not raw.strip():
            print("[GEMINI] Pass 2 empty after retry — keeping Pass 1 transcript")
            _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite retry")
            return pass1_messages, contact_name, _pass2_meta(
                False, gres, "empty_after_retry"
            )

    try:
        name2, msgs2, _ledger2 = _parse_gemini_full_vision_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 2 JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name, _pass2_meta(False, gres, "json_parse_failed")

    if not msgs2 or len(msgs2) != len(pass1_messages):
        _pv(
            f"[GEMINI] Pass 2 message count mismatch ({len(msgs2)} vs {len(pass1_messages)}) — "
            "keeping Pass 1 transcript"
        )
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name, _pass2_meta(
            False, gres, "message_count_mismatch"
        )

    role_drift_indices = []
    for i, m in enumerate(msgs2):
        r2 = _canonicalize_gemini_role(m.get("role") or "")
        if r2 not in ("contact", "user", "system"):
            r2 = "system"
        if r2 != pass1_messages[i]["role"]:
            role_drift_indices.append(i)

    if role_drift_indices:
        _append_gemini_debug_pass2(
            prompt,
            raw + f"\n[ROLE DRIFT IGNORED: {role_drift_indices}]\n",
            "PASS 2 — OCR-guided rewrite",
        )
        _pv(
            f"[GEMINI] Pass 2 role drift at indices {role_drift_indices[:8]} "
            f"(keeping Pass 1 roles, using Pass 2 text)"
        )

    merged = [
        {
            "role": pass1_messages[i]["role"],
            "text_src": (
                msgs2[i].get("text_src")
                or pass1_messages[i].get("text_src")
                or pass1_messages[i].get("text_en")
                or ""
            ).strip(),
            "text_en": "",
        }
        for i in range(len(pass1_messages))
    ]

    if ocr_match_verified is not None:
        for i in range(len(pass1_messages)):
            if not ocr_match_verified[i]:
                merged[i]["text_src"] = (
                    pass1_messages[i].get("text_src") or pass1_messages[i].get("text_en") or ""
                ).strip()

    final_name = (name2 or contact_name or "").strip() or contact_name
    if not role_drift_indices:
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
    return merged, final_name, _pass2_meta(True, gres)


def _default_status_bar_info(contact_hint: str) -> dict:
    """Shape matches :func:`_gemini_status_bar_pass` output for render pipeline."""
    return {
        "contact_name": (contact_hint or "").strip(),
        "status_text": "",
        "avatar_image_index": 0,
        "avatar_bbox": None,
    }


def _status_bar_info_from_pass3_merged_response(data: Optional[dict], contact_hint: str) -> dict:
    """Read header fields from Pass 3 merged JSON (same request as reference resolution)."""
    if not isinstance(data, dict):
        return _default_status_bar_info(contact_hint)
    name = str(
        data.get("header_contact_name")
        or data.get("header_name")
        or ""
    ).strip()
    st = str(
        data.get("header_status_text")
        or data.get("header_status")
        or ""
    ).strip()
    bbox = data.get("header_avatar_bbox")
    if bbox is None:
        bbox = data.get("avatar_bbox")
    cleaned = None
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            cleaned = [int(bbox[i]) for i in range(4)]
        except (TypeError, ValueError):
            cleaned = None
    fb = (contact_hint or "").strip()
    return {
        "contact_name": name or fb,
        "status_text": st,
        "avatar_image_index": 0,
        "avatar_bbox": cleaned,
    }


def _pass3_fallback_messages_from_pass2(pass2_messages: list) -> list:
    """Final chat rows when Pass 3 is skipped or fails: English from ``translate_to_en(text_src)``."""
    return [
        {
            "role": m.get("role"),
            "text_src": (m.get("text_src") or "").strip(),
            "text_en": translate_to_en(m.get("text_src") or "") or (m.get("text_src") or ""),
        }
        for m in (pass2_messages or [])
    ]


def _gemini_reference_resolution_pass(
    contact_name: str,
    pass2_messages: list,
    timeout: int,
    status_bar_b64: Optional[str] = None,
):
    """Pass 3: reference resolution + final English; optionally same call reads one status-bar crop."""
    def _pass3_meta(
        applied: bool, gres: Optional[GeminiApiResult] = None, reason: str = ""
    ) -> Dict[str, Any]:
        gmeta = (getattr(gres, "meta", None) or {}) if gres is not None else {}
        return {
            "applied": bool(applied),
            "successful_attempt": gmeta.get("successful_attempt"),
            "status": gmeta.get("status"),
            "reason": reason or None,
        }

    if not pass2_messages:
        return (
            pass2_messages,
            contact_name,
            [],
            _default_status_bar_info(contact_name),
            _pass3_meta(False, reason="missing_pass2_messages"),
        )

    def _normalize_reference_text(text: str) -> list:
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    def _has_visible_ambiguity_markers(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        if re.search(r"\b\w+\s*/\s*\w+\b", t):
            return True
        if re.search(r"\b(i|you|he|she|we|they|me|him|her|my|your|his|their|our|us)\s+or\s+(i|you|he|she|we|they|me|him|her|my|your|his|their|our|us)\b", t):
            return True
        if re.search(r"\[[^\]]*\]", t):
            return True
        return False

    def _choose_reference_english(literal_gloss: str, resolved_text: str, confidence: str, ambiguity: str) -> tuple[str, str]:
        literal = (literal_gloss or "").strip()
        resolved = (resolved_text or "").strip()
        conf = (confidence or "").strip().lower()
        amb = (ambiguity or "").strip().lower()
        if _has_visible_ambiguity_markers(resolved):
            return literal or resolved, "literal_resolved_visible_ambiguity"
        if _has_visible_ambiguity_markers(literal):
            return resolved or literal, "resolved_literal_visible_ambiguity"
        if not resolved:
            return literal, "literal_missing_resolved"
        if conf != "high":
            return literal or resolved, "literal_not_high_confidence"
        if amb not in ("", "none", "low"):
            return literal or resolved, "literal_ambiguous"

        lit_tokens = _normalize_reference_text(literal)
        res_tokens = _normalize_reference_text(resolved)
        if lit_tokens and res_tokens:
            lit_set = set(lit_tokens)
            res_set = set(res_tokens)
            overlap = len(lit_set & res_set) / max(1, len(lit_set | res_set))
            length_ratio = len(res_tokens) / max(1, len(lit_tokens))
            if overlap < 0.45 or length_ratio > 1.45:
                return literal or resolved, "literal_resolved_too_different"
        return resolved, "resolved_high_confidence"

    payload_messages = []
    for i, m in enumerate(pass2_messages):
        src = (m.get("text_src") or m.get("text_en") or "").strip()
        prev_src = [
            (pass2_messages[j].get("text_src") or pass2_messages[j].get("text_en") or "").strip()
            for j in range(max(0, i - 2), i)
        ]
        next_src = [
            (pass2_messages[j].get("text_src") or pass2_messages[j].get("text_en") or "").strip()
            for j in range(i + 1, min(len(pass2_messages), i + 3))
        ]
        payload_messages.append({
            "message_index": i,
            "role": m.get("role"),
            "text_src": src,
            "prev_context_src": prev_src,
            "next_context_src": next_src,
        })

    payload = {
        "contact_name": contact_name,
        "messages": payload_messages,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    expected_n = len(pass2_messages)

    _header_section = ""
    _header_json_suffix = ""
    if status_bar_b64:
        _header_section = """
## Attached image: Messenger status/header bar
The **first attached image** (JPEG) is a **crop of the top status/header bar** from the **first** screenshot of this chat. It is **not** a message bubble.

Add these keys to the **same** root JSON object as `contact_name` and `messages`. Fill them **only** from that image. They must **not** change your reference-resolution or translation work on `messages` in any way.
- `header_contact_name` (string): chat title or contact name visible in the bar — empty if unreadable.
- `header_status_text` (string): short status line (Active now, Online, last seen, etc.) — empty if none.
- `header_avatar_bbox` (array of four integers, or `null`): pixel box `[x1,y1,x2,y2]` **within that crop** around the circular profile photo, or `null` if unclear.
"""
        _header_json_suffix = (
            ',"header_contact_name":"<string>","header_status_text":"<string>",'
            '"header_avatar_bbox":[x1,y1,x2,y2]'
        )

    _p3_intro = "You are PASS 3 of 3.\n" + (_header_section + "\n" if status_bar_b64 else "")

    prompt = f"""{_p3_intro}You receive a chat transcript after OCR correction. Your task is to translate it to English and resolve references only when the context clearly supports it.

Input JSON:
{payload_json}

Rules:
- Keep the same number of messages: {expected_n}.
- Keep the same order.
- Keep the same role for each message.
- Use surrounding messages to resolve who is speaking about whom.
- Distinguish carefully between first person, second person, and third person references.
- The message `role` only tells you who sent the bubble; it does not by itself tell you who the omitted subject refers to.
- If the source is ambiguous, preserve that ambiguity in English instead of guessing.
- Do not invent a subject if it is unclear.
- Prefer neutral English like "Coming now." or "Call me back." when that is safer than a wrong explicit subject.
- Do not assume the omitted subject in a message automatically refers to the speaker. It may refer to the other person or a third party.
- Never show uncertainty as visible alternatives in the final English.
- Do not output forms like `I/you`, `he/she`, `bring it / bring change`, bracketed options, or multiple candidate readings.
- If you are not confident, keep the wording general and natural instead of exposing ambiguity.
- `text_src` should remain the corrected source-language message.
- `literal_gloss_en` should be a cautious, close English gloss of `text_src`.
- `resolved_text_en` should be the English after minimal reference resolution.
- `resolved_text_en` must preserve the same message content as `literal_gloss_en`.
- Only make minimal reference adjustments such as clarifying an omitted subject, object, or addressee when the context is truly clear.
- Do not add new actions, motives, explanations, emotional tone, or extra detail that is not already in the source.
- Do not rewrite the sentence more fluently if that changes wording beyond the reference clarification itself.
- Use `resolution_confidence` = `high|medium|low`.
- Only use `high` when the surrounding context clearly supports the reference interpretation.
- If the change would require more than minimal pronoun/subject clarification, use `medium` or `low`.
- If not high confidence, keep `resolved_text_en` the same as or extremely close to `literal_gloss_en`.

For each message, also return:
- `subject_role`: `contact|user|third_party|unknown`
- `target_role`: `contact|user|third_party|unknown`
- `ambiguity`: `none|low|medium|high`
- `note`: short explanation if useful

Output JSON only:
{{"contact_name":"<string>","messages":[{{"message_index":0,"role":"contact|user|system","text_src":"<source reading>","literal_gloss_en":"<cautious English gloss>","resolved_text_en":"<reference-resolved English>","resolution_confidence":"high|medium|low","subject_role":"contact|user|third_party|unknown","target_role":"contact|user|third_party|unknown","ambiguity":"none|low|medium|high","note":"<short note>"}}]{_header_json_suffix}}}"""

    _pass3_dbg = "PASS 3 — Reference + header (merged)" if status_bar_b64 else "PASS 3 — Reference resolution"
    _write_gemini_prompt_file(
        "gemini_prompt_pass3_reference_header.txt" if status_bar_b64 else "gemini_prompt_pass3_reference.txt",
        "Pass 3 (reference + header)" if status_bar_b64 else "Pass 3 (reference resolution)",
        prompt,
    )
    if _compact_verbose_logs():
        print("", flush=True)
        print("*******Pass 3 — SPEAKER REFERENCE******", flush=True)

    def _pass3_fallback_rows():
        return _pass3_fallback_messages_from_pass2(pass2_messages)

    img_list = [status_bar_b64] if status_bar_b64 else None
    try:
        gres = _gemini_generate(
            prompt,
            image_b64_list=img_list,
            timeout=timeout,
            pass_num=3,
        )
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 3 reference failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}", _pass3_dbg)
        return (
            _pass3_fallback_messages_from_pass2(pass2_messages),
            contact_name,
            [],
            _default_status_bar_info(contact_name),
            _pass3_meta(False, reason="request_failed"),
        )

    if not raw.strip():
        _pv("[GEMINI] Pass 3 reference empty — keeping Pass 2 translation fallback")
        _append_gemini_debug_pass2(prompt, raw, _pass3_dbg)
        return (
            _pass3_fallback_rows(),
            contact_name,
            [],
            _default_status_bar_info(contact_name),
            _pass3_meta(False, gres, "empty_response"),
        )

    try:
        payload_text = _extract_json_object(raw) or raw.strip()
        data = json.loads(payload_text)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 3 reference JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw, _pass3_dbg)
        return (
            _pass3_fallback_rows(),
            contact_name,
            [],
            _default_status_bar_info(contact_name),
            _pass3_meta(False, gres, "json_parse_failed"),
        )

    name2 = (data.get("contact_name") or contact_name or "").strip() or contact_name
    msgs2 = data.get("messages") or []
    if not isinstance(msgs2, list) or len(msgs2) != len(pass2_messages):
        _pv(
            f"[GEMINI] Pass 3 reference message count mismatch ({len(msgs2)} vs {len(pass2_messages)}) — "
            "keeping Pass 2 translation fallback"
        )
        _append_gemini_debug_pass2(prompt, raw, _pass3_dbg)
        return (
            _pass3_fallback_rows(),
            contact_name,
            [],
            _status_bar_info_from_pass3_merged_response(data, contact_name),
            _pass3_meta(False, gres, "message_count_mismatch"),
        )

    status_info = _status_bar_info_from_pass3_merged_response(data, name2 or contact_name)

    role_drift_indices = []
    resolved = []
    debug_rows = []
    for i, m in enumerate(msgs2):
        if not isinstance(m, dict):
            continue
        r2 = _canonicalize_gemini_role(m.get("role") or "")
        if r2 not in ("contact", "user", "system"):
            r2 = "system"
        if r2 != pass2_messages[i]["role"]:
            role_drift_indices.append(i)
        text_src = (
            m.get("text_src")
            or pass2_messages[i].get("text_src")
            or pass2_messages[i].get("text_en")
            or ""
        ).strip()
        literal_gloss_en = (m.get("literal_gloss_en") or "").strip()
        resolved_text_en = (m.get("resolved_text_en") or m.get("text_en") or "").strip()
        resolution_confidence = (m.get("resolution_confidence") or "").strip().lower()
        fallback_literal = translate_to_en(text_src) or text_src
        ambiguity = (m.get("ambiguity") or "").strip()
        chosen_en, choice_reason = _choose_reference_english(
            literal_gloss_en or fallback_literal,
            resolved_text_en,
            resolution_confidence,
            ambiguity,
        )
        if _has_visible_ambiguity_markers(chosen_en):
            chosen_en = fallback_literal
            choice_reason = "fallback_literal_hidden_ambiguity"
        if not chosen_en:
            chosen_en = fallback_literal
        resolved.append({
            "role": pass2_messages[i]["role"],
            "text_src": text_src,
            "text_en": _prefer_english_surface(chosen_en, fallback_literal),
        })
        debug_rows.append({
            "message_index": i,
            "role": pass2_messages[i]["role"],
            "text_src": text_src,
            "literal_gloss_en": literal_gloss_en,
            "resolved_text_en": resolved_text_en,
            "chosen_text_en": chosen_en,
            "choice_reason": choice_reason,
            "resolution_confidence": resolution_confidence or "low",
            "subject_role": (m.get("subject_role") or "unknown").strip(),
            "target_role": (m.get("target_role") or "unknown").strip(),
            "ambiguity": ambiguity,
            "note": (m.get("note") or "").strip(),
        })

    if role_drift_indices:
        _append_gemini_debug_pass2(
            prompt,
            raw + f"\n[ROLE DRIFT IGNORED: {role_drift_indices}]\n",
            _pass3_dbg,
        )
    else:
        _append_gemini_debug_pass2(prompt, raw, _pass3_dbg)
    return resolved, name2, debug_rows, status_info, _pass3_meta(True, gres)


def _parse_gemini_status_bar_json(raw: str) -> dict:
    payload_text = _extract_json_object(raw or "") or (raw or "").strip()
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        return {}

    name = str(
        payload.get("contact_name")
        or payload.get("speaker_name")
        or payload.get("name")
        or ""
    ).strip()
    status_text = str(
        payload.get("status_text")
        or payload.get("speaker_status")
        or payload.get("status")
        or ""
    ).strip()

    avatar_image_index = payload.get("avatar_image_index")
    try:
        avatar_image_index = int(avatar_image_index)
    except (TypeError, ValueError):
        avatar_image_index = 0

    avatar_bbox = payload.get("avatar_bbox")
    if not (isinstance(avatar_bbox, list) and len(avatar_bbox) >= 4):
        avatar_bbox = None
    else:
        cleaned = []
        for value in avatar_bbox[:4]:
            try:
                cleaned.append(int(value))
            except (TypeError, ValueError):
                cleaned.append(0)
        avatar_bbox = cleaned

    return {
        "contact_name": name,
        "status_text": status_text,
        "avatar_image_index": avatar_image_index,
        "avatar_bbox": avatar_bbox,
    }


def _gemini_status_bar_pass(
    status_bar_images,
    contact_hint: str,
    timeout: int,
):
    """Pass 4: extract final header information from status-bar crops."""
    if not status_bar_images:
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    img_b64_list = []
    for img in list(status_bar_images or []):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        b64 = _jpeg_b64_from_bgr(img, quality=92)
        if b64:
            img_b64_list.append(b64)

    if not img_b64_list:
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    prompt = f"""You are PASS 4 of 4.

These images are the top Messenger header/status-bar crops from the same conversation, in chronological order.

Task:
- Infer the best contact name shown in the header.
- Infer the short status text if visible, such as Active now, Online, last active text, or similar.
- Choose the image with the clearest profile icon and return its 0-based image index.
- Return the approximate avatar bounding box within that chosen crop as [x1, y1, x2, y2].

Guidelines:
- Prefer text actually visible in the status-bar crops.
- Use "{(contact_hint or '').strip()}" only as a weak fallback hint for the name.
- If the status text is not readable, return an empty string.
- If the avatar is unclear, still choose the best image and return your best approximate bounding box around the visible avatar circle.

Return JSON only:
{{"contact_name":"<string>","status_text":"<string>","avatar_image_index":0,"avatar_bbox":[x1,y1,x2,y2]}}"""

    _write_gemini_prompt_file("gemini_prompt_pass4_statusbar.txt", "Pass 4 status bar", prompt)

    try:
        gres = _gemini_generate(
            prompt, image_b64_list=img_b64_list, timeout=timeout, pass_num=4
        )
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 4 status-bar failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}", "PASS 4 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    if not raw.strip():
        _append_gemini_debug_pass2(prompt, raw, "PASS 4 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    try:
        info = _parse_gemini_status_bar_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 4 status-bar JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw, "PASS 4 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    if not info.get("contact_name"):
        info["contact_name"] = (contact_hint or "").strip()
    _append_gemini_debug_pass2(prompt, raw, "PASS 4 — Status bar extraction")
    return info


def _append_gemini_debug_pass2(prompt: str, response: str, label: str = "PASS 3 — FINAL ENGLISH TRANSLATION (text-only)"):
    path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(label + "\n")
            f.write("=" * 60 + "\n\n")
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(response or "(empty)\n")
    except Exception as exc:
        _pv(f"[GEMINI] Could not append pass-2 debug: {exc}")


_ROLE_TO_RENDER_TYPE = {
    "contact": "receiver",
    "user": "sender",
    "system": "timestamp",
}


def _meta_from_gemini_messages(gemini_messages):
    """Build render_chat-compatible dicts with synthetic bbox + order."""
    out = []
    for i, gm in enumerate(gemini_messages):
        role = _canonicalize_gemini_role(gm.get("role") or "")
        if role not in ("contact", "user", "system"):
            role = "system"
        typ = _ROLE_TO_RENDER_TYPE.get(role, "timestamp")
        src = (gm.get("text_src") or gm.get("text_en") or "").strip()
        text_en = _prefer_english_surface(
            gm.get("text_en") or "",
            translate_to_en(src) or src,
        )
        out.append({
            "type": typ,
            "bbox": [0, i * 40, 400, i * 40 + 35],
            "text_th": "",
            "text_en": text_en,
            "order": i,
            "ocr_source": "gemini_full_vision",
            "ocr_span_count": 0,
            "ocr_validated": True,
            "ocr_trust_score": 1.0,
            "ocr_low_confidence": False,
            "ocr_reasons": [],
        })
    return out


def _write_gemini_debug_vision_only(prompt, response, footer="", api_payload=None):
    path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("GEMINI — stitched chat transcription / OCR rewrite / translation\n\n")
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(response or "(empty)\n")
            if api_payload is not None:
                f.write("\n\n=== GEMINI API (parsed JSON) ===\n")
                try:
                    f.write(json.dumps(api_payload, indent=2, ensure_ascii=False))
                except (TypeError, ValueError):
                    f.write(repr(api_payload))
                f.write("\n")
            if footer:
                f.write("\n=== NOTE ===\n")
                f.write(footer)
        _pv(f"[GEMINI] Debug file → {path}")
    except Exception as exc:
        _pv(f"[GEMINI] Could not write debug file: {exc}")


def translate_conversation_gemini_multimodal(
    page_images,
    combined_img,
    pass1_message_images=None,
    pass1_role_hints=None,
    contact_hint="",
    source_language_name=None,
    ocr_hints=None,
    ocr_hints_format="none",
    precomputed_stitch_bands=None,
    precomputed_stitch_meta=None,
    pass1_exclude_ocr_hints: bool = False,
    craft_expected_message_rows: Optional[int] = None,
    craft_vertical_bands_markdown: Optional[str] = None,
    ocr_pass2_by_message: Optional[str] = None,
):
    """Chat images → English (page-image Pass 1, crop transcript Pass 2, fusion Pass 3).

    Returns ``(success, contact_name, meta_list_for_render_chat, pre_ocr_english_meta_for_render_chat, pass_debug)``.
    """
    def _pass1_failure(
        reason: str,
        contact_name_out: str = "",
        pre_ocr_meta_out=None,
        extra: Optional[dict] = None,
    ):
        payload = {"failure_reason": reason}
        if extra:
            payload.update(extra)
        return False, contact_name_out, [], list(pre_ocr_meta_out or []), payload

    if not _gemini_discover_if_needed():
        return _pass1_failure("gemini_not_configured")

    timeout = gemini_pass_timeout_sec(1)
    b64_list = []
    pass1_message_images = list(pass1_message_images or [])
    for page_idx, img in enumerate(list(page_images or [])):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        b64 = _jpeg_b64_from_bgr(img, quality=90)
        if b64:
            b64_list.append(b64)
            _pv(
                f"[GEMINI] Pass 1 page {page_idx + 1}/{len(page_images)}: "
                f"{img.shape[1]}x{img.shape[0]}px ({len(b64) // 1024}KB base64)"
            )

    if b64_list and _compact_verbose_logs():
        print("", flush=True)

    if not b64_list:
        print("[GEMINI] No page images to send")
        return _pass1_failure("no_page_images")

    intro = "these are images representing a conversation on facebook messenger between two speakers"
    hint = (contact_hint or "").strip()
    _n_exp = int(craft_expected_message_rows) if craft_expected_message_rows is not None else 0
    craft_section = (
        f"i count {_n_exp} message bubbles in total\n\n"
        if _n_exp > 0
        else ""
    )
    bubble_section = (
        f"{craft_vertical_bands_markdown.strip()}\n\n"
        if craft_vertical_bands_markdown and str(craft_vertical_bands_markdown).strip()
        else ""
    )
    _src_hint = (source_language_name or "").strip()
    _src_lang_section = (
        f"source language hint: message bubbles are expected to be in {_src_hint}. "
        "put each bubble's text in text_src in that language; if the image clearly shows another language, use what you see.\n\n"
        if _src_hint
        else ""
    )
    prompt = f"""{intro}

{_src_lang_section}transcribe the conversation in the original language
the first image is the oldest

{craft_section}{bubble_section}transcribe what you see
the added information above is guidance about the actual text bubbles only
for actual chat bubbles, map receiver -> role "contact" and sender -> role "user"
keep the actual chat bubbles in that same top-to-bottom order
if you clearly see timestamps, missed calls, transfer notices, or other centered/system conversation rows, include them too as role "system"
those system rows are metadata only and must not change the receiver/sender bubble order above
for role "system" only you may add "event_timestamp": the exact time/date line shown for that row in the UI if any (e.g. under a call card); otherwise ""

return json in this structure:
{{"contact_name":"<header or empty>","messages":[{{"role":"contact|user|system","text_src":"<original-language transcription>","event_timestamp":""}}]}}

output json only."""

    _write_gemini_prompt_file("gemini_prompt_pass1.txt", "Pass 1 (vision)", prompt)
    if _compact_verbose_logs():
        print("", flush=True)

    pass1_started = time.time()

    def _finish_warn(pay):
        fr = _gemini_candidate_finish_reason(pay)
        if fr == "MAX_TOKENS" and not _compact_verbose_logs():
            _pv(
                f"[GEMINI] Pass 1 finishReason={fr!r} — if JSON fails, internal "
                "thinking may have consumed the output token budget (see PIPELINE.md)."
            )
        return fr

    try:
        gres = _gemini_generate(
            prompt,
            image_b64_list=b64_list,
            timeout=timeout,
            pass_num=1,
        )
        raw = (gres.text if gres else "") or ""
        api_payload = gres.response_json if gres else None
    except Exception as e:
        print(f"[GEMINI] Request failed: {e}")
        _write_gemini_debug_vision_only(prompt, f"ERROR: {e}", "")
        return _pass1_failure(
            "request_failed",
            hint or "",
            extra={"exception": str(e)},
        )

    if not raw or not raw.strip():
        print("[GEMINI] Empty response from full-vision translation")
        _write_gemini_debug_vision_only(prompt, raw or "", "", api_payload=api_payload)
        return _pass1_failure("empty_response", hint or "")

    _finish_warn(api_payload)
    parse_err = None
    try:
        gname, gmsgs, amb_ledger = _parse_gemini_full_vision_json(raw)
    except json.JSONDecodeError as e:
        parse_err = e
        gname, gmsgs, amb_ledger = "", [], []

    if parse_err is not None:
        print(f"[GEMINI] JSON parse failed: {parse_err}")
        _write_gemini_debug_vision_only(
            prompt, raw, f"JSON ERROR: {parse_err}", api_payload=api_payload
        )
        return _pass1_failure(
            "json_parse_failed",
            hint or "",
            extra={"parse_error": str(parse_err)},
        )

    gmsgs = _filter_pass1_messages(gmsgs)
    pass1_system_msgs = []
    pass1_chat_msgs = []
    chat_seen = 0
    for m in gmsgs:
        role = _canonicalize_gemini_role(m.get("role") or "")
        text_src = (m.get("text_src") or m.get("text_en") or "").strip()
        if not text_src:
            continue
        if role == "system":
            pass1_system_msgs.append({
                "role": "system",
                "text_src": text_src,
                "insert_before_chat_index": chat_seen,
                "event_timestamp": (m.get("event_timestamp") or "").strip(),
            })
            continue
        mm = dict(m)
        mm["role"] = role
        pass1_chat_msgs.append(mm)
        chat_seen += 1

    non_system_count = len(pass1_chat_msgs)
    system_count = len(pass1_system_msgs)

    if _n_exp > 0 and non_system_count != _n_exp:
        _pv(
            f"[GEMINI] Pass 1 bubble count mismatch: expected {_n_exp}, got {non_system_count} "
            f"(plus {system_count} system row(s))."
        )
    elif _n_exp > 0:
        _pv(f"[GEMINI] Pass 1 bubble count OK: {_n_exp} bubbles, {system_count} system row(s).")

    contact_name = gname or hint or ""
    if not pass1_chat_msgs:
        print("[GEMINI] Parsed JSON but no messages")
        _write_gemini_debug_vision_only(prompt, raw, "NO MESSAGES PARSED", api_payload=api_payload)
        return _pass1_failure(
            "no_messages_parsed",
            contact_name,
            extra={
                "raw_message_count": len(gmsgs),
                "system_message_count": system_count,
            },
        )

    pass1_footer = json.dumps(
        {
            "contact_name": contact_name,
            "n_messages": len(gmsgs),
            "bubble_messages": non_system_count,
            "system_messages": system_count,
            "ambiguity_ledger": amb_ledger,
            "has_ref_placeholders": _conversation_has_ref_placeholders(pass1_chat_msgs),
        },
        indent=2,
        ensure_ascii=False,
    )
    _write_gemini_debug_vision_only(prompt, raw, pass1_footer, api_payload=api_payload)

    if not _compact_verbose_logs():
        _pv(f"[GEMINI] Pass 1 (vision): {len(gmsgs)} rows total")
        _pv(f"[TIMER] Pass 1 Gemini in {time.time()-pass1_started:.1f}s")
    pass1_source_msgs = [
        {
            "role": m.get("role"),
            "text_en": ((m.get("text_src") or m.get("text_en") or "").strip()),
            "text_src": (m.get("text_src") or m.get("text_en") or "").strip(),
        }
        for m in pass1_chat_msgs
    ]
    pre_ocr_meta = _meta_from_gemini_messages(
        [
            {
                "role": m.get("role"),
                "text_en": _prefer_english_surface(
                    translate_to_en(m.get("text_src") or ""), m.get("text_src") or ""
                ),
            }
            for m in pass1_source_msgs
        ]
    )
    pass2_effective_msgs = []
    pass2_effective_count = 0
    pass2_meta = {
        "enabled": False,
        "skipped": True,
        "reason": "ocr_hints_only_mode",
        "input_messages": len(pass1_source_msgs),
        "input_crops": 0,
        "output_messages": 0,
    }
    final_msgs = [
        {
            "role": m.get("role"),
            "text_src": (m.get("text_src") or "").strip(),
            "text_en": _prefer_english_surface(
                translate_to_en(m.get("text_src") or ""),
                m.get("text_src") or "",
            ),
        }
        for m in pass1_source_msgs
    ]

    meta = _meta_from_gemini_messages(final_msgs)
    if not meta:
        return _pass1_failure(
            "meta_build_empty",
            contact_name,
            pre_ocr_meta_out=pre_ocr_meta,
            extra={
                "pass1_count": len(pass1_source_msgs),
                "pass1_total_count": len(pass1_source_msgs),
                "pass1_system_count": system_count,
                "pass2_count": len(pass2_effective_msgs),
                "pass2_effective_count": pass2_effective_count,
                "pass2_crop_refine": pass2_meta,
                "pass3_count": 0,
                "first_pass_only": False,
                "third_pass_skipped": True,
                "pass1_messages": pass1_source_msgs,
                "pass1_system_messages": pass1_system_msgs,
                "pass2_crop_messages": pass2_effective_msgs,
                "pass3_messages": [],
            },
        )

    if _compact_verbose_logs():
        print(
            f"\n*************[GEMINI] Pass 1 final output OK → {len(meta)} chat rows, "
            f"contact={contact_name!r} ***********",
            flush=True,
        )
    else:
        _pv(f"[GEMINI] Pass 1 final output OK → {len(meta)} chat rows, contact={contact_name!r}")
    return True, contact_name, meta, pre_ocr_meta, {
        "pass1_count": non_system_count,
        "pass1_total_count": len(pass1_source_msgs),
        "pass1_system_count": system_count,
        "pass2_count": len(pass2_effective_msgs),
        "pass2_effective_count": pass2_effective_count,
        "pass2_crop_refine": pass2_meta,
        "pass3_count": 0,
        "first_pass_only": False,
        "third_pass_skipped": True,
        "pass1_messages": pass1_source_msgs,
        "pass1_system_messages": pass1_system_msgs,
        "pass2_crop_messages": pass2_effective_msgs,
        "pass3_messages": [],
    }


def _contact_ocr_unreliable_for_prompt(name: str) -> bool:
    """True when status-bar OCR is clearly junk — do not put it in the Gemini prompt."""
    if not name or not str(name).strip():
        return True
    n = str(name).strip()
    if n.lower() in {"person a", "unknown"}:
        return False
    if any(ord(c) > 0x024F for c in n if not c.isspace()):
        return True
    if "ชั่วโมง" in n:
        return True
    if "...." in n or (n.count(".") >= 3 and len(n) > 15):
        return True
    return False


def _prompt_person_a_side_label(contact_name: str, has_image: bool) -> str:
    if has_image and _contact_ocr_unreliable_for_prompt(contact_name):
        return "Person A (LEFT grey bubbles)"
    return (contact_name or "Person A").strip() or "Person A"


def _parse_gemini_slots_response(raw: str):
    """Extract NAME, CONTEXT, and slot_num → (is_timestamp, english_text) from Gemini output."""
    gemini_name = ""
    context_line = ""
    for _cl in raw.splitlines():
        cl = _cl.strip()
        if not gemini_name and cl.upper().startswith("NAME:"):
            gemini_name = cl[len("NAME:"):].strip()
        if not context_line and cl.upper().startswith("CONTEXT:"):
            context_line = cl[len("CONTEXT:"):].strip()
        if gemini_name and context_line:
            break

    parsed = {}
    lines = raw.splitlines()
    for line_idx, line in enumerate(lines):
        line = line.strip()
        m = re.match(r"^(\d+)\.\s*\[(.*?)\]:\s*(.+)$", line)
        if not m:
            continue
        slot_num = int(m.group(1)) - 1
        label = m.group(2).strip().lower()
        captured = m.group(3).strip()
        is_ts = "timestamp" in label

        if not is_ts and _is_non_latin(captured):
            eng_match = re.search(
                r"(?:\*\s*English\s*:|→|->|=>)\s*([A-Za-z].+)$",
                captured, re.IGNORECASE
            )
            if eng_match:
                captured = eng_match.group(1).strip()
            elif line_idx + 1 < len(lines):
                next_line = lines[line_idx + 1].strip()
                eng_next = re.match(
                    r"^\*?\s*(?:English|Translation)\s*:\s*(.+)$",
                    next_line, re.IGNORECASE
                )
                if eng_next:
                    captured = eng_next.group(1).strip()
                elif not _is_non_latin(next_line) and next_line:
                    captured = next_line
                else:
                    continue

        if captured and (is_ts or not _is_non_latin(captured)):
            parsed[slot_num] = (is_ts, captured)

    return gemini_name, context_line, parsed


def _apply_gemini_parsed_to_objects(objects, slots, parsed):
    """Write parsed English lines back into *objects* using *slots* index mapping."""
    for slot_idx, (orig_idx, _, _th) in enumerate(slots):
        result = parsed.get(slot_idx)
        if result:
            is_ts, en = result
            objects[orig_idx]["text_en"] = en
            if is_ts:
                objects[orig_idx]["type"] = "timestamp"


def _gemini_recovery_pass(image_b64, draft_text: str, num_entries: int, person_a_label: str):
    """Second pass: compare draft to image and expand lines that omit visible detail."""
    prompt = (
        "The attached image is the same chat screenshot as before.\n"
        "Below is a DRAFT English translation (NAME, CONTEXT, and numbered lines).\n\n"
        "TASK — Compare each NUMBERED line to the matching bubble or timestamp in the image "
        "(same order, top to bottom). For any line where the visible text includes names, "
        "hotels, places, addresses, numbers, or clauses that are missing, shortened, or wrong "
        "in the draft, REVISE that line so the English fully reflects what is shown.\n\n"
        "If the draft CONTEXT or any line describes people, places, or events that do NOT appear "
        "in this image, replace them with what the image actually shows — do not preserve a wrong story.\n\n"
        "RULES:\n"
        "- Output ONLY English.\n"
        "- Do NOT add or remove numbered entries. Do NOT merge or split lines.\n"
        "- Keep the same speaker labels and numbers as in the draft.\n"
        "- Fix NAME:/CONTEXT: if they contradict the image; CONTEXT must only summarize this screenshot.\n"
        "- No commentary, no bullet list of changes — only the final transcript.\n\n"
        f"Expected entry count: {num_entries} numbered lines after CONTEXT.\n\n"
        "--- DRAFT ---\n"
        f"{draft_text.strip()}"
    )
    gres = _gemini_generate(prompt, image_b64=image_b64)
    text = gres.text if gres else None
    return text, prompt


def _append_gemini_debug_recovery_section(recovery_prompt, recovery_response):
    path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("GEMINI DEBUG — RECOVERY PASS (INPUT)\n")
            f.write("=" * 60 + "\n\n")
            f.write(recovery_prompt or "")
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("GEMINI DEBUG — RECOVERY PASS (RAW RESPONSE)\n")
            f.write("=" * 60 + "\n\n")
            f.write(recovery_response or "")
            f.write("\n")
    except OSError:
        pass


def refine_and_translate_with_gemini(objects, combined_img=None, contact_name="",
                                     raw_ocr_spans=None):
    """Send conversation structure + screenshot to Gemini; write English into *objects*.

    Strategy: **image-first** — the prompt uses a numbered scaffold (speaker roles only)
    so Gemini reads bubble text from pixels, not from noisy OCR strings.  A second
    **recovery** multimodal call (set ``GEMINI_RECOVERY_PASS=0`` to disable) compares
    the draft translation to the image again to expand omitted names, places, etc.

    *raw_ocr_spans* is accepted for API compatibility with *main.py*; it is not sent
    in the prompt (vision + recovery replace it).

    Returns ``(True, name)`` on success.
    """
    _ = raw_ocr_spans  # not sent in prompt; kept for caller compatibility
    if not objects:
        return False, ""

    if not _gemini_discover_if_needed():
        return False, ""

    # Encode combined image for multimodal input.
    # IMPORTANT: very tall stitched screenshots must NOT be scaled by max(h,w) alone —
    # that crushes width to ~100px and Gemini cannot read text (hallucinations / wrong topic).
    image_b64 = None
    if combined_img is not None:
        try:
            import base64 as _b64
            img_to_encode = _resize_image_for_gemini_vision(combined_img)
            _, buf = cv2.imencode(".jpg", img_to_encode,
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")
            _gw = img_to_encode.shape[1]
            _gh = img_to_encode.shape[0]
            _pv(f"[GEMINI] Image attached: {_gw}x{_gh}px "
                  f"({len(image_b64)//1024}KB base64)")
            if _gw < 600:
                _pv(
                    "[GEMINI] WARN: image width is very small — bubble text may be unreadable; "
                    "check combined_ocr_input.png and consider fewer pages per run."
                )
        except Exception as e:
            _pv(f"[GEMINI] Could not encode image, falling back to text-only: {e}")
            image_b64 = None

    # Display / contact name (may be junk from status-bar OCR)
    person_a_label = contact_name.strip() if contact_name.strip() else "Person A"
    person_b_label = "Person B (you)"

    has_image = image_b64 is not None
    prompt_a_label = _prompt_person_a_side_label(person_a_label, has_image)

    # Build slots — chat bubbles AND timestamps.  When an image is attached, include
    # every bubble even if OCR left text_th empty (Gemini reads from the image).
    slots = []
    for i, obj in enumerate(objects):
        th = (obj.get("text_th") or "").strip()
        obj_type = (obj.get("type") or "receiver").lower()
        if obj_type in {"status_bar", "bottom_artifact", "bottom_bar", "keyboard"}:
            continue
        if obj_type == "timestamp":
            continue
        if not has_image:
            if not th or _looks_like_link_text(th):
                continue
        if obj_type == "timestamp":
            role = "Timestamp"
        elif obj_type == "sender":
            role = person_b_label
        else:
            role = prompt_a_label
        slots.append((i, role, th))

    if not slots:
        return False, ""

    # Keep slot order = CRAFT → grouping → classify order (do NOT re-sort by bbox:
    # center-y sorts scramble left/right turns that sit on similar y).

    # Scaffold: do not inject noisy OCR text into the prompt (it anchors the model wrong).
    # With image, each line only names the role; Gemini reads content from pixels.
    _scaffold_hint = (
        "Read the matching bubble or timestamp row in the image; translate all visible text "
        "to natural English."
    )
    dialogue = "\n".join(
        f"{idx + 1}. [{role}]: {_scaffold_hint if has_image else th}"
        for idx, (_, role, th) in enumerate(slots)
    )

    lang_name = _source_lang_name
    if has_image and _contact_ocr_unreliable_for_prompt(person_a_label):
        person_a_desc = (
            "Person A is the other participant. In the screenshot their bubbles are on the "
            "LEFT (usually grey); yours (Person B) are on the RIGHT (usually pink). "
            "Read Person A's real name from the chat header for the NAME: line.\n"
        )
    else:
        person_a_desc = (
            f"Person A is '{person_a_label}' (the other person in this chat)."
            if person_a_label != "Person A"
            else "Person A is the other person."
        )

    primary_block = (
        "PRIMARY SOURCE — the attached chat screenshot.\n"
        "Read every bubble directly from the image. "
        f"Left-aligned bubbles are {prompt_a_label}; "
        "right-aligned bubbles are Person B (you). "
        "The image is the authority for spelling of names, hotels, addresses, and URLs.\n\n"
        if has_image else
        "No image attached — translate the source text in the structured list below.\n\n"
    )

    numbering_block = (
        "NUMBERING: Entries 1..N are in PIPELINE ORDER (the same order rows were detected, "
        "roughly top-to-bottom). Line N corresponds to the N-th chat row in that sequence — "
        "translate that bubble only. The speaker label is a hint; if it conflicts with left/right "
        "alignment for that row, trust the image alignment.\n\n"
        if has_image else ""
    )

    secondary_block = (
        numbering_block
        + f"SCAFFOLD — conversation structure only ({len(slots)} entries).\n"
        "Each line gives the speaker role and entry number. "
        + (
            "It does NOT include the message text — read each bubble from the image.\n\n"
            if has_image else
            "The text after each colon is the OCR extract to translate.\n\n"
        )
        + f"ENTRIES ({lang_name}, {len(slots)} lines):\n"
        + f"{dialogue}\n"
    )

    _step2a = (
        "Read the corresponding bubble or timestamp from the image "
        "(use the scaffold only for numbering and speaker side).\n"
        if has_image else
        "Use the source text after the colon on that scaffold line.\n"
    )
    _step3 = (
        "STEP 3 — Before you finish, scan the image top-to-bottom once more: "
        "if any bubble still has visible wording not reflected in your English line, "
        "revise that line. (A second automated pass may follow — be thorough.)\n\n"
        if has_image else
        "STEP 3 — Re-read your lines: ensure you did not drop names, numbers, or URLs "
        "from the source text.\n\n"
    )
    prompt = (
        f"You are translating a real {lang_name} two-person chat conversation to English.\n"
        f"{person_a_desc} Person B is the phone owner (you).\n\n"
        "WORK THROUGH THESE STEPS IN ORDER:\n\n"
        "STEP 1 — Examine the screenshot only (not guesses from memory).\n"
        "Write these two lines:\n"
        "NAME: [Person A's display name as shown in the chat header or bubbles — English/romanized; "
        "if unreadable write 'Person A']\n"
        "CONTEXT: [ONE sentence summarizing ONLY topics and facts that actually appear in THIS image — "
        "places, people, and plans visible in the bubbles. Do NOT invent a different trip, city, or "
        "names that are not readable in the screenshot.]\n\n"
        "STEP 2 — For each numbered entry:\n"
        "  a) "
        + _step2a
        + "  b) Translate ONLY the text visible in that bubble/timestamp — facts (names, hotels, boats, "
        "cities, URLs, numbers) must match what you see there.\n"
        + "  c) CONTEXT is for tone only; it must NOT override or replace what a bubble actually says. "
        "If CONTEXT conflicts with a bubble, ignore CONTEXT for that line.\n"
        + "  d) You may smooth grammar lightly, but do not substitute a different location, person, or "
        "story than the image shows.\n"
        + "  e) Output only the final English line for that number.\n\n"
        + _step3
        + "RULES:\n"
        + f"- Output ONLY English. Zero {lang_name} characters anywhere.\n"
        + "- NAME and CONTEXT lines must come first, before the numbered entries.\n"
        + "- Preserve speaker labels and entry numbers exactly as in the scaffold.\n"
        + "- [Timestamp] lines: readable English (e.g. 'Sun. 11:50 AM').\n"
        + "- Every entry must appear — do not skip or merge lines.\n"
        + "- No asterisks, footnotes, or extra blank lines.\n\n"
        + "OUTPUT FORMAT:\n"
        + "NAME: Phee Klaew\n"
        + "CONTEXT: brief summary here\n"
        + f"1. [{prompt_a_label}]: natural English text\n"
        + "2. [Person B (you)]: natural English text\n"
        + "3. [Timestamp]: Sun. 11:50 AM\n\n"
        + primary_block
        + secondary_block
    )

    # Write debug input file before the API call so it's always available
    _write_gemini_debug(prompt, None, slots)

    if image_b64:
        _pv(
            f"[GEMINI] Call 1: {len(slots)} scaffold rows — speaker role + order only "
            f"(no OCR text in prompt); bubble content read from image"
        )
    else:
        _pv(
            f"[GEMINI] Sending {len(slots)} scaffold rows with OCR text per line (no image)"
        )
    t_gemini_start = time.time()

    for attempt in range(3):
        try:
            t_attempt = time.time()
            gres = _gemini_generate(prompt, image_b64=image_b64)
            raw = (gres.text if gres else "") or ""
            if not raw:
                raise ValueError("Empty response")
            raw = raw.strip()
            elapsed = time.time() - t_attempt
            _pv(f"[GEMINI] Response received in {elapsed:.1f}s  ({len(raw)} chars)")

            gemini_name, context_line, parsed = _parse_gemini_slots_response(raw)
            if gemini_name:
                _pv(f"[GEMINI] Contact name identified: {gemini_name}")
            if context_line:
                _pv(f"[GEMINI] Conversation context: {context_line}")

            _pv(f"[GEMINI] Response preview:\n{raw[:500]}")

            if not parsed:
                raise ValueError(f"Could not parse any lines from response:\n{raw[:400]}")

            # Second multimodal pass: compare draft to pixels to restore omitted detail.
            # Disable with GEMINI_RECOVERY_PASS=0
            _recovery_on = (
                image_b64
                and os.environ.get("GEMINI_RECOVERY_PASS", "1").lower()
                not in ("0", "false", "no", "off")
            )
            if _recovery_on:
                _pv("[GEMINI] Recovery pass: comparing draft to image...")
                t_rec = time.time()
                rec_raw, recovery_prompt_text = _gemini_recovery_pass(
                    image_b64, raw, len(slots), person_a_label
                )
                if rec_raw and rec_raw.strip():
                    r2 = rec_raw.strip()
                    gn2, ctx2, parsed2 = _parse_gemini_slots_response(r2)
                    if len(parsed2) >= len(parsed):
                        raw = r2
                        gemini_name = gn2 or gemini_name
                        context_line = ctx2 or context_line
                        parsed = parsed2
                        _pv(
                            f"[GEMINI] Recovery pass applied in {time.time()-t_rec:.1f}s "
                            f"({len(parsed2)} slots)"
                        )
                    else:
                        _pv(
                            f"[GEMINI] Recovery pass skipped "
                            f"(draft {len(parsed)} slots vs recovery {len(parsed2)})"
                        )
                    _append_gemini_debug_recovery_section(
                        recovery_prompt_text, rec_raw or ""
                    )
                else:
                    _pv("[GEMINI] Recovery pass returned empty — keeping draft")
                    _append_gemini_debug_recovery_section(
                        recovery_prompt_text, "(empty response)"
                    )

            _apply_gemini_parsed_to_objects(objects, slots, parsed)

            _pv(f"[GEMINI] Parsed {len(parsed)}/{len(slots)} items  "
                  f"(total Gemini time: {time.time()-t_gemini_start:.1f}s)")

            # Write debug file with the full round-trip
            _write_gemini_debug(prompt, raw, slots, parsed, context_line)
            return True, gemini_name

        except Exception as e:
            err = str(e)
            delay_match = re.search(r"(\d+)s", err)
            suggested = int(delay_match.group(1)) if delay_match else 0
            wait = max(suggested + 5, 15)
            if attempt < 2:
                _pv(f"[GEMINI] Attempt {attempt+1} failed ({err[:120]}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[GEMINI] All 3 attempts failed: {err[:200]}")
                _write_gemini_debug(prompt, f"ERROR: {err}", slots)

    return False, ""


def _write_gemini_debug(prompt, response, slots, parsed=None, context_line=""):
    """Write a human-readable debug file with the full Gemini round-trip."""
    debug_path = os.path.join(OUTPUT_DIR, "gemini_debug.txt")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("GEMINI DEBUG — INPUT PROMPT\n")
            f.write("=" * 60 + "\n\n")
            f.write(prompt)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("GEMINI DEBUG — RAW RESPONSE\n")
            f.write("=" * 60 + "\n\n")
            if response is None:
                f.write("(not yet received)\n")
            else:
                f.write(response)

            if parsed is not None:
                if context_line:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write("GEMINI CONVERSATION CONTEXT\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"  {context_line}\n")

                f.write("\n\n" + "=" * 60 + "\n")
                f.write("GEMINI DEBUG — PARSED MAPPING  (src → EN)\n")
                f.write("=" * 60 + "\n\n")
                for slot_idx, (orig_idx, role, src_text) in enumerate(slots):
                    result = parsed.get(slot_idx)
                    en = result[1] if result else "[NOT PARSED]"
                    f.write(f"  slot {slot_idx+1:>2}  [{role}]\n")
                    f.write(f"    SRC : {src_text}\n")
                    f.write(f"    EN  : {en}\n\n")

        _pv(f"[GEMINI] Debug file → {debug_path}")
    except Exception as exc:
        _pv(f"[GEMINI] Could not write debug file: {exc}")


def refine_thai_with_gemini(objects):
    """Kept for compatibility — calls the unified fix+translate function."""
    refine_and_translate_with_gemini(objects)
    return objects


def refine_conversation_with_gemini(messages):
    """Kept for compatibility — no longer used in main pipeline."""
    return messages


def translate_conversation_block(texts):
    """Translate a list of source-language strings as a single conversation block.

    Google Translate resolves ambiguous words much better when it can see
    the surrounding context. We join all messages with a unique separator, send
    one request, then split the result back.

    Returns a list of English strings the same length as *texts*.
    """
    if not texts:
        return []

    preserved = []  # indices that must not be sent to translator
    slots = []      # the texts that will be joined and sent
    slot_map = {}   # slot_index → original index

    for i, t in enumerate(texts):
        if not t or not t.strip() or _looks_like_link_text(t):
            preserved.append(i)
        else:
            slot_map[len(slots)] = i
            slots.append(t)

    results = [""] * len(texts)

    # Preserve non-Thai / link texts as-is
    for i in preserved:
        results[i] = texts[i]

    if not slots:
        return results

    joined = _BLOCK_SEP.join(slots)

    # Cache key for the whole block
    cache_key = f"__block__:{joined}"
    if cache_key in translation_cache:
        translated_block = translation_cache[cache_key]
    else:
        try:
            translated_block = translator.translate(joined)
            translation_cache[cache_key] = translated_block
        except Exception:
            translated_block = None

    if translated_block:
        parts = translated_block.split(_BLOCK_SEP.strip())
        # Separator may be translated or partially eaten; fall back gracefully
        if len(parts) != len(slots):
            # Try splitting on any variant of the separator
            for sep_variant in ["<<<MSG>>>", "<<MSG>>", "<MSG>", "MSG"]:
                parts = [p.strip() for p in translated_block.split(sep_variant) if p.strip()]
                if len(parts) == len(slots):
                    break
            else:
                parts = None  # fallback to per-message below

        if parts and len(parts) == len(slots):
            for slot_idx, en in enumerate(parts):
                orig_idx = slot_map[slot_idx]
                results[orig_idx] = en.strip()
            print(f"[TRANSLATE] Block translation succeeded ({len(slots)} messages)")
            return results

    # Fallback: translate each message individually
    print(f"[TRANSLATE] Block separator lost; falling back to per-message translation")
    for slot_idx, th in enumerate(slots):
        orig_idx = slot_map[slot_idx]
        results[orig_idx] = translate_to_en(th)

    return results

# --------------------------------------------------
# REGION EXTRACTION (UNCHANGED)
# --------------------------------------------------

def extract_region(img, rect, idx):
    x1, y1, x2, y2 = rect

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = max(12, int(min(width, height) * 0.22))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    crop = img[y1:y2, x1:x2].copy()

    debug_path = os.path.join(DEBUG_DIR, f"crop_{idx}.png")
    cv2.imwrite(debug_path, crop)

    return crop

# --------------------------------------------------
# OCR
# --------------------------------------------------

def _normalize_ocr_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


COMMON_FOREIGN_WORDS = {
    "a", "an", "and", "are", "at", "be", "call", "can", "come", "do", "for",
    "go", "good", "hello", "help", "hi", "i", "if", "in", "is", "it", "me",
    "message", "my", "no", "not", "now", "of", "ok", "okay", "on", "or",
    "please", "see", "send", "sorry", "thanks", "thank", "the", "this", "to",
    "today", "tomorrow", "transfer", "we", "what", "when", "where", "yes",
    "you", "your",
}
MAX_OCR_RETRIES = 2
_OCR_ENGINE_VALIDATED = False


def _contains_thai(text):
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


def _contains_latin(text):
    return any(("a" <= ch.lower() <= "z") for ch in text)


def _latin_tokens(text):
    return re.findall(r"[A-Za-z]+", text or "")


def _looks_like_known_foreign_text(text):
    tokens = [token.lower() for token in _latin_tokens(text)]
    if not tokens:
        return False
    return all(token in COMMON_FOREIGN_WORDS for token in tokens)


def _looks_like_link_text(text):
    lowered = (text or "").lower()
    return (
        "http" in lowered
        or "www." in lowered
        or ".com" in lowered
        or ".co/" in lowered
        or ".net" in lowered
        or ".org" in lowered
        or lowered.count("/") >= 2
    )


def _looks_like_wrong_language(text):
    compact = (text or "").replace(" ", "")
    if len(compact) < 3:
        return False
    if _contains_thai(compact):
        return False
    if not _contains_latin(compact):
        return False
    if _looks_like_known_foreign_text(text):
        return False
    return True


def _normalize_thai_spacing(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _thai_tokens(text, keep_whitespace=False):
    cleaned = _normalize_thai_spacing(text)
    if not cleaned:
        return []

    if word_tokenize is None:
        if keep_whitespace:
            return re.findall(r"\s+|[^\s]+", cleaned)
        return [token for token in re.split(r"\s+", cleaned) if token]

    try:
        return word_tokenize(cleaned, keep_whitespace=keep_whitespace)
    except TypeError:
        tokens = word_tokenize(cleaned)
        if keep_whitespace:
            return tokens
        return [token for token in tokens if token and not token.isspace()]


def _looks_like_suspicious_thai(text):
    compact = (text or "").replace(" ", "")
    if len(compact) < 3:
        return False
    if not _contains_thai(compact):
        return False
    thai_chars = sum(1 for ch in compact if "\u0e00" <= ch <= "\u0e7f")
    non_thai_chars = len(compact) - thai_chars
    return non_thai_chars >= max(2, thai_chars // 2)


def _thai_sentence_plausibility(text):
    cleaned = _normalize_thai_spacing(text)
    compact = cleaned.replace(" ", "")
    if not compact or not _contains_thai(compact):
        return 0.0

    tokens = [token for token in _thai_tokens(cleaned) if token and not token.isspace()]
    thai_tokens = [token for token in tokens if _contains_thai(token)]
    thai_chars = sum(1 for ch in compact if "\u0e00" <= ch <= "\u0e7f")
    non_thai_chars = max(0, len(compact) - thai_chars)

    thai_ratio = thai_chars / max(1, len(compact))
    meaningful_ratio = (
        sum(1 for token in thai_tokens if len(re.sub(r"\W+", "", token)) >= 2) / max(1, len(thai_tokens))
    )
    single_char_ratio = (
        sum(1 for token in thai_tokens if len(re.sub(r"\W+", "", token)) <= 1) / max(1, len(thai_tokens))
    )
    mixed_token_ratio = (
        sum(1 for token in thai_tokens if any(not ("\u0e00" <= ch <= "\u0e7f") for ch in token if ch.strip()))
        / max(1, len(thai_tokens))
    )
    punctuation_ratio = non_thai_chars / max(1, len(compact))

    score = (
        thai_ratio * 0.45
        + meaningful_ratio * 0.35
        + (1.0 - single_char_ratio) * 0.15
        - mixed_token_ratio * 0.20
        - punctuation_ratio * 0.10
    )
    return max(0.0, min(1.0, score))


def _correct_thai_text(text):
    cleaned = _normalize_thai_spacing(text)
    if not cleaned:
        return ""

    if thai_normalize is None or word_tokenize is None or thai_spell_correct is None:
        return cleaned

    normalized = thai_normalize(cleaned)
    tokens = _thai_tokens(normalized, keep_whitespace=True)
    corrected_tokens = []
    for token in tokens:
        if not token or token.isspace():
            corrected_tokens.append(token)
            continue
        if any("\u0e00" <= ch <= "\u0e7f" for ch in token):
            corrected_tokens.append(thai_spell_correct(token))
        else:
            corrected_tokens.append(token)
    corrected = "".join(corrected_tokens).strip()
    if corrected:
        corrected = _normalize_thai_spacing(corrected)
    return corrected or normalized


def _should_repair_thai_sentence(text, translated_text):
    compact = (text or "").replace(" ", "")
    if not compact or not _contains_thai(compact):
        return False
    if _looks_like_link_text(text):
        return False

    plausibility = _thai_sentence_plausibility(text)
    if _looks_like_suspicious_thai(text):
        return True
    if not translated_text and plausibility < 0.82:
        return True
    if plausibility < 0.58:
        return True
    return False

def _region_scale(region):
    h, w = region.shape[:2]
    short_side = min(h, w)
    if short_side <= 22:
        return 3.8
    if short_side <= 32:
        return 3.2
    if short_side <= 48:
        return 2.5
    if short_side <= 72:
        return 2.0
    return 1.6

def _prepare_base_region(region):
    scale = _region_scale(region)
    prepared = cv2.resize(
        region,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Mild denoising + sharpening improves cracked-screen captures
    prepared = cv2.bilateralFilter(prepared, 5, 30, 30)
    sharpened = cv2.addWeighted(
        prepared,
        1.22,
        cv2.GaussianBlur(prepared, (0, 0), 1.0),
        -0.22,
        0,
    )
    return sharpened


def _prepare_candidate_regions(region):
    base = _prepare_base_region(region)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(6, 6)).apply(gray)
    binary = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        9,
    )

    candidates = [("base", base)]
    h, w = region.shape[:2]
    short_side = min(h, w)
    if short_side <= 90 or gray.std() < 45:
        candidates.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))
    if short_side <= 56 or gray.std() < 32:
        candidates.append(("thresh", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))

    return candidates


def _prepare_full_image_variants(img):
    if img is None or getattr(img, "size", 0) == 0:
        return []

    denoised = cv2.bilateralFilter(img, 7, 45, 45)
    sharpened = cv2.addWeighted(
        denoised,
        1.16,
        cv2.GaussianBlur(denoised, (0, 0), 1.1),
        -0.16,
        0,
    )
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        ("raw", img),
        ("enhanced", sharpened),
        ("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)),
        ("thresh", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
    ]

def _prepare_retry_region(region):
    scale = max(3.0, _region_scale(region) + 0.9)
    enlarged = cv2.resize(
        region,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    denoised = cv2.bilateralFilter(enlarged, 7, 40, 40)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(6, 6)).apply(gray)
    return denoised, clahe


def _prepare_retry_attempt(region, attempt):
    denoised, clahe = _prepare_retry_region(region)

    if attempt == 1:
        return "retry_clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

    binary = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        9,
    )
    return "retry_thresh", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def _extract_prediction(result):
    if not result:
        return "", 0.0

    item = result[0]
    if not isinstance(item, dict):
        return "", 0.0

    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or []

    cleaned_texts = [_normalize_ocr_text(t) for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned_texts:
        return "", 0.0

    if isinstance(scores, (list, tuple)) and scores:
        valid_scores = [float(s) for s in scores[:len(cleaned_texts)] if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    else:
        avg_score = 0.0

    return " ".join(cleaned_texts), avg_score


def _box_to_rect(box):
    if box is None:
        return None

    arr = np.asarray(box)
    if arr.size == 0:
        return None

    if arr.ndim == 1 and arr.shape[0] >= 4:
        x1, y1, x2, y2 = arr[:4]
        return (int(x1), int(y1), int(x2), int(y2))

    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        xs = arr[..., 0]
        ys = arr[..., 1]
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    return None


def _extract_full_image_detections(result):
    if not result:
        return []

    item = result[0]
    if not isinstance(item, dict):
        return []

    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or []
    boxes = item.get("rec_boxes")
    if boxes is None or len(boxes) == 0:
        boxes = item.get("rec_polys")
    if boxes is None or len(boxes) == 0:
        boxes = item.get("dt_polys")
    if boxes is None:
        boxes = []

    detections = []
    for idx, raw_box in enumerate(boxes):
        rect = _box_to_rect(raw_box)
        if rect is None:
            continue

        text = _normalize_ocr_text(texts[idx]) if idx < len(texts) else ""
        score = 0.0
        if isinstance(scores, (list, tuple, np.ndarray)) and idx < len(scores) and scores[idx] is not None:
            score = float(scores[idx])

        detections.append({
            "bbox": rect,
            "text": text,
            "score": score,
        })

    return detections


def _rect_area(rect):
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_1d(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def _gap_1d(a1, a2, b1, b2):
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def _expand_rect(rect, pad_x, pad_y):
    x1, y1, x2, y2 = rect
    return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def _assignment_score(object_bbox, text_bbox):
    overlap = _intersection_area(object_bbox, text_bbox)
    if overlap > 0:
        return 1.0 + (overlap / max(1, _rect_area(text_bbox)))

    expanded = _expand_rect(object_bbox, pad_x=28, pad_y=16)
    center_x = (text_bbox[0] + text_bbox[2]) / 2
    center_y = (text_bbox[1] + text_bbox[3]) / 2
    if (
        expanded[0] <= center_x <= expanded[2]
        and expanded[1] <= center_y <= expanded[3]
    ):
        return 0.8

    y_overlap = _overlap_1d(object_bbox[1], object_bbox[3], text_bbox[1], text_bbox[3])
    min_h = max(1, min(object_bbox[3] - object_bbox[1], text_bbox[3] - text_bbox[1]))
    y_overlap_ratio = y_overlap / min_h
    if y_overlap_ratio <= 0:
        return 0.0

    x_gap = _gap_1d(object_bbox[0], object_bbox[2], text_bbox[0], text_bbox[2])
    if y_overlap_ratio >= 0.45 and x_gap <= 34:
        return 0.3 + y_overlap_ratio - (x_gap / 200.0)

    return 0.0


def _assigned_text_confidence(assigned):
    if not assigned:
        return 0.0
    scores = [float(item.get("score", 0.0)) for item in assigned if item.get("score") is not None]
    return (sum(scores) / len(scores)) if scores else 0.0


def _assign_predictions_to_objects(objects, predictions):
    assignments = [[] for _ in objects]
    for prediction in predictions:
        best_idx = None
        best_score = 0.0
        for idx, obj in enumerate(objects):
            score = _assignment_score(obj["bbox"], prediction["bbox"])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.22:
            assignments[best_idx].append(prediction)

    return assignments

def _ocr_quality_score(text, confidence, prefer_thai=False):
    if not text:
        return -1.0

    compact = text.replace(" ", "")
    digit_bonus = 0.05 if any(ch.isdigit() for ch in compact) else 0.0
    has_thai = _contains_thai(compact)
    has_latin = _contains_latin(compact)
    thai_bonus = 0.08 if has_thai else 0.0
    length_bonus = min(len(compact), 20) * 0.01
    punctuation_penalty = 0.06 if len(compact) <= 2 and not any(ch.isdigit() for ch in compact) else 0.0
    foreign_word_bonus = 0.04 if _looks_like_known_foreign_text(text) else 0.0
    thai_preference_bonus = 0.14 if prefer_thai and has_thai else 0.0
    thai_preference_penalty = 0.12 if prefer_thai and has_latin and not has_thai else 0.0
    return (
        confidence
        + digit_bonus
        + thai_bonus
        + length_bonus
        + foreign_word_bonus
        + thai_preference_bonus
        - punctuation_penalty
        - thai_preference_penalty
    )


def _normalized_compare_text(text):
    return "".join(ch for ch in (text or "").lower() if ch.isalnum() or ("\u0e00" <= ch <= "\u0e7f"))


def _word_tokens(text):
    return [token for token in re.split(r"\s+", (text or "").strip()) if token]


def _text_similarity(a, b):
    a_norm = _normalized_compare_text(a)
    b_norm = _normalized_compare_text(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _word_overlap_score(a, b):
    a_tokens = set(_word_tokens(a))
    b_tokens = set(_word_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def _consensus_bonus(text, other_texts):
    if not text:
        return -0.2
    total = 0.0
    count = 0
    for other in other_texts:
        if not other:
            continue
        char_score = _text_similarity(text, other)
        word_score = _word_overlap_score(text, other)
        total += (char_score * 0.7) + (word_score * 0.3)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _candidate_quality(text, confidence, other_texts):
    prefer_thai = not _looks_like_link_text(text)
    quality = _ocr_quality_score(text, confidence, prefer_thai=prefer_thai)
    quality += _consensus_bonus(text, other_texts) * 0.35
    if _looks_like_wrong_language(text) and prefer_thai:
        quality -= 0.18
    if _looks_like_suspicious_thai(text):
        quality -= 0.10
    if _looks_like_link_text(text):
        quality += 0.08
    return quality


def _assess_ocr_text(text, confidence=0.0, other_texts=None, span_count=0):
    text = _normalize_ocr_text(text)
    other_texts = [candidate for candidate in (other_texts or []) if candidate]

    if not text:
        return {
            "trust_score": 0.0,
            "low_confidence": True,
            "block_translation": True,
            "reasons": ["empty_text"],
            "plausibility": 0.0,
            "consensus": 0.0,
        }

    compact = text.replace(" ", "")
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    plausibility = _thai_sentence_plausibility(text) if _contains_thai(compact) else 0.0
    consensus = _consensus_bonus(text, other_texts) if other_texts else 0.0
    reasons = []

    trust = 0.28 + (confidence * 0.32)

    if _looks_like_link_text(text):
        trust += 0.18
    else:
        if _looks_like_wrong_language(text):
            reasons.append("wrong_language")
            trust -= 0.38
        if _looks_like_suspicious_thai(text):
            reasons.append("mixed_thai_noise")
            trust -= 0.22
        if _contains_thai(compact):
            if plausibility < 0.55:
                reasons.append("implausible_thai_sentence")
                trust -= 0.22 + ((0.55 - plausibility) * 0.55)
            elif plausibility < 0.72:
                reasons.append("weak_thai_sentence")
                trust -= 0.08

    if span_count <= 0:
        reasons.append("no_detected_spans")
        trust -= 0.22
    elif span_count >= 2:
        trust += 0.05

    if other_texts:
        trust += consensus * 0.20
        if consensus < 0.22:
            reasons.append("low_variant_agreement")
            trust -= 0.14

    tokens = _word_tokens(text)
    if len(compact) <= 2 and not any(ch.isdigit() for ch in compact):
        reasons.append("too_short")
        trust -= 0.18
    elif tokens:
        short_tokens = sum(1 for token in tokens if len(re.sub(r"\W+", "", token)) <= 1)
        if short_tokens >= max(2, len(tokens) // 2):
            reasons.append("fragmented_tokens")
            trust -= 0.10

    trust = max(0.0, min(1.0, trust))
    severe_reasons = {"wrong_language", "implausible_thai_sentence", "mixed_thai_noise"}
    low_confidence = trust < 0.68 or any(reason in severe_reasons for reason in reasons)
    block_translation = (
        not _looks_like_link_text(text)
        and (
            trust < 0.58
            or "wrong_language" in reasons
            or "implausible_thai_sentence" in reasons
        )
    )

    return {
        "trust_score": trust,
        "low_confidence": low_confidence,
        "block_translation": block_translation,
        "reasons": reasons,
        "plausibility": plausibility,
        "consensus": consensus,
    }


def _recognize_text_from_region(region, idx=None):
    if region is None or region.size == 0:
        return "", -1.0

    text = run_ocr_on_region(region)
    text, _ = _retry_ocr_if_needed(region, text, "", idx=idx)
    text, _ = _repair_thai_text_if_needed(text, "", idx=idx)
    quality = _ocr_quality_score(text, 0.0, prefer_thai=True)
    return text.strip(), quality


def _run_ocr_once(prepared, prefer_thai=False):
    global _OCR_ENGINE_VALIDATED
    if not _OCR_ENGINE_VALIDATED:
        if not hasattr(ocr_engine, "predict"):
            raise RuntimeError(
                f"OCR engine {ocr_engine.__class__.__name__} has no predict() method."
            )
        print(f"[OCR] Using engine {ocr_engine.__class__.__module__}.{ocr_engine.__class__.__name__}")
        _OCR_ENGINE_VALIDATED = True

    is_vision = ocr_engine.__class__.__name__ == "GoogleVisionOCR"
    if is_vision:
        # Vision API on a small region crop: join all detected words
        detections = ocr_engine.predict(prepared)
        text = " ".join(d["text"] for d in detections if d.get("text", "").strip())
        text = _normalize_ocr_text(text)
        confidence = float(sum(d.get("score", 1.0) for d in detections) / max(len(detections), 1)) if detections else 0.0
    else:
        result = ocr_engine.predict(prepared)
        text, confidence = _extract_prediction(result)

    quality = _ocr_quality_score(text, confidence, prefer_thai=prefer_thai)
    return text, quality

def run_ocr_on_region(region):
    if region.size == 0:
        return ""

    best_text = ""
    best_quality = -1.0
    for _, prepared in _prepare_candidate_regions(region):
        text, quality = _run_ocr_once(prepared)
        if quality > best_quality:
            best_quality = quality
            best_text = text

    return best_text


def _should_retry_ocr(text, translated_text):
    if _looks_like_wrong_language(text):
        return "wrong_language"
    return ""


def _retry_ocr_if_needed(region, text, translated_text, idx=None):
    reason = _should_retry_ocr(text, translated_text)
    if not reason:
        return text, translated_text

    label = f"crop_{idx}" if idx is not None else "crop"
    current_text = text
    current_translation = translated_text

    for attempt in range(1, MAX_OCR_RETRIES + 1):
        variant_name, prepared = _prepare_retry_attempt(region, attempt)
        retry_text, retry_quality = _run_ocr_once(prepared, prefer_thai=True)
        final_text = current_text
        final_translation = current_translation

        if retry_text:
            retry_translation = translate_th_to_en(retry_text)
            if reason == "wrong_language":
                if _contains_thai(retry_text) or retry_quality >= 0.92:
                    final_text = retry_text
                    final_translation = retry_translation
            elif retry_translation and retry_text != current_text:
                final_text = retry_text
                final_translation = retry_translation

        changed = final_text != current_text
        print(
            f"[OCR RETRY] {label} attempt={attempt}/{MAX_OCR_RETRIES} "
            f"variant={variant_name} reason={reason} changed={'yes' if changed else 'no'} "
            f"before='{current_text}' after='{final_text}'"
        )

        current_text = final_text
        current_translation = final_translation
        if changed:
            break

    return current_text, current_translation


def _repair_thai_text_if_needed(text, translated_text, idx=None):
    if not _should_repair_thai_sentence(text, translated_text):
        return text, translated_text

    original_translation = translated_text or _translate_or_preserve_text(text)
    original_score = _thai_sentence_plausibility(text)
    corrected_text = _correct_thai_text(text)
    if not corrected_text:
        return text, original_translation

    corrected_score = _thai_sentence_plausibility(corrected_text)
    corrected_translation = _translate_or_preserve_text(corrected_text)
    changed = corrected_text != text
    label = f"crop_{idx}" if idx is not None else "crop"

    should_accept = False
    if corrected_translation and not original_translation:
        should_accept = True
    elif corrected_score >= original_score + 0.08 and changed:
        should_accept = True
    elif changed and _looks_like_suspicious_thai(text) and corrected_score > original_score:
        should_accept = True

    if changed and should_accept:
        print(
            f"[OCR THAI FIX] {label} "
            f"before='{text}' after='{corrected_text}' "
            f"score_before={original_score:.2f} score_after={corrected_score:.2f}"
        )

    if should_accept and corrected_translation:
        return corrected_text, corrected_translation
    return text, original_translation


def ocr_and_translate_region(img, rect, idx=None):
    if img is None or rect is None:
        return "", ""

    region = extract_region(img, rect, idx if idx is not None else "region")
    text_th, _ = _recognize_text_from_region(region, idx=idx)
    if _source_lang_code == "th":
        text_th = _repair_thai_text_if_needed(text_th, "", idx=idx)[0]
    return text_th, ""   # text_en filled in main.py after Gemini step


_OCR_UPSCALE = 2.0
_OCR_MAX_SIDE = 3800  # PaddleOCR hard limit is 4000; stay safely below it


def _upscale_for_ocr(img, scale=_OCR_UPSCALE):
    h, w = img.shape[:2]
    max_side = max(h, w)
    safe_scale = min(scale, _OCR_MAX_SIDE / max_side)
    if safe_scale < 1.1:
        # Upscaling less than 10% adds noise without benefit; run at native size
        return img, 1.0
    upscaled = cv2.resize(img, (int(w * safe_scale), int(h * safe_scale)), interpolation=cv2.INTER_CUBIC)
    return upscaled, safe_scale


def ocr_and_translate_full_image(objects, img, progress_label="OCR"):
    """Run OCR on *img*, assign text to each object, and return (objects, raw_spans).

    *raw_spans* is the full list of Vision-detected text segments sorted top-to-bottom,
    regardless of whether they were matched to a CRAFT object.  Callers can pass this
    to Gemini so it has access to every word the OCR found, not just the grouped ones.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return objects, []

    from config import ocr_engine as _engine
    is_vision_api = hasattr(_engine, "__class__") and _engine.__class__.__name__ == "GoogleVisionOCR"

    if is_vision_api:
        # Vision API works best at native resolution; no upscaling needed
        print(f"\n[{progress_label}] Running Google Vision OCR ({img.shape[1]}x{img.shape[0]})...")
        detections = _engine.predict(img)
        scale = 1.0
        # Propagate detected language so Gemini prompt and Thai NLP use the right language
        _set_source_language(_engine.detected_language, _engine.detected_language_name)
    else:
        ocr_img, scale = _upscale_for_ocr(img)
        print(
            f"\n[{progress_label}] Running OCR on concatenated image "
            f"({img.shape[1]}x{img.shape[0]} → {ocr_img.shape[1]}x{ocr_img.shape[0]}, scale={scale}x)..."
        )
        result = ocr_engine.predict(ocr_img)
        detections = _extract_full_image_detections(result)

    print(f"[{progress_label}] Detected {len(detections)} spans")

    predictions = []
    for item in detections:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        x1, y1, x2, y2 = item["bbox"]
        predictions.append({
            "text": text,
            "bbox": (x1 / scale, y1 / scale, x2 / scale, y2 / scale),
            "score": item.get("score", 0.0),
        })
    assignments = _assign_predictions_to_objects(objects, predictions)

    # Assemble raw OCR text per object — translation happens later in main.py
    # (after Gemini corrects the Thai, so we only translate once on clean text)
    for idx, obj in enumerate(objects):
        assigned = sorted(assignments[idx], key=lambda item: (item["bbox"][1], item["bbox"][0]))
        text_th = " ".join(item["text"] for item in assigned).strip()
        if _source_lang_code == "th":
            text_th = _repair_thai_text_if_needed(text_th, "", idx=idx)[0]
        confidence = _assigned_text_confidence(assigned)
        span_count = len(assigned)
        assessment = _assess_ocr_text(text_th, confidence=confidence, span_count=span_count)

        obj["text_th"] = text_th
        obj["text_en"] = ""          # filled in main.py after Gemini step
        obj["ocr_span_count"] = span_count
        obj["ocr_source"] = "combined_canvas"
        obj["ocr_validated"] = bool(text_th)
        obj["ocr_trust_score"] = assessment["trust_score"]
        obj["ocr_low_confidence"] = assessment["low_confidence"]
        obj["ocr_reasons"] = assessment["reasons"]
        obj["translation_blocked"] = False

    # Return all OCR spans sorted top-to-bottom so the caller can pass them to
    # Gemini as a supplementary dump of everything the Vision API found.
    raw_spans = sorted(predictions, key=lambda p: (p["bbox"][1], p["bbox"][0]))
    return objects, raw_spans

# --------------------------------------------------
# OCR + TRANSLATION LOOP (ONLY PLACE WE ADD LOGIC)
# --------------------------------------------------

def _format_runtime(seconds):
    minutes, secs = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def _print_progress(prefix, current, total, start_time):
    total = max(total, 1)
    percent = current / total
    bar_length = 36
    filled = int(bar_length * percent)
    bar = "#" * filled + "-" * (bar_length - filled)

    elapsed = time.time() - start_time
    avg = elapsed / current if current else 0
    eta = avg * (total - current)

    sys.stdout.write(
        "\r"
        f"[{prefix}] |{bar}| {percent * 100:6.2f}% "
        f"({current}/{total}) "
        f"elapsed={_format_runtime(elapsed)} "
        f"eta={_format_runtime(eta)}"
    )
    sys.stdout.flush()


def ocr_and_translate(objects, img, progress_label="OCR"):
    total = len(objects)
    start_time = time.time()

    if total == 0:
        print(f"[{progress_label}] No objects found for OCR")
        return objects

    print(f"\n[{progress_label}] Processing {total} objects...")

    for i, obj in enumerate(objects):
        rect = obj["bbox"]
        text_th, text_en = ocr_and_translate_region(img, rect, idx=i)

        obj["text_th"] = text_th
        obj["text_en"] = text_en
        assessment = _assess_ocr_text(text_th, span_count=1)
        obj["ocr_trust_score"] = assessment["trust_score"]
        obj["ocr_low_confidence"] = assessment["low_confidence"]
        obj["ocr_reasons"] = assessment["reasons"]
        obj["translation_blocked"] = False

        _print_progress(progress_label, i + 1, total, start_time)

    runtime = time.time() - start_time

    print(
        f"\n[{progress_label}] Completed in {_format_runtime(runtime)} "
        f"({runtime:.2f}s)\n"
    )

    return objects