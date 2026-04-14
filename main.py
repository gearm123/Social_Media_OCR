import os
import sys
import json
import time
import copy
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import cv2
import numpy as np

# Allow Thai / Unicode characters to print on Windows without crashing
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load project `.env` before `config` (same as `web_app.py`) so `python main.py` picks up
# GEMINI_API_KEY / GOOGLE_VISION_API_KEY without exporting them in the shell.
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent
    if os.environ.get("RENDER", "").strip().lower() != "true":
        load_dotenv(_root / ".env")
except ImportError:
    pass

from config import INPUT_DIR, OUTPUT_DIR, JSON_DIR, RENDER_DIR, ocr_engine, translate_en_to
from output_languages import OutputLanguage, parse_output_language
import config as runtime_config
import ocr_translate as ocr_translate_module
from ocr_translate import (
    translate_to_en,
    translate_conversation_gemini_multimodal,
    build_pass2_per_message_ocr_hints,
    _gemini_ocr_hints_refine_pass,
    _gemini_reference_resolution_pass,
    get_gemini_pass_outcomes,
    _pass3_fallback_messages_from_pass2,
    _jpeg_b64_from_bgr,
    _meta_from_gemini_messages,
    reset_gemini_pass_outcomes,
    _set_source_language,
    gemini_pass_timeout_sec,
    _compact_verbose_logs,
    _prefer_english_surface,
    _default_status_bar_info,
    _looks_like_link_text,
)
from pipeline import prepare_image_crop_info
from chat_renderer import composite_chat_below_top_banner, render_chat, resolve_top_banner_path

PAGE_GAP_PX = 48
PASS1_BUBBLE_INPUT_PATH = Path(__file__).resolve().parent / "pass1_bubble_input.txt"


class JobCancelledError(Exception):
    """Raised when a hosted job is cancelled mid-pipeline (see web_app cancel flag)."""


def _pipeline_verbose() -> bool:
    return os.environ.get("PIPELINE_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _vprint(*args, **kwargs) -> None:
    if _pipeline_verbose():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _format_pass_attempt_log(pass_name: str, meta: Optional[Dict[str, Any]]) -> str:
    """Build a compact production log line for non-ideal pass attempts."""
    if not isinstance(meta, dict) or not meta:
        return ""
    interesting = any(
        (
            int(meta.get("successful_attempt") or 0) > 1,
            int(meta.get("timed_out_attempts") or 0) > 0,
            int(meta.get("transient_status_retry_count") or 0) > 0,
            int(meta.get("model_failovers") or 0) > 0,
            str(meta.get("status") or "").strip() in ("failed", "timeout_exhausted"),
            str(meta.get("reason") or "").strip() in ("wrapper_exception", "request_failed"),
        )
    )
    if not interesting:
        return ""
    details = [f"status={meta.get('status') or 'unknown'}"]
    successful_attempt = meta.get("successful_attempt")
    max_tries = meta.get("max_tries")
    if successful_attempt is not None:
        attempt_text = str(successful_attempt)
        if max_tries is not None:
            attempt_text += f"/{max_tries}"
        details.append(f"attempt={attempt_text}")
    elif max_tries is not None:
        details.append(f"max_tries={max_tries}")
    for key in (
        "timed_out_attempts",
        "transient_status_retry_count",
        "final_http_status",
        "model",
        "model_failovers",
        "reason",
    ):
        value = meta.get(key)
        if value is None or value == "":
            continue
        details.append(f"{key}={value}")
    return f"[pipeline] {pass_name} attempts summary: " + " ".join(details)


def _log_pass_attempts(pass_name: str, meta: Optional[Dict[str, Any]]) -> None:
    line = _format_pass_attempt_log(pass_name, meta)
    if line:
        print(line, flush=True)


# UI strings that appear in phone status bars but are NOT contact names
_STATUS_BAR_NOISE = {
    "am", "pm", "lte", "5g", "4g", "3g", "wifi", "ok",
    "back", "call", "video", "mute", "search",
}


def _extract_contact_name(info):
    """OCR the status bar of an image and extract the contact/chat name.

    The contact name is usually the largest, most central text in the header bar.
    We filter out obvious phone-UI noise (time, signal, battery) and return the
    most likely name string.
    """
    status_bar_info = info.get("status_bar_info")
    if not status_bar_info:
        return ""
    bbox = status_bar_info.get("bbox")
    if not bbox:
        return ""
    img = info["img"]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return ""

    try:
        detections = ocr_engine.predict(region)
    except RuntimeError:
        return ""
    except Exception as e:
        print(f"[NAME] Status bar OCR failed: {e}")
        return ""

    if not detections:
        return ""

    img_w = x2 - x1
    img_cx = img_w / 2

    # Score each detection: prefer central, longer, non-noise words
    candidates = []
    for det in detections:
        text = det.get("text", "").strip()
        if not text or len(text) < 2:
            continue
        # Skip pure numbers (time, battery %)
        if text.replace(":", "").replace("%", "").replace(".", "").isdigit():
            continue
        # Skip known UI noise
        if text.lower() in _STATUS_BAR_NOISE:
            continue
        # Skip very short all-caps (signal indicators like "LTE", "5G")
        if len(text) <= 3 and text.isupper():
            continue

        bx1, _, bx2, _ = det["bbox"]
        det_cx = (bx1 + bx2) / 2
        # Centrality score: how close is the detection to the horizontal centre
        centrality = 1.0 - abs(det_cx - img_cx) / max(img_cx, 1)
        length_score = min(len(text) / 20, 1.0)
        score = centrality * 0.7 + length_score * 0.3
        candidates.append((score, det_cx, text))

    if not candidates:
        return ""

    # Sort by score desc, group horizontally-adjacent words into a name phrase
    candidates.sort(key=lambda c: -c[0])
    # Take the top-scored word as the anchor, then collect adjacent words
    best = candidates[0]
    name_words = [best[2]]
    for score, cx, text in candidates[1:]:
        if abs(cx - best[1]) < img_w * 0.35:   # within 35% of image width
            name_words.append(text)
        if len(name_words) >= 4:
            break

    # Sort collected words by x position for correct reading order
    top_dets = [(d[1], d[2]) for d in candidates if d[2] in name_words]
    top_dets.sort(key=lambda t: t[0])
    contact_name = " ".join(t[1] for t in top_dets)
    return contact_name


def _crop_bounds_from_info(info, keep_status_bar=False):
    """Compute crop top/bottom from artifact detection info (no objects needed)."""
    img = info["img"]
    top = 0
    bottom = img.shape[0]

    if not keep_status_bar:
        status_bar_info = info.get("status_bar_info")
        if status_bar_info:
            sb_bbox = status_bar_info.get("bbox")
            if sb_bbox:
                top = int(sb_bbox[3]) + 1

    bottom_artifact_info = info.get("bottom_artifact_info")
    if bottom_artifact_info:
        bottom = min(bottom, int(bottom_artifact_info.get("conversation_end_y", img.shape[0] - 1)) + 1)

    if bottom <= top:
        top = 0
        bottom = img.shape[0]

    return top, bottom


def extract_page_segment_images(crop_infos):
    """One cropped BGR image per input file (status bar kept only on page 0)."""
    out = []
    for page_index, info in enumerate(crop_infos):
        keep_sb = page_index == 0
        crop_top, crop_bottom = _crop_bounds_from_info(info, keep_status_bar=keep_sb)
        seg = info["img"][crop_top:crop_bottom, :].copy()
        if seg.size:
            out.append(seg)
    return out


def extract_status_bar_images(crop_infos):
    """Extract the status/header bar region from every input image."""
    out = []
    for info in crop_infos or []:
        img = info.get("img")
        sb = (info.get("status_bar_info") or {}).get("bbox")
        if img is None or sb is None or len(sb) < 4:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in sb[:4]]
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        crop = img[y1:y2, x1:x2].copy()
        if crop.size:
            out.append(crop)
    return out


def _map_bubble_role_token(token: str):
    tok = str(token or "").strip().lower()
    if tok in ("s", "sender"):
        return "sender"
    if tok in ("r", "receiver", "recevier", "reciever"):
        return "receiver"
    return None


def _build_bubble_summary_from_specs(specs):
    lines = []
    total_bubbles = 0
    page_specs = []
    for i, spec in enumerate(specs):
        order = list(spec.get("order") or [])
        count = int(spec.get("count") or len(order))
        total_bubbles += count
        page_specs.append({
            "page_index": i,
            "count": count,
            "order": list(order),
        })
        lines.append(f"in image {i + 1}")
        lines.append(f"i count {count} message bubbles")
        lines.extend(order if order else ["none"])
        if i < len(specs) - 1:
            lines.append("")
    return "\n".join(lines), total_bubbles, page_specs


def _load_pass1_bubble_summary_file(num_images: int):
    path = PASS1_BUBBLE_INPUT_PATH
    if not path.exists():
        return None
    try:
        raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except Exception as e:
        print(f"[pipeline] Could not read {path.name}: {e}")
        return None

    lines = [line for line in raw_lines if line]
    if not lines:
        return None
    if len(lines) % 2 != 0:
        print(f"[pipeline] {path.name} is invalid: expected count/order line pairs.")
        return None

    specs = []
    for i in range(0, len(lines), 2):
        count_line = lines[i]
        order_line = lines[i + 1]
        try:
            count = int(count_line)
        except ValueError:
            print(f"[pipeline] {path.name} has an invalid count line: '{count_line}'")
            return None

        if "," in order_line:
            tokens = [p.strip() for p in order_line.replace(";", ",").split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in order_line.split() if p.strip()]

        mapped = []
        for tok in tokens:
            role = _map_bubble_role_token(tok)
            if role is None:
                print(f"[pipeline] {path.name} has an invalid role token: '{tok}'")
                return None
            mapped.append(role)

        if count != len(mapped):
            print(
                f"[pipeline] {path.name}: count {count} does not match order length {len(mapped)} "
                f"for image {(i // 2) + 1}; using order length."
            )
            count = len(mapped)

        specs.append({"count": count, "order": mapped})

    if len(specs) != num_images:
        print(
            f"[pipeline] {path.name} has {len(specs)} image specs but current input has {num_images} images."
        )
        return None

    return _build_bubble_summary_from_specs(specs)


def _merge_text_spans(spans, gap_px=24):
    if not spans:
        return []
    spans = sorted((int(a), int(b)) for a, b in spans if b > a)
    merged = [list(spans[0])]
    for y0, y1 in spans[1:]:
        prev = merged[-1]
        if y0 <= (prev[1] + gap_px):
            prev[1] = max(prev[1], y1)
        else:
            merged.append([y0, y1])
    return [(a, b) for a, b in merged]


def _estimate_manual_message_bands(page_img, count: int):
    """Estimate vertical message bands using OCR text spans, falling back to equal slices."""
    if page_img is None or getattr(page_img, "size", 0) == 0 or count <= 0:
        return []

    h, _w = page_img.shape[:2]
    spans = []
    try:
        detections = ocr_engine.predict(page_img)
    except Exception:
        detections = []

    for det in detections or []:
        text = (det.get("text") or "").strip()
        bbox = det.get("bbox") or []
        if not text or len(bbox) < 4:
            continue
        y0 = max(0, int(round(float(bbox[1]))))
        y1 = min(h, int(round(float(bbox[3]))))
        if y1 - y0 >= 8:
            spans.append((y0, y1))

    merged = _merge_text_spans(spans)
    if len(merged) >= count:
        bands = []
        for i in range(count):
            start_idx = int(round(i * len(merged) / count))
            end_idx = max(start_idx + 1, int(round((i + 1) * len(merged) / count)))
            bucket = merged[start_idx:end_idx]
            y0 = min(a for a, _ in bucket)
            y1 = max(b for _, b in bucket)
            bands.append((y0, y1))
        return bands

    slice_h = max(80, int(round(h / max(count, 1))))
    bands = []
    for i in range(count):
        cy = int(round((i + 0.5) * h / count))
        y0 = max(0, cy - slice_h // 2)
        y1 = min(h, y0 + slice_h)
        y0 = max(0, y1 - slice_h)
        bands.append((y0, y1))
    return bands


def _tight_content_bbox(roi):
    """Shrink a crop around dense non-white content (bubble color or dark text)."""
    if roi is None or getattr(roi, "size", 0) == 0:
        return None

    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    color_mask = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] <= 250)
    dark_mask = gray <= 210
    mask = np.where(color_mask | dark_mask, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    row_counts = (mask > 0).sum(axis=1)
    col_counts = (mask > 0).sum(axis=0)
    row_thresh = max(10, int(round(w * 0.03)))
    col_thresh = max(8, int(round(h * 0.04)))

    rows = np.where(row_counts >= row_thresh)[0]
    cols = np.where(col_counts >= col_thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    y0 = max(0, int(rows[0]) - 14)
    y1 = min(h, int(rows[-1]) + 15)
    x0 = max(0, int(cols[0]) - 18)
    x1 = min(w, int(cols[-1]) + 19)

    if y1 - y0 < 40 or x1 - x0 < 80:
        return None
    return x0, y0, x1, y1


def build_manual_message_context_crops(page_images, page_specs):
    """Build approximate per-message crops from manual bubble counts and order."""
    crops = []
    entries = []
    global_idx = 0

    for page_idx, page_img in enumerate(page_images or []):
        spec = page_specs[page_idx] if page_idx < len(page_specs) else {"count": 0, "order": []}
        count = int(spec.get("count") or 0)
        order = list(spec.get("order") or [])
        if page_img is None or getattr(page_img, "size", 0) == 0 or count <= 0:
            continue

        h, w = page_img.shape[:2]
        bands = _estimate_manual_message_bands(page_img, count)
        pad_y = 28
        pad_x = 24

        for local_idx, (band_y0, band_y1) in enumerate(bands):
            role = order[local_idx] if local_idx < len(order) else ""
            top = max(0, int(band_y0) - pad_y)
            bottom = min(h, int(band_y1) + pad_y)
            left = 0
            right = w
            if role == "receiver":
                right = min(w, int(round(w * 0.84)) + pad_x)
            elif role == "sender":
                left = max(0, int(round(w * 0.16)) - pad_x)

            row_roi = page_img[top:bottom, left:right].copy()
            tight_bbox = _tight_content_bbox(row_roi)
            if tight_bbox is not None:
                tx0, ty0, tx1, ty1 = tight_bbox
                left += tx0
                right = left + (tx1 - tx0)
                top += ty0
                bottom = top + (ty1 - ty0)

            crop = page_img[top:bottom, left:right].copy()
            if crop.size == 0:
                continue
            crops.append(crop)
            entries.append({
                "message_index": global_idx,
                "page_index": page_idx,
                "page_local_index": local_idx,
                "role_hint": role,
                "estimated_bbox_page": [left, top, right, bottom],
            })
            global_idx += 1

    return crops, entries


def collect_page_ocr_debug(page_images):
    """Run OCR on each cleaned page image and collect text/confidence debug output.

    Threshold defaults to ``GEMINI_PASS2_OCR_MIN_CONF`` (default **0.92**), aligned with Pass 2 bucketing.
    """
    lines = []
    entries = []
    min_conf = float(os.environ.get("GEMINI_PASS2_OCR_MIN_CONF", "0.92"))
    min_conf = max(0.0, min(1.0, min_conf))

    for page_idx, page_img in enumerate(page_images or []):
        lines.append(f"image {page_idx + 1}")
        lines.append(f"min_confidence {min_conf:.2f}")
        page_entry = {"page_index": page_idx, "detections": []}
        if page_img is None or getattr(page_img, "size", 0) == 0:
            lines.append("(empty image)")
            lines.append("")
            entries.append(page_entry)
            continue
        try:
            detections = ocr_engine.predict(page_img)
        except Exception as e:
            lines.append(f"(ocr failed: {e})")
            lines.append("")
            page_entry["error"] = str(e)
            entries.append(page_entry)
            continue

        detections = sorted(
            detections or [],
            key=lambda d: (
                float((d.get("bbox") or [0, 0, 0, 0])[1]),
                float((d.get("bbox") or [0, 0, 0, 0])[0]),
            ),
        )
        if not detections:
            lines.append("(no text detected)")
            lines.append("")
            entries.append(page_entry)
            continue

        for det in detections:
            text = (det.get("text") or "").strip()
            bbox = det.get("bbox") or []
            score = float(det.get("score") or 0.0)
            if not text or len(bbox) < 4 or score < min_conf:
                continue
            bbox_int = [int(round(float(v))) for v in bbox[:4]]
            page_entry["detections"].append(
                {"text": text, "confidence": score, "bbox": bbox_int}
            )
            lines.append(
                f'- conf={score:.3f} bbox={bbox_int} text={json.dumps(text, ensure_ascii=False)}'
            )
        if not page_entry["detections"]:
            lines.append("(no hints kept at this confidence threshold)")
        lines.append("")
        entries.append(page_entry)

    return "\n".join(lines).strip() + "\n", entries


def build_combined_image(crop_infos, background=248, global_first_page_index=0):
    """Concatenate cropped images (status bar kept only on global first page of the run).

    *global_first_page_index* is the original index of ``crop_infos[0]`` in the full input list
    (0 for the first chunk, >0 for continuation chunks — those pages never keep the status bar).

    Returns (combined_img, page_ranges) where page_ranges is a list of
    (start_y, end_y, page_index, crop_top, status_bar_info, bottom_artifact_info).
    """
    segments = []
    page_ranges = []
    current_y = 0

    for local_i, info in enumerate(crop_infos):
        page_index = global_first_page_index + local_i
        keep_sb = (page_index == 0)
        crop_top, crop_bottom = _crop_bounds_from_info(info, keep_status_bar=keep_sb)
        if not _compact_verbose_logs():
            _vprint(
                f"[COMBINE] page={page_index} keep_status_bar={keep_sb} "
                f"crop_top={crop_top} crop_bottom={crop_bottom} "
                f"img_h={info['img'].shape[0]} "
                f"status_bar={info.get('status_bar_info', {}) and info.get('status_bar_info', {}).get('bbox')}"
            )
        segment = info["img"][crop_top:crop_bottom, :].copy()
        if segment.size == 0:
            continue

        seg_h = segment.shape[0]
        page_ranges.append((current_y, current_y + seg_h, page_index, crop_top,
                            info["status_bar_info"], info["bottom_artifact_info"]))
        segments.append(segment)
        current_y += seg_h + PAGE_GAP_PX

    if not segments:
        return None, []

    total_height = current_y - PAGE_GAP_PX
    max_width = max(s.shape[1] for s in segments)
    channels = segments[0].shape[2]
    combined = np.full((total_height, max_width, channels), background, dtype=np.uint8)

    y = 0
    for idx, seg in enumerate(segments):
        h, w = seg.shape[:2]
        combined[y:y + h, :w] = seg
        y += h
        if idx < len(segments) - 1:
            y += PAGE_GAP_PX

    return combined, page_ranges


_KNOWN_LANGUAGES = {
    "th": "Thai", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "ru": "Russian", "he": "Hebrew", "hi": "Hindi",
    "vi": "Vietnamese", "id": "Indonesian", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "tr": "Turkish",
}


def _parse_difficulty_arg(value: str) -> int:
    try:
        n = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("difficulty must be 1, 2, or 3") from exc
    if n not in (1, 2, 3):
        raise argparse.ArgumentTypeError("difficulty must be 1, 2, or 3")
    return n


def _coerce_pipeline_difficulty(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("difficulty must be 1, 2, or 3") from None
    if n not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2, or 3")
    return n


def _pass2_messages_from_pass1(pass1_msgs: list) -> list:
    """Same {role, text_src, text_en} shape as Gemini Pass 2 when Pass 2 is skipped."""
    out = []
    for m in pass1_msgs or []:
        src = (m.get("text_src") or m.get("text_en") or "").strip()
        out.append({
            "role": m.get("role"),
            "text_src": src,
            "text_en": _prefer_english_surface(translate_to_en(src), src),
        })
    return out


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Gemini vision chat translator (stitched + per-page screenshots)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Full pipeline and Gemini logs (HTTP wait lines, pass summaries, debug paths). "
            "Same as PIPELINE_VERBOSE=1. Default layout is compact; set "
            "PIPELINE_VERBOSE_LAYOUT=classic (or full/old) for the longer log style. "
            "When unset, GEMINI_PASS_SUMMARY and GEMINI_WAIT_UI follow verbose unless you set them explicitly. "
            "Also appends rolling HTTP timing stats under timing_debug/ "
            "(short pass_timing_debug.txt vs pass_timing_debug_hurry_up.txt when --hurry-up; "
            "full state in matching *_state.json). Only pass blocks for passes that ran under "
            "--difficulty are updated (1→pass1 only, 2→pass1–2, 3→pass1–3)."
        ),
    )
    parser.add_argument(
        "-l", "--source-language",
        metavar="CODE",
        dest="source_language",
        default=None,
        help=(
            "Optional source language BCP-47 hint for Gemini transcription "
            "(e.g. th, zh, ja). Default: infer from screenshots."
        ),
    )
    parser.add_argument(
        "--language",
        dest="output_language",
        type=_parse_output_language_cli,
        default="english",
        metavar="LANG",
        help=(
            "Final translation language for translated_conversation.json and translated_conversation.png "
            "(default: english). The model still resolves meaning in English internally; this only "
            "localizes the delivered artifact. Per-pass debug/compare PNGs stay in English. "
            "Romance + Germanic only: english, spanish, french, german, dutch, swedish, … "
            "(see output_languages.OutputLanguage)."
        ),
    )
    parser.add_argument(
        "--difficulty",
        type=_parse_difficulty_arg,
        default=3,
        metavar="N",
        help=(
            "Gemini depth: 1 = Pass 1 only (final image uses Pass 1 + English gloss); "
            "2 = Passes 1–2; 3 = full Passes 1–3 (default). "
            "Same output paths (translated_conversation.png, JSON, per-pass PNGs). "
            "Compare debug: 1 = Pass 1 panel; 2 = Pass 1 | Pass 2; 3 = Pass 1 | Pass 2 | Pass 3."
        ),
    )
    parser.add_argument(
        "--hurry-up",
        action="store_true",
        help=(
            "Use shorter HTTP timeouts and fixed thinking budgets per pass/attempt (Gemini 2.5). "
            "Pass 1: 1920/30s then 960/30s; Pass 2: 1920/20s then 480/15s; Pass 3: 960/20s (one try). "
            "Default off — normal behavior uses env-based timeouts and existing thinking rules. "
            "Verbose timing log + state use timing_debug/pass_timing_debug_hurry_up*. "
            "Delete both .txt and _state.json for that mode to reset counts."
        ),
    )
    return parser.parse_args()


def _compose_labeled_chat_panels(panels):
    """Place one or more rendered chats side-by-side with simple labels above each."""
    gap = 20
    pad = 12
    label_h = 34
    bg = 248

    valid_panels = [(img, label) for img, label in panels if img is not None and getattr(img, "size", 0) != 0]
    if not valid_panels:
        return None

    panel_w = max(img.shape[1] for img, _label in valid_panels)
    panel_h = max(img.shape[0] + label_h for img, _label in valid_panels)
    canvas_h = panel_h + pad * 2
    canvas_w = panel_w * len(valid_panels) + gap * (len(valid_panels) - 1) + pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    def _draw_panel(src, label, x0):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 1
        color = (55, 55, 55)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x0 + max(8, (panel_w - tw) // 2)
        ty = pad + th + 2
        cv2.putText(canvas, label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

        y1 = pad + label_h
        h, w = src.shape[:2]
        canvas[y1:y1 + h, x0:x0 + w] = src

    for idx, (img, label) in enumerate(valid_panels):
        _draw_panel(img, label, pad + idx * (panel_w + gap))
    return canvas


def _wrap_text_lines(text, max_chars=44):
    words = str(text or "").split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _render_text_report_image(title, sections, width=900):
    """Render a simple readable text report image."""
    bg = np.full((2000, width, 3), 248, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (45, 45, 45)
    y = 40

    def _put(line, scale=0.62, thickness=1):
        nonlocal y
        cv2.putText(bg, line, (24, y), font, scale, color, thickness, cv2.LINE_AA)
        y += int(28 * max(scale / 0.62, 1.0))

    _put(title, scale=0.8, thickness=2)
    y += 10
    for heading, lines in sections:
        _put(heading, scale=0.68, thickness=2)
        for line in lines:
            for wrapped in _wrap_text_lines(line, max_chars=52):
                _put(wrapped, scale=0.58, thickness=1)
        y += 12

    return bg[: min(y + 20, bg.shape[0]), :, :]


def _render_safe_contact_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "Person A"
    if any(ord(c) > 0x024F for c in n if not c.isspace()):
        return "Person A"
    return n


def _is_call_action_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"โทรอีกครั้ง", "โทรกลับ", "call again", "call back"}


def _looks_like_call_notice(src_text: str, en_text: str) -> bool:
    s = (src_text or "").lower()
    e = (en_text or "").lower()
    call_markers = (
        "การโทร",
        "ไม่ได้รับ",
        "โทรด้วยเสียง",
        "voice call",
        "audio call",
        "video call",
        "missed",
        "call",
    )
    return any(marker in s or marker in e for marker in call_markers)


def _build_call_notice_meta(
    src_text: str,
    en_text: str,
    button_text: str = "Call back",
    event_timestamp: Optional[str] = None,
):
    """Subtitle: ``event_timestamp`` from Pass 1 (clock time or call duration) is translated to English.

    Gemini sometimes puts a Thai duration line (e.g. minutes/seconds) in ``event_timestamp`` instead of a
    clock time; that still needs ``translate_to_en`` so units render in English. If translation fails,
    the raw string is kept. If ``event_timestamp`` is empty, subtitle is extra lines from merged text."""
    src = (src_text or "").strip()
    en = (en_text or "").strip()
    combined = en or src
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    src_l = src.lower()
    full_l = combined.lower()
    missed = ("ไม่ได้รับ" in src_l) or ("missed" in full_l)
    video = ("วิดีโอ" in src_l) or ("video" in full_l)
    title = "Missed audio call" if missed else ("Video call" if video else "Audio call")
    if event_timestamp is not None and str(event_timestamp).strip():
        raw_ts = str(event_timestamp).strip()
        subtitle_en = translate_to_en(raw_ts)
        subtitle = subtitle_en.strip() if subtitle_en.strip() else raw_ts
    else:
        subtitle = "\n".join(lines[1:]).strip() if len(lines) >= 2 else ""
    return {
        "type": "call_notice",
        "bbox": [0, 0, 400, 35],
        "text_th": "",
        "text_en": title,
        "subtitle": subtitle,
        "button_text": button_text,
        "missed": missed,
        "ocr_source": "gemini_full_vision",
        "ocr_span_count": 0,
        "ocr_validated": True,
        "ocr_trust_score": 1.0,
        "ocr_low_confidence": False,
        "ocr_reasons": [],
    }


def _merge_chat_with_system_metadata(chat_meta, system_messages):
    systems_by_before = {}
    sys_items = list(system_messages or [])
    i = 0
    while i < len(sys_items):
        sys_msg = sys_items[i]
        before_idx = int(sys_msg.get("insert_before_chat_index") or 0)
        src_text = sys_msg.get("text_src") or ""
        text_en = translate_to_en(src_text) or src_text

        if _looks_like_call_notice(src_text, text_en):
            button_text = "Call back"
            chunk_msgs = [sys_msg]
            chunks = [(src_text or "").strip()]
            j = i + 1
            while j < len(sys_items):
                if int(sys_items[j].get("insert_before_chat_index") or 0) != before_idx:
                    break
                t2 = (sys_items[j].get("text_src") or "").strip()
                e2 = translate_to_en(t2) or t2
                if _is_call_action_text(t2) or _is_call_action_text(e2):
                    j += 1
                    break
                if _looks_like_call_notice(t2, e2):
                    break
                chunk_msgs.append(sys_items[j])
                chunks.append(t2)
                j += 1
            merged_src = "\n".join(chunks)
            merged_en = translate_to_en(merged_src) or merged_src
            ev_ts = None
            for cm in chunk_msgs:
                v = (cm.get("event_timestamp") or "").strip()
                if v:
                    ev_ts = v
                    break
            systems_by_before.setdefault(before_idx, []).append(
                _build_call_notice_meta(
                    merged_src,
                    merged_en,
                    button_text=button_text,
                    event_timestamp=ev_ts,
                )
            )
            i = j
            continue
        else:
            systems_by_before.setdefault(before_idx, []).append({
                "type": "timestamp",
                "bbox": [0, 0, 400, 35],
                "text_th": "",
                "text_en": text_en,
                "ocr_source": "gemini_full_vision",
                "ocr_span_count": 0,
                "ocr_validated": True,
                "ocr_trust_score": 1.0,
                "ocr_low_confidence": False,
                "ocr_reasons": [],
            })
        i += 1

    merged = []
    order = 0
    chat_items = list(chat_meta or [])
    for idx in range(len(chat_items) + 1):
        for sys_meta in systems_by_before.get(idx, []):
            item = dict(sys_meta)
            item["order"] = order
            merged.append(item)
            order += 1
        if idx < len(chat_items):
            item = dict(chat_items[idx])
            item["order"] = order
            merged.append(item)
            order += 1
    return merged


def _localize_render_meta_for_output(meta: list, lang: OutputLanguage) -> None:
    """Mutates *meta* in place: English ``text_en`` (and call-card strings) → *lang*."""
    if lang == OutputLanguage.ENGLISH:
        return
    code = lang.value
    for item in meta or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "call_notice":
            for field in ("text_en", "subtitle", "button_text"):
                v = item.get(field)
                if isinstance(v, str) and v.strip() and not _looks_like_link_text(v):
                    item[field] = translate_en_to(v, code) or v
            continue
        v = item.get("text_en")
        if isinstance(v, str) and v.strip() and not _looks_like_link_text(v):
            item["text_en"] = translate_en_to(v, code) or v


def _parse_output_language_cli(value: str | OutputLanguage) -> OutputLanguage:
    if isinstance(value, OutputLanguage):
        return value
    p = parse_output_language(value)
    if p is None:
        raise argparse.ArgumentTypeError(
            f"unknown output language {value!r}; "
            "use a name or code such as english, spanish, french, german, portuguese, dutch "
            "(full list: output_languages.OutputLanguage)."
        )
    return p


def _parse_bubble_summary_text(raw_text: str, num_images: int):
    raw_lines = [line.strip() for line in (raw_text or "").splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return "", 0, []
    if len(lines) % 2 != 0:
        raise ValueError("Bubble summary text must contain count/order line pairs.")

    specs = []
    for i in range(0, len(lines), 2):
        try:
            count = int(lines[i])
        except ValueError as exc:
            raise ValueError(f"Invalid bubble count line: {lines[i]!r}") from exc

        order_line = lines[i + 1]
        if "," in order_line:
            tokens = [p.strip() for p in order_line.replace(";", ",").split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in order_line.split() if p.strip()]

        mapped = []
        for tok in tokens:
            role = _map_bubble_role_token(tok)
            if role is None:
                raise ValueError(f"Invalid bubble role token: {tok!r}")
            mapped.append(role)

        if count != len(mapped):
            count = len(mapped)
        specs.append({"count": count, "order": mapped})

    if len(specs) != num_images:
        raise ValueError(
            f"Bubble summary has {len(specs)} image specs but this job has {num_images} images."
        )
    return _build_bubble_summary_from_specs(specs)


@contextmanager
def _override_runtime_dirs(
    input_dir,
    output_dir,
    json_dir,
    render_dir,
    debug_dir,
    pipeline_timing_difficulty: Optional[int] = None,
    hurry_up: bool = False,
):
    """When *pipeline_timing_difficulty* is set (1–3), ``PIPELINE_TIMING_DIFFICULTY`` is exposed so
    verbose pass timing logs only update pass blocks for passes that ran in this job.

    ``PIPELINE_HURRY_UP`` is set for the job duration so Gemini timeouts/thinking and timing log path
    follow ``--hurry-up``."""
    original = {
        "main.INPUT_DIR": globals()["INPUT_DIR"],
        "main.OUTPUT_DIR": globals()["OUTPUT_DIR"],
        "main.JSON_DIR": globals()["JSON_DIR"],
        "main.RENDER_DIR": globals()["RENDER_DIR"],
        "config.INPUT_DIR": runtime_config.INPUT_DIR,
        "config.OUTPUT_DIR": runtime_config.OUTPUT_DIR,
        "config.JSON_DIR": runtime_config.JSON_DIR,
        "config.RENDER_DIR": runtime_config.RENDER_DIR,
        "config.DEBUG_DIR": runtime_config.DEBUG_DIR,
        "ocr_translate.OUTPUT_DIR": ocr_translate_module.OUTPUT_DIR,
        "ocr_translate.DEBUG_DIR": ocr_translate_module.DEBUG_DIR,
    }
    globals()["INPUT_DIR"] = Path(input_dir)
    globals()["OUTPUT_DIR"] = Path(output_dir)
    globals()["JSON_DIR"] = Path(json_dir)
    globals()["RENDER_DIR"] = Path(render_dir)
    runtime_config.INPUT_DIR = Path(input_dir)
    runtime_config.OUTPUT_DIR = Path(output_dir)
    runtime_config.JSON_DIR = Path(json_dir)
    runtime_config.RENDER_DIR = Path(render_dir)
    runtime_config.DEBUG_DIR = Path(debug_dir)
    ocr_translate_module.OUTPUT_DIR = Path(output_dir)
    ocr_translate_module.DEBUG_DIR = Path(debug_dir)
    prev_ptd: Optional[str] = None
    if pipeline_timing_difficulty is not None:
        prev_ptd = os.environ.get("PIPELINE_TIMING_DIFFICULTY")
        os.environ["PIPELINE_TIMING_DIFFICULTY"] = str(
            max(1, min(3, int(pipeline_timing_difficulty)))
        )
    prev_hurry: Optional[str] = os.environ.get("PIPELINE_HURRY_UP")
    os.environ["PIPELINE_HURRY_UP"] = "1" if hurry_up else "0"
    try:
        yield
    finally:
        globals()["INPUT_DIR"] = original["main.INPUT_DIR"]
        globals()["OUTPUT_DIR"] = original["main.OUTPUT_DIR"]
        globals()["JSON_DIR"] = original["main.JSON_DIR"]
        globals()["RENDER_DIR"] = original["main.RENDER_DIR"]
        runtime_config.INPUT_DIR = original["config.INPUT_DIR"]
        runtime_config.OUTPUT_DIR = original["config.OUTPUT_DIR"]
        runtime_config.JSON_DIR = original["config.JSON_DIR"]
        runtime_config.RENDER_DIR = original["config.RENDER_DIR"]
        runtime_config.DEBUG_DIR = original["config.DEBUG_DIR"]
        ocr_translate_module.OUTPUT_DIR = original["ocr_translate.OUTPUT_DIR"]
        ocr_translate_module.DEBUG_DIR = original["ocr_translate.DEBUG_DIR"]
        if pipeline_timing_difficulty is not None:
            if prev_ptd is None:
                os.environ.pop("PIPELINE_TIMING_DIFFICULTY", None)
            else:
                os.environ["PIPELINE_TIMING_DIFFICULTY"] = prev_ptd
        if prev_hurry is None:
            os.environ.pop("PIPELINE_HURRY_UP", None)
        else:
            os.environ["PIPELINE_HURRY_UP"] = prev_hurry


@contextmanager
def _install_gemini_retry_notifier(cb):
    ocr_translate_module.set_gemini_retry_notifier(cb)
    try:
        yield
    finally:
        ocr_translate_module.clear_gemini_retry_notifier()


def run_pipeline_job(
    image_paths,
    work_dir,
    language=None,
    bubble_summary_text=None,
    use_project_pass1_bubble_file: bool = False,
    on_stage: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    difficulty: int = 3,
    hurry_up: bool = False,
    output_language: Optional[OutputLanguage] = None,
):
    def _check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise JobCancelledError("Job cancelled by user request")

    difficulty = _coerce_pipeline_difficulty(difficulty)
    target_output_lang = output_language if output_language is not None else OutputLanguage.ENGLISH
    pipeline_start = time.time()
    retry_eta_extra_sec = 0.0
    reset_gemini_pass_outcomes()
    pass_outcomes: Dict[str, Any] = {}
    current_phase_state: Dict[str, Any] = {
        "phase": "starting",
        "label": "Starting…",
        "progress": 0.0,
    }

    _quiet_phase_stdout = {
        "pass_1": "Pass 1…",
        "pass_2": "Pass 2…",
        "pass_3": "Pass 3…",
        "rendering": "Generating image…",
        "completed": "Done.",
    }

    def _emit_phase(phase_id: str, label: str, progress: float, **extra_fields: Any) -> None:
        """Notify API/UI and print status (full line if verbose; short milestones otherwise)."""
        _check_cancel()
        elapsed = round(time.time() - pipeline_start, 1)
        prog = max(0.0, min(1.0, float(progress)))
        current_phase_state["phase"] = phase_id
        current_phase_state["label"] = label
        current_phase_state["progress"] = prog
        payload: Dict[str, Any] = {
            "phase": phase_id,
            "label": label,
            "progress": prog,
            "elapsed_sec": elapsed,
            "eta_extra_sec": round(retry_eta_extra_sec, 1),
        }
        payload.update(extra_fields)
        if on_stage is not None:
            try:
                on_stage(payload)
            except Exception:
                pass
        if not _pipeline_verbose():
            quiet = _quiet_phase_stdout.get(phase_id)
            if quiet:
                print(quiet, flush=True)
            return
        if _compact_verbose_logs():
            _skip = {
                "artifact_cleaning",
                "status_bar_extract",
                "pass_1",
                "pass_2_prep",
                "pass_3",
                "finalizing",
            }
            if phase_id in _skip:
                return
            if phase_id == "rendering":
                print("[pipeline] Rendering final images…", flush=True)
                return
            if phase_id == "completed":
                print(f"[pipeline] Final image generated  elapsed={elapsed}s", flush=True)
                return
            if phase_id == "pass_2":
                print(
                    f"[pipeline] {label}  progress={int(round(prog * 100))}%  elapsed={elapsed}s",
                    flush=True,
                )
                return
            return
        print(
            f"[pipeline] {label}  progress={int(round(prog * 100))}%  elapsed={elapsed}s",
            flush=True,
        )

    def _on_gemini_retry(payload: Dict[str, Any]) -> None:
        nonlocal retry_eta_extra_sec
        if bool(payload.get("clear_retry_label")):
            retry_eta_extra_sec = 0.0
            _check_cancel()
            clear_payload: Dict[str, Any] = {
                "phase": str(current_phase_state.get("phase") or "running"),
                "label": str(current_phase_state.get("label") or "Processing…"),
                "progress": float(current_phase_state.get("progress") or 0.0),
                "elapsed_sec": round(time.time() - pipeline_start, 1),
                "eta_extra_sec": round(retry_eta_extra_sec, 1),
            }
            if on_stage is not None:
                try:
                    on_stage(clear_payload)
                except Exception:
                    pass
            return
        eta_extra = payload.get("eta_extra_sec")
        if eta_extra is None:
            added_eta = max(0.0, float(payload.get("added_eta_sec") or 0.0))
            retry_eta_extra_sec = round(retry_eta_extra_sec + added_eta, 1)
        else:
            retry_eta_extra_sec = round(max(0.0, float(eta_extra)), 1)
        _check_cancel()
        retry_payload: Dict[str, Any] = {
            "phase": str(current_phase_state.get("phase") or "running"),
            "label": str(payload.get("label") or current_phase_state.get("label") or "Retrying…"),
            "progress": float(current_phase_state.get("progress") or 0.0),
            "elapsed_sec": round(time.time() - pipeline_start, 1),
            "eta_extra_sec": round(retry_eta_extra_sec, 1),
            "reset_phase_started_at": bool(payload.get("reset_phase_started_at")),
        }
        if on_stage is not None:
            try:
                on_stage(retry_payload)
            except Exception:
                pass

    work_dir = Path(work_dir)
    input_dir = work_dir / "input_images"
    output_dir = work_dir / "result"
    json_dir = work_dir / "result_json"
    render_dir = work_dir / "rendered_chat"
    debug_dir = work_dir / "debug_crops"
    for path in (input_dir, output_dir, json_dir, render_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_lang_note = None
    if language:
        code = language.lower()
        name = _KNOWN_LANGUAGES.get(code, code.upper())
        _set_source_language(code, name)
        source_lang_note = name

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required for this pipeline. "
            "Copy .env.example to .env in the project root, set GEMINI_API_KEY, then run `python main.py` again "
            "(the CLI loads .env automatically; Google Vision is optional)."
        )

    gemini_pass_sec: Dict[str, float] = {}

    images = [Path(p) for p in image_paths]
    if not images:
        raise ValueError("No input images were provided.")

    _check_cancel()

    with _override_runtime_dirs(
        input_dir,
        output_dir,
        json_dir,
        render_dir,
        debug_dir,
        pipeline_timing_difficulty=difficulty,
        hurry_up=hurry_up,
    ), _install_gemini_retry_notifier(_on_gemini_retry):
        if not _compact_verbose_logs():
            _vprint(f"[JOB] job_dir={work_dir}")
        if _compact_verbose_logs():
            _vprint("**************")
        _vprint(
            f"[GEMINI] HTTP timeouts: pass1={gemini_pass_timeout_sec(1)}s "
            f"pass2={gemini_pass_timeout_sec(2)}s pass3={gemini_pass_timeout_sec(3)}s"
        )
        if not _pipeline_verbose():
            print("Starting up…", flush=True)

        _emit_phase("artifact_cleaning", "Cleaning artifacts…", 0.06)
        crop_infos = [prepare_image_crop_info(str(path)) for path in images]
        if not crop_infos:
            raise RuntimeError("No usable images after preprocessing.")

        _check_cancel()

        _emit_phase("status_bar_extract", "Extracting status bar…", 0.12)
        contact_name = _extract_contact_name(crop_infos[0]) if crop_infos else ""
        page_images = extract_page_segment_images(crop_infos)
        status_bar_images = extract_status_bar_images(crop_infos)
        combined_img, _page_ranges = build_combined_image(crop_infos, global_first_page_index=0)
        if combined_img is None:
            raise RuntimeError("Empty combined image.")

        ch, cw = combined_img.shape[0], combined_img.shape[1]
        _ready = (
            f"[pipeline] Input ready: {len(images)} file(s) loaded, "
            f"{len(page_images)} page segment(s), combined canvas {cw}×{ch} px"
        )
        _vprint(_ready if _compact_verbose_logs() else f"{_ready} — OK to use")
        if _compact_verbose_logs():
            _vprint("**************")

        _check_cancel()

        ocr_hints = None
        ocr_hints_format = "none"
        ocr_pass2_by_message = None
        pass1_exclude_ocr = True
        pass1_message_images = []
        craft_expected_n = None
        pass1_bubble_context = ""
        _manual_page_specs: list = []
        bs = (bubble_summary_text or "").strip()
        if bs:
            try:
                pass1_bubble_context, craft_expected_n, _manual_page_specs = _parse_bubble_summary_text(
                    bs, len(page_images)
                )
            except ValueError as exc:
                print(f"[pipeline] Bubble summary from request ignored: {exc}", flush=True)
        elif use_project_pass1_bubble_file:
            loaded = _load_pass1_bubble_summary_file(len(page_images))
            if loaded is not None:
                pass1_bubble_context, craft_expected_n, _manual_page_specs = loaded
                _vprint(
                    f"\n[pipeline] Using bubble guidance file {PASS1_BUBBLE_INPUT_PATH.name} for Pass 1…",
                )

        if pass1_bubble_context:
            bubble_debug_path = os.path.join(OUTPUT_DIR, "pass1_bubble_summary.txt")
            with open(bubble_debug_path, "w", encoding="utf-8") as f:
                f.write(pass1_bubble_context)

        _emit_phase("pass_1", "Pass 1 — vision transcription…", 0.18)
        if not _compact_verbose_logs():
            _vprint(
                "[pipeline] Pass 1 — waiting for Gemini (vision). "
                "The next line shows a live timer until the API responds.",
            )
        _t_g1 = time.time()
        ok, g_contact, all_meta, pre_ocr_meta, pass_debug = translate_conversation_gemini_multimodal(
            page_images,
            combined_img,
            pass1_message_images=pass1_message_images,
            craft_vertical_bands_markdown=pass1_bubble_context,
            contact_hint=contact_name,
            source_language_name=source_lang_note,
            ocr_hints=ocr_hints,
            ocr_hints_format=ocr_hints_format,
            pass1_exclude_ocr_hints=pass1_exclude_ocr,
            craft_expected_message_rows=craft_expected_n,
            ocr_pass2_by_message=ocr_pass2_by_message,
        )
        gemini_pass_sec["pass1"] = round(time.time() - _t_g1, 2)
        pass_outcomes["pass1"] = get_gemini_pass_outcomes().get(1) or {}
        _log_pass_attempts("pass1", pass_outcomes["pass1"])
        if not ok or not all_meta:
            failure_reason = str((pass_debug or {}).get("failure_reason") or "").strip()
            if failure_reason == "request_failed":
                detail = str((pass_debug or {}).get("exception") or "").strip()
                if detail.startswith("SERVERS_OVERLOADED:"):
                    raise RuntimeError(detail)
                raise RuntimeError(
                    "Gemini Pass 1 request failed."
                    + (f" {detail}" if detail else "")
                )
            if failure_reason == "empty_response":
                raise RuntimeError("Gemini Pass 1 returned an empty response.")
            if failure_reason == "json_parse_failed":
                detail = str((pass_debug or {}).get("parse_error") or "").strip()
                raise RuntimeError(
                    "Gemini Pass 1 returned unreadable JSON."
                    + (f" {detail}" if detail else "")
                )
            if failure_reason == "no_messages_parsed":
                raw_n = (pass_debug or {}).get("raw_message_count")
                sys_n = (pass_debug or {}).get("system_message_count")
                raise RuntimeError(
                    "Gemini Pass 1 returned JSON, but no usable chat messages were parsed."
                    + (
                        f" rows={raw_n} system_rows={sys_n}"
                        if raw_n is not None and sys_n is not None
                        else ""
                    )
                )
            if failure_reason == "no_page_images":
                raise RuntimeError("No usable page images were available for Gemini Pass 1.")
            if failure_reason == "gemini_not_configured":
                raise RuntimeError("Gemini is not configured for this pipeline.")
            if failure_reason == "meta_build_empty":
                raise RuntimeError("Gemini Pass 1 produced messages, but render metadata was empty.")
            raise RuntimeError("Gemini did not return a usable conversation.")

        _check_cancel()

        pass1_for_pass2 = (pass_debug or {}).get("pass1_messages") or []
        contact_name = (g_contact or contact_name or "Person A").strip() or "Person A"
        pass1_meta = list(all_meta or [])
        for i, item in enumerate(pre_ocr_meta or []):
            item["order"] = i
        for i, item in enumerate(pass1_meta):
            item["order"] = i

        pass2_ocr_debug_path = os.path.join(OUTPUT_DIR, "pass2_ocr_debug.txt")
        if difficulty >= 2:
            _emit_phase("pass_2_prep", "Clustering OCR + matching to Pass 1…", 0.50)
            try:
                pass2_ocr_text, ocr_match_verified, pass2_ocr_meta = build_pass2_per_message_ocr_hints(
                    page_images,
                    pass1_for_pass2,
                )
            except Exception as exc:
                print(
                    f"[pipeline] Pass 2 OCR clustering failed ({exc}); "
                    "continuing without per-message OCR hints.",
                    flush=True,
                )
                pass2_ocr_text = ""
                ocr_match_verified = [False] * len(pass1_for_pass2)
                pass2_ocr_meta = {"skipped": True, "reason": "ocr_clustering_exception", "error": str(exc)}
            with open(pass2_ocr_debug_path, "w", encoding="utf-8") as f:
                f.write(pass2_ocr_text or "")
                f.write("\n\n--- pass2_ocr_meta.json ---\n")
                f.write(json.dumps(pass2_ocr_meta, ensure_ascii=False, indent=2))

            _check_cancel()

            _emit_phase("pass_2", "Pass 2 — OCR-guided transcript refine…", 0.56)
            if _compact_verbose_logs():
                _vprint("[pipeline] Pass 2 — waiting for Gemini (OCR-guided refine).")
            else:
                _vprint(
                    "[pipeline] Pass 2 — waiting for Gemini (OCR-guided refine). "
                    "Live timer on the line below.",
                )

            _t_g2 = time.time()
            try:
                pass2_messages, contact_name2, pass2_meta = _gemini_ocr_hints_refine_pass(
                    contact_name,
                    pass1_for_pass2,
                    pass2_ocr_text,
                    timeout=gemini_pass_timeout_sec(2),
                    ocr_match_verified=ocr_match_verified,
                )
            except Exception as exc:
                print(
                    f"[pipeline] Pass 2 Gemini refine failed ({exc}); using Pass 1 transcript.",
                    flush=True,
                )
                pass2_messages = [dict(m) for m in pass1_for_pass2]
                contact_name2 = None
                last_p2 = get_gemini_pass_outcomes().get(2) or {}
                pass2_meta = {
                    "applied": False,
                    "successful_attempt": last_p2.get("successful_attempt"),
                    "status": last_p2.get("status"),
                    "max_tries": last_p2.get("max_tries"),
                    "timed_out_attempts": last_p2.get("timed_out_attempts"),
                    "transient_status_retry_count": last_p2.get("transient_status_retry_count"),
                    "final_http_status": last_p2.get("final_http_status"),
                    "model": last_p2.get("model"),
                    "model_failovers": last_p2.get("model_failovers"),
                    "reason": "wrapper_exception",
                }
            gemini_pass_sec["pass2"] = round(time.time() - _t_g2, 2)
            pass_outcomes["pass2"] = pass2_meta
            _log_pass_attempts("pass2", pass2_meta)
            if contact_name2:
                contact_name = contact_name2
        else:
            with open(pass2_ocr_debug_path, "w", encoding="utf-8") as f:
                f.write(
                    "Pass 2 OCR hints were not built (pipeline difficulty=1: Gemini Pass 2 disabled).\n"
                )
                f.write("\n\n--- pass2_ocr_meta.json ---\n")
                f.write(
                    json.dumps({"skipped": True, "reason": "difficulty=1"}, ensure_ascii=False, indent=2)
                )
            pass2_messages = _pass2_messages_from_pass1(pass1_for_pass2)
            pass_outcomes["pass2"] = {
                "applied": False,
                "successful_attempt": None,
                "status": None,
                "reason": "difficulty=1",
            }

        all_meta = _meta_from_gemini_messages(pass2_messages)
        for i, item in enumerate(all_meta or []):
            item["order"] = i

        _check_cancel()

        if difficulty >= 3:
            _emit_phase("pass_3", "Pass 3 — reference resolution & header…", 0.78)
            if not _compact_verbose_logs():
                _vprint(
                    "[pipeline] Pass 3 — waiting for Gemini (reference resolution + status bar from first image). "
                    "Live timer on the line below.",
                )
            status_bar_b64 = None
            if status_bar_images:
                _sb0 = status_bar_images[0]
                if _sb0 is not None and getattr(_sb0, "size", 0) != 0:
                    _sb_b64 = _jpeg_b64_from_bgr(_sb0, quality=92)
                    status_bar_b64 = _sb_b64 if _sb_b64 else None
            _t_g3 = time.time()
            try:
                final_messages, contact_name3, pass3_reference_debug, status_bar_info, pass3_meta = (
                    _gemini_reference_resolution_pass(
                        contact_name,
                        pass2_messages,
                        timeout=gemini_pass_timeout_sec(3),
                        status_bar_b64=status_bar_b64,
                    )
                )
            except Exception as exc:
                print(
                    f"[pipeline] Pass 3 failed ({exc}); using Pass 2 transcript with machine translation.",
                    flush=True,
                )
                final_messages = _pass3_fallback_messages_from_pass2(pass2_messages)
                contact_name3 = None
                pass3_reference_debug = []
                status_bar_info = _default_status_bar_info(contact_name)
                last_p3 = get_gemini_pass_outcomes().get(3) or {}
                pass3_meta = {
                    "applied": False,
                    "successful_attempt": last_p3.get("successful_attempt"),
                    "status": last_p3.get("status"),
                    "max_tries": last_p3.get("max_tries"),
                    "timed_out_attempts": last_p3.get("timed_out_attempts"),
                    "transient_status_retry_count": last_p3.get("transient_status_retry_count"),
                    "final_http_status": last_p3.get("final_http_status"),
                    "model": last_p3.get("model"),
                    "model_failovers": last_p3.get("model_failovers"),
                    "reason": "wrapper_exception",
                }
            gemini_pass_sec["pass3"] = round(time.time() - _t_g3, 2)
            pass_outcomes["pass3"] = pass3_meta
            _log_pass_attempts("pass3", pass3_meta)
            if contact_name3:
                contact_name = contact_name3
        else:
            final_messages = [dict(m) for m in pass2_messages]
            pass3_reference_debug = []
            status_bar_info = _default_status_bar_info(contact_name)
            pass_outcomes["pass3"] = {
                "applied": False,
                "successful_attempt": None,
                "status": None,
                "reason": "difficulty<3",
            }

        final_chat_meta = _meta_from_gemini_messages(final_messages)
        for i, item in enumerate(final_chat_meta or []):
            item["order"] = i
        final_render_meta = _merge_chat_with_system_metadata(
            final_chat_meta,
            (pass_debug or {}).get("pass1_system_messages") or [],
        )

        _check_cancel()
        final_contact_name_src = (status_bar_info.get("contact_name") or contact_name or "Person A").strip() or "Person A"
        final_status_text_src = (status_bar_info.get("status_text") or "").strip()
        final_contact_name = (translate_to_en(final_contact_name_src) or final_contact_name_src).strip() or "Person A"
        final_status_text = (translate_to_en(final_status_text_src) or final_status_text_src).strip()
        # Profile photo from screenshots disabled for now (generic stub in renderer).
        profile_image = None

        json_and_final_meta = final_render_meta
        display_contact_name = final_contact_name
        display_status_text = final_status_text
        if target_output_lang != OutputLanguage.ENGLISH:
            json_and_final_meta = copy.deepcopy(final_render_meta)
            _localize_render_meta_for_output(json_and_final_meta, target_output_lang)
            oc = final_contact_name
            if oc and not _looks_like_link_text(oc):
                oc = translate_en_to(oc, target_output_lang.value) or oc
            display_contact_name = (oc or "Person A").strip() or "Person A"
            st = final_status_text
            if st and not _looks_like_link_text(st):
                st = translate_en_to(st, target_output_lang.value) or st
            display_status_text = st.strip()

        _check_cancel()

        _emit_phase("finalizing", "Writing outputs & debug JSON…", 0.92)
        pass1_source_debug_path = os.path.join(JSON_DIR, "pass1_transcript_debug.json")
        pass3_reference_debug_path = os.path.join(JSON_DIR, "pass3_reference_debug.json")
        pass1_source_debug = []
        for i, m in enumerate((pass_debug or {}).get("pass1_messages") or []):
            pass1_source_debug.append({
                "message_index": i,
                "role": m.get("role"),
                "text_src": m.get("text_src") or "",
                "text_en_debug": translate_to_en(m.get("text_src") or "") or "",
            })
        with open(pass1_source_debug_path, "w", encoding="utf-8") as f:
            json.dump(pass1_source_debug, f, indent=2, ensure_ascii=False)
        with open(pass3_reference_debug_path, "w", encoding="utf-8") as f:
            json.dump(pass3_reference_debug or [], f, indent=2, ensure_ascii=False)

        combined_json_path = os.path.join(JSON_DIR, "translated_conversation.json")
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(json_and_final_meta, f, indent=2, ensure_ascii=False)

        _emit_phase("rendering", "Rendering final images…", 0.96)
        # Per-pass / compare PNGs stay English in the header; only the main artifact uses localized names.
        _render_header_debug = dict(
            profile_image=profile_image,
            contact_name=final_contact_name,
            header_status=final_status_text,
        )
        _render_header_final = dict(
            profile_image=profile_image,
            contact_name=display_contact_name,
            header_status=display_status_text,
        )
        _top_banner = resolve_top_banner_path()
        if not _top_banner:
            print(
                "[pipeline] No top banner image found — Facebook bar is skipped. "
                "Add and commit `assets/top_banner.png`, or set env `CHAT_TOP_BANNER_PATH` "
                "to a PNG path on this machine.",
                flush=True,
            )
        pass1_raw = render_chat(pass1_meta, **_render_header_debug)
        pass2_raw = render_chat(all_meta, **_render_header_debug)
        pass3_raw = render_chat(final_chat_meta, **_render_header_debug)

        def _write_required_image(path: str, image, artifact_label: str) -> None:
            ok = bool(cv2.imwrite(path, image))
            if not ok or not os.path.exists(path):
                raise RuntimeError(f"{artifact_label} could not be written.")

        pass1_chat = composite_chat_below_top_banner(pass1_raw, _top_banner)
        pass1_path = os.path.join(RENDER_DIR, "translated_conversation_pass1.png")
        _write_required_image(pass1_path, pass1_chat, "Pass 1 image")

        pass2_img = composite_chat_below_top_banner(pass2_raw, _top_banner)
        pass2_path = os.path.join(RENDER_DIR, "translated_conversation_pass2.png")
        _write_required_image(pass2_path, pass2_img, "Pass 2 image")

        pass3_img = composite_chat_below_top_banner(pass3_raw, _top_banner)
        pass3_path = os.path.join(RENDER_DIR, "translated_conversation_pass3.png")
        _write_required_image(pass3_path, pass3_img, "Pass 3 image")

        if difficulty >= 3:
            _compare_panels = [
                (pass1_raw, "Pass 1 Translation"),
                (pass2_raw, "Pass 2 OCR Source Polish (Debug EN)"),
                (pass3_raw, "Pass 3 Reference Resolution"),
            ]
        elif difficulty == 2:
            _compare_panels = [
                (pass1_raw, "Pass 1 Translation"),
                (pass2_raw, "Pass 2 OCR Source Polish (Debug EN)"),
            ]
        else:
            _compare_panels = [
                (pass1_raw, "Pass 1 Translation"),
            ]
        compare_img = _compose_labeled_chat_panels(_compare_panels)
        compare_path = os.path.join(RENDER_DIR, "translated_conversation_compare.png")
        if compare_img is not None:
            _write_required_image(compare_path, compare_img, "Compare image")

        combined_path = os.path.join(RENDER_DIR, "translated_conversation.png")
        final_chat = render_chat(json_and_final_meta, **_render_header_final)
        final_chat = composite_chat_below_top_banner(final_chat, _top_banner)
        _write_required_image(combined_path, final_chat, "Final image")
        _emit_phase("completed", "Final image generated", 1.0)
        if _compact_verbose_logs():
            _vprint(f"\nfinal_image={combined_path}\n")
        else:
            _vprint(f"[JOB] messages={len(json_and_final_meta)} final_image={combined_path}")
            p1 = gemini_pass_sec.get("pass1")
            p2 = gemini_pass_sec.get("pass2")
            p3 = gemini_pass_sec.get("pass3")
            if p1 is not None and p2 is not None and p3 is not None:
                _vprint(
                    f"[pipeline] Gemini pass wall times (this run): "
                    f"pass1={p1:.1f}s  pass2={p2:.1f}s  pass3={p3:.1f}s  "
                    f"(sum={p1 + p2 + p3:.1f}s; see [gemini] lines for API thinkingBudget / thoughts_tokens)",
                )

    return {
        "job_dir": str(work_dir),
        "language": language,
        "output_language": target_output_lang.value,
        "difficulty": difficulty,
        "hurry_up": bool(hurry_up),
        "contact_name": display_contact_name,
        "status_text": display_status_text,
        "images_processed": len(images),
        "messages_rendered": len(json_and_final_meta),
        "total_runtime_sec": round(time.time() - pipeline_start, 2),
        "gemini_pass_sec": dict(gemini_pass_sec),
        "pass_outcomes": pass_outcomes,
        "artifacts": {
            "json": str(json_dir / "translated_conversation.json"),
            "pass1_debug": str(json_dir / "pass1_transcript_debug.json"),
            "pass3_debug": str(json_dir / "pass3_reference_debug.json"),
            "final_image": str(render_dir / "translated_conversation.png"),
            "compare_image": str(render_dir / "translated_conversation_compare.png"),
            "pass1_image": str(render_dir / "translated_conversation_pass1.png"),
            "pass2_image": str(render_dir / "translated_conversation_pass2.png"),
            "pass3_image": str(render_dir / "translated_conversation_pass3.png"),
            "ocr_debug": str(output_dir / "pass2_ocr_debug.txt"),
            "gemini_debug": str(output_dir / "gemini_debug.txt"),
        },
    }


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    args = _parse_args()
    if args.verbose:
        os.environ["PIPELINE_VERBOSE"] = "1"
    images = sorted(
        p
        for p in Path(INPUT_DIR).glob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        print(
            "[pipeline] Add .png / .jpg / .webp / .bmp files to input_images/ (see README).",
            file=sys.stderr,
        )
        sys.exit(1)
    if _compact_verbose_logs():
        _vprint("\n******[pipeline] Gemini full-vision chat translator************\n")
        _vprint("******************")
        if args.source_language:
            _vprint(f"[pipeline] Source language hint: {args.source_language}")
        else:
            _vprint("[pipeline] Source language: infer from screenshots")
        _vprint(f"[pipeline] Found {len(images)} image(s)")
        _vprint(
            f"[pipeline] --language (output)={args.output_language.value} "
            f"({args.output_language.name.lower().replace('_', ' ')})"
        )
        _vprint(
            f"[pipeline] difficulty={args.difficulty} "
            f"(1=Pass1 only, 2=Pass1–2, 3=full Pass1–3)  "
            f"hurry_up={'on' if args.hurry_up else 'off'}"
        )
        _vprint("*****************\n")
    else:
        _vprint("\n[pipeline] Gemini full-vision chat translator\n")
        if args.source_language:
            _vprint(f"[pipeline] Source language hint: {args.source_language}")
        else:
            _vprint("[pipeline] Source language: infer from screenshots")
        _vprint(f"[pipeline] Found {len(images)} image(s) under {INPUT_DIR}")
        _vprint(
            f"[pipeline] --language (output)={args.output_language.value} "
            f"({args.output_language.name.lower().replace('_', ' ')})"
        )
        _vprint(
            f"[pipeline] difficulty={args.difficulty} "
            f"(1=Pass1 only, 2=Pass1–2, 3=full Pass1–3)  "
            f"hurry_up={'on' if args.hurry_up else 'off'}\n"
        )
    result = run_pipeline_job(
        images,
        Path(__file__).resolve().parent,
        language=args.source_language,
        use_project_pass1_bubble_file=True,
        difficulty=args.difficulty,
        hurry_up=args.hurry_up,
        output_language=args.output_language,
    )
    if not _compact_verbose_logs():
        _vprint(f"[pipeline] Output JSON → {result['artifacts']['json']}")
        _vprint(f"[pipeline] Output image → {result['artifacts']['final_image']}")
        _vprint(f"[pipeline] Output compare → {result['artifacts']['compare_image']}")
    gps = result.get("gemini_pass_sec") or {}
    g1, g2, g3 = gps.get("pass1"), gps.get("pass2"), gps.get("pass3")
    gemini_line = ""
    if g1 is not None and g2 is not None and g3 is not None:
        gemini_line = f"\n  Gemini passes    : pass1={g1:.1f}s  pass2={g2:.1f}s  pass3={g3:.1f}s"
    _vprint(f"""
╔══════════════════════════════════════════╗
  Pipeline summary (full-vision Gemini)
  Images processed : {result['images_processed']}
  Messages rendered: {result['messages_rendered']}
  Contact name     : {result['contact_name']}
  Total runtime    : {result['total_runtime_sec']:.1f}s{gemini_line}
╚══════════════════════════════════════════╝""")

if __name__ == "__main__":
    main()