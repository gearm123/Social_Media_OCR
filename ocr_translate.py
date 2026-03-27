import os
import json
import cv2
import time
import sys
import re
from difflib import SequenceMatcher
from typing import NamedTuple, Optional, Any, Dict
import numpy as np
from config import (
    DEBUG_DIR,
    GOOGLE_VISION_API_KEY,
    OUTPUT_DIR,
    ocr_engine,
    translator,
    translation_cache,
)

# --------------------------------------------------
# SOURCE LANGUAGE  (auto-detected from Vision OCR)
# --------------------------------------------------
# Populated after the first OCR call from ocr_and_translate_full_image.
_source_lang_code = "auto"
_source_lang_name = "the source language"


def _set_source_language(code: str, name: str):
    global _source_lang_code, _source_lang_name
    _source_lang_code = code
    _source_lang_name = name


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
            print(f"[OCR HINT] Page {i + 1} Vision failed: {e}")
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
            print(f"[OCR STRUCT] Page {pi + 1} Vision failed: {e}")
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
        print(
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
            print(f"[OCR STITCH] Page {pi + 1} Vision failed: {e}")
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
        print(f"[OCR BY-MSG] Vision on stitched image failed: {e}")
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
            print(f"[OCR BY-MSG] Vision on crop {midx} failed: {e}")
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
            print(f"[OCR BY-MSG] Vision on page {page_idx} failed: {e}")
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
            print(f"[GEMINI] ListModels error ({ver}): {e}")
    return None, None


class GeminiApiResult(NamedTuple):
    """Result of a ``generateContent`` call (text + full parsed JSON for debugging)."""

    text: str
    response_json: Optional[Dict[str, Any]]


def _gemini_discover_if_needed() -> bool:
    """Load API key from env and pick a model. Returns False if Gemini is unavailable."""
    global _gemini_api_key, _gemini_active_model

    if _gemini_api_key is None:
        _gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
    if not _gemini_api_key:
        return False
    if _gemini_active_model is None:
        print("[GEMINI] Discovering available models...")
        model, ver = _gemini_discover_model(_gemini_api_key)
        if model:
            _gemini_active_model = (model, ver)
            print(f"[GEMINI] Using model: {model} (API {ver})")
        else:
            print("[GEMINI] No generateContent-capable model found — refinement disabled")
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


def _gemini_generate(
    prompt,
    image_b64=None,
    image_b64_list=None,
    timeout=120,
    max_output_tokens_override: Optional[int] = None,
):
    """Call Gemini REST API directly (text-only or multimodal).

    Pass *image_b64* (single JPEG base64) or *image_b64_list* (several JPEGs in order)
    before the text prompt.

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
        max_out = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "32768"))
    gen_cfg: Dict[str, Any] = {
        "maxOutputTokens": max(1024, min(max_out, 65536)),
        "temperature": 0.0,
        "topP": 1.0,
    }
    # Gemini 2.5 **Flash** (not Pro): internal "thinking" shares the output token cap → truncated JSON.
    # Default thinkingBudget=0 for OCR/translation (set GEMINI_THINKING_BUDGET=-1 for dynamic thinking).
    # 2.5 Pro cannot disable thinking per API docs — omit thinkingConfig for Pro.
    _mname = (model or "").lower()
    if "2.5" in _mname and "pro" not in _mname:
        _tb = os.environ.get("GEMINI_THINKING_BUDGET", "0").strip()
        try:
            thinking_budget = int(_tb)
        except ValueError:
            thinking_budget = 0
        gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    # Optional: force JSON body (no markdown fence). Disable if the API rejects your model/payload.
    _jm = os.environ.get("GEMINI_JSON_OUTPUT", "0").strip().lower()
    if _jm not in ("0", "false", "no", "off"):
        gen_cfg["responseMimeType"] = "application/json"
    payload: Dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": gen_cfg,
    }
    # Chat screenshots often trip default safety (meds, travel, etc.) → empty candidates.
    _rel = os.environ.get("GEMINI_SAFETY_RELAXED", "1").strip().lower()
    if _rel not in ("0", "false", "no", "off"):
        payload["safetySettings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

    image_count = len(image_b64_list or ([] if not image_b64 else [image_b64]))
    prompt_chars = len(prompt or "")
    print(
        f"[GEMINI] Request start: model={model} images={image_count} "
        f"prompt_chars={prompt_chars} max_tokens={gen_cfg['maxOutputTokens']} timeout={timeout}s",
        flush=True,
    )
    t_http = time.time()
    r = _req.post(url, json=payload, timeout=timeout)
    print(
        f"[GEMINI] HTTP response received in {time.time()-t_http:.1f}s with status {r.status_code}",
        flush=True,
    )
    r.raise_for_status()
    t_json = time.time()
    data = r.json()
    print(
        f"[GEMINI] Response JSON parsed in {time.time()-t_json:.1f}s",
        flush=True,
    )

    pf = data.get("promptFeedback")
    if pf:
        print(f"[GEMINI] promptFeedback: {pf}")

    candidates = data.get("candidates") or []
    if not candidates:
        print(
            "[GEMINI] No candidates in API response — often prompt blocked or request too large. "
            "See result/gemini_debug.txt (=== GEMINI API ===)."
        )
        return GeminiApiResult("", data)

    c0 = candidates[0]
    fr = c0.get("finishReason")
    usage = data.get("usageMetadata") or {}
    thoughts_tokens = usage.get("thoughtsTokenCount")
    total_tokens = usage.get("totalTokenCount")
    print(
        f"[GEMINI] Candidate summary: finishReason={fr or 'unknown'} "
        f"totalTokens={total_tokens if total_tokens is not None else 'n/a'} "
        f"thoughtsTokens={thoughts_tokens if thoughts_tokens is not None else 'n/a'}",
        flush=True,
    )
    if fr and fr not in ("STOP", "MAX_TOKENS", "FINISH_REASON_STOP", "FINISH_REASON_MAX_TOKENS"):
        print(f"[GEMINI] finishReason: {fr}")

    content = c0.get("content") or {}
    resp_parts = content.get("parts") or []
    text = "".join(
        p.get("text", "") for p in resp_parts if isinstance(p, dict)
    )
    sr = c0.get("safetyRatings")
    if sr and not text.strip():
        print(f"[GEMINI] safetyRatings: {sr}")
    if not text.strip():
        print(
            "[GEMINI] Empty model text — check finishReason / promptFeedback in "
            "result/gemini_debug.txt (=== GEMINI API ===)."
        )
    return GeminiApiResult(text, data)


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
            print(f"[GEMINI] Stitch prepared as {n} vertical band(s)")
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
        print(f"[GEMINI] Prompt file → {path}")
    except Exception as exc:
        print(f"[GEMINI] Could not write prompt file {filename}: {exc}")


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
            print(
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
        normalized.append(
            {
                "role": role,
                "text_en": te or text_src or text_en_debug,
                "text_src": text_src,
                "crop_text_src": crop_text_src,
                "text_en_debug": text_en_debug,
                "legibility": (m.get("legibility") or "").strip(),
                "note": (m.get("note") or "").strip(),
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
        gres = _gemini_generate(prompt, timeout=timeout)
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
        print(
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
            print(
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
        print("[GEMINI] Pass 2 left some ⟦REF⟧ tokens — check gemini_debug.txt")
    else:
        print("[GEMINI] Pass 2 applied (final English translation).")
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
        print(f"[GEMINI] Could not append OCR-refine debug: {exc}")


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
        print(f"[GEMINI] Could not append crop-refine debug: {exc}")


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
            gres = _gemini_generate(prompt, image_b64_list=img_b64_list, timeout=timeout)
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
        gres = _gemini_generate(prompt, image_b64_list=img_b64_list, timeout=timeout)
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
        print(
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
        print(
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


def _gemini_ocr_hints_refine_pass(
    contact_name: str,
    pass1_messages: list,
    ocr_page_text: str,
    timeout: int,
):
    """Pass 2: rewrite Pass 1 transcript using OCR page hints."""
    if not pass1_messages:
        return pass1_messages, contact_name

    payload = {
        "contact_name": contact_name,
        "messages": [
            {
                "message_index": i,
                "role": m.get("role"),
                "pass1_text_src": (m.get("text_src") or m.get("text_en") or "").strip(),
            }
            for i, m in enumerate(pass1_messages)
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    expected_n = len(pass1_messages)
    hints_full = (ocr_page_text or "").strip()
    max_hint_chars = int(os.environ.get("GEMINI_PASS2_OCR_MAX_CHARS", "24000"))
    retry_hint_chars = int(os.environ.get("GEMINI_PASS2_OCR_RETRY_MAX_CHARS", "12000"))

    def _truncate_hints(text: str, limit: int) -> str:
        text = (text or "").strip()
        if not text or len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n\n[... OCR hints truncated ...]"

    def _build_prompt(hints_text: str, retry_note: str = "") -> str:
        return f"""You are PASS 2 of 2.

You receive:
1. A conversation transcript from Pass 1.
2. High-confidence OCR hints grouped by image, in chronological order.

Input JSON:
{payload_json}

OCR hints from the cleaned images:
{hints_text if hints_text else "(no OCR hints available)"}

Task:
- Rewrite the conversation using the OCR hints only as soft supporting evidence.
- Keep the same number of messages: {expected_n}.
- Keep the same order.
- Keep the same role for each message.
- Improve words, names, URLs, numbers, and short phrases when the OCR clearly supports a better reading.
- Do not add, remove, merge, or split messages.
- Ignore OCR fragments that look like UI noise or make the message worse.
- Return English in `text_en`.
{retry_note}

Output JSON only:
{{"contact_name":"<string>","messages":[{{"message_index":0,"role":"contact|user|system","text_src":"<final source reading>","text_en":"<final English line>"}}]}}"""

    hints = _truncate_hints(hints_full, max_hint_chars)
    prompt = _build_prompt(hints)
    _write_gemini_prompt_file("gemini_prompt_pass2.txt", "Pass 2 (OCR-guided rewrite)", prompt)
    try:
        gres = _gemini_generate(prompt, timeout=timeout)
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 2 failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}", "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name

    if not raw.strip():
        print("[GEMINI] Pass 2 empty — retrying once with shorter OCR hints")
        retry_hints = _truncate_hints(hints_full, retry_hint_chars)
        retry_prompt = _build_prompt(
            retry_hints,
            retry_note="- If OCR support is weak, keep the original wording and only fix lines where the OCR clearly helps.",
        )
        try:
            gres = _gemini_generate(retry_prompt, timeout=timeout)
            raw = (gres.text if gres else "") or ""
            prompt = retry_prompt
        except Exception as e:
            print(f"[GEMINI] Pass 2 retry failed: {e}")
            _append_gemini_debug_pass2(retry_prompt, f"ERROR: {e}", "PASS 2 — OCR-guided rewrite retry")
            return pass1_messages, contact_name
        if not raw.strip():
            print("[GEMINI] Pass 2 empty after retry — keeping Pass 1 transcript")
            _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite retry")
            return pass1_messages, contact_name

    try:
        name2, msgs2, _ledger2 = _parse_gemini_full_vision_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 2 JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name

    if not msgs2 or len(msgs2) != len(pass1_messages):
        print(
            f"[GEMINI] Pass 2 message count mismatch ({len(msgs2)} vs {len(pass1_messages)}) — "
            "keeping Pass 1 transcript"
        )
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
        return pass1_messages, contact_name

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
        print(
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
    for i, mm in enumerate(merged):
        src = (mm.get("text_src") or "").strip()
        en_raw = (msgs2[i].get("text_en") or "").strip()
        mm["text_en"] = _prefer_english_surface(en_raw, translate_to_en(src) or src)

    final_name = (name2 or contact_name or "").strip() or contact_name
    if not role_drift_indices:
        _append_gemini_debug_pass2(prompt, raw, "PASS 2 — OCR-guided rewrite")
    return merged, final_name


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
    """Pass 3: extract final header information from status-bar crops."""
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

    prompt = f"""You are PASS 3 of 3.

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

    _write_gemini_prompt_file("gemini_prompt_pass3_statusbar.txt", "Pass 3 status bar", prompt)

    try:
        gres = _gemini_generate(prompt, image_b64_list=img_b64_list, timeout=timeout)
        raw = (gres.text if gres else "") or ""
    except Exception as e:
        print(f"[GEMINI] Pass 3 status-bar failed: {e}")
        _append_gemini_debug_pass2(prompt, f"ERROR: {e}", "PASS 3 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    if not raw.strip():
        _append_gemini_debug_pass2(prompt, raw, "PASS 3 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    try:
        info = _parse_gemini_status_bar_json(raw)
    except json.JSONDecodeError as e:
        print(f"[GEMINI] Pass 3 status-bar JSON parse failed: {e}")
        _append_gemini_debug_pass2(prompt, raw, "PASS 3 — Status bar extraction")
        return {
            "contact_name": (contact_hint or "").strip(),
            "status_text": "",
            "avatar_image_index": 0,
            "avatar_bbox": None,
        }

    if not info.get("contact_name"):
        info["contact_name"] = (contact_hint or "").strip()
    _append_gemini_debug_pass2(prompt, raw, "PASS 3 — Status bar extraction")
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
        print(f"[GEMINI] Could not append pass-2 debug: {exc}")


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
        print(f"[GEMINI] Debug file → {path}")
    except Exception as exc:
        print(f"[GEMINI] Could not write debug file: {exc}")


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
    if not _gemini_discover_if_needed():
        return False, "", [], [], {}

    timeout = int(os.environ.get("GEMINI_REQUEST_TIMEOUT_SEC", "600"))
    b64_list = []
    pass1_message_images = list(pass1_message_images or [])
    for page_idx, img in enumerate(list(page_images or [])):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        b64 = _jpeg_b64_from_bgr(img, quality=90)
        if b64:
            b64_list.append(b64)
            print(
                f"[GEMINI] Pass 1 page {page_idx + 1}/{len(page_images)}: "
                f"{img.shape[1]}x{img.shape[0]}px ({len(b64) // 1024}KB base64)"
            )

    if not b64_list:
        print("[GEMINI] No page images to send")
        return False, "", [], [], {}

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
    prompt = f"""{intro}

transcribe the conversation in the original language
the first image is the oldest

{craft_section}{bubble_section}transcribe what you see
the added information above is guidance about the actual text bubbles only
for actual chat bubbles, map receiver -> role "contact" and sender -> role "user"
keep the actual chat bubbles in that same top-to-bottom order
if you clearly see timestamps, missed calls, transfer notices, or other centered/system conversation rows, include them too as role "system"
those system rows are metadata only and must not change the receiver/sender bubble order above

return json in this structure:
{{"contact_name":"<header or empty>","messages":[{{"role":"contact|user|system","text_src":"<original-language transcription>"}}]}}

output json only."""

    _write_gemini_prompt_file("gemini_prompt_pass1.txt", "Pass 1 (vision)", prompt)

    pass1_started = time.time()
    try:
        gres = _gemini_generate(prompt, image_b64_list=b64_list, timeout=timeout)
        raw = (gres.text if gres else "") or ""
        api_payload = gres.response_json if gres else None
    except Exception as e:
        print(f"[GEMINI] Request failed: {e}")
        _write_gemini_debug_vision_only(prompt, f"ERROR: {e}", "")
        return False, hint or "", [], [], {}

    if not raw or not raw.strip():
        print("[GEMINI] Empty response from full-vision translation")
        _write_gemini_debug_vision_only(prompt, raw or "", "", api_payload=api_payload)
        return False, hint or "", [], [], {}

    def _finish_warn(pay):
        fr = _gemini_candidate_finish_reason(pay)
        if fr == "MAX_TOKENS":
            print(
                f"[GEMINI] Pass 1 finishReason={fr!r} — if JSON fails, internal "
                "thinking may have consumed the output token budget (see PIPELINE.md)."
            )
        return fr

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
        return False, hint or "", [], [], {}

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
            })
            continue
        mm = dict(m)
        mm["role"] = role
        pass1_chat_msgs.append(mm)
        chat_seen += 1

    non_system_count = len(pass1_chat_msgs)
    system_count = len(pass1_system_msgs)

    if _n_exp > 0 and non_system_count != _n_exp:
        print(
            f"[GEMINI] Pass 1 bubble count mismatch: expected {_n_exp}, got {non_system_count} "
            f"(plus {system_count} system row(s))."
        )
    elif _n_exp > 0:
        print(f"[GEMINI] Pass 1 bubble count OK: {_n_exp} bubbles, {system_count} system row(s).")

    contact_name = gname or hint or ""
    if not pass1_chat_msgs:
        print("[GEMINI] Parsed JSON but no messages")
        _write_gemini_debug_vision_only(prompt, raw, "NO MESSAGES PARSED", api_payload=api_payload)
        return False, contact_name, [], [], {}

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

    print(f"[GEMINI] Pass 1 (vision): {len(gmsgs)} rows total")
    print(f"[TIMER] Pass 1 Gemini in {time.time()-pass1_started:.1f}s")
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
        return False, contact_name, [], pre_ocr_meta, {
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
        }

    print(f"[GEMINI] Pass 1 final output OK → {len(meta)} chat rows, contact={contact_name!r}")
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
            print(f"[GEMINI] Image attached: {_gw}x{_gh}px "
                  f"({len(image_b64)//1024}KB base64)")
            if _gw < 600:
                print(
                    "[GEMINI] WARN: image width is very small — bubble text may be unreadable; "
                    "check combined_ocr_input.png and consider fewer pages per run."
                )
        except Exception as e:
            print(f"[GEMINI] Could not encode image, falling back to text-only: {e}")
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
        print(
            f"[GEMINI] Call 1: {len(slots)} scaffold rows — speaker role + order only "
            f"(no OCR text in prompt); bubble content read from image"
        )
    else:
        print(
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
            print(f"[GEMINI] Response received in {elapsed:.1f}s  ({len(raw)} chars)")

            gemini_name, context_line, parsed = _parse_gemini_slots_response(raw)
            if gemini_name:
                print(f"[GEMINI] Contact name identified: {gemini_name}")
            if context_line:
                print(f"[GEMINI] Conversation context: {context_line}")

            print(f"[GEMINI] Response preview:\n{raw[:500]}")

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
                print("[GEMINI] Recovery pass: comparing draft to image...")
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
                        print(
                            f"[GEMINI] Recovery pass applied in {time.time()-t_rec:.1f}s "
                            f"({len(parsed2)} slots)"
                        )
                    else:
                        print(
                            f"[GEMINI] Recovery pass skipped "
                            f"(draft {len(parsed)} slots vs recovery {len(parsed2)})"
                        )
                    _append_gemini_debug_recovery_section(
                        recovery_prompt_text, rec_raw or ""
                    )
                else:
                    print("[GEMINI] Recovery pass returned empty — keeping draft")
                    _append_gemini_debug_recovery_section(
                        recovery_prompt_text, "(empty response)"
                    )

            _apply_gemini_parsed_to_objects(objects, slots, parsed)

            print(f"[GEMINI] Parsed {len(parsed)}/{len(slots)} items  "
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
                print(f"[GEMINI] Attempt {attempt+1} failed ({err[:120]}), "
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

        print(f"[GEMINI] Debug file → {debug_path}")
    except Exception as exc:
        print(f"[GEMINI] Could not write debug file: {exc}")


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