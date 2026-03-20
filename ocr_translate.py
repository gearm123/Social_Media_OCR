import os
import cv2
import time
import sys
import re
from difflib import SequenceMatcher
import numpy as np
from config import DEBUG_DIR, ocr_engine, translator, translation_cache

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


def _is_non_latin(text: str) -> bool:
    """Return True if *text* contains characters outside the Latin/ASCII range.

    Used as a language-agnostic replacement for the old _is_thai() guard:
    if Gemini returns source-language text instead of English, the captured
    text will contain non-Latin characters and we know to look elsewhere for
    the English translation.
    """
    return any(ord(c) > 0x024F for c in text if not c.isspace())

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
            # Pick highest-priority model
            for pref in _GEMINI_PREFER:
                for name in capable:
                    if pref in name and "flash" in name:
                        return name, ver
            # Fallback: first capable model
            return capable[0], ver
        except Exception as e:
            print(f"[GEMINI] ListModels error ({ver}): {e}")
    return None, None


def _gemini_generate(prompt):
    """Call Gemini REST API directly — no SDK package required."""
    global _gemini_api_key, _gemini_active_model
    import requests as _req

    if _gemini_api_key is None:
        _gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None

    if not _gemini_api_key:
        return None

    if _gemini_active_model is None:
        print("[GEMINI] Discovering available models...")
        model, ver = _gemini_discover_model(_gemini_api_key)
        if model:
            _gemini_active_model = (model, ver)
            print(f"[GEMINI] Using model: {model} (API {ver})")
        else:
            print("[GEMINI] No generateContent-capable model found — refinement disabled")
            _gemini_api_key = ""
            return None

    model, ver = _gemini_active_model
    url = (
        f"https://generativelanguage.googleapis.com/{ver}/models/"
        f"{model}:generateContent?key={_gemini_api_key}"
    )
    r = _req.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=120,
    )
    r.raise_for_status()
    candidates = r.json().get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def refine_and_translate_with_gemini(objects):
    """Send the full labelled conversation to Gemini in one call.

    Gemini receives every message with its sender/receiver role (from bounding-box
    classification), fixes any OCR character errors using conversation context,
    and returns a natural English translation — all in one step.

    Writes both corrected "text_th" and translated "text_en" back into each object.
    Returns True if successful so the caller can skip Google Translate.
    """
    if not objects:
        return False

    # Trigger model discovery
    if _gemini_api_key is None:
        _gemini_generate("discovery")
    if not _gemini_active_model:
        return False

    # Build slots: only chat bubbles (skip timestamps, empty, links)
    slots = []  # (orig_idx, role_label, thai_text)
    for i, obj in enumerate(objects):
        th = (obj.get("text_th") or "").strip()
        obj_type = (obj.get("type") or "receiver").lower()
        if not th or _looks_like_link_text(th) or obj_type in {"timestamp", "status_bar", "bottom_artifact"}:
            continue
        role = "Person B (you)" if obj_type == "sender" else "Person A (other)"
        slots.append((i, role, th))

    if not slots:
        return False

    dialogue = "\n".join(
        f"{idx+1}. [{role}]: {th}"
        for idx, (_, role, th) in enumerate(slots)
    )

    lang_name = _source_lang_name  # e.g. "Thai", "Chinese", "Arabic" …
    prompt = (
        f"You are translating a {lang_name} chat conversation that was extracted via OCR "
        f"(OCR may have misread some {lang_name} characters).\n"
        "Person A is the other person; Person B is the phone owner.\n\n"
        "TASK: For each numbered message below, output ONLY the English translation "
        "of that message — nothing else.\n\n"
        "STRICT RULES:\n"
        f"- Your output must be ONLY English. Do NOT include any {lang_name} characters.\n"
        "- Use the conversation context to fix obvious OCR mistakes before translating.\n"
        "- Keep exactly the same numbering and speaker labels.\n"
        "- Do NOT add explanations, notes, asterisks, or extra lines.\n\n"
        "OUTPUT FORMAT (follow exactly):\n"
        "1. [Person A (other)]: English text here\n"
        "2. [Person B (you)]: English text here\n\n"
        "MESSAGES TO TRANSLATE:\n"
        f"{dialogue}"
    )

    print(f"[GEMINI] Sending {len(slots)} messages for OCR fix + translation...")
    for attempt in range(3):
        try:
            raw = _gemini_generate(prompt)
            if not raw:
                raise ValueError("Empty response")
            raw = raw.strip()
            print(f"[GEMINI] Response preview (first 300 chars): {raw[:300]}")

            parsed = {}
            lines = raw.splitlines()
            for line_idx, line in enumerate(lines):
                line = line.strip()
                m = re.match(r"^(\d+)\.\s*\[.*?\]:\s*(.+)$", line)
                if not m:
                    continue
                slot_num = int(m.group(1)) - 1
                captured = m.group(2).strip()

                # If Gemini returned source-language text in the numbered line,
                # look for the English translation on the same line (after a
                # separator) or on the immediately following line.
                if _is_non_latin(captured):
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
                            continue  # genuinely no English found, skip

                if not _is_non_latin(captured) and captured:
                    parsed[slot_num] = captured

            if not parsed:
                raise ValueError(f"Could not parse any English lines from response:\n{raw[:400]}")

            for slot_idx, (orig_idx, _, th) in enumerate(slots):
                en = parsed.get(slot_idx, "")
                if en:
                    objects[orig_idx]["text_en"] = en

            print(f"[GEMINI] Translated {len(parsed)}/{len(slots)} messages")
            return True

        except Exception as e:
            err = str(e)
            delay_match = re.search(r"(\d+)s", err)
            suggested = int(delay_match.group(1)) if delay_match else 0
            wait = max(suggested + 5, 15)
            if attempt < 2:
                print(f"[GEMINI] Error, waiting {wait}s before retry {attempt+2}/3: {err[:100]}")
                time.sleep(wait)
            else:
                print(f"[GEMINI] Failed after 3 attempts: {err[:200]}")

    return False


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
    if img is None or getattr(img, "size", 0) == 0:
        return objects

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

    return objects

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