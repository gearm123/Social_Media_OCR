import os
import cv2
import time
import sys
from config import DEBUG_DIR, ocr_engine, translator, translation_cache

# --------------------------------------------------
# TRANSLATION (UNCHANGED)
# --------------------------------------------------

def translate_th_to_en(text):
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

# --------------------------------------------------
# REGION EXTRACTION (UNCHANGED)
# --------------------------------------------------

def extract_region(img, rect, idx):
    x1, y1, x2, y2 = rect

    pad = 12

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

def _region_scale(region):
    h, w = region.shape[:2]
    short_side = min(h, w)
    if short_side <= 22:
        return 2.4
    if short_side <= 32:
        return 2.1
    if short_side <= 48:
        return 1.8
    return 1.55

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
    sharpened = cv2.addWeighted(prepared, 1.18, cv2.GaussianBlur(prepared, (0, 0), 1.1), -0.18, 0)
    return sharpened

def _prepare_fallback_variants(region):
    base = _prepare_base_region(region)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

    thresh = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    variants = [("base", base)]

    # Only pay for fallback passes when the crop is small or visually noisy.
    h, w = region.shape[:2]
    if min(h, w) <= 56 or gray.std() < 42:
        variants.append(("clahe", clahe_bgr))
    if min(h, w) <= 44 or gray.std() < 30:
        variants.append(("thresh", thresh_bgr))

    return variants

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

def _ocr_quality_score(text, confidence):
    if not text:
        return -1.0

    compact = text.replace(" ", "")
    digit_bonus = 0.05 if any(ch.isdigit() for ch in compact) else 0.0
    thai_bonus = 0.08 if any("\u0e00" <= ch <= "\u0e7f" for ch in compact) else 0.0
    length_bonus = min(len(compact), 20) * 0.01
    punctuation_penalty = 0.06 if len(compact) <= 2 and not any(ch.isdigit() for ch in compact) else 0.0
    return confidence + digit_bonus + thai_bonus + length_bonus - punctuation_penalty

def run_ocr_on_region(region):
    if region.size == 0:
        return ""

    best_text = ""
    best_quality = -1.0

    for idx, (_, prepared) in enumerate(_prepare_fallback_variants(region)):
        result = ocr_engine.predict(prepared)
        text, confidence = _extract_prediction(result)
        quality = _ocr_quality_score(text, confidence)

        if quality > best_quality:
            best_quality = quality
            best_text = text

        # Stop early when the first pass is already solid.
        if idx == 0 and confidence >= 0.84 and len(text.replace(" ", "")) >= 4:
            return text

    return best_text

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

        region = extract_region(img, rect, i)

        text_th = run_ocr_on_region(region)
        text_en = translate_th_to_en(text_th)

        obj["text_th"] = text_th
        obj["text_en"] = text_en

        _print_progress(progress_label, i + 1, total, start_time)

    runtime = time.time() - start_time

    print(
        f"\n[{progress_label}] Completed in {_format_runtime(runtime)} "
        f"({runtime:.2f}s)\n"
    )

    return objects