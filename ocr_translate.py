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
# OCR (UNCHANGED)
# --------------------------------------------------

def run_ocr_on_region(region):
    if region.size == 0:
        return ""

    region = cv2.resize(
        region,
        None,
        fx=1.6,
        fy=1.6,
        interpolation=cv2.INTER_CUBIC
    )

    result = ocr_engine.predict(region)

    if not result:
        return ""

    r = result[0]

    if isinstance(r, dict) and "rec_texts" in r:
        return " ".join(
            [t for t in r["rec_texts"] if isinstance(t, str) and t.strip()]
        )

    return ""

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