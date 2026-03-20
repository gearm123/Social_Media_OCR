import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Allow Thai / Unicode characters to print on Windows without crashing
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import INPUT_DIR, OUTPUT_DIR, JSON_DIR, RENDER_DIR, load_craft, ocr_engine
from ocr_translate import (
    ocr_and_translate_full_image, ocr_and_translate_region,
    translate_conversation_block, refine_and_translate_with_gemini,
    _set_source_language,
)
from pipeline import prepare_image_crop_info, run_craft_and_group_on_combined
from grouping import classify_object_type
from chat_renderer import render_chat

PAGE_GAP_PX = 48

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


def combine_images_vertically(images, background=248):
    valid_images = [img for img in images if img is not None and getattr(img, "size", 0) > 0]
    if not valid_images:
        return None

    max_width = max(img.shape[1] for img in valid_images)
    total_height = sum(img.shape[0] for img in valid_images)
    channels = valid_images[0].shape[2]
    combined = np.full((total_height, max_width, channels), background, dtype=np.uint8)

    y = 0
    for img in valid_images:
        height, width = img.shape[:2]
        combined[y:y + height, :width] = img
        y += height

    return combined


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


def build_combined_image(crop_infos, background=248):
    """Concatenate cropped images (status bar kept only on first).

    Returns (combined_img, page_ranges) where page_ranges is a list of
    (start_y, end_y, page_index, crop_top, status_bar_info, bottom_artifact_info).
    """
    segments = []
    page_ranges = []
    current_y = 0

    for page_index, info in enumerate(crop_infos):
        keep_sb = (page_index == 0)
        crop_top, crop_bottom = _crop_bounds_from_info(info, keep_status_bar=keep_sb)
        print(
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


def validate_all_objects(all_objects, crop_infos, page_ranges):
    """Fallback OCR for any object that has no text after combined OCR."""
    total_objects = len(all_objects)
    assigned_objects = 0
    fallback_objects = 0

    for obj_idx, obj in enumerate(all_objects):
        text_th = (obj.get("text_th") or "").strip()
        if text_th:
            assigned_objects += 1
            obj["ocr_validated"] = True
            obj["ocr_source"] = obj.get("ocr_source", "combined_canvas")
            obj["ocr_trust_score"] = float(obj.get("ocr_trust_score", 0.0) or 0.0)
            obj["ocr_low_confidence"] = bool(obj.get("ocr_low_confidence", False))
            obj["ocr_reasons"] = list(obj.get("ocr_reasons", []) or [])
            obj["translation_blocked"] = False
            continue

        page_index = obj.get("page_index", 0)
        info = crop_infos[page_index]
        start_y, _, _, crop_top, _, _ = page_ranges[page_index]

        # Map bbox from combined coords back to original image coords
        cx1, cy1, cx2, cy2 = obj["bbox"]
        orig_bbox = (cx1, cy1 - start_y + crop_top, cx2, cy2 - start_y + crop_top)

        fallback_th, fallback_en = ocr_and_translate_region(
            info["img"], orig_bbox,
            idx=f"validate_{page_index}_{obj_idx}",
        )
        if fallback_th:
            obj["text_th"] = fallback_th
            obj["text_en"] = fallback_en
            obj["ocr_source"] = "validation_fallback"
            obj["ocr_validated"] = True
            obj["ocr_trust_score"] = 0.45
            obj["ocr_low_confidence"] = not bool(fallback_en)
            obj["ocr_reasons"] = ["validation_fallback_missing_object"]
            obj["translation_blocked"] = False
            fallback_objects += 1
            assigned_objects += 1
        else:
            obj["ocr_source"] = obj.get("ocr_source", "combined_canvas")
            obj["ocr_validated"] = False
            obj["ocr_trust_score"] = 0.0
            obj["ocr_low_confidence"] = True
            obj["ocr_reasons"] = ["missing_after_validation"]
            obj["translation_blocked"] = False

    print(
        f"[OCR VALIDATE] assigned={assigned_objects}/{total_objects} "
        f"fallbacks={fallback_objects} missing={max(0, total_objects - assigned_objects)}"
    )

_KNOWN_LANGUAGES = {
    "th": "Thai", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "ru": "Russian", "he": "Hebrew", "hi": "Hindi",
    "vi": "Vietnamese", "id": "Indonesian", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "tr": "Turkish",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="OCR + Gemini chat translator"
    )
    parser.add_argument(
        "-l", "--language",
        metavar="CODE",
        default=None,
        help=(
            "Source language BCP-47 code to override auto-detection "
            "(e.g. th, zh, ja, ar, ru). "
            "Default: auto-detect from Vision OCR response."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    pipeline_start = time.time()
    print("\n[PIPELINE] Starting Messenger OCR pipeline\n")

    # Language override from CLI
    if args.language:
        code = args.language.lower()
        name = _KNOWN_LANGUAGES.get(code, code.upper())
        _set_source_language(code, name)
        print(f"[LANG] Source language forced to: {name} ({code})")
    else:
        print("[LANG] Source language: auto-detect from Vision OCR")

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        print(f"[GEMINI] Key found — conversation refinement ENABLED")
    else:
        print("[GEMINI] No GEMINI_API_KEY set — skipping conversation refinement")
        print("[GEMINI] To enable: $env:GEMINI_API_KEY = 'your_key'  (or set it permanently below)")
        print("[GEMINI] Permanent:  [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY','your_key','User')")

    t_craft_load = time.time()
    craft_net = load_craft()
    print(f"[TIMER] CRAFT model loaded in {time.time()-t_craft_load:.1f}s")

    images = sorted(Path(INPUT_DIR).glob("*"))
    print(f"[INPUT] Found {len(images)} images")

    # ── Step 1: read images + detect artifacts (no CRAFT yet) ─────────────
    t1 = time.time()
    print("\n[STEP 1/7] Reading images and detecting artifacts...")
    crop_infos = []
    for path in images:
        print(f"  {Path(path).name}")
        crop_infos.append(prepare_image_crop_info(str(path)))
    print(f"[TIMER] Step 1 done in {time.time()-t1:.1f}s")

    # Extract contact name from the first image's status bar
    contact_name = _extract_contact_name(crop_infos[0]) if crop_infos else ""
    if contact_name:
        print(f"[NAME] Contact name detected: '{contact_name}'")
    else:
        print("[NAME] Contact name not detected — will use 'Person A'")

    # ── Step 2: concatenate cropped images ────────────────────────────────
    t2 = time.time()
    print("\n[STEP 2/7] Building combined image...")
    combined_img, page_ranges = build_combined_image(crop_infos)
    if combined_img is None:
        print("[ERROR] No images to process.")
        return

    combined_ocr_path = os.path.join(OUTPUT_DIR, "combined_ocr_input.png")
    cv2.imwrite(combined_ocr_path, combined_img)
    print(f"[COMBINE] {combined_img.shape[1]}x{combined_img.shape[0]}px "
          f"from {len(page_ranges)} pages → {combined_ocr_path}")
    print(f"[TIMER] Step 2 done in {time.time()-t2:.1f}s")

    # ── Step 3: CRAFT + grouping on the combined image ────────────────────
    t3 = time.time()
    print("\n[STEP 3/7] CRAFT text detection + grouping on combined image...")
    all_objects = run_craft_and_group_on_combined(combined_img, craft_net, page_ranges)
    if not all_objects:
        print("[WARN] No objects detected in combined image.")
        return
    print(f"[TIMER] Step 3 done in {time.time()-t3:.1f}s  "
          f"→ {len(all_objects)} objects")

    # ── Step 4: Vision OCR on the combined image ──────────────────────────
    t4 = time.time()
    print("\n[STEP 4/7] Google Vision OCR on combined image...")
    all_objects = ocr_and_translate_full_image(
        all_objects, combined_img, progress_label="OCR combined"
    )
    texts_found = sum(1 for o in all_objects if o.get("text_th"))
    print(f"[TIMER] Step 4 done in {time.time()-t4:.1f}s  "
          f"→ {texts_found}/{len(all_objects)} objects have text")

    # ── Step 5: fallback OCR for any missed objects ────────────────────────
    t5 = time.time()
    print("\n[STEP 5/7] Validating OCR (fallback for missed objects)...")
    validate_all_objects(all_objects, crop_infos, page_ranges)
    print(f"[TIMER] Step 5 done in {time.time()-t5:.1f}s")

    # ── Step 6: classify sender/receiver for every object ─────────────────
    t6 = time.time()
    print("\n[STEP 6/7] Classifying sender / receiver / timestamp...")
    for obj in all_objects:
        if "type" not in obj:
            obj["type"] = classify_object_type(
                combined_img, obj["bbox"],
                obj.get("text_th", ""), obj.get("text_en", "")
            )
    type_counts = {}
    for obj in all_objects:
        t = obj.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[CLASSIFY] {dict(type_counts)}")
    print(f"[TIMER] Step 6 done in {time.time()-t6:.1f}s")

    # ── Step 7: Gemini fix+translate in one shot ──────────────────────────
    t7 = time.time()
    print("\n[STEP 7/7] Gemini OCR-fix + translation...")
    gemini_ok, gemini_name = refine_and_translate_with_gemini(
        all_objects, combined_img=combined_img, contact_name=contact_name
    )
    # Prefer the name Gemini inferred from conversation context over the
    # OCR-extracted status-bar text (which may contain garbled UI text)
    if gemini_name and gemini_name.lower() not in {"person a", "unknown", ""}:
        contact_name = gemini_name
        print(f"[NAME] Using Gemini-identified name: '{contact_name}'")

    if not gemini_ok:
        print("[TRANSLATE] Gemini unavailable — falling back to Google Translate")
        all_thai = [obj.get("text_th", "") for obj in all_objects]
        all_en = translate_conversation_block(all_thai)
        for obj, en in zip(all_objects, all_en):
            if en:
                obj["text_en"] = en
    else:
        # Fill any remaining gaps (links or objects Gemini skipped)
        gaps = [obj for obj in all_objects if not obj.get("text_en") and obj.get("text_th")]
        if gaps:
            gap_texts = [obj["text_th"] for obj in gaps]
            gap_en = translate_conversation_block(gap_texts)
            for obj, en in zip(gaps, gap_en):
                if en:
                    obj["text_en"] = en
    translated = sum(1 for o in all_objects if o.get("text_en"))
    print(f"[TIMER] Step 7 done in {time.time()-t7:.1f}s  "
          f"→ {translated}/{len(all_objects)} objects translated")

    # ── Step 8: collect metadata for all objects in conversation order ────────
    t8 = time.time()
    print("\n[STEP 8/8] Building conversation metadata...")
    all_meta = []
    for obj in all_objects:
        typ = obj.get("type", "receiver")
        if typ in {"status_bar", "bottom_artifact"}:
            continue
        all_meta.append({
            "type": typ,
            "bbox": [int(v) for v in obj["bbox"]],
            "text_th": obj.get("text_th", ""),
            "text_en": obj.get("text_en", ""),
            "ocr_source": obj.get("ocr_source", ""),
            "ocr_span_count": int(obj.get("ocr_span_count", 0) or 0),
            "ocr_validated": bool(obj.get("ocr_validated", False)),
            "ocr_trust_score": float(obj.get("ocr_trust_score", 0.0) or 0.0),
            "ocr_low_confidence": bool(obj.get("ocr_low_confidence", False)),
            "ocr_reasons": list(obj.get("ocr_reasons", []) or []),
        })
    print(f"[TIMER] Step 8 done in {time.time()-t8:.3f}s  "
          f"→ {len(all_meta)} conversation items")

    # ── Step 9: single unified chat render for the full conversation ──────────
    t9 = time.time()
    print("\n[STEP 9] Rendering translated conversation...")

    # Stamp conversation order so render_chat doesn't re-sort by bbox y
    for i, item in enumerate(all_meta):
        item["order"] = i

    combined_json_path = os.path.join(JSON_DIR, "translated_conversation.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    combined_chat = render_chat(all_meta, contact_name=contact_name or "Person A")
    combined_path = os.path.join(RENDER_DIR, "translated_conversation.png")
    cv2.imwrite(combined_path, combined_chat)
    print(f"[OUTPUT] Translated conversation → {combined_path}")
    print(f"[TIMER] Step 9 done in {time.time()-t9:.1f}s")

    total_runtime = time.time() - pipeline_start
    print(f"""
╔══════════════════════════════════════════╗
  PIPELINE SUMMARY
  Images processed : {len(images)}
  Objects detected : {len(all_objects)}
  Objects translated: {sum(1 for o in all_meta if o.get('text_en'))}
  Total runtime    : {total_runtime:.1f}s
╚══════════════════════════════════════════╝""")

if __name__ == "__main__":
    main()