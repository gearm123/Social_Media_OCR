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

from config import INPUT_DIR, OUTPUT_DIR, JSON_DIR, RENDER_DIR, load_craft
from ocr_translate import (
    ocr_and_translate_full_image, ocr_and_translate_region,
    translate_conversation_block, refine_and_translate_with_gemini,
    _set_source_language,
)
from pipeline import finalize_image_layout, prepare_image_crop_info, run_craft_and_group_on_combined
from grouping import classify_object_type
from chat_renderer import render_chat

PAGE_GAP_PX = 48


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

    craft_net = load_craft()

    images = sorted(Path(INPUT_DIR).glob("*"))
    print(f"[INPUT] Found {len(images)} images")

    # ── Step 1: read images + detect artifacts (no CRAFT yet) ─────────────
    print("\n[STEP] Reading images and detecting artifacts...")
    crop_infos = []
    for path in images:
        print(f"  {Path(path).name}")
        crop_infos.append(prepare_image_crop_info(str(path)))

    # ── Step 2: concatenate cropped images ────────────────────────────────
    combined_img, page_ranges = build_combined_image(crop_infos)
    if combined_img is None:
        print("[ERROR] No images to process.")
        return

    combined_ocr_path = os.path.join(OUTPUT_DIR, "combined_ocr_input.png")
    cv2.imwrite(combined_ocr_path, combined_img)
    print(f"[OUTPUT] Saved OCR input canvas to {combined_ocr_path}")
    print(f"[COMBINE] Combined image: {combined_img.shape[1]}x{combined_img.shape[0]} "
          f"from {len(page_ranges)} pages")

    # ── Step 3: CRAFT + grouping on the combined image ────────────────────
    all_objects = run_craft_and_group_on_combined(combined_img, craft_net, page_ranges)
    if not all_objects:
        print("[WARN] No objects detected in combined image.")
        return

    # ── Step 4: Vision OCR on the combined image ──────────────────────────
    all_objects = ocr_and_translate_full_image(
        all_objects, combined_img, progress_label="OCR combined"
    )

    # ── Step 5: fallback OCR for any missed objects ────────────────────────
    validate_all_objects(all_objects, crop_infos, page_ranges)

    # ── Step 6: classify sender/receiver for every object ─────────────────
    for obj in all_objects:
        if "type" not in obj:
            obj["type"] = classify_object_type(
                combined_img, obj["bbox"],
                obj.get("text_th", ""), obj.get("text_en", "")
            )

    # ── Step 7: Gemini fix+translate in one shot ──────────────────────────
    gemini_ok = refine_and_translate_with_gemini(all_objects)

    if not gemini_ok:
        print("[TRANSLATE] Gemini unavailable — falling back to Google Translate")
        all_thai = [obj.get("text_th", "") for obj in all_objects]
        all_en = translate_conversation_block(all_thai)
        for obj, en in zip(all_objects, all_en):
            if en:
                obj["text_en"] = en
    else:
        # Fill any gaps (timestamps, links) that Gemini skipped
        for obj in all_objects:
            if not obj.get("text_en") and obj.get("text_th"):
                obj["text_en"] = translate_conversation_block([obj["text_th"]])[0]

    # ── Step 8: per-image overlays (bounding boxes on originals) + JSON ──────
    all_meta = []   # collects result dicts for ALL objects in conversation order

    for page_index, info in enumerate(crop_infos):
        image_start = time.time()
        path = Path(info["path"])
        start_y, _, _, crop_top, _, _ = page_ranges[page_index]

        # Filter objects belonging to this page
        page_objects = [obj for obj in all_objects if obj.get("page_index") == page_index]

        # Map bboxes from combined coords to original image coords for overlay drawing
        orig_objects = []
        for obj in page_objects:
            cx1, cy1, cx2, cy2 = obj["bbox"]
            orig_obj = dict(obj)
            orig_obj["bbox"] = (cx1, cy1 - start_y + crop_top, cx2, cy2 - start_y + crop_top)
            orig_objects.append(orig_obj)

        virtual_layout = {
            "path": str(path),
            "img": info["img"],
            "objects": orig_objects,
            "status_bar_info": info["status_bar_info"],
            "bottom_artifact_info": info["bottom_artifact_info"],
            "status_bar_result": None,
        }
        overlay, meta, _ = finalize_image_layout(virtual_layout)
        all_meta.extend(meta)   # accumulate in conversation order

        out = os.path.join(OUTPUT_DIR, path.name)
        cv2.imwrite(out, overlay)

        json_path = os.path.join(JSON_DIR, path.stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        image_runtime = time.time() - image_start
        print(f"[OUTPUT] Saved overlay for {path.name} in {image_runtime:.2f}s")

    # ── Step 9: single unified chat render for the full conversation ──────────
    # all_meta contains every object across all pages in order — render once.
    combined_json_path = os.path.join(JSON_DIR, "combined_chat.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    combined_chat = render_chat(all_meta)
    combined_path = os.path.join(RENDER_DIR, "combined_chat.png")
    cv2.imwrite(combined_path, combined_chat)
    print(f"[OUTPUT] Saved combined conversation chat to {combined_path}")

    total_runtime = time.time() - pipeline_start
    print(f"\n[PIPELINE] Finished processing all images in {total_runtime:.2f}s")

if __name__ == "__main__":
    main()