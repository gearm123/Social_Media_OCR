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

from config import INPUT_DIR, OUTPUT_DIR, JSON_DIR, RENDER_DIR, ocr_engine
from ocr_translate import (
    translate_to_en,
    translate_conversation_gemini_multimodal,
    _gemini_ocr_hints_refine_pass,
    _gemini_status_bar_pass,
    _meta_from_gemini_messages,
    _set_source_language,
)
from pipeline import prepare_image_crop_info
from chat_renderer import render_chat

PAGE_GAP_PX = 48
PASS1_BUBBLE_INPUT_PATH = Path(__file__).resolve().parent / "pass1_bubble_input.txt"

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


def crop_avatar_from_status_bar(crop_infos, status_bar_images, status_info):
    """Crop the chosen avatar region, preferring detected avatar geometry."""
    if not status_bar_images:
        return None
    image_index = status_info.get("avatar_image_index")
    bbox = status_info.get("avatar_bbox")
    try:
        image_index = int(image_index)
    except (TypeError, ValueError):
        image_index = 0
    if not (0 <= image_index < len(status_bar_images)):
        image_index = 0
    img = status_bar_images[image_index]
    if img is None or getattr(img, "size", 0) == 0:
        return None
    h, w = img.shape[:2]

    rect = None
    info = crop_infos[image_index] if 0 <= image_index < len(crop_infos or []) else None
    status_meta = (info or {}).get("status_bar_info") or {}
    detected_avatar_rect = status_meta.get("avatar_rect")
    status_bbox = status_meta.get("bbox")

    if isinstance(detected_avatar_rect, (list, tuple)) and len(detected_avatar_rect) >= 4:
        try:
            ax1, ay1, ax2, ay2 = [int(v) for v in detected_avatar_rect[:4]]
            if isinstance(status_bbox, (list, tuple)) and len(status_bbox) >= 4:
                sx1, sy1, _sx2, _sy2 = [int(v) for v in status_bbox[:4]]
                rect = [ax1 - sx1, ay1 - sy1, ax2 - sx1, ay2 - sy1]
            else:
                rect = [ax1, ay1, ax2, ay2]
        except (TypeError, ValueError):
            rect = None

    if rect is None and isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
        # Gemini may return full-image coordinates; convert them to status-crop-local coords.
        if (x2 > w or y2 > h) and isinstance(status_bbox, (list, tuple)) and len(status_bbox) >= 4:
            sx1, sy1, _sx2, _sy2 = [int(v) for v in status_bbox[:4]]
            x1, y1, x2, y2 = x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1
        rect = [x1, y1, x2, y2]

    if rect is None:
        return None

    x1, y1, x2, y2 = rect
    pad = 6
    x1 = max(0, min(w - 1, x1 - pad))
    x2 = max(x1 + 1, min(w, x2 + pad))
    y1 = max(0, min(h - 1, y1 - pad))
    y2 = max(y1 + 1, min(h, y2 + pad))
    avatar = img[y1:y2, x1:x2].copy()
    return avatar if avatar.size else None


def prompt_manual_bubble_summary(num_images: int):
    """Ask the user for bubble counts and sender/receiver order per cleaned image."""
    if num_images <= 0:
        return "", 0, []

    file_summary = _load_pass1_bubble_summary_file(num_images)
    if file_summary is not None:
        print(f"\n[STEP 1a] Using Pass 1 bubble summary from {PASS1_BUBBLE_INPUT_PATH.name}…")
        return file_summary

    print("\n[STEP 1a] Manual bubble summary for Pass 1…")
    print(f"[INPUT] You can also prefill {PASS1_BUBBLE_INPUT_PATH.name} to skip prompts.")
    print("[INPUT] Enter the number of actual text bubbles for each cleaned image.")
    print("[INPUT] Then enter their order using sender/receiver or s/r, comma-separated.")
    print("[INPUT] Do not count timestamps, call notices, reactions, headers, or other UI artifacts.")

    lines = []
    total_bubbles = 0
    page_specs = []

    for i in range(num_images):
        while True:
            raw_n = input(f"Image {i + 1} bubble count: ").strip()
            try:
                count = int(raw_n)
            except ValueError:
                print("Please enter a whole number.")
                continue
            if count < 0:
                print("Please enter 0 or a positive number.")
                continue
            break

        order = []
        if count > 0:
            while True:
                raw_seq = input(
                    f"Image {i + 1} bubble order ({count} items, e.g. s,s,r): "
                ).strip()
                parts = [p.strip().lower() for p in raw_seq.replace(";", ",").split(",") if p.strip()]
                mapped = []
                bad = False
                for p in parts:
                    if p in ("s", "sender"):
                        mapped.append("sender")
                    elif p in ("r", "receiver", "recevier", "reciever"):
                        mapped.append("receiver")
                    else:
                        bad = True
                        break
                if bad or len(mapped) != count:
                    print("Sequence must match the count and use only sender/receiver (or s/r).")
                    continue
                order = mapped
                break

        total_bubbles += count
        page_specs.append({
            "page_index": i,
            "count": count,
            "order": list(order),
        })

        lines.append(f"in image {i + 1}")
        lines.append(f"i count {count} message bubbles")
        lines.extend(order if order else ["none"])
        if i < num_images - 1:
            lines.append("")

    return "\n".join(lines), total_bubbles, page_specs


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
        print(f"[INPUT] Could not read {path.name}: {e}")
        return None

    lines = [line for line in raw_lines if line]
    if not lines:
        return None
    if len(lines) % 2 != 0:
        print(f"[INPUT] {path.name} is invalid: expected count/order line pairs.")
        return None

    specs = []
    for i in range(0, len(lines), 2):
        count_line = lines[i]
        order_line = lines[i + 1]
        try:
            count = int(count_line)
        except ValueError:
            print(f"[INPUT] {path.name} has an invalid count line: '{count_line}'")
            return None

        if "," in order_line:
            tokens = [p.strip() for p in order_line.replace(";", ",").split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in order_line.split() if p.strip()]

        mapped = []
        for tok in tokens:
            role = _map_bubble_role_token(tok)
            if role is None:
                print(f"[INPUT] {path.name} has an invalid role token: '{tok}'")
                return None
            mapped.append(role)

        if count != len(mapped):
            print(
                f"[INPUT] {path.name}: count {count} does not match order length {len(mapped)} "
                f"for image {(i // 2) + 1}; using order length."
            )
            count = len(mapped)

        specs.append({"count": count, "order": mapped})

    if len(specs) != num_images:
        print(
            f"[INPUT] {path.name} has {len(specs)} image specs but current input has {num_images} images."
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
    """Run OCR on each cleaned page image and collect text/confidence debug output."""
    lines = []
    entries = []
    min_conf = 0.94

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


_KNOWN_LANGUAGES = {
    "th": "Thai", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "ru": "Russian", "he": "Hebrew", "hi": "Hindi",
    "vi": "Vietnamese", "id": "Indonesian", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "tr": "Turkish",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Gemini vision chat translator (stitched + per-page screenshots)"
    )
    parser.add_argument(
        "-l", "--language",
        metavar="CODE",
        default=None,
        help=(
            "Source language BCP-47 hint passed to Gemini "
            "(e.g. th, zh, ja). Default: infer from screenshots."
        ),
    )
    return parser.parse_args()


def _compose_labeled_chat_comparison(left_img, right_img, left_label, right_label):
    """Place two rendered chats side-by-side with simple labels."""
    gap = 20
    pad = 12
    label_h = 34
    bg = 248

    if left_img is None or getattr(left_img, "size", 0) == 0:
        return right_img
    if right_img is None or getattr(right_img, "size", 0) == 0:
        return left_img

    panel_w = max(left_img.shape[1], right_img.shape[1])
    left_h = left_img.shape[0] + label_h
    right_h = right_img.shape[0] + label_h
    panel_h = max(left_h, right_h)

    canvas_h = panel_h + pad * 2
    canvas_w = panel_w * 2 + gap + pad * 2
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

    _draw_panel(left_img, left_label, pad)
    _draw_panel(right_img, right_label, pad + panel_w + gap)
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


def _merge_chat_with_system_metadata(chat_meta, system_messages):
    systems_by_before = {}
    for sys_msg in list(system_messages or []):
        before_idx = int(sys_msg.get("insert_before_chat_index") or 0)
        systems_by_before.setdefault(before_idx, []).append({
            "type": "timestamp",
            "bbox": [0, 0, 400, 35],
            "text_th": "",
            "text_en": translate_to_en(sys_msg.get("text_src") or "") or (sys_msg.get("text_src") or ""),
            "ocr_source": "gemini_full_vision",
            "ocr_span_count": 0,
            "ocr_validated": True,
            "ocr_trust_score": 1.0,
            "ocr_low_confidence": False,
            "ocr_reasons": [],
        })

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


def main():
    args = _parse_args()
    pipeline_start = time.time()
    print("\n[PIPELINE] Gemini full-vision chat translator\n")

    source_lang_note = None
    if args.language:
        code = args.language.lower()
        name = _KNOWN_LANGUAGES.get(code, code.upper())
        _set_source_language(code, name)
        source_lang_note = name
        print(f"[LANG] Source language hint for Gemini: {name} ({code})")
    else:
        print("[LANG] Source language: infer from screenshots")

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY is required for this pipeline.")
        print("  Set: $env:GEMINI_API_KEY = 'your_key'")
        return
    print("[GEMINI] API key found — multimodal translation enabled")

    images = sorted(Path(INPUT_DIR).glob("*"))
    print(f"[INPUT] Found {len(images)} images")

    t1 = time.time()
    print("\n[STEP 1] Reading images and detecting artifacts...")
    crop_infos = []
    for path in images:
        print(f"  {Path(path).name}")
        crop_infos.append(prepare_image_crop_info(str(path)))
    print(f"[TIMER] Step 1 done in {time.time()-t1:.1f}s")

    if not crop_infos:
        print("[ERROR] No input images.")
        return

    contact_name = _extract_contact_name(crop_infos[0]) if crop_infos else ""
    if contact_name:
        print(f"[NAME] Status-bar hint: '{contact_name}'")
    else:
        print("[NAME] No status-bar name (Vision disabled or unreadable) — Gemini may infer")

    t2 = time.time()
    page_images = extract_page_segment_images(crop_infos)
    status_bar_images = extract_status_bar_images(crop_infos)
    combined_img, page_ranges = build_combined_image(
        crop_infos, global_first_page_index=0
    )
    if combined_img is None:
        print("[ERROR] Empty combined image.")
        return

    print(
        f"[COMBINE] {combined_img.shape[1]}×{combined_img.shape[0]} px, "
        f"{len(page_ranges)} page band(s)"
    )
    print(f"[COMBINE] {len(page_images)} page crop(s) for OCR / internal use")
    print(f"[STATUS] {len(status_bar_images)} status-bar crop(s) ready for Pass 3")
    print(f"[TIMER] Stitch done in {time.time()-t2:.1f}s")

    ocr_hints = None
    ocr_hints_format = "none"
    ocr_pass2_by_message = None
    pass1_bubble_context = ""
    pass1_exclude_ocr = True
    pass1_message_images = []
    craft_expected_n = None
    manual_page_specs = []
    pass1_bubble_context, craft_expected_n, manual_page_specs = prompt_manual_bubble_summary(len(page_images))

    if pass1_bubble_context:
        bubble_debug_path = os.path.join(OUTPUT_DIR, "pass1_bubble_summary.txt")
        with open(bubble_debug_path, "w", encoding="utf-8") as f:
            f.write(pass1_bubble_context)
        print(f"[OUTPUT] Pass 1 bubble summary → {bubble_debug_path}")


    t3 = time.time()
    print("\n[STEP 2] Gemini multimodal…")
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
    print(f"[TIMER] Gemini done in {time.time()-t3:.1f}s")

    if not ok or not all_meta:
        print("[ERROR] Gemini did not return a usable conversation. See result/gemini_debug.txt")
        return

    t_ocr2 = time.time()
    print("\n[STEP 2b] OCR pass on cleaned images…")
    pass2_ocr_text, pass2_ocr_entries = collect_page_ocr_debug(page_images)
    pass2_ocr_debug_path = os.path.join(OUTPUT_DIR, "pass2_ocr_debug.txt")
    with open(pass2_ocr_debug_path, "w", encoding="utf-8") as f:
        f.write(pass2_ocr_text)
    print(f"[OUTPUT] Pass 2 OCR debug → {pass2_ocr_debug_path}")
    print(f"[TIMER] Pass 2 OCR in {time.time()-t_ocr2:.1f}s")

    contact_name = (g_contact or contact_name or "Person A").strip() or "Person A"
    pass1_meta = list(all_meta or [])
    for i, item in enumerate(pre_ocr_meta or []):
        item["order"] = i
    for i, item in enumerate(pass1_meta):
        item["order"] = i
    pass1_output_count = int((pass_debug or {}).get("pass1_count") or 0)
    pass1_system_count = int((pass_debug or {}).get("pass1_system_count") or 0)
    print(
        f"[ALIGN] Counts: manual_expected={craft_expected_n} "
        f"pass1_bubbles={pass1_output_count} pass1_system={pass1_system_count}"
    )

    t_pass2 = time.time()
    print("\n[STEP 2c] Gemini rewrite with OCR hints…")
    final_messages, contact_name2 = _gemini_ocr_hints_refine_pass(
        contact_name,
        (pass_debug or {}).get("pass1_messages") or [],
        pass2_ocr_text,
        timeout=int(os.environ.get("GEMINI_REQUEST_TIMEOUT_SEC", "600")),
    )
    if contact_name2:
        contact_name = contact_name2
    all_meta = _meta_from_gemini_messages(final_messages)
    for i, item in enumerate(all_meta or []):
        item["order"] = i
    final_render_meta = _merge_chat_with_system_metadata(
        all_meta,
        (pass_debug or {}).get("pass1_system_messages") or [],
    )
    print(f"[TIMER] Pass 2 Gemini rewrite in {time.time()-t_pass2:.1f}s")

    t_pass3 = time.time()
    print("\n[STEP 3] Gemini status-bar pass…")
    status_bar_info = _gemini_status_bar_pass(
        status_bar_images,
        contact_hint=contact_name,
        timeout=int(os.environ.get("GEMINI_REQUEST_TIMEOUT_SEC", "600")),
    )
    final_contact_name_src = (status_bar_info.get("contact_name") or contact_name or "Person A").strip() or "Person A"
    final_status_text_src = (status_bar_info.get("status_text") or "").strip()
    final_contact_name = (translate_to_en(final_contact_name_src) or final_contact_name_src).strip() or "Person A"
    final_status_text = (translate_to_en(final_status_text_src) or final_status_text_src).strip()
    # The detected Messenger avatar is too small/noisy to preserve reliably, so keep the generic header avatar.
    profile_image = None
    contact_name = final_contact_name
    print(
        f"[STATUS] Final header: name='{final_contact_name}'"
        + (f", status='{final_status_text}'" if final_status_text else "")
    )
    print(f"[TIMER] Pass 3 status bar in {time.time()-t_pass3:.1f}s")

    pass1_source_debug_path = os.path.join(JSON_DIR, "pass1_transcript_debug.json")

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
    print(f"[OUTPUT] Pass 1 transcript debug → {pass1_source_debug_path}")

    t4 = time.time()
    print("\n[STEP 4] Saving JSON + rendered chat...")
    combined_json_path = os.path.join(JSON_DIR, "translated_conversation.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(final_render_meta, f, indent=2, ensure_ascii=False)

    pass1_chat = render_chat(pass1_meta)
    pass1_path = os.path.join(RENDER_DIR, "translated_conversation_pass1.png")
    cv2.imwrite(pass1_path, pass1_chat)

    pass2_img = render_chat(all_meta)
    pass2_path = os.path.join(RENDER_DIR, "translated_conversation_pass2.png")
    cv2.imwrite(pass2_path, pass2_img)

    compare_img = _compose_labeled_chat_comparison(
        pass1_chat,
        pass2_img,
        "Pass 1 Translation",
        "Pass 2 With OCR Hints",
    )
    compare_path = os.path.join(RENDER_DIR, "translated_conversation_compare.png")
    cv2.imwrite(compare_path, compare_img)

    combined_path = os.path.join(RENDER_DIR, "translated_conversation.png")
    final_chat = render_chat(
        final_render_meta,
        profile_image=profile_image,
        contact_name=final_contact_name,
        header_status=final_status_text,
    )
    cv2.imwrite(combined_path, final_chat)
    print(f"[OUTPUT] JSON → {combined_json_path}")
    print(f"[OUTPUT] Image → {combined_path}")
    print(f"[OUTPUT] Pass 1 image → {pass1_path}")
    print(f"[OUTPUT] Pass 2 image → {pass2_path}")
    print(f"[OUTPUT] Compare image → {compare_path}")
    print(f"[TIMER] Render done in {time.time()-t4:.1f}s")

    total_runtime = time.time() - pipeline_start
    print(f"""
╔══════════════════════════════════════════╗
  PIPELINE SUMMARY (full-vision Gemini)
  Images processed : {len(images)}
  Messages rendered: {len(all_meta)}
  Contact name     : {contact_name}
  Total runtime    : {total_runtime:.1f}s
╚══════════════════════════════════════════╝""")

if __name__ == "__main__":
    main()