import cv2
import os
import re
import time
from pathlib import Path

from artifacts_cleaning import (
    detect_bottom_artifacts,
    detect_top_status_bar,
    process_status_bar,
    split_artifacts_from_conversation,
)
from detection import detect_text
from grouping import (
    box_to_rect,
    group_rows,
    group_objects,
    classify_object_type,
    classify_sender_receiver,
    side_hint,
)
from ocr_translate import ocr_and_translate, run_ocr_on_region, translate_th_to_en
from timestamp_detection import parse_timestamp_text

def draw_rect(img, rect, color, label):
    x1, y1, x2, y2 = rect

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        img,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def to_serializable_rect(rect):
    x1, y1, x2, y2 = rect
    return [int(x1), int(y1), int(x2), int(y2)]


def _append_artifact_results(overlay, img, status_bar_info, bottom_artifact_info, status_bar_result):
    return []


def prepare_image_crop_info(path):
    """Read image and detect artifacts — no CRAFT, no grouping.
    Returns the minimal info needed to crop and concatenate images.
    """
    img = cv2.imread(path)
    status_bar_info = detect_top_status_bar(img)
    bottom_artifact_info = detect_bottom_artifacts(img)

    if status_bar_info:
        print(
            f"[ARTIFACT] {Path(path).name}: status bar "
            f"{status_bar_info.get('bbox')} via {status_bar_info.get('method','unknown')}"
        )
    if bottom_artifact_info:
        print(
            f"[ARTIFACT] {Path(path).name}: bottom artifact "
            f"{bottom_artifact_info.get('bbox')}"
        )

    return {
        "path": path,
        "img": img,
        "status_bar_info": status_bar_info,
        "bottom_artifact_info": bottom_artifact_info,
    }


def run_craft_and_group_on_combined(combined_img, craft_net, page_ranges):
    """Run CRAFT text detection + grouping on the combined image.

    CRAFT degrades badly on very tall images (rescales too aggressively).
    To avoid this, we run CRAFT on each page slice individually and then
    re-map every detected box back to combined-image coordinates.

    *page_ranges* is a list of (start_y, end_y, page_index, crop_top,
    status_bar_info, bottom_artifact_info) tuples.

    Returns a flat list of objects with "page_index" and "bbox" in combined coords.
    """
    _craft_verbose = os.environ.get("CRAFT_VERBOSE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if _craft_verbose:
        print("\n[STEP] Running CRAFT text detection per page slice...")

    # CRAFT max height — images taller than this cause detail loss
    CRAFT_MAX_HEIGHT = 2000
    img_h, img_w = combined_img.shape[:2]

    all_conversation_rects = []   # in combined-image coordinates
    total_boxes = 0

    for idx, (start_y, end_y, page_idx, crop_top,
              status_bar_info, bottom_artifact_info) in enumerate(page_ranges):

        # Crop out this page's slice from the combined image
        slice_img = combined_img[start_y:end_y, :]

        slice_h = slice_img.shape[0]
        if slice_h == 0:
            continue

        # If the slice is still too tall, downscale for CRAFT then map back
        scale = 1.0
        if slice_h > CRAFT_MAX_HEIGHT:
            scale = CRAFT_MAX_HEIGHT / slice_h
            slice_for_craft = cv2.resize(
                slice_img,
                (int(img_w * scale), CRAFT_MAX_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        else:
            slice_for_craft = slice_img

        raw_boxes = detect_text(slice_for_craft, craft_net)
        total_boxes += len(raw_boxes)

        # Convert CRAFT polygons → axis-aligned rects, rescale if needed
        local_rects = []
        for b in raw_boxes:
            x1, y1, x2, y2 = box_to_rect(b)
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            # These are in slice-local coords; shift to crop_top-relative for
            # artifact detection (matches how status_bar_info bbox was measured)
            local_rects.append((x1, y1 + crop_top, x2, y2 + crop_top))

        conv_rects, top_art, bot_art = split_artifacts_from_conversation(
            local_rects, status_bar_info, bottom_artifact_info
        )
        if _craft_verbose:
            print(
                f"[CRAFT] Page {page_idx}: {len(raw_boxes)} boxes → "
                f"{len(conv_rects)} conversation, {len(top_art)} top artifacts"
            )

        # Shift conversation rects to combined-image coordinates
        for x1, y1, x2, y2 in conv_rects:
            # y is currently crop_top-relative; remove crop_top offset, add start_y
            all_conversation_rects.append((
                x1,
                y1 - crop_top + start_y,
                x2,
                y2 - crop_top + start_y,
            ))

    if _craft_verbose:
        print(f"[CRAFT] Total: {total_boxes} boxes detected across {len(page_ranges)} pages")

    rows = group_rows(all_conversation_rects, img_w, combined_img)
    objects = group_objects(rows, img_w, combined_img)

    # Tag each object with its page_index
    for obj in objects:
        cy = (obj["bbox"][1] + obj["bbox"][3]) / 2
        obj["page_index"] = 0
        for start_y, end_y, page_idx, _, _, _ in page_ranges:
            if start_y <= cy < end_y:
                obj["page_index"] = page_idx
                break

    if _craft_verbose:
        print(f"[GROUP] Created {len(objects)} chat objects in combined image")
    return objects


def filter_timestamp_chat_objects(objects, image_width, img=None):
    """Remove real date/time rows; fix messages wrongly classified as *timestamp*.

    - Drops objects that are genuinely timestamps (or time-only OCR on bubbles).
    - If type is *timestamp* but OCR text does not look like a date/time, re-classify
      using bubble colour (sender/receiver) so real messages are not lost.

    Returns ``(kept_list, num_dropped)``.
    """
    from timestamp_detection import is_timestamp

    if not objects:
        return [], 0

    kept = []
    dropped = 0
    for obj in objects:
        th = (obj.get("text_th") or "").strip()
        typ = (obj.get("type") or "receiver").lower()
        bbox = obj.get("bbox")

        if typ == "timestamp":
            # Message mis-tagged as timestamp — use bubble colour only (avoid timestamp heuristics)
            if th and not is_timestamp(th) and len(th) > 8 and img is not None and bbox:
                obj["type"] = classify_sender_receiver(img, bbox)
            if (obj.get("type") or "receiver").lower() == "timestamp":
                dropped += 1
                continue

        if th and is_timestamp(th):
            side = side_hint(bbox, image_width) if bbox and image_width else "unknown"
            if side == "center":
                dropped += 1
                continue
            if len(th) <= 40:
                dropped += 1
                continue

        kept.append(obj)

    return kept, dropped


_GAMBLING_ASCII = re.compile(
    r"(78win|vg98|win\s*98|pg\s*slot|slot\s*online|bet\s*flix|ufa\d+|lsm\d+)",
    re.I,
)


def _is_gambling_spam_ocr(text_th, text_en):
    """True if OCR text looks like an in-screen gambling/ad overlay, not chat."""
    th = (text_th or "").strip()
    en = (text_en or "").strip()
    blob = f"{th} {en}"
    blob_l = blob.lower()
    if _GAMBLING_ASCII.search(blob_l):
        return True
    # Thai promos: "เครดิต … ฟรี" with promo digits / short banner
    if "เครดิต" in th and "ฟรี" in th:
        if len(th) <= 120 and (
            re.search(r"\d", th)
            or th.strip().startswith("(")
            or re.search(r"เครดิต\s*ฟรี", th)
        ):
            return True
    return False


def filter_gambling_overlay_objects(objects):
    """Remove objects whose OCR matches known gambling/ad overlay strings.

    Returns ``(kept_list, num_dropped)``.
    """
    if not objects:
        return [], 0
    kept = []
    dropped = 0
    for obj in objects:
        if _is_gambling_spam_ocr(obj.get("text_th"), obj.get("text_en")):
            dropped += 1
            continue
        kept.append(obj)
    return kept, dropped


def prepare_image_layout(path, craft_net, keep_status_bar_context=True):
    image_start = time.time()
    print(f"\n[IMAGE] Processing {Path(path).name}")

    img = cv2.imread(path)
    status_bar_info = detect_top_status_bar(img)
    bottom_artifact_info = detect_bottom_artifacts(img)
    status_bar_result = (
        process_status_bar(img, status_bar_info, craft_net)
        if keep_status_bar_context else None
    )

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        print(
            f"[ARTIFACT] UI-based top status/header band detected "
            f"via {status_bar_info.get('method', 'unknown')}: {status_bbox}"
        )
    if bottom_artifact_info:
        print(
            f"[ARTIFACT] UI-based bottom artifact band detected "
            f"via {bottom_artifact_info.get('method', 'unknown')}: "
            f"{bottom_artifact_info['bbox']}"
        )

    print("[STEP] Running CRAFT text detection...")
    raw_boxes = detect_text(img, craft_net)
    print(f"[CRAFT] Detected {len(raw_boxes)} text boxes")

    rects = [box_to_rect(b) for b in raw_boxes]
    rects, top_artifact_rects, bottom_artifact_rects = split_artifacts_from_conversation(
        rects,
        status_bar_info,
        bottom_artifact_info,
    )

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        print(
            f"[ARTIFACT] Top status/header band detected: "
            f"{status_bbox}"
        )
        print(
            f"[ARTIFACT] Preserving {len(top_artifact_rects)} top artifact boxes "
            f"and processing {len(rects)} conversation boxes"
        )
    if bottom_artifact_info:
        print(
            f"[ARTIFACT] Preserving {len(bottom_artifact_rects)} bottom artifact boxes"
        )

    rows = group_rows(rects, img.shape[1], img)
    objects = group_objects(rows, img.shape[1], img)

    print(f"[GROUP] Created {len(objects)} chat objects")
    print(
        f"[IMAGE] Layout ready for {Path(path).name} in "
        f"{time.time() - image_start:.2f}s"
    )

    return {
        "path": path,
        "img": img,
        "objects": objects,
        "conversation_rects": rects,
        "conversation_top_y": min((rect[1] for rect in rects), default=0),
        "conversation_bottom_y": max((rect[3] for rect in rects), default=img.shape[0] - 1),
        "status_bar_info": status_bar_info,
        "bottom_artifact_info": bottom_artifact_info,
        "status_bar_result": status_bar_result,
    }


def finalize_image_layout(layout):
    img = layout["img"]
    overlay = img.copy()
    results = _append_artifact_results(
        overlay,
        img,
        layout.get("status_bar_info"),
        layout.get("bottom_artifact_info"),
        layout.get("status_bar_result"),
    )

    for obj in layout["objects"]:
        rect = obj["bbox"]
        text_th = obj.get("text_th", "")
        text_en = obj.get("text_en", "")
        typ = classify_object_type(img, rect, text_th, text_en)

        if typ == "timestamp":
            timestamp_info = parse_timestamp_text(text_th) or parse_timestamp_text(text_en)
            if timestamp_info and timestamp_info.get("format_type") == "text_time":
                text_part = timestamp_info.get("text_part", "").strip()
                time_text = timestamp_info.get("time_text", "").strip()
                if text_part and time_text:
                    translated_prefix = translate_th_to_en(text_part) or text_part
                    text_en = f"{translated_prefix} {time_text}".strip()
                elif time_text:
                    text_en = time_text
            elif not text_en:
                text_en = text_th

        if typ == "sender":
            draw_rect(overlay, rect, (0, 0, 0), "sender")
        elif typ == "timestamp":
            draw_rect(overlay, rect, (80, 180, 80), "timestamp")
        else:
            draw_rect(overlay, rect, (255, 0, 0), "receiver")

        results.append({
            "type": typ,
            "bbox": to_serializable_rect(rect),
            "text_th": text_th,
            "text_en": text_en,
            "ocr_source": obj.get("ocr_source", ""),
            "ocr_span_count": int(obj.get("ocr_span_count", 0) or 0),
            "ocr_validated": bool(obj.get("ocr_validated", False)),
            "ocr_trust_score": float(obj.get("ocr_trust_score", 0.0) or 0.0),
            "ocr_low_confidence": bool(obj.get("ocr_low_confidence", False)),
            "ocr_reasons": list(obj.get("ocr_reasons", []) or []),
        })

    render_context = {
        "speaker_text_en": "",
        "profile_image": None,
    }

    return overlay, results, render_context

def process_image(path, craft_net):
    image_start = time.time()
    layout = prepare_image_layout(path, craft_net)
    print("[STEP] Running PaddleOCR on extracted regions...")
    layout["objects"] = ocr_and_translate(
        layout["objects"],
        layout["img"],
        progress_label=f"OCR {Path(path).name}"
    )
    overlay, results, render_context = finalize_image_layout(layout)
    print(
        f"[IMAGE] Finished {Path(path).name} in "
        f"{time.time() - image_start:.2f}s"
    )
    return overlay, results, render_context