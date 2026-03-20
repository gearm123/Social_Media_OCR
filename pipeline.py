import cv2
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
    """Run CRAFT text detection + grouping on an already-concatenated image.

    *page_ranges* is a list of (start_y, end_y, page_index, crop_top, status_bar_info,
    bottom_artifact_info) tuples describing each image slice inside combined_img.

    Returns a flat list of objects with "page_index" and "bbox" in combined coords,
    plus per-page artifact splits for rendering.
    """
    print("\n[STEP] Running CRAFT text detection on combined image...")
    raw_boxes = detect_text(combined_img, craft_net)
    print(f"[CRAFT] Detected {len(raw_boxes)} text boxes in combined image")

    all_rects = [box_to_rect(b) for b in raw_boxes]

    # Split rects per page using y-ranges, apply artifact removal per page
    per_page_rects = [[] for _ in page_ranges]
    for rect in all_rects:
        cy = (rect[1] + rect[3]) / 2
        for idx, (start_y, end_y, page_idx, crop_top, _, _) in enumerate(page_ranges):
            if start_y <= cy < end_y:
                per_page_rects[idx].append(rect)
                break

    all_conversation_rects = []
    for idx, (start_y, end_y, page_idx, crop_top, status_bar_info, bottom_artifact_info) in enumerate(page_ranges):
        rects = per_page_rects[idx]

        # Shift rects to page-local coordinates for artifact detection
        local_rects = [(x1, y1 - start_y + crop_top, x2, y2 - start_y + crop_top)
                       for (x1, y1, x2, y2) in rects]

        conv_rects, top_art, bot_art = split_artifacts_from_conversation(
            local_rects, status_bar_info, bottom_artifact_info
        )
        if status_bar_info:
            print(
                f"[ARTIFACT] Page {page_idx}: preserving {len(top_art)} artifact boxes, "
                f"{len(conv_rects)} conversation boxes"
            )

        # Shift back to combined coords
        for r in conv_rects:
            x1, y1, x2, y2 = r
            all_conversation_rects.append((x1, y1 + start_y - crop_top, x2, y2 + start_y - crop_top))

    rows = group_rows(all_conversation_rects, combined_img.shape[1], combined_img)
    objects = group_objects(rows, combined_img.shape[1], combined_img)

    # Tag each object with its page_index
    for obj in objects:
        cy = (obj["bbox"][1] + obj["bbox"][3]) / 2
        obj["page_index"] = 0
        for start_y, end_y, page_idx, _, _, _ in page_ranges:
            if start_y <= cy < end_y:
                obj["page_index"] = page_idx
                break

    print(f"[GROUP] Created {len(objects)} chat objects in combined image")
    return objects


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