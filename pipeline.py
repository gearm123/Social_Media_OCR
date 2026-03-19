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
from ocr_translate import ocr_and_translate, run_ocr_on_region

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

def process_image(path, craft_net):
    image_start = time.time()
    print(f"\n[IMAGE] Processing {Path(path).name}")

    img = cv2.imread(path)
    status_bar_info = detect_top_status_bar(img)
    bottom_artifact_info = detect_bottom_artifacts(img)
    status_bar_result = process_status_bar(img, status_bar_info, craft_net)

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

    overlay = img.copy()
    results = []

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        sx1, sy1, sx2, sy2 = status_bbox
        status_crop = img[sy1:sy2, sx1:sx2].copy()
        status_text = run_ocr_on_region(status_crop)
        draw_rect(overlay, status_bbox, (0, 165, 255), "status_bar")
        speaker_bbox = status_bar_result.get("speaker_bbox") if status_bar_result else None
        if speaker_bbox:
            draw_rect(overlay, speaker_bbox, (0, 200, 0), "speaker")

        results.append({
            "type": "status_bar",
            "bbox": to_serializable_rect(status_bbox),
            "text_th": status_text,
            "text_en": status_text,
            "speaker_text_th": status_bar_result.get("speaker_text_th", "") if status_bar_result else "",
            "speaker_text_en": status_bar_result.get("speaker_text_en", "") if status_bar_result else "",
            "speaker_bbox": to_serializable_rect(speaker_bbox) if speaker_bbox else None,
        })

    if bottom_artifact_info:
        bottom_bbox = bottom_artifact_info["bbox"]
        bx1, by1, bx2, by2 = bottom_bbox
        bottom_crop = img[by1:by2, bx1:bx2].copy()
        bottom_text = run_ocr_on_region(bottom_crop)
        draw_rect(overlay, bottom_bbox, (0, 140, 255), "bottom_artifact")
        results.append({
            "type": "bottom_artifact",
            "bbox": to_serializable_rect(bottom_bbox),
            "text_th": bottom_text,
            "text_en": bottom_text,
        })

        keyboard_bbox = bottom_artifact_info.get("keyboard_bbox")
        if keyboard_bbox:
            draw_rect(overlay, keyboard_bbox, (60, 60, 220), "keyboard")
            results.append({
                "type": "keyboard",
                "bbox": to_serializable_rect(keyboard_bbox),
                "text_th": "",
                "text_en": "",
            })

        bottom_bar_bbox = bottom_artifact_info.get("bottom_bar_bbox")
        if bottom_bar_bbox:
            draw_rect(overlay, bottom_bar_bbox, (0, 220, 220), "bottom_bar")
            results.append({
                "type": "bottom_bar",
                "bbox": to_serializable_rect(bottom_bar_bbox),
                "text_th": "",
                "text_en": "",
            })

    print("[STEP] Running PaddleOCR on extracted regions...")
    objects = ocr_and_translate(
        objects,
        img,
        progress_label=f"OCR {Path(path).name}"
    )

    for idx, obj in enumerate(objects):
        rect = obj["bbox"]
        text_th = obj.get("text_th", "")
        text_en = obj.get("text_en", "")
        typ = classify_object_type(img, rect, text_th, text_en)

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
            "text_en": text_en
        })

    image_runtime = time.time() - image_start
    print(
        f"[IMAGE] Finished {Path(path).name} in "
        f"{image_runtime:.2f}s"
    )

    render_context = {
        "speaker_text_en": status_bar_result.get("speaker_text_en", "") if status_bar_result else "",
        "profile_image": status_bar_result.get("profile_image") if status_bar_result else None,
    }

    return overlay, results, render_context