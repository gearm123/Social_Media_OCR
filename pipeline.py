import cv2
import time
from pathlib import Path

from artifacts_cleaning import (
    detect_top_status_bar,
    split_artifacts_from_conversation,
)
from detection import detect_text
from grouping import (
    box_to_rect,
    group_rows,
    group_objects,
    classify_sender_receiver,
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

def process_image(path, craft_net):
    image_start = time.time()
    print(f"\n[IMAGE] Processing {Path(path).name}")

    img = cv2.imread(path)
    status_bar_info = detect_top_status_bar(img)

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        print(
            f"[ARTIFACT] UI-based top status/header band detected "
            f"via {status_bar_info.get('method', 'unknown')}: {status_bbox}"
        )

    print("[STEP] Running CRAFT text detection...")
    raw_boxes = detect_text(img, craft_net)
    print(f"[CRAFT] Detected {len(raw_boxes)} text boxes")

    rects = [box_to_rect(b) for b in raw_boxes]
    if not status_bar_info:
        status_bar_info = detect_top_status_bar(img, rects)

    rects, artifact_rects = split_artifacts_from_conversation(rects, status_bar_info)

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        print(
            f"[ARTIFACT] Top status/header band detected: "
            f"{status_bbox}"
        )
        print(
            f"[ARTIFACT] Preserving {len(artifact_rects)} top artifact boxes "
            f"and processing {len(rects)} conversation boxes"
        )

    rows = group_rows(rects)
    objects = group_objects(rows)

    print(f"[GROUP] Created {len(objects)} chat objects")

    overlay = img.copy()
    results = []

    if status_bar_info:
        status_bbox = status_bar_info["bbox"]
        sx1, sy1, sx2, sy2 = status_bbox
        status_crop = img[sy1:sy2, sx1:sx2].copy()
        status_text = run_ocr_on_region(status_crop)
        draw_rect(overlay, status_bbox, (0, 165, 255), "status_bar")
        results.append({
            "type": "status_bar",
            "bbox": status_bbox,
            "text_th": status_text,
            "text_en": status_text,
        })

    print("[STEP] Running PaddleOCR on extracted regions...")
    objects = ocr_and_translate(
        objects,
        img,
        progress_label=f"OCR {Path(path).name}"
    )

    for idx, obj in enumerate(objects):
        rect = obj["bbox"]
        typ = classify_sender_receiver(img, rect)
        text_th = obj.get("text_th", "")
        text_en = obj.get("text_en", "")

        if typ == "sender":
            draw_rect(overlay, rect, (0, 0, 0), "sender")
        else:
            draw_rect(overlay, rect, (255, 0, 0), "receiver")

        results.append({
            "type": typ,
            "bbox": rect,
            "text_th": text_th,
            "text_en": text_en
        })

    image_runtime = time.time() - image_start
    print(
        f"[IMAGE] Finished {Path(path).name} in "
        f"{image_runtime:.2f}s"
    )

    return overlay, results