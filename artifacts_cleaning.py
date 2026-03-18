from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Rect = Tuple[int, int, int, int]


def _merge_top_bands(rects: List[Rect], image_height: int) -> List[Rect]:
    if not rects:
        return []

    merge_gap = max(12, int(image_height * 0.015))
    bands: List[Rect] = []

    for rect in sorted(rects, key=lambda r: (r[1], r[0])):
        x1, y1, x2, y2 = rect

        if not bands:
            bands.append(rect)
            continue

        bx1, by1, bx2, by2 = bands[-1]
        if y1 - by2 <= merge_gap:
            bands[-1] = (
                min(bx1, x1),
                min(by1, y1),
                max(bx2, x2),
                max(by2, y2),
            )
        else:
            bands.append(rect)

    return bands


def _component_rects(mask, x_offset=0, y_offset=0, min_area=40):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    rects: List[Rect] = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        rects.append((x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset))

    return rects


def _blue_icon_candidates(img, top_limit):
    height, width = img.shape[:2]
    roi = img[:top_limit, int(width * 0.52):]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    rects = _component_rects(
        mask,
        x_offset=int(width * 0.52),
        y_offset=0,
        min_area=max(35, int(width * height * 0.00003))
    )

    filtered: List[Rect] = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        if w <= 6 or h <= 6:
            continue
        if y1 > int(height * 0.24):
            continue
        if w > int(width * 0.16) or h > int(height * 0.10):
            continue
        filtered.append(rect)

    return filtered


def _avatar_candidate(img, top_limit):
    height, width = img.shape[:2]
    roi = img[:top_limit, :int(width * 0.42)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    min_radius = max(10, int(min(height, width) * 0.018))
    max_radius = max(min_radius + 4, int(min(height, width) * 0.055))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_radius * 2),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    best_rect = None
    best_score = -1.0

    for circle in circles[0]:
        cx, cy, radius = circle
        cx = int(cx)
        cy = int(cy)
        radius = int(radius)

        if cx > int(width * 0.28):
            continue
        if cy < int(height * 0.04) or cy > int(height * 0.22):
            continue

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(roi.shape[1] - 1, cx + radius)
        y2 = min(roi.shape[0] - 1, cy + radius)
        crop = roi[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
        score = float(np.mean(saturation)) + radius
        if score > best_score:
            best_score = score
            best_rect = (x1, y1, x2, y2)

    return best_rect


def _build_status_bbox(image_shape, avatar_rect, icon_rects):
    image_height, image_width = image_shape[:2]

    y_candidates = []
    if avatar_rect:
        y_candidates.extend([avatar_rect[1], avatar_rect[3]])
    for rect in icon_rects:
        y_candidates.extend([rect[1], rect[3]])

    if not y_candidates:
        return None

    top = max(0, min(y_candidates) - max(8, int(image_height * 0.01)))
    bottom = min(
        image_height - 1,
        max(y_candidates) + max(10, int(image_height * 0.015))
    )

    min_height = max(48, int(image_height * 0.09))
    if bottom - top < min_height:
        bottom = min(image_height - 1, top + min_height)

    return (0, top, image_width - 1, bottom)


def _detect_by_ui(img) -> Optional[Dict[str, object]]:
    if img is None:
        return None

    image_height, image_width = img.shape[:2]
    top_limit = int(image_height * 0.30)

    avatar_rect = _avatar_candidate(img, top_limit)
    icon_rects = _blue_icon_candidates(img, top_limit)

    if avatar_rect is None and not icon_rects:
        return None

    status_bbox = _build_status_bbox(img.shape, avatar_rect, icon_rects)
    if status_bbox is None:
        return None

    return {
        "bbox": status_bbox,
        "conversation_start_y": min(image_height - 1, status_bbox[3] + 1),
        "avatar_rect": avatar_rect,
        "icon_rects": icon_rects,
        "method": "ui",
    }


def _detect_by_rects(rects: List[Rect], image_shape) -> Optional[Dict[str, object]]:
    if not rects or image_shape is None:
        return None

    image_height, image_width = image_shape[:2]
    top_scan_limit = int(image_height * 0.32)
    top_rects = [rect for rect in rects if rect[1] < top_scan_limit]

    if not top_rects:
        return None

    bands = _merge_top_bands(top_rects, image_height)
    if not bands:
        return None

    max_gap = max(18, int(image_height * 0.02))
    max_bottom = int(image_height * 0.24)

    artifact_top = bands[0][1]
    artifact_bottom = bands[0][3]

    for next_band in bands[1:]:
        gap = next_band[1] - artifact_bottom
        if gap > max_gap:
            break
        if next_band[1] > max_bottom:
            break
        artifact_bottom = max(artifact_bottom, next_band[3])

    top_padding = max(2, int(image_height * 0.004))
    bottom_padding = max(6, int(image_height * 0.008))

    full_width_bbox: Rect = (
        0,
        max(0, artifact_top - top_padding),
        image_width - 1,
        min(image_height - 1, artifact_bottom + bottom_padding),
    )

    conversation_start_y = min(image_height - 1, full_width_bbox[3] + 1)

    return {
        "bbox": full_width_bbox,
        "conversation_start_y": conversation_start_y,
        "bands": bands,
        "method": "rects",
    }


def detect_top_status_bar(img, rects: Optional[List[Rect]] = None) -> Optional[Dict[str, object]]:
    ui_result = _detect_by_ui(img)
    if ui_result:
        return ui_result

    if rects is not None:
        return _detect_by_rects(rects, img.shape if img is not None else None)

    return None


def split_artifacts_from_conversation(
    rects: List[Rect],
    status_bar_info: Optional[Dict[str, object]],
) -> Tuple[List[Rect], List[Rect]]:
    if not rects or not status_bar_info:
        return rects, []

    cutoff_y = int(status_bar_info["conversation_start_y"])
    conversation_rects: List[Rect] = []
    artifact_rects: List[Rect] = []

    for rect in rects:
        _, y1, _, y2 = rect
        center_y = (y1 + y2) / 2
        if center_y < cutoff_y:
            artifact_rects.append(rect)
        else:
            conversation_rects.append(rect)

    return conversation_rects, artifact_rects
