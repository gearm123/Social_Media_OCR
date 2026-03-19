from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detection import detect_text
from grouping import box_to_rect, group_objects, group_rows
from ocr_translate import run_ocr_on_region, translate_th_to_en


Rect = Tuple[int, int, int, int]

STATUS_SPEAKER_VERTICAL_DELTA_PX = 64
STATUS_SPEAKER_HORIZONTAL_GAP_PX = 190
STATUS_EXPAND_VERTICAL_GAP_PX = 76
STATUS_EXPAND_LEFT_DELTA_PX = 95


def _to_python_int_rect(rect) -> Rect:
    x1, y1, x2, y2 = rect
    return (int(x1), int(y1), int(x2), int(y2))


def _offset_rect(rect: Rect, dx: int = 0, dy: int = 0) -> Rect:
    x1, y1, x2, y2 = rect
    return _to_python_int_rect((x1 + dx, y1 + dy, x2 + dx, y2 + dy))


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
            bands[-1] = _to_python_int_rect((
                min(bx1, x1),
                min(by1, y1),
                max(bx2, x2),
                max(by2, y2),
            ))
        else:
            bands.append(_to_python_int_rect(rect))

    return bands


def _component_rects(mask, x_offset=0, y_offset=0, min_area=40):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    rects: List[Rect] = []

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        rects.append(_to_python_int_rect((
            x + x_offset,
            y + y_offset,
            x + w + x_offset,
            y + h + y_offset,
        )))

    return rects


def _merge_rects(rects: List[Rect], y_gap: int = 12) -> Optional[Rect]:
    if not rects:
        return None

    x1 = min(rect[0] for rect in rects)
    y1 = min(rect[1] for rect in rects)
    x2 = max(rect[2] for rect in rects)
    y2 = max(rect[3] for rect in rects)
    return _to_python_int_rect((x1, y1, x2, y2))


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


def _bottom_blue_icon_candidates(img, bottom_start):
    height, width = img.shape[:2]
    roi = img[bottom_start:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([145, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    rects = _component_rects(
        mask,
        x_offset=0,
        y_offset=bottom_start,
        min_area=max(30, int(width * height * 0.00002))
    )

    filtered: List[Rect] = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        if w <= 6 or h <= 6:
            continue
        if y1 < int(height * 0.68):
            continue
        if w > int(width * 0.10) or h > int(height * 0.08):
            continue
        filtered.append(rect)

    return filtered


def _find_input_field_candidates(img, search_top: int, search_bottom: int) -> List[Rect]:
    height, width = img.shape[:2]
    if search_bottom <= search_top:
        return []

    roi = img[search_top:search_bottom, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Messenger composer field is usually a light, low-saturation rounded band.
    mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 70, 255]))
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    rects = _component_rects(
        mask,
        x_offset=0,
        y_offset=search_top,
        min_area=max(120, int(width * height * 0.00008)),
    )

    filtered: List[Rect] = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        width_ratio = w / max(1, width)
        aspect_ratio = w / max(1, h)
        if w < int(width * 0.18):
            continue
        if h < 18 or h > int(height * 0.08):
            continue
        if y1 < search_top or y2 > search_bottom + 4:
            continue
        if y1 < max(search_top - 4, int(height * 0.62)):
            continue
        if width_ratio < 0.18 or width_ratio > 0.86:
            continue
        if aspect_ratio < 2.3:
            continue
        filtered.append(rect)

    return filtered


def _detect_keyboard(img) -> Optional[Rect]:
    height, width = img.shape[:2]
    start_y = int(height * 0.55)
    roi = img[start_y:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    row_mean = gray.mean(axis=1)
    dark_rows = np.where(row_mean < 105)[0]
    if len(dark_rows) == 0:
        return None

    groups = np.split(dark_rows, np.where(np.diff(dark_rows) > 6)[0] + 1)
    min_band_height = max(70, int(height * 0.12))

    best_band = None
    for group in groups:
        if len(group) == 0:
            continue
        band_top = int(group[0]) + start_y
        band_bottom = int(group[-1]) + start_y
        if band_bottom - band_top < min_band_height:
            continue
        if band_top < int(height * 0.62):
            continue
        best_band = (0, band_top, width - 1, min(height - 1, band_bottom + 8))
        break

    return _to_python_int_rect(best_band) if best_band else None


def _detect_bottom_bar(
    img,
    search_top: int,
    search_bottom: int,
    has_keyboard: bool = False,
) -> Optional[Rect]:
    height, width = img.shape[:2]
    if search_bottom <= search_top:
        return None

    icon_rects = _bottom_blue_icon_candidates(img, search_top)
    icon_rects = [
        rect for rect in icon_rects
        if rect[1] >= search_top - 4 and rect[3] <= search_bottom + 4
    ]
    input_rects = _find_input_field_candidates(img, search_top, search_bottom)

    if not icon_rects and not input_rects:
        return None

    best_input = None
    if input_rects:
        best_input = max(
            input_rects,
            key=lambda rect: (
                (rect[2] - rect[0]) / max(1, width),
                rect[1],
                -(rect[3] - rect[1]),
            ),
        )

    if best_input is not None:
        if search_bottom >= int(height * 0.90) and not icon_rects and not has_keyboard:
            return None

        input_top = best_input[1]
        nearby_icons = [
            rect for rect in icon_rects
            if rect[1] >= input_top - int(height * 0.03)
            and rect[3] <= search_bottom + int(height * 0.02)
        ]

        relevant_rects = [best_input] + nearby_icons
        merged = _merge_rects(relevant_rects)
        if merged is None:
            return None

        _, _, _, y2 = merged
        top = max(0, input_top - max(6, int(height * 0.006)))
        bottom = min(search_bottom, max(y2, best_input[3]) + max(6, int(height * 0.008)))
        min_height = max(26, int(height * 0.035))
        if bottom - top < min_height:
            bottom = min(search_bottom, top + min_height)

        return _to_python_int_rect((0, top, width - 1, bottom))

    if not has_keyboard:
        return None

    if len(icon_rects) < 2:
        return None

    x_positions = [((rect[0] + rect[2]) / 2) for rect in icon_rects]
    spread = max(x_positions) - min(x_positions)
    if spread < width * 0.16:
        return None

    merged_icons = _merge_rects(icon_rects)
    if merged_icons is None:
        return None

    _, y1, _, y2 = merged_icons
    top = max(0, y1 - max(6, int(height * 0.006)))
    bottom = min(search_bottom, y2 + max(10, int(height * 0.012)))
    min_height = max(24, int(height * 0.03))
    if bottom - top < min_height:
        bottom = min(search_bottom, top + min_height)

    return _to_python_int_rect((0, top, width - 1, bottom))


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
            best_rect = _to_python_int_rect((x1, y1, x2, y2))

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

    top = max(0, min(y_candidates) - max(4, int(image_height * 0.006)))
    bottom = min(
        image_height - 1,
        max(y_candidates) + max(5, int(image_height * 0.007))
    )

    min_height = max(42, int(image_height * 0.065))
    if bottom - top < min_height:
        bottom = min(image_height - 1, top + min_height)

    max_height = max(58, int(image_height * 0.10))
    if bottom - top > max_height:
        bottom = min(image_height - 1, top + max_height)

    return _to_python_int_rect((0, top, image_width - 1, bottom))


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
        "bbox": _to_python_int_rect(status_bbox),
        "conversation_start_y": int(min(image_height - 1, status_bbox[3] + 1)),
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

    full_width_bbox: Rect = _to_python_int_rect((
        0,
        max(0, artifact_top - top_padding),
        image_width - 1,
        min(image_height - 1, artifact_bottom + bottom_padding),
    ))

    conversation_start_y = int(min(image_height - 1, full_width_bbox[3] + 1))

    return {
        "bbox": _to_python_int_rect(full_width_bbox),
        "conversation_start_y": conversation_start_y,
        "bands": [_to_python_int_rect(band) for band in bands],
        "method": "rects",
    }


def detect_top_status_bar(img, rects: Optional[List[Rect]] = None) -> Optional[Dict[str, object]]:
    ui_result = _detect_by_ui(img)
    if ui_result:
        return ui_result

    if rects is not None:
        return _detect_by_rects(rects, img.shape if img is not None else None)

    return None


def detect_bottom_artifacts(img) -> Optional[Dict[str, object]]:
    if img is None:
        return None

    height, width = img.shape[:2]
    keyboard_bbox = _detect_keyboard(img)

    if keyboard_bbox:
        search_bottom = keyboard_bbox[1]
        search_top = max(int(height * 0.54), search_bottom - int(height * 0.18))
    else:
        search_bottom = int(height * 0.97)
        search_top = int(height * 0.72)

    bottom_bar_bbox = _detect_bottom_bar(
        img,
        search_top,
        search_bottom,
        has_keyboard=keyboard_bbox is not None,
    )

    if not keyboard_bbox and not bottom_bar_bbox:
        return None

    artifact_top = (
        bottom_bar_bbox[1] if bottom_bar_bbox
        else keyboard_bbox[1] if keyboard_bbox
        else None
    )
    artifact_bottom = (
        keyboard_bbox[3] if keyboard_bbox
        else bottom_bar_bbox[3] if bottom_bar_bbox
        else None
    )

    if artifact_top is None or artifact_bottom is None:
        return None

    full_bbox = _to_python_int_rect((0, artifact_top, width - 1, artifact_bottom))

    return {
        "bbox": full_bbox,
        "conversation_end_y": int(max(0, artifact_top - 1)),
        "keyboard_bbox": keyboard_bbox,
        "bottom_bar_bbox": bottom_bar_bbox,
        "method": "ui",
    }


def _select_speaker_row(rows: List[Dict[str, object]], avatar_rect: Optional[Rect]) -> Optional[Rect]:
    if not rows or avatar_rect is None:
        return None

    ax1, ay1, ax2, ay2 = avatar_rect
    avatar_center_y = (ay1 + ay2) / 2
    best_bbox = None
    best_score = None
    fallback_bbox = None
    fallback_score = None

    for row in rows:
        bbox = _to_python_int_rect(row["bbox"])
        x1, y1, x2, y2 = bbox
        row_center_y = (y1 + y2) / 2
        vertical_delta = abs(row_center_y - avatar_center_y)
        horizontal_gap = max(0, x1 - ax2)
        row_width = x2 - x1

        if x2 < ax2 - 12:
            continue

        score = (horizontal_gap, vertical_delta, -row_width)
        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_bbox = bbox

        if vertical_delta > STATUS_SPEAKER_VERTICAL_DELTA_PX:
            continue
        if horizontal_gap > STATUS_SPEAKER_HORIZONTAL_GAP_PX:
            continue

        if best_score is None or score < best_score:
            best_score = score
            best_bbox = bbox

    return best_bbox or fallback_bbox


def _expand_speaker_bbox(rows: List[Dict[str, object]], base_bbox: Optional[Rect], avatar_rect: Optional[Rect]) -> Optional[Rect]:
    if base_bbox is None:
        return None

    bx1, by1, bx2, by2 = _to_python_int_rect(base_bbox)
    merged = [base_bbox]
    current_bottom = by2

    for row in sorted(rows, key=lambda r: r["bbox"][1]):
        rect = _to_python_int_rect(row["bbox"])
        x1, y1, x2, y2 = rect

        if rect == base_bbox:
            continue
        if y1 < by1:
            continue

        vertical_gap = y1 - current_bottom
        if vertical_gap < 0:
            vertical_gap = 0
        if vertical_gap > STATUS_EXPAND_VERTICAL_GAP_PX:
            continue

        left_delta = abs(x1 - bx1)
        overlap = max(0, min(x2, bx2) - max(x1, bx1))
        overlap_ratio = overlap / max(1, min(x2 - x1, bx2 - bx1))

        if left_delta > STATUS_EXPAND_LEFT_DELTA_PX and overlap_ratio < 0.25:
            continue

        if (x2 - x1) > (bx2 - bx1) * 1.55:
            continue

        merged.append(rect)
        current_bottom = max(current_bottom, y2)

    x1 = min(rect[0] for rect in merged)
    y1 = min(rect[1] for rect in merged)
    x2 = max(rect[2] for rect in merged)
    y2 = max(rect[3] for rect in merged)
    return _to_python_int_rect((x1, y1, x2, y2))


def extract_profile_image(img, avatar_rect: Optional[Rect]):
    if img is None or avatar_rect is None:
        return None

    x1, y1, x2, y2 = _to_python_int_rect(avatar_rect)
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    size = int(max(width, height) * 1.08)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - size // 2)
    ny1 = max(0, cy - size // 2)
    nx2 = min(img.shape[1], nx1 + size)
    ny2 = min(img.shape[0], ny1 + size)

    # Re-center after clamping to preserve a square crop.
    nx1 = max(0, nx2 - size)
    ny1 = max(0, ny2 - size)

    crop = img[ny1:ny2, nx1:nx2].copy()
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    if crop_h != crop_w:
        size = max(crop_h, crop_w)
        pad_top = (size - crop_h) // 2
        pad_bottom = size - crop_h - pad_top
        pad_left = (size - crop_w) // 2
        pad_right = size - crop_w - pad_left
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    return crop


def process_status_bar(img, status_bar_info: Optional[Dict[str, object]], craft_net):
    if img is None or not status_bar_info:
        return None

    status_bbox = _to_python_int_rect(status_bar_info["bbox"])
    sx1, sy1, sx2, sy2 = status_bbox
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    status_crop = img[sy1:sy2, sx1:sx2].copy()
    if status_crop.size == 0:
        return None

    raw_boxes = detect_text(status_crop, craft_net)
    local_rects = [box_to_rect(box) for box in raw_boxes]
    local_rows = group_rows(local_rects)
    _ = group_objects(local_rows)
    absolute_rows = [
        {"bbox": _offset_rect(row["bbox"], dx=sx1, dy=sy1)}
        for row in local_rows
    ]

    avatar_rect = status_bar_info.get("avatar_rect")
    speaker_bbox = _select_speaker_row(absolute_rows, avatar_rect)
    speaker_bbox = _expand_speaker_bbox(absolute_rows, speaker_bbox, avatar_rect)

    full_text = run_ocr_on_region(status_crop)
    speaker_text_th = ""
    speaker_text_en = ""

    if speaker_bbox:
        x1, y1, x2, y2 = speaker_bbox
        speaker_crop = img[y1:y2, x1:x2].copy()
        speaker_text_th = run_ocr_on_region(speaker_crop)
        speaker_text_en = translate_th_to_en(speaker_text_th)

    return {
        "bbox": status_bbox,
        "status_image": status_crop,
        "profile_image": extract_profile_image(img, avatar_rect),
        "text_th": full_text,
        "text_en": full_text,
        "speaker_bbox": _to_python_int_rect(speaker_bbox) if speaker_bbox else None,
        "speaker_text_th": speaker_text_th,
        "speaker_text_en": speaker_text_en,
    }


def split_artifacts_from_conversation(
    rects: List[Rect],
    top_artifact_info: Optional[Dict[str, object]],
    bottom_artifact_info: Optional[Dict[str, object]] = None,
) -> Tuple[List[Rect], List[Rect], List[Rect]]:
    if not rects:
        return rects, [], []

    top_cutoff = int(top_artifact_info["conversation_start_y"]) if top_artifact_info else 0
    bottom_cutoff = int(bottom_artifact_info["conversation_end_y"]) if bottom_artifact_info else 10**9
    conversation_rects: List[Rect] = []
    top_artifact_rects: List[Rect] = []
    bottom_artifact_rects: List[Rect] = []

    for rect in rects:
        _, y1, _, y2 = rect
        center_y = (y1 + y2) / 2
        if center_y < top_cutoff:
            top_artifact_rects.append(rect)
        elif center_y > bottom_cutoff:
            bottom_artifact_rects.append(rect)
        else:
            conversation_rects.append(rect)

    return conversation_rects, top_artifact_rects, bottom_artifact_rects
