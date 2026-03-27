import cv2
import numpy as np
from timestamp_detection import is_timestamp, parse_timestamp_text
import re

ROW_VERTICAL_GAP_PX = 30
ROW_VERTICAL_GAP_WITH_X_OVERLAP_PX = 42
ROW_HORIZONTAL_GAP_PX = 110
ROW_HORIZONTAL_GAP_WITH_Y_OVERLAP_PX = 68
OBJECT_VERTICAL_GAP_PX = 34
OBJECT_VERTICAL_GAP_WITH_X_OVERLAP_PX = 48
OBJECT_HORIZONTAL_GAP_PX = 72
OBJECT_HORIZONTAL_GAP_WITH_Y_OVERLAP_PX = 82

TIMESTAMP_CENTER_TOLERANCE_RATIO = 0.16
TIMESTAMP_NUMERIC_FORMAT_RE = re.compile(r"^[0-9\s:.,\-\/]+$")
TIMESTAMP_HAS_FORMATTING_RE = re.compile(r"[:.,\-\/]")

def box_to_rect(box):
    xs = box[:, 0]
    ys = box[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def rect_center_y(r):
    return (r[1] + r[3]) / 2

def rect_center_x(r):
    return (r[0] + r[2]) / 2

def rect_width(r):
    return max(1, r[2] - r[0])

def rect_height(r):
    return max(1, r[3] - r[1])

def union_rect(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    )

def horizontal_gap(a, b):
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0

def horizontal_overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))

def vertical_gap(a, b):
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0

def vertical_overlap(a, b):
    return max(0, min(a[3], b[3]) - max(a[1], b[1]))

def _pairwise_metrics(rects, rect):
    x_overlaps = [horizontal_overlap(existing, rect) for existing in rects]
    y_overlaps = [vertical_overlap(existing, rect) for existing in rects]
    x_gaps = [horizontal_gap(existing, rect) for existing in rects]
    y_gaps = [vertical_gap(existing, rect) for existing in rects]
    return {
        "max_x_overlap": max(x_overlaps, default=0),
        "has_y_overlap": any(overlap > 0 for overlap in y_overlaps),
        "min_x_gap": min(x_gaps, default=0),
        "min_y_gap": min(y_gaps, default=0),
    }


def sides_are_compatible(a, b):
    if a == "unknown" or b == "unknown":
        return True
    if a == b:
        return True
    # Center lane = date/time rows — must NOT merge with left/right message bubbles.
    if a == "center" and b == "center":
        return True
    if a == "center" or b == "center":
        return False
    return False


def _any_compatible_side(rects, rect, image_width=None):
    rect_side = side_hint(rect, image_width)
    known_sides = [side_hint(existing, image_width) for existing in rects]
    known_sides = [side for side in known_sides if side != "unknown"]

    if not known_sides or rect_side == "unknown":
        return True

    return any(sides_are_compatible(existing_side, rect_side) for existing_side in known_sides)


def _any_compatible_visual(rects, rect, img=None):
    rect_visual = visual_message_type(img, rect)
    known_visuals = [visual_message_type(img, existing) for existing in rects]
    known_visuals = [visual for visual in known_visuals if visual != "unknown"]

    if not known_visuals or rect_visual == "unknown":
        return True

    return any(existing_visual == rect_visual for existing_visual in known_visuals)

def side_hint(rect, image_width=None):
    if image_width is None or image_width <= 0:
        return "unknown"

    center_x = rect_center_x(rect)
    width = rect_width(rect)
    image_center = image_width / 2

    if width <= image_width * 0.34 and abs(center_x - image_center) <= image_width * 0.14:
        return "center"
    if center_x >= image_width * 0.56:
        return "right"
    if center_x <= image_width * 0.44:
        return "left"
    return "unknown"

def visual_message_type(img, rect):
    if img is None:
        return "unknown"
    return classify_sender_receiver(img, rect)

def _can_join_row(row_boxes, row_bbox, rect, image_width=None, img=None):
    metrics = _pairwise_metrics(row_boxes, rect)
    allowed_y_gap = (
        ROW_VERTICAL_GAP_WITH_X_OVERLAP_PX
        if metrics["max_x_overlap"] > 0
        else ROW_VERTICAL_GAP_PX
    )
    if metrics["min_y_gap"] > allowed_y_gap:
        return False

    row_side = side_hint(row_bbox, image_width)
    rect_side = side_hint(rect, image_width)
    if not sides_are_compatible(row_side, rect_side):
        return False

    row_visual = visual_message_type(img, row_bbox)
    rect_visual = visual_message_type(img, rect)
    if row_visual != "unknown" and rect_visual != "unknown" and row_visual != rect_visual:
        return False

    allowed_x_gap = (
        ROW_HORIZONTAL_GAP_WITH_Y_OVERLAP_PX
        if metrics["has_y_overlap"]
        else ROW_HORIZONTAL_GAP_PX
    )
    if metrics["min_x_gap"] > allowed_x_gap:
        return False

    return True

def group_rows(rects, image_width=None, img=None):
    rects = sorted(rects, key=lambda r: (rect_center_y(r), r[0]))
    rows = []

    for r in rects:
        placed = False
        for row in rows:
            if _can_join_row(row["boxes"], row["bbox"], r, image_width, img):
                row["boxes"].append(r)
                row["bbox"] = union_rect(row["bbox"], r)
                placed = True
                break

        if not placed:
            rows.append({
                "boxes": [r],
                "bbox": r
            })

    return rows

def _can_join_object(obj_rows, obj_bbox, row_bbox, image_width=None, img=None):
    row_rects = [existing_row["bbox"] for existing_row in obj_rows]
    if not _any_compatible_side(row_rects, row_bbox, image_width):
        return False

    if not _any_compatible_visual(row_rects, row_bbox, img):
        return False

    metrics = _pairwise_metrics(row_rects, row_bbox)
    allowed_y_gap = (
        OBJECT_VERTICAL_GAP_WITH_X_OVERLAP_PX
        if metrics["max_x_overlap"] > 0
        else OBJECT_VERTICAL_GAP_PX
    )
    if metrics["min_y_gap"] > allowed_y_gap:
        return False

    overlap = metrics["max_x_overlap"]
    overlap_ratio = overlap / max(1, min(rect_width(obj_bbox), rect_width(row_bbox)))
    if overlap_ratio >= 0.18:
        return True

    if not metrics["has_y_overlap"]:
        return True

    allowed_x_gap = (
        OBJECT_HORIZONTAL_GAP_WITH_Y_OVERLAP_PX
        if metrics["has_y_overlap"]
        else OBJECT_HORIZONTAL_GAP_PX
    )
    return metrics["min_x_gap"] <= allowed_x_gap

def group_objects(rows, image_width=None, img=None):
    rows = sorted(rows, key=lambda r: r["bbox"][1])
    objects = []

    for row in rows:
        placed = False
        for obj in objects:
            if _can_join_object(obj["rows"], obj["bbox"], row["bbox"], image_width, img):
                obj["rows"].append(row)
                obj["bbox"] = union_rect(obj["bbox"], row["bbox"])
                placed = True
                break

        if not placed:
            objects.append({
                "rows": [row],
                "bbox": row["bbox"]
            })

    return objects

def blue_purple_ratio(img, rect):
    x1, y1, x2, y2 = rect
    region = img[y1:y2, x1:x2]

    if region.size == 0:
        return 0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([90, 40, 40]),
        np.array([140, 255, 255])
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([141, 40, 40]),
        np.array([175, 255, 255])
    )

    mask = cv2.bitwise_or(mask1, mask2)

    return np.count_nonzero(mask) / mask.size

def classify_sender_receiver(img, rect):
    blue_ratio = blue_purple_ratio(img, rect)

    if blue_ratio >= 0.14:
        return "sender"

    return "receiver"

def _is_center_aligned_timestamp_text(text, rect, img):
    if not text or img is None:
        return False

    image_width = img.shape[1]
    center_delta = abs(rect_center_x(rect) - (image_width / 2))
    if center_delta > image_width * TIMESTAMP_CENTER_TOLERANCE_RATIO:
        return False

    return parse_timestamp_text(text) is not None


def _is_center_aligned_timestamp_candidate(rect, img, text_th="", text_en=""):
    if img is None:
        return False

    image_width = img.shape[1]
    image_height = img.shape[0]
    center_delta = abs(rect_center_x(rect) - (image_width / 2))
    if center_delta > image_width * 0.14:
        return False

    if rect_width(rect) > image_width * 0.22:
        return False
    if rect_height(rect) > max(58, int(image_height * 0.035)):
        return False

    parsed = parse_timestamp_text(text_th) or parse_timestamp_text(text_en)
    if parsed:
        return True

    compact = f"{text_th} {text_en}".replace(" ", "").strip()
    if compact and len(compact) <= 12 and any(ch.isdigit() for ch in compact):
        return True

    if (
        compact
        and len(compact) <= 14
        and blue_purple_ratio(img, rect) < 0.08
    ):
        return True

    return False

def classify_object_type(img, rect, text_th="", text_en=""):
    if (
        is_timestamp(text_th)
        or is_timestamp(text_en)
        or _is_center_aligned_timestamp_text(text_th, rect, img)
        or _is_center_aligned_timestamp_text(text_en, rect, img)
        or _is_center_aligned_timestamp_candidate(rect, img, text_th, text_en)
    ):
        return "timestamp"
    return classify_sender_receiver(img, rect)