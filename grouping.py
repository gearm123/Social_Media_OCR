import cv2
import numpy as np

def box_to_rect(box):
    xs = box[:, 0]
    ys = box[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def rect_center_y(r):
    return (r[1] + r[3]) / 2

def union_rect(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    )

def group_rows(rects):
    rects = sorted(rects, key=lambda r: (rect_center_y(r), r[0]))
    rows = []

    for r in rects:
        placed = False

        for row in rows:
            if abs(rect_center_y(r) - rect_center_y(row["bbox"])) < 20:
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

def group_objects(rows):
    rows = sorted(rows, key=lambda r: r["bbox"][1])
    objects = []

    for row in rows:
        placed = False

        for obj in objects:
            if abs(row["bbox"][1] - obj["bbox"][3]) < 40:
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