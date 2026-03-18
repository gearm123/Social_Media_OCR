import cv2
import numpy as np

# --------------------------------------------------
# SORT OBJECTS (TOP → BOTTOM)
# --------------------------------------------------

def sort_objects(objects):
    return sorted(objects, key=lambda o: o["bbox"][1])


# --------------------------------------------------
# DRAW HELPERS
# --------------------------------------------------

def wrap_text(text, font, scale, thickness, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word
        (w, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [text]


def draw_rounded_rect(img, x1, y1, x2, y2, color, radius):
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def draw_avatar(img, center_x, center_y, fill_color, accent_color):
    cv2.circle(img, (center_x, center_y), 16, fill_color, -1)
    cv2.circle(img, (center_x, center_y - 4), 6, accent_color, -1)
    cv2.ellipse(
        img,
        (center_x, center_y + 8),
        (9, 6),
        0,
        0,
        180,
        accent_color,
        -1
    )


def draw_timestamp(img, text, width, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.47
    thickness = 1
    color = (120, 120, 120)

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - text_w) // 2
    cv2.putText(img, text, (x, y + text_h), font, scale, color, thickness, cv2.LINE_AA)
    return y + text_h + 18


def draw_bubble(img, text, x, y, max_width, align="left", bubble_color=(200, 200, 200)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    padding_x = 16
    padding_y = 12
    line_height = 22
    radius = 18
    avatar_space = 34 if align == "left" else 0

    lines = wrap_text(text, font, scale, thickness, max_width - padding_x * 2)
    line_widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(line_widths) + padding_x * 2
    box_h = len(lines) * line_height + padding_y * 2

    if align == "right":
        x = img.shape[1] - box_w - 18
        text_color = (255, 255, 255)
    else:
        x = x + avatar_space
        text_color = (25, 25, 25)

    draw_rounded_rect(img, x, y, x + box_w, y + box_h, bubble_color, radius)

    for i, line in enumerate(lines):
        text_y = y + padding_y + (i + 1) * line_height - 6
        cv2.putText(
            img,
            line,
            (x + padding_x, text_y),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    if align == "left":
        avatar_y = y + box_h - 10
        draw_avatar(
            img,
            x - 18,
            avatar_y,
            (210, 190, 170),
            (120, 90, 70)
        )

    return y + box_h + 10


def estimate_canvas_height(objects):
    base_height = 70
    for obj in objects:
        text = obj.get("text_en", "")
        if not text:
            continue

        if obj.get("type") == "timestamp":
            base_height += 38
        else:
            lines = max(1, len(text) // 28 + 1)
            base_height += 20 + lines * 22

    return max(320, base_height + 30)


# --------------------------------------------------
# RENDER CHAT
# --------------------------------------------------

def render_chat(objects, width=600):
    objects = sort_objects(objects)
    canvas_height = estimate_canvas_height(objects)

    canvas = np.ones((canvas_height, width, 3), dtype=np.uint8) * 248

    y = 18
    rendered_count = 0

    for obj in objects:
        text = obj.get("text_en", "")
        obj_type = obj.get("type", "receiver")

        if obj_type == "status_bar":
            continue

        if not text:
            continue

        rendered_count += 1

        if obj_type == "timestamp":
            y = draw_timestamp(canvas, text, width, y)

        elif obj_type == "sender":
            y = draw_bubble(
                canvas,
                text,
                x=20,
                y=y,
                max_width=int(width * 0.64),
                align="right",
                bubble_color=(240, 73, 255)
            )

        else:  # receiver
            y = draw_bubble(
                canvas,
                text,
                x=20,
                y=y,
                max_width=int(width * 0.62),
                align="left",
                bubble_color=(235, 235, 235)
            )

    if rendered_count == 0:
        cv2.putText(
            canvas,
            "No translated text available",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2
        )
        y = 80

    final_height = min(max(y + 20, 120), canvas.shape[0])
    return canvas[:final_height, :]