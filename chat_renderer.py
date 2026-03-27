import os

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


_PIL_FONT_CACHE = {}
_UNICODE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\LeelawUI.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

# --------------------------------------------------
# SORT OBJECTS (TOP → BOTTOM)
# --------------------------------------------------

def sort_objects(objects):
    # Use explicit conversation order when present (multi-page combined render).
    # Fall back to bbox y-coordinate for single-page renders.
    if not objects:
        return objects
    if any("order" in o for o in objects):
        # int() avoids string sort bugs ("10" before "2") if order ever stringifies.
        return sorted(objects, key=lambda o: int(o.get("order", 0) or 0))
    return sorted(objects, key=lambda o: o["bbox"][1])


# --------------------------------------------------
# DRAW HELPERS
# --------------------------------------------------

def _contains_non_latin(text):
    return any(ord(c) > 0x024F for c in (text or "") if not c.isspace())


def _load_unicode_font(size):
    if ImageFont is None:
        return None
    size = max(12, int(size))
    if size in _PIL_FONT_CACHE:
        return _PIL_FONT_CACHE[size]
    for fp in _UNICODE_FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                _PIL_FONT_CACHE[size] = ImageFont.truetype(fp, size=size)
                return _PIL_FONT_CACHE[size]
            except Exception:
                continue
    try:
        _PIL_FONT_CACHE[size] = ImageFont.load_default()
        return _PIL_FONT_CACHE[size]
    except Exception:
        return None


def _measure_text(text, font, scale, thickness):
    if _contains_non_latin(text):
        pil_font = _load_unicode_font(22 if scale >= 0.55 else 18)
        if pil_font is not None:
            bbox = pil_font.getbbox(text or " ")
            return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]), pil_font
    (w, h), _ = cv2.getTextSize(text or " ", font, scale, thickness)
    return max(1, w), max(1, h), None


def _draw_text(img, text, x, y_top, font, scale, thickness, color):
    w, h, pil_font = _measure_text(text, font, scale, thickness)
    if pil_font is not None and Image is not None and ImageDraw is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y_top), text, font=pil_font, fill=(int(color[2]), int(color[1]), int(color[0])))
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return h
    cv2.putText(img, text, (x, y_top + h), font, scale, color, thickness, cv2.LINE_AA)
    return h


def wrap_text(text, font, scale, thickness, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word
        w, _h, _pil_font = _measure_text(test, font, scale, thickness)
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


def draw_avatar_image(img, avatar_image, center_x, center_y, size=32):
    if avatar_image is None or getattr(avatar_image, "size", 0) == 0:
        draw_avatar(img, center_x, center_y, (210, 190, 170), (120, 90, 70))
        return

    src_h, src_w = avatar_image.shape[:2]
    square_size = max(src_h, src_w)
    pad_top = (square_size - src_h) // 2
    pad_bottom = square_size - src_h - pad_top
    pad_left = (square_size - src_w) // 2
    pad_right = square_size - src_w - pad_left
    square_avatar = cv2.copyMakeBorder(
        avatar_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )

    avatar = cv2.resize(square_avatar, (size, size), interpolation=cv2.INTER_LINEAR)
    top = max(0, center_y - size // 2)
    left = max(0, center_x - size // 2)
    bottom = min(img.shape[0], top + size)
    right = min(img.shape[1], left + size)

    avatar = avatar[:bottom - top, :right - left]
    if avatar.size == 0:
        draw_avatar(img, center_x, center_y, (210, 190, 170), (120, 90, 70))
        return

    mask = np.zeros((avatar.shape[0], avatar.shape[1]), dtype=np.uint8)
    cv2.circle(
        mask,
        (avatar.shape[1] // 2, avatar.shape[0] // 2),
        min(avatar.shape[0], avatar.shape[1]) // 2,
        255,
        -1
    )

    roi = img[top:bottom, left:right]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(avatar, avatar, mask=mask)
    img[top:bottom, left:right] = cv2.add(bg, fg)


def draw_speaker_title(img, text, width, y):
    if not text:
        return y

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    color = (45, 45, 45)
    text_w, text_h, _ = _measure_text(text, font, scale, thickness)
    x = max(18, (width - text_w) // 2)
    _draw_text(img, text, x, y, font, scale, thickness, color)
    return y + text_h + 12


def draw_chat_header(img, contact_name, width):
    """Draw a Messenger-style header bar with the contact name at the top."""
    header_h = 56
    header_bg  = (75, 90, 200)   # deep blue-indigo (BGR)
    avatar_bg  = (210, 190, 170)
    avatar_acc = (120, 90, 70)

    cv2.rectangle(img, (0, 0), (width, header_h), header_bg, -1)

    # Avatar circle on the left
    av_cx, av_cy = 36, header_h // 2
    draw_avatar(img, av_cx, av_cy, avatar_bg, avatar_acc)

    # Contact name
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    tw, th, _ = _measure_text(contact_name, font, scale, thickness)
    tx = av_cx + 26
    ty = max(6, (header_h - th) // 2)
    _draw_text(img, contact_name, tx, ty, font, scale, thickness, (255, 255, 255))
    return header_h


def draw_name_label(img, name, x, y):
    """Draw a small name label just above a receiver bubble."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1
    color = (90, 90, 90)
    tw, th, _ = _measure_text(name, font, scale, thickness)
    _draw_text(img, name, x, y, font, scale, thickness, color)
    return y + th + 4


def draw_timestamp(img, text, width, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.47
    thickness = 1
    color = (120, 120, 120)

    text_w, text_h, _ = _measure_text(text, font, scale, thickness)
    x = (width - text_w) // 2
    _draw_text(img, text, x, y, font, scale, thickness, color)
    return y + text_h + 18


def draw_bubble(img, text, x, y, max_width, align="left", bubble_color=(200, 200, 200), profile_image=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    padding_x = 16
    padding_y = 12
    line_gap = 6
    radius = 18
    avatar_size = 34
    avatar_space = 0

    lines = wrap_text(text, font, scale, thickness, max_width - padding_x * 2)
    line_sizes = [_measure_text(line, font, scale, thickness) for line in lines]
    line_widths = [s[0] for s in line_sizes]
    line_heights = [s[1] for s in line_sizes]
    box_w = max(line_widths) + padding_x * 2
    text_h_total = sum(line_heights) + max(0, len(lines) - 1) * line_gap
    box_h = text_h_total + padding_y * 2

    if align == "left":
        avatar_size = min(52, max(40, box_h - 10))
        avatar_space = avatar_size + 12

    if align == "right":
        x = img.shape[1] - box_w - 18
        text_color = (255, 255, 255)
    else:
        x = x + avatar_space
        text_color = (25, 25, 25)

    draw_rounded_rect(img, x, y, x + box_w, y + box_h, bubble_color, radius)

    text_cursor_y = y + padding_y
    for i, line in enumerate(lines):
        lh = _draw_text(
            img,
            line,
            x + padding_x,
            text_cursor_y,
            font,
            scale,
            thickness,
            text_color,
        )
        text_cursor_y += lh + line_gap

    if align == "left":
        avatar_x = x - 8 - (avatar_size // 2)
        avatar_y = y + box_h - (avatar_size // 2) - 2
        draw_avatar_image(img, profile_image, avatar_x, avatar_y, size=avatar_size)

    return y + box_h + 10


def estimate_canvas_height(objects):
    base_height = 70
    for obj in objects:
        text_en = (obj.get("text_en") or "").strip()
        text_th = (obj.get("text_th") or "").strip()
        text = text_en or text_th
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

def render_chat(objects, width=600, speaker_text="", profile_image=None,
                contact_name=""):
    objects = sort_objects(objects)

    # Reserve space for Messenger-style header if we have a contact name
    header_h = 56 if contact_name else 0
    speaker_title_height = 26 if speaker_text else 0
    top_gap = 12 if speaker_text else 0
    canvas_height = (
        estimate_canvas_height(objects)
        + header_h
        + speaker_title_height
        + top_gap
        + 60  # extra padding for name labels above receiver bubbles
    )

    canvas = np.ones((canvas_height, width, 3), dtype=np.uint8) * 248

    # Draw the header bar (returns header height so we start content below it)
    y = 0
    if contact_name:
        y = draw_chat_header(canvas, contact_name, width)
        y += 12  # breathing room below header

    if speaker_text:
        y = draw_speaker_title(canvas, speaker_text, width, y)

    rendered_count = 0
    prev_type = None  # track previous bubble type to show name label only on first receiver run

    for obj in objects:
        text_en = (obj.get("text_en") or "").strip()
        text_th = (obj.get("text_th") or "").strip()
        obj_type = obj.get("type", "receiver")

        if obj_type in {"status_bar", "bottom_artifact", "bottom_bar", "keyboard"}:
            continue

        # Use English translation; if missing but Thai exists, show placeholder
        if text_en:
            text = text_en
        elif text_th:
            text = "[untranslated]"
        else:
            continue

        rendered_count += 1

        if obj_type == "timestamp":
            y = draw_timestamp(canvas, text, width, y)
            prev_type = "timestamp"

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
            prev_type = "sender"

        else:  # receiver
            # Show name label above the first bubble in each consecutive receiver run
            if contact_name and prev_type != "receiver":
                avatar_space = 52 + 12  # matches draw_bubble left indent
                y = draw_name_label(canvas, contact_name, 20 + avatar_space, y)

            y = draw_bubble(
                canvas,
                text,
                x=20,
                y=y,
                max_width=int(width * 0.62),
                align="left",
                bubble_color=(235, 235, 235),
                profile_image=profile_image,
            )
            prev_type = "receiver"

    if rendered_count == 0:
        cv2.putText(
            canvas,
            "No translated text available",
            (20, header_h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2
        )
        y = header_h + 80

    final_height = min(max(y + 20, 120), canvas.shape[0])
    return canvas[:final_height, :]