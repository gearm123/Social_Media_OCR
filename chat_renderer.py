import math
import os
import unicodedata

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

# Segoe UI (Windows) / fallbacks — Messenger-style call cards (vector text, clearer than cv2.putText).
_CALL_CARD_FONT_REGULAR = [
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_CALL_CARD_FONT_BOLD = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\segoeuisemibold.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_CALL_CARD_FONT_CACHE: dict = {}

# Messenger-style call disks (PNG includes circular background).
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_CALL_ICON_MISSED_PNG = os.path.join(_ASSETS_DIR, "call_icon_missed.png")
_CALL_ICON_INCOMING_PNG = os.path.join(_ASSETS_DIR, "call_icon_incoming.png")
_CALL_ICON_RGBA_CACHE: dict = {}


def _prepare_call_icon_rgba(path: str, diameter: int):
    """Load and resize PNG to ``diameter``×``diameter`` BGRA; alpha × circular mask for crisp disks."""
    d = max(1, int(diameter))
    key = (path, d)
    if key in _CALL_ICON_RGBA_CACHE:
        return _CALL_ICON_RGBA_CACHE[key]
    if not path or not os.path.isfile(path):
        _CALL_ICON_RGBA_CACHE[key] = None
        return None
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        _CALL_ICON_RGBA_CACHE[key] = None
        return None
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        raw[:, :, 3] = 255
    elif raw.shape[2] == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        raw[:, :, 3] = 255
    elif raw.shape[2] != 4:
        _CALL_ICON_RGBA_CACHE[key] = None
        return None
    scaled = cv2.resize(raw, (d, d), interpolation=cv2.INTER_AREA)
    h, w = scaled.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    rad = min(w, h) / 2.0 - 0.5
    inside = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= rad ** 2
    a = scaled[:, :, 3].astype(np.float32) * inside.astype(np.float32)
    scaled[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    _CALL_ICON_RGBA_CACHE[key] = scaled
    return scaled


def _composite_rgba_on_bgr(img, center_x, center_y, rgba):
    """Alpha-blend BGRA patch centered at (center_x, center_y) onto BGR image."""
    h, w = rgba.shape[:2]
    left = int(round(center_x - w / 2.0))
    top = int(round(center_y - h / 2.0))
    H, W = img.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(W, left + w), min(H, top + h)
    if x2 <= x1 or y2 <= y1:
        return
    ix1, iy1 = x1 - left, y1 - top
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)
    roi = img[y1:y2, x1:x2]
    patch = rgba[iy1:iy2, ix1:ix2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    fg = patch[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    roi[:] = np.clip(blended, 0, 255).astype(np.uint8)


_CHAT_THEMES = {
    "messenger_light": {
        "canvas_bg": (255, 255, 255),
        "header_bg": (255, 255, 255),
        "header_divider": (233, 236, 239),
        "header_title": (36, 36, 36),
        "header_subtitle": (116, 116, 120),
        "header_icon": (250, 48, 168),  # purple-blue accent in BGR family
        "receiver_bubble": (240, 242, 245),
        "receiver_text": (28, 30, 33),
        "sender_text": (255, 255, 255),
        "sender_gradient_left": (247, 57, 173),
        "sender_gradient_right": (255, 112, 34),
        "timestamp_text": (108, 112, 118),
        "avatar_fill": (224, 230, 238),
        "avatar_accent": (102, 118, 145),
        "presence_active": (58, 192, 98),
        "presence_ring": (255, 255, 255),
        # Messenger call / system cards (hex → BGR) — match facebook.com/web Messenger
        "call_card_bg": (245, 242, 240),  # #F0F2F5
        "call_button_bg": (235, 230, 228),  # #E4E6EB
        "call_title": (5, 5, 5),  # #050505 (near-black; reads sharper than #1C1E21)
        "call_subtitle": (107, 103, 101),  # #65676B
        "call_missed_icon_fill": (62, 62, 250),  # #FA3E3E
        "call_active_icon_fill": (255, 102, 8),  # legacy (header); call cards use incoming gray
        "call_incoming_icon_fill": (235, 230, 228),  # #E4E6EB — Facebook disk (not Messenger blue)
        "call_missed_icon": (62, 62, 250),
        "call_audio_icon": (255, 102, 8),
    },
}


def _get_chat_theme(theme_name="messenger_light"):
    return dict(_CHAT_THEMES.get(theme_name, _CHAT_THEMES["messenger_light"]))

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


def _rounded_rect_mask(height, width, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    draw_rounded_rect(mask, 0, 0, width, height, 255, radius)
    return mask


def draw_gradient_rounded_rect(img, x1, y1, x2, y2, color_left, color_right, radius):
    img_h, img_w = img.shape[:2]
    x1 = max(0, min(int(x1), img_w))
    x2 = max(0, min(int(x2), img_w))
    y1 = max(0, min(int(y1), img_h))
    y2 = max(0, min(int(y2), img_h))
    if x2 <= x1 or y2 <= y1:
        return

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    grad = np.zeros((height, width, 3), dtype=np.uint8)
    left = np.array(color_left, dtype=np.float32)
    right = np.array(color_right, dtype=np.float32)
    if width == 1:
        grad[:, 0] = left.astype(np.uint8)
    else:
        for xi in range(width):
            alpha = xi / float(width - 1)
            grad[:, xi] = (left * (1.0 - alpha) + right * alpha).astype(np.uint8)
    mask = _rounded_rect_mask(height, width, radius)
    roi = img[y1:y2, x1:x2]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(grad, grad, mask=mask)
    img[y1:y2, x1:x2] = cv2.add(bg, fg)


def draw_avatar(img, center_x, center_y, fill_color, accent_color, presence_color=None):
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
    if presence_color is not None:
        cv2.circle(img, (center_x + 11, center_y + 11), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (center_x + 11, center_y + 11), 4, presence_color, -1, cv2.LINE_AA)


def _draw_header_icon(img, center_x, center_y, kind="phone", color=(10, 124, 255)):
    if kind == "phone":
        cv2.circle(img, (center_x, center_y), 14, color, 2, cv2.LINE_AA)
        cv2.ellipse(img, (center_x, center_y), (6, 8), 35, 210, 330, color, 2, cv2.LINE_AA)
    elif kind == "video":
        cv2.circle(img, (center_x, center_y), 14, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (center_x - 6, center_y - 4), (center_x + 3, center_y + 4), color, 2)
        pts = np.array([[center_x + 4, center_y - 4], [center_x + 10, center_y - 7], [center_x + 10, center_y + 7], [center_x + 4, center_y + 4]], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (center_x, center_y), 14, color, 2, cv2.LINE_AA)
        for dy in (-5, 0, 5):
            cv2.circle(img, (center_x, center_y + dy), 1, color, -1, cv2.LINE_AA)


def draw_avatar_image(img, avatar_image, center_x, center_y, size=32):
    if avatar_image is None or getattr(avatar_image, "size", 0) == 0:
        draw_avatar(img, center_x, center_y, (224, 230, 238), (102, 118, 145))
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


def draw_speaker_title(img, text, width, y, theme):
    if not text:
        return y

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    color = theme["header_title"]
    text_w, text_h, _ = _measure_text(text, font, scale, thickness)
    x = max(18, (width - text_w) // 2)
    _draw_text(img, text, x, y, font, scale, thickness, color)
    return y + text_h + 12


def draw_chat_header(img, contact_name, width, status_text="", avatar_image=None, theme=None):
    """Draw a chat header with theme styling."""
    theme = theme or _get_chat_theme()
    header_h = 78 if status_text else 64
    header_bg = theme["header_bg"]
    avatar_bg = theme["avatar_fill"]
    avatar_acc = theme["avatar_accent"]
    accent = theme["header_icon"]
    divider = theme["header_divider"]

    cv2.rectangle(img, (0, 0), (width, header_h), header_bg, -1)
    cv2.line(img, (0, header_h - 1), (width, header_h - 1), divider, 1, cv2.LINE_AA)

    # Avatar circle on the left
    av_cx, av_cy = 40, header_h // 2
    if avatar_image is not None and getattr(avatar_image, "size", 0) != 0:
        draw_avatar_image(img, avatar_image, av_cx, av_cy, size=38)
    else:
        presence_color = theme["presence_active"] if "active" in (status_text or "").lower() else None
        draw_avatar(img, av_cx, av_cy, avatar_bg, avatar_acc, presence_color=presence_color)

    # Contact name
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    tw, th, _ = _measure_text(contact_name, font, scale, thickness)
    tx = av_cx + 30
    ty = 10 if status_text else max(10, (header_h - th) // 2)
    _draw_text(img, contact_name, tx, ty, font, scale, thickness, theme["header_title"])
    if status_text:
        _draw_text(img, status_text, tx, ty + th + 5, font, 0.42, 1, theme["header_subtitle"])

    icon_y = header_h // 2
    _draw_header_icon(img, width - 116, icon_y, "phone", accent)
    _draw_header_icon(img, width - 74, icon_y, "video", accent)
    _draw_header_icon(img, width - 32, icon_y, "info", accent)
    return header_h


def draw_name_label(img, name, x, y, theme):
    """Draw a small name label just above a receiver bubble."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1
    color = theme["header_subtitle"]
    tw, th, _ = _measure_text(name, font, scale, thickness)
    _draw_text(img, name, x, y, font, scale, thickness, color)
    return y + th + 4


def draw_timestamp(img, text, width, y, theme):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    color = theme["timestamp_text"]

    text_w, text_h, _ = _measure_text(text, font, scale, thickness)
    x = (width - text_w) // 2
    _draw_text(img, text, x, y, font, scale, thickness, color)
    return y + text_h + 20


def _bgr_to_rgb_for_pil(bgr):
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _call_card_pil_font(size_px: int, bold: bool):
    if ImageFont is None:
        return None
    key = ("call_ui", size_px, bold)
    if key in _CALL_CARD_FONT_CACHE:
        return _CALL_CARD_FONT_CACHE[key]
    paths = _CALL_CARD_FONT_BOLD if bold else _CALL_CARD_FONT_REGULAR
    for fp in paths:
        if os.path.isfile(fp):
            try:
                f = ImageFont.truetype(fp, size=size_px)
                _CALL_CARD_FONT_CACHE[key] = f
                return f
            except OSError:
                continue
    f = ImageFont.load_default()
    _CALL_CARD_FONT_CACHE[key] = f
    return f


def _draw_call_notice_phone_glyph(img, cx, cy, color, incoming_arrow=False, scale=1):
    """Filled classic handset (two rounded lobes + bridge), like Messenger web — not a hollow outline."""
    s = float(scale)
    tilt = 46.0
    ang = math.radians(tilt)

    def pr(dx, dy):
        x = dx * math.cos(ang) - dy * math.sin(ang)
        y = dx * math.sin(ang) + dy * math.cos(ang)
        return (int(round(cx + x * s)), int(round(cy + y * s)))

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Earpiece (top-right when drawn), mouthpiece (bottom-left) — filled ellipses + thick joining stroke.
    cv2.ellipse(
        mask,
        pr(5.0, -6.0),
        (max(1, int(round(4.6 * s))), max(1, int(round(5.5 * s)))),
        tilt,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        mask,
        pr(-5.2, 6.2),
        (max(1, int(round(5.0 * s))), max(1, int(round(6.0 * s)))),
        tilt,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    br = max(1, int(round(5.5 * s)))
    cv2.line(mask, pr(-0.5, 0.8), pr(2.8, -2.2), 255, br, cv2.LINE_AA)
    ksz = max(3, int(round(2.5 * s)))
    if ksz % 2 == 0:
        ksz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    sel = mask > 0
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    img[:, :, 0][sel] = b
    img[:, :, 1][sel] = g
    img[:, :, 2][sel] = r

    if incoming_arrow:
        # Thin separate stroke + wedge (Messenger: arrow approaches earpiece from upper-right).
        t = max(1, int(round(1.25 * s)))
        tip = pr(6.5, -6.8)
        tail = pr(12.8, -13.2)
        cv2.line(img, tail, tip, color, t, cv2.LINE_AA)
        cv2.line(img, tip, pr(5.2, -6.0), color, t, cv2.LINE_AA)
        cv2.line(img, tip, pr(6.8, -8.2), color, t, cv2.LINE_AA)


def _draw_call_notice_video_glyph(img, cx, cy, color, scale=1):
    """Simple video camera on gray disk (non-missed video calls)."""
    s = float(scale)
    t = max(1, int(round(2 * s)))
    x1, y1 = cx - int(round(9 * s)), cy - int(round(6 * s))
    x2, y2 = cx + int(round(7 * s)), cy + int(round(7 * s))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
    cv2.circle(img, (cx - int(round(1 * s)), cy + int(round(1 * s))), int(round(3 * s)), color, t, cv2.LINE_AA)
    pts = np.array(
        [
            [x2, y1 + int(round(2 * s))],
            [x2 + int(round(5 * s)), cy],
            [x2, y2 - int(round(2 * s))],
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], False, color, t, cv2.LINE_AA)


def _draw_call_notice_messenger_icon_on_patch(patch, cx, cy, missed, theme, video=False, scale=1):
    """Draw call icon centered at (cx, cy) on patch; ``scale`` upsamples strokes for INTER_AREA downscale."""
    r = int(round(22 * scale))
    if missed:
        fill = theme.get("call_missed_icon_fill", theme.get("call_missed_icon", (62, 62, 250)))
        cv2.circle(patch, (cx, cy), r, fill, -1, cv2.LINE_AA)
        white = (255, 255, 255)
        _draw_call_notice_phone_glyph(patch, cx, cy, white, incoming_arrow=False, scale=scale)
        # Small × at upper-right of handset (Messenger missed-call asset).
        t = max(2, int(round(2 * scale)))
        ang = math.radians(46)

        def wp(dx, dy):
            x = dx * math.cos(ang) - dy * math.sin(ang)
            y = dx * math.sin(ang) + dy * math.cos(ang)
            return (int(round(cx + x * scale)), int(round(cy + y * scale)))

        ax, ay = wp(9, -10)
        br = int(round(3.0 * scale))
        cv2.line(patch, (ax - br, ay - br), (ax + br, ay + br), white, t, cv2.LINE_AA)
        cv2.line(patch, (ax + br, ay - br), (ax - br, ay + br), white, t, cv2.LINE_AA)
        return

    fill = theme.get(
        "call_incoming_icon_fill",
        theme.get("call_button_bg", (235, 230, 228)),
    )
    glyph = theme.get("call_incoming_icon_glyph", theme["call_title"])
    cv2.circle(patch, (cx, cy), r, fill, -1, cv2.LINE_AA)
    if video:
        _draw_call_notice_video_glyph(patch, cx, cy, glyph, scale=scale)
    else:
        _draw_call_notice_phone_glyph(patch, cx, cy, glyph, incoming_arrow=True, scale=scale)


def _draw_call_notice_messenger_icon(img, cx, cy, missed, theme, video=False):
    """Circular Messenger call disk: PNG assets when present; else procedural (e.g. video)."""
    r = 22
    d = 2 * r
    if not video:
        path = _CALL_ICON_MISSED_PNG if missed else _CALL_ICON_INCOMING_PNG
        icon_rgba = _prepare_call_icon_rgba(path, d)
        if icon_rgba is not None:
            _composite_rgba_on_bgr(img, cx, cy, icon_rgba)
            return
    _draw_call_notice_messenger_icon_vector(img, cx, cy, missed, theme, video=video)


def _draw_call_notice_messenger_icon_vector(img, cx, cy, missed, theme, video=False):
    """Procedural icon when PNGs are missing or for video-call rows."""
    r = 22
    margin = 6
    sup = 2
    patch_side = 2 * (r + margin) * sup
    patch = np.full((patch_side, patch_side, 3), theme["call_card_bg"], dtype=np.uint8)
    pcx = patch_side // 2
    pcy = patch_side // 2
    _draw_call_notice_messenger_icon_on_patch(patch, pcx, pcy, missed, theme, video=video, scale=float(sup))
    small_side = patch_side // sup
    small = cv2.resize(patch, (small_side, small_side), interpolation=cv2.INTER_AREA)
    left = int(cx) - small_side // 2
    top = int(cy) - small_side // 2
    h, w = small.shape[:2]
    H, W = img.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(W, left + w), min(H, top + h)
    sx1, sy1 = x1 - left, y1 - top
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = small[sy1:sy2, sx1:sx2]


def _pil_text_width(draw, text, font):
    if not text:
        return 0
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    w, _h = draw.textsize(text, font=font)
    return w


def _wrap_text_pil(draw, text, font, max_w):
    """Simple word-wrap using PIL font metrics."""
    words = (text or "").split()
    if not words:
        return []
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip() if cur else word
        tw = _pil_text_width(draw, trial, font)
        if tw <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text or ""]


def _call_notice_card_width(canvas_w: int, card_x: int) -> int:
    """Messenger web: wide column card, slightly under full thread width (between old narrow and ~52%)."""
    margin_right = 14
    inner_max = max(100, canvas_w - card_x - margin_right)
    by_canvas = int(round(canvas_w * 0.46))
    by_inner = int(round(inner_max * 0.80))
    w = min(by_canvas, by_inner, inner_max)
    w = max(220, w)
    return w


def _clean_call_notice_subtitle(text):
    """Normalize and drop characters that often render as tofu with UI fonts (emoji, replacement, controls)."""
    s = unicodedata.normalize("NFC", (text or "").strip())
    out = []
    for ch in s:
        o = ord(ch)
        cat = unicodedata.category(ch)
        if ch == "\ufffd" or cat in ("Cc", "Cf", "Cs"):
            continue
        # Emoji / misc symbols that Segoe UI often maps to empty boxes next to Latin digits.
        if 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0xFE00 <= o <= 0xFE0F:
            continue
        out.append(ch)
    return "".join(out).strip()


def draw_call_notice_card(
    img, title, subtitle, button_text, y, theme, missed=False, video=False, profile_image=None
):
    """Messenger-style call / missed-call card (#F0F2F5, vector text via PIL when available)."""
    subtitle = _clean_call_notice_subtitle(subtitle)
    # Same horizontal rhythm as left-aligned bubbles: avatar column then card.
    avatar_size = 36
    avatar_space = avatar_size + 10
    card_x = 20 + avatar_space
    card_w = _call_notice_card_width(img.shape[1], card_x)
    pad = 14 if card_w < 198 else 16
    card_radius = 19
    button_radius = 9
    icon_cx = card_x + pad + 22
    icon_cy = y + pad + 22
    text_left = card_x + pad + 50
    text_max_w = card_w - pad * 2 - 54

    # Slightly larger than before — closer to Messenger web and reads cleaner when hinted.
    title_px = 17 if card_w < 198 else 18
    sub_px = 14 if card_w < 198 else 15
    btn_px = 15 if card_w < 198 else 16
    title_f = _call_card_pil_font(title_px, bold=True)
    # Subtitle: Tahoma/Leelawadi etc. first — better Thai/Latin time strings than Segoe-only for mixed OCR.
    sub_f = _load_unicode_font(sub_px) or _call_card_pil_font(sub_px, bold=False)
    btn_f = _call_card_pil_font(btn_px, bold=True)
    use_pil = Image is not None and ImageDraw is not None and title_f is not None

    if use_pil:
        _measure_img = Image.new("RGB", (max(1, text_max_w + 2), 400), (255, 255, 255))
        _md = ImageDraw.Draw(_measure_img)
        title_lines = _wrap_text_pil(_md, title or "", title_f, text_max_w)
        subtitle_lines = (
            _wrap_text_pil(_md, subtitle or "", sub_f, text_max_w)
            if (subtitle or "").strip()
            else []
        )

        def _pil_line_height(draw, fnt, line):
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), line, font=fnt)
                return bbox[3] - bbox[1]
            _w, h = draw.textsize(line, font=fnt)
            return h

        title_h = sum(_pil_line_height(_md, title_f, ln) for ln in title_lines) + max(
            0, len(title_lines) - 1
        ) * 3
        subtitle_h = sum(_pil_line_height(_md, sub_f, ln) for ln in subtitle_lines) + max(
            0, len(subtitle_lines) - 1
        ) * 2
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale, sub_scale = 0.58, 0.42
        title_thickness, sub_thickness = 2, 1
        title_lines = wrap_text(title or "", font, title_scale, title_thickness, text_max_w)
        subtitle_lines = (
            wrap_text(subtitle or "", font, sub_scale, sub_thickness, text_max_w)
            if (subtitle or "").strip()
            else []
        )
        title_h = sum(
            _measure_text(line, font, title_scale, title_thickness)[1] for line in title_lines
        ) + max(0, len(title_lines) - 1) * 4
        subtitle_h = sum(
            _measure_text(line, font, sub_scale, sub_thickness)[1] for line in subtitle_lines
        ) + max(0, len(subtitle_lines) - 1) * 4

    button_h = 42 if button_text else 0
    top_block_h = pad + max(48, title_h + (6 if subtitle_h else 0) + subtitle_h) + pad // 2
    card_h = max(96, top_block_h + (12 if button_h else 0) + button_h + pad)

    draw_rounded_rect(img, card_x, y, card_x + card_w, y + card_h, theme["call_card_bg"], card_radius)
    _draw_call_notice_messenger_icon(img, icon_cx, icon_cy, missed, theme, video=video)

    col_title = _bgr_to_rgb_for_pil(theme["call_title"])
    col_sub = _bgr_to_rgb_for_pil(theme["call_subtitle"])

    if use_pil:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        ty = float(y + pad + 2)
        tx = float(text_left)
        for line in title_lines:
            draw.text((int(round(tx)), int(round(ty))), line, font=title_f, fill=col_title)
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((int(round(tx)), int(round(ty))), line, font=title_f)
                ty = float(bbox[3] + 4)
            else:
                _w, h = draw.textsize(line, font=title_f)
                ty += float(h + 4)
        if subtitle_lines:
            ty += 2.0
            for line in subtitle_lines:
                draw.text((int(round(tx)), int(round(ty))), line, font=sub_f, fill=col_sub)
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((int(round(tx)), int(round(ty))), line, font=sub_f)
                    ty = float(bbox[3] + 2)
                else:
                    _w, h = draw.textsize(line, font=sub_f)
                    ty += float(h + 2)
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if button_text:
            btn_x1 = card_x + pad
            btn_x2 = card_x + card_w - pad
            btn_y1 = y + card_h - pad - button_h
            btn_y2 = y + card_h - pad
            draw_rounded_rect(img, btn_x1, btn_y1, btn_x2, btn_y2, theme["call_button_bg"], button_radius)
            if btn_f is not None:
                rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil2 = Image.fromarray(rgb2)
                dr2 = ImageDraw.Draw(pil2)
                if hasattr(dr2, "textbbox"):
                    bb = dr2.textbbox((0, 0), button_text, font=btn_f)
                    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                else:
                    bw, bh = dr2.textsize(button_text, font=btn_f)
                btx = int(btn_x1 + max(0, ((btn_x2 - btn_x1) - bw) // 2))
                bty = int(btn_y1 + max(0, ((button_h - bh) // 2) - 1))
                dr2.text((btx, bty), button_text, font=btn_f, fill=col_title)
                img[:] = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)
            else:
                bw, bh, _ = _measure_text(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                _draw_text(
                    img,
                    button_text,
                    btn_x1 + max(8, ((btn_x2 - btn_x1) - bw) // 2),
                    btn_y1 + max(6, ((button_h - bh) // 2) - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    1,
                    theme["call_title"],
                )
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        ty = y + pad
        for line in title_lines:
            lh = _draw_text(img, line, text_left, ty, font, 0.58, 2, theme["call_title"])
            ty += lh + 4
        if subtitle_lines:
            ty += 2
            for line in subtitle_lines:
                lh = _draw_text(
                    img, line, text_left, ty, font, 0.42, 1, theme["call_subtitle"]
                )
                ty += lh + 4
        if button_text:
            btn_x1 = card_x + pad
            btn_x2 = card_x + card_w - pad
            btn_y1 = y + card_h - pad - button_h
            btn_y2 = y + card_h - pad
            draw_rounded_rect(img, btn_x1, btn_y1, btn_x2, btn_y2, theme["call_button_bg"], button_radius)
            bw, bh, _ = _measure_text(button_text, font, 0.52, 1)
            _draw_text(
                img,
                button_text,
                btn_x1 + max(8, ((btn_x2 - btn_x1) - bw) // 2),
                btn_y1 + max(6, ((button_h - bh) // 2) - 1),
                font,
                0.52,
                1,
                theme["call_title"],
            )

    av_cx = card_x - 6 - (avatar_size // 2)
    av_cy = y + card_h - (avatar_size // 2) - 2
    draw_avatar_image(img, profile_image, av_cx, av_cy, size=avatar_size)

    return y + card_h + 12


def draw_bubble(img, text, x, y, max_width, align="left", bubble_color=(200, 200, 200), profile_image=None, theme=None):
    theme = theme or _get_chat_theme()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.57
    thickness = 1
    padding_x = 15
    padding_y = 11
    line_gap = 5
    radius = 20
    avatar_size = 32
    avatar_space = 0

    lines = wrap_text(text, font, scale, thickness, max_width - padding_x * 2)
    line_sizes = [_measure_text(line, font, scale, thickness) for line in lines]
    line_widths = [s[0] for s in line_sizes]
    line_heights = [s[1] for s in line_sizes]
    box_w = max(line_widths) + padding_x * 2
    text_h_total = sum(line_heights) + max(0, len(lines) - 1) * line_gap
    box_h = text_h_total + padding_y * 2

    if align == "left":
        avatar_size = min(44, max(34, box_h - 12))
        avatar_space = avatar_size + 10

    if align == "right":
        x = img.shape[1] - box_w - 16
        text_color = theme["sender_text"]
    else:
        x = x + avatar_space
        text_color = theme["receiver_text"]

    if align == "right":
        draw_gradient_rounded_rect(
            img,
            x,
            y,
            x + box_w,
            y + box_h,
            theme["sender_gradient_left"],
            theme["sender_gradient_right"],
            radius,
        )
    else:
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
        avatar_x = x - 6 - (avatar_size // 2)
        avatar_y = y + box_h - (avatar_size // 2) - 2
        draw_avatar_image(img, profile_image, avatar_x, avatar_y, size=avatar_size)

    return y + box_h + 12


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
        elif obj.get("type") == "call_notice":
            base_height += 140
        else:
            lines = max(1, len(text) // 28 + 1)
            base_height += 20 + lines * 22

    return max(320, base_height + 30)


# --------------------------------------------------
# RENDER CHAT
# --------------------------------------------------

def render_chat(objects, width=600, speaker_text="", profile_image=None,
                contact_name="", header_status="", theme_name="messenger_light"):
    objects = sort_objects(objects)
    theme = _get_chat_theme(theme_name)

    # Reserve space for Messenger-style header if we have a contact name
    header_h = (78 if header_status else 64) if contact_name else 0
    speaker_title_height = 26 if speaker_text else 0
    top_gap = 12 if speaker_text else 0
    canvas_height = (
        estimate_canvas_height(objects)
        + header_h
        + speaker_title_height
        + top_gap
        + 60  # extra padding for name labels above receiver bubbles
    )

    canvas = np.full((canvas_height, width, 3), theme["canvas_bg"], dtype=np.uint8)

    # Draw the header bar (returns header height so we start content below it)
    y = 0
    if contact_name:
        y = draw_chat_header(canvas, contact_name, width, status_text=header_status, avatar_image=profile_image, theme=theme)
        y += 12

    if speaker_text:
        y = draw_speaker_title(canvas, speaker_text, width, y, theme)

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
            y = draw_timestamp(canvas, text, width, y, theme)
            prev_type = "timestamp"
        elif obj_type == "call_notice":
            y = draw_call_notice_card(
                canvas,
                text,
                (obj.get("subtitle") or "").strip(),
                (obj.get("button_text") or "").strip(),
                y,
                theme,
                missed=bool(obj.get("missed")),
                video=bool(obj.get("video")),
                profile_image=profile_image,
            )
            prev_type = "timestamp"

        elif obj_type == "sender":
            y = draw_bubble(
                canvas,
                text,
                x=20,
                y=y,
                max_width=int(width * 0.64),
                align="right",
                bubble_color=theme["sender_gradient_right"],
                theme=theme,
            )
            prev_type = "sender"

        else:  # receiver
            y = draw_bubble(
                canvas,
                text,
                x=20,
                y=y,
                max_width=int(width * 0.62),
                align="left",
                bubble_color=theme["receiver_bubble"],
                profile_image=profile_image,
                theme=theme,
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