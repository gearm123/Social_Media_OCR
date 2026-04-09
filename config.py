import os
import sys
import base64
import requests as _requests
from pathlib import Path
import torch
from deep_translator import GoogleTranslator

from output_languages import google_translate_target_code

# --------------------------------------------------
# BASE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def _resolve_existing_path(candidates):
    fallback = None
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if fallback is None:
            fallback = candidate_path
        if candidate_path.exists():
            return candidate_path
    return fallback or BASE_DIR


CRAFT_REPO = _resolve_existing_path([
    os.environ.get("CRAFT_REPO"),
    BASE_DIR.parent / "CRAFT-pytorch",
    BASE_DIR.parent / "image_ocr_project" / "CRAFT-pytorch",
])
CRAFT_WEIGHTS = _resolve_existing_path([
    os.environ.get("CRAFT_WEIGHTS"),
    BASE_DIR.parent / "weights" / "craft_mlt_25k.pth",
    BASE_DIR.parent / "image_ocr_project" / "weights" / "craft_mlt_25k.pth",
])

INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "result"
JSON_DIR = BASE_DIR / "result_json"
DEBUG_DIR = BASE_DIR / "debug_crops"
RENDER_DIR = BASE_DIR / "rendered_chat"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)

# --------------------------------------------------
# ADD CRAFT TO PYTHON PATH
# --------------------------------------------------

CRAFT = None
if CRAFT_REPO.exists():
    sys.path.insert(0, str(CRAFT_REPO))
    try:
        from craft import CRAFT
    except ModuleNotFoundError:
        CRAFT = None

if CRAFT is None:
    print(
        f"[CONFIG] CRAFT module not available at {CRAFT_REPO}. "
        "CRAFT-dependent features stay disabled until CRAFT_REPO is fixed."
    )

# --------------------------------------------------
# LOAD CRAFT MODEL
# --------------------------------------------------

def copyStateDict(state_dict):
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return {".".join(k.split(".")[start_idx:]): v for k, v in state_dict.items()}


def load_craft():
    print("[INIT] Loading CRAFT model...")
    if CRAFT is None:
        raise ModuleNotFoundError(
            "CRAFT module not found. Set CRAFT_REPO or place CRAFT-pytorch next to the project."
        )
    if not CRAFT_WEIGHTS.exists():
        raise FileNotFoundError(f"CRAFT weights not found at: {CRAFT_WEIGHTS}")
    net = CRAFT()
    state_dict = torch.load(str(CRAFT_WEIGHTS), map_location="cpu", weights_only=False)
    net.load_state_dict(copyStateDict(state_dict))
    net.eval()
    print("[INIT] CRAFT ready")
    return net


# --------------------------------------------------
# OCR ENGINE — Google Cloud Vision API
# Set GOOGLE_VISION_API_KEY in your environment to use.
# Get a free key: https://console.cloud.google.com → Vision API → Credentials
# Free quota: 1000 requests / month
# --------------------------------------------------

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()

_VISION_URL = None
if GOOGLE_VISION_API_KEY:
    _VISION_URL = (
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    )
else:
    print(
        "[CONFIG] GOOGLE_VISION_API_KEY not set — Cloud Vision OCR is disabled.\n"
        "         Gemini-only translation still works if GEMINI_API_KEY is set."
    )


# Maps BCP-47 language codes → human-readable names used in prompts
LANGUAGE_NAMES = {
    "th": "Thai", "zh": "Chinese", "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "ru": "Russian", "uk": "Ukrainian", "he": "Hebrew",
    "hi": "Hindi", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "en": "English",
}


class GoogleVisionOCR:
    """Drop-in replacement for PaddleOCR that calls Google Cloud Vision API."""

    def __init__(self):
        self.detected_language = "auto"    # BCP-47 code of dominant source language
        self.detected_language_name = "the source language"

    def predict(self, img):
        import cv2

        if not GOOGLE_VISION_API_KEY or not _VISION_URL:
            raise RuntimeError(
                "Google Cloud Vision is not configured. Set GOOGLE_VISION_API_KEY "
                "or use Gemini-only mode (no OCR)."
            )

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

        resp = _requests.post(_VISION_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        responses = data.get("responses", [{}])
        full_text_ann = responses[0].get("fullTextAnnotation") if responses else None
        if not full_text_ann:
            return []

        # Tally language codes across all blocks (weighted by block confidence)
        lang_votes: dict = {}
        detections = []
        for page in full_text_ann.get("pages", []):
            for block in page.get("blocks", []):
                for lang in block.get("property", {}).get("detectedLanguages", []):
                    code = (lang.get("languageCode") or "").lower()
                    conf = float(lang.get("confidence") or 1.0)
                    if code and code != "en":
                        lang_votes[code] = lang_votes.get(code, 0.0) + conf

                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        symbols = word.get("symbols", [])
                        if not symbols:
                            continue
                        text = "".join(s.get("text", "") for s in symbols)
                        verts = word.get("boundingBox", {}).get("vertices", [])
                        if len(verts) < 4:
                            continue
                        xs = [v.get("x", 0) for v in verts]
                        ys = [v.get("y", 0) for v in verts]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                        conf = word.get("confidence", 1.0) or 1.0
                        detections.append({
                            "text": text,
                            "bbox": bbox,
                            "score": float(conf),
                        })

        # Store dominant non-English language detected
        if lang_votes:
            dominant = max(lang_votes, key=lang_votes.__getitem__)
            self.detected_language = dominant
            self.detected_language_name = LANGUAGE_NAMES.get(
                dominant, LANGUAGE_NAMES.get(dominant.split("-")[0], dominant.upper())
            )
            print(f"[OCR] Detected source language: {self.detected_language_name} ({dominant})")

        return detections

    def document_plain_text(self, img):
        """Full page text from DOCUMENT_TEXT_DETECTION (Vision's reading order).

        Lighter than *predict()* when you only need a transcript hint for another model.
        """
        import cv2

        if not GOOGLE_VISION_API_KEY or not _VISION_URL:
            return ""

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

        resp = _requests.post(_VISION_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        responses = data.get("responses", [{}])
        full_text_ann = responses[0].get("fullTextAnnotation") if responses else None
        if not full_text_ann:
            return ""
        return (full_text_ann.get("text") or "").strip()

    def document_paragraph_boxes(self, img):
        """Paragraph-level text + union bbox, sorted top → bottom (then left → right).

        Used to infer **chat side** from horizontal position (contact left / user right).
        """
        import cv2

        if not GOOGLE_VISION_API_KEY or not _VISION_URL:
            return []

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                }
            ]
        }

        resp = _requests.post(_VISION_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        responses = data.get("responses", [{}])
        full_text_ann = responses[0].get("fullTextAnnotation") if responses else None
        if not full_text_ann:
            return []

        out = []
        for page in full_text_ann.get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    words_text = []
                    word_boxes = []
                    for word in para.get("words", []):
                        symbols = word.get("symbols", [])
                        if not symbols:
                            continue
                        chunk = "".join(s.get("text", "") for s in symbols)
                        verts = word.get("boundingBox", {}).get("vertices", [])
                        if len(verts) < 4:
                            continue
                        xs = [v.get("x", 0) for v in verts]
                        ys = [v.get("y", 0) for v in verts]
                        word_boxes.append((min(xs), min(ys), max(xs), max(ys)))
                        words_text.append(chunk)
                    if not words_text:
                        continue
                    combined = "".join(words_text).strip()
                    if len(combined) < 1:
                        continue
                    x1 = min(b[0] for b in word_boxes)
                    y1 = min(b[1] for b in word_boxes)
                    x2 = max(b[2] for b in word_boxes)
                    y2 = max(b[3] for b in word_boxes)
                    out.append({"text": combined, "bbox": (x1, y1, x2, y2)})

        out.sort(
            key=lambda d: (
                (d["bbox"][1] + d["bbox"][3]) / 2.0,
                (d["bbox"][0] + d["bbox"][2]) / 2.0,
            )
        )
        return out


if GOOGLE_VISION_API_KEY:
    print("[INIT] Google Cloud Vision OCR ready")
else:
    print("[INIT] Google Cloud Vision OCR not loaded (no API key)")
ocr_engine = GoogleVisionOCR()


# --------------------------------------------------
# TRANSLATION
# --------------------------------------------------

translator = GoogleTranslator(source="auto", target="en")
translation_cache = {}


def translate_text(text):
    if not text or not text.strip():
        return ""
    if text in translation_cache:
        return translation_cache[text]
    try:
        translated = translator.translate(text)
        translation_cache[text] = translated
        return translated
    except Exception:
        return ""


# English → target (final output localization). Separate cache from auto→en above.
_en_to_target_cache: dict[tuple[str, str], str] = {}


def translate_en_to(text: str, target_code: str) -> str:
    """Translate *text* from English to *target_code* (Google Translate)."""
    if not text or not str(text).strip():
        return ""
    tgt_raw = (target_code or "en").strip()
    tgt_lower = tgt_raw.lower()
    if tgt_lower in ("en", "en-us", "en-gb"):
        return text
    tgt = google_translate_target_code(tgt_raw)
    key = (tgt, text)
    if key in _en_to_target_cache:
        return _en_to_target_cache[key]
    try:
        out = GoogleTranslator(source="en", target=tgt).translate(text)
        if not out:
            return text
        _en_to_target_cache[key] = out
        return out
    except Exception:
        return text
