import os
import sys
import base64
import requests as _requests
from pathlib import Path
import torch
from deep_translator import GoogleTranslator

# --------------------------------------------------
# BASE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CRAFT_REPO = BASE_DIR.parent / "CRAFT-pytorch"
CRAFT_WEIGHTS = BASE_DIR.parent / "weights" / "craft_mlt_25k.pth"

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

sys.path.insert(0, str(CRAFT_REPO))
from craft import CRAFT

# --------------------------------------------------
# LOAD CRAFT MODEL
# --------------------------------------------------

def copyStateDict(state_dict):
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return {".".join(k.split(".")[start_idx:]): v for k, v in state_dict.items()}


def load_craft():
    print("[INIT] Loading CRAFT model...")
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

if not GOOGLE_VISION_API_KEY:
    raise RuntimeError(
        "\n\n[CONFIG] GOOGLE_VISION_API_KEY is not set.\n"
        "  1. Go to https://console.cloud.google.com\n"
        "  2. Create a project → enable 'Cloud Vision API'\n"
        "  3. Credentials → Create API key\n"
        "  4. Set the environment variable:\n"
        "       $env:GOOGLE_VISION_API_KEY = 'YOUR_KEY_HERE'\n"
        "     then re-run the project.\n"
    )

_VISION_URL = (
    f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
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


print("[INIT] Google Cloud Vision OCR ready")
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
