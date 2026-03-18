import os
import sys
from pathlib import Path
import torch
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

# --------------------------------------------------
# ENV FIXES (silence paddle warnings / slow checks)
# --------------------------------------------------

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "2"
os.environ["FLAGS_logtostderr"] = "0"
os.environ["FLAGS_v"] = "0"

# --------------------------------------------------
# BASE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# 🔥 CRAFT is ONE LEVEL ABOVE
CRAFT_REPO = BASE_DIR.parent / "CRAFT-pytorch"

# 🔥 weights also ONE LEVEL ABOVE
CRAFT_WEIGHTS = BASE_DIR.parent / "weights" / "craft_mlt_25k.pth"

# local project folders
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "result"
JSON_DIR = BASE_DIR / "result_json"
DEBUG_DIR = BASE_DIR / "debug_crops"
RENDER_DIR = BASE_DIR / "rendered_chat"

# ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RENDER_DIR, exist_ok=True)

# --------------------------------------------------
# ADD CRAFT TO PYTHON PATH
# --------------------------------------------------

sys.path.insert(0, str(CRAFT_REPO))

# now import
from craft import CRAFT

# --------------------------------------------------
# LOAD CRAFT MODEL
# --------------------------------------------------

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = {}

    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v

    return new_state_dict


def load_craft():
    print("[INIT] Loading CRAFT model...")

    if not CRAFT_WEIGHTS.exists():
        raise FileNotFoundError(f"CRAFT weights not found at: {CRAFT_WEIGHTS}")

    net = CRAFT()

    state_dict = torch.load(
        str(CRAFT_WEIGHTS),
        map_location="cpu",
        weights_only=False
    )

    net.load_state_dict(copyStateDict(state_dict))
    net.eval()

    print("[INIT] CRAFT ready")
    return net


# --------------------------------------------------
# OCR ENGINE (Paddle)
# --------------------------------------------------

print("[INIT] Initializing PaddleOCR...")

ocr_engine = PaddleOCR(
    lang="th",  # Thai + English auto works well
    use_angle_cls=False
)

print("[INIT] PaddleOCR ready")


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