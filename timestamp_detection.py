import re

# --------------------------------------------------
# STRONG PATTERNS (HIGH CONFIDENCE)
# --------------------------------------------------

THAI_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?น\.?")
THAI_DATE = re.compile(r"\b\d{1,2}\s?[ก-ฮ\.]+\s?\d{4}")
THAI_DAY_TIME = re.compile(r"[ก-ฮ]\.\s?\d{1,2}:\d{2}")

EN_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)\b")
EN_DATE = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}")
EN_DAY_TIME = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b.*\d{1,2}:\d{2}")

# --------------------------------------------------
# WEAK / FUZZY PATTERNS (OCR tolerant)
# --------------------------------------------------

WEAK_TIME = re.compile(r"\d{1,2}\s?[:\.]\s?\d{2}")
WEAK_YEAR = re.compile(r"(20\d{2}|25\d{2})")  # 2026 / 2569

# Thai month fragments (very permissive)
THAI_MONTH_HINT = [
    "ม.ค", "ก.พ", "มี.ค", "เม.ย", "พ.ค", "มิ.ย",
    "ก.ค", "ส.ค", "ก.ย", "ต.ค", "พ.ย", "ธ.ค"
]

# English month hints
EN_MONTH_HINT = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# --------------------------------------------------
# MAIN DETECTION FUNCTION
# --------------------------------------------------

def is_timestamp(text: str) -> bool:
    if not text:
        return False

    t = text.strip()

    score = 0

    # ---------------------------
    # Strong matches
    # ---------------------------
    if THAI_TIME.search(t):
        score += 2
    if THAI_DATE.search(t):
        score += 2
    if THAI_DAY_TIME.search(t):
        score += 2

    if EN_TIME.search(t):
        score += 2
    if EN_DATE.search(t):
        score += 2
    if EN_DAY_TIME.search(t):
        score += 2

    # ---------------------------
    # Weak matches
    # ---------------------------
    if WEAK_TIME.search(t):
        score += 1
    if WEAK_YEAR.search(t):
        score += 1

    # Month hints
    if any(m in t for m in THAI_MONTH_HINT):
        score += 1
    if any(m in t for m in EN_MONTH_HINT):
        score += 1

    # Thai "น." indicator
    if "น." in t:
        score += 2

    # ---------------------------
    # Length filter (important)
    # ---------------------------
    if len(t) > 40:
        return False  # timestamps are short

    # ---------------------------
    # Final decision
    # ---------------------------
    return score >= 2