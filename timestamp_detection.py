import re

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
OCR_CONFUSABLES = str.maketrans({
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "Z": "2",
    "S": "5",
    "s": "5",
    "B": "8",
    "A": "4",
    "a": "4",
})
TIMESTAMP_FORMAT_TOLERANCE = 1

THAI_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?น\.?")
THAI_DATE = re.compile(r"\b\d{1,2}\s?[ก-ฮ\.]+\s?\d{4}")
THAI_DAY_TIME = re.compile(r"[ก-ฮ][ก-ฮ\.]*.*\d{1,2}[:\.]\d{2}")

EN_TIME = re.compile(r"\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)\b")
EN_DATE = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}")
EN_DAY_TIME = re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b.*\d{1,2}:\d{2}")

WEAK_TIME = re.compile(r"\d{1,2}\s?[:\.]\s?\d{2}")
WEAK_YEAR = re.compile(r"(20\d{2}|25\d{2})")
NUMERIC_DATE = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?\b")
TIME_PATTERN = re.compile(r"\b(\d{1,2})\s*[:\.]\s*(\d{2})\b")

THAI_MONTH_HINT = [
    "ม.ค", "ก.พ", "มี.ค", "เม.ย", "พ.ค", "มิ.ย",
    "ก.ค", "ส.ค", "ก.ย", "ต.ค", "พ.ย", "ธ.ค"
]

EN_MONTH_HINT = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

THAI_MONTH_REGEX = "|".join(re.escape(month) for month in THAI_MONTH_HINT)
THAI_TEXT_DATE = re.compile(
    rf"\b(\d{{1,2}})\s*({THAI_MONTH_REGEX})(?:\s*(\d{{2,4}}))?\b"
)

RELATIVE_DAY_PATTERNS = [
    (re.compile(r"วันนี้"), "Today"),
    (re.compile(r"เมื่อวาน"), "Yesterday"),
]

DAY_PATTERNS = [
    (re.compile(r"(วันจันทร์|จันทร์|จ\.)"), "Mon"),
    (re.compile(r"(วันอังคาร|อังคาร|อ\.)"), "Tue"),
    (re.compile(r"(วันพุธ|พุธ|พ\.)"), "Wed"),
    (re.compile(r"(วันพฤหัสบดี|พฤหัสบดี|พฤ\.)"), "Thu"),
    (re.compile(r"(วันศุกร์|ศุกร์|ศ\.)"), "Fri"),
    (re.compile(r"(วันเสาร์|เสาร์|ส\.)"), "Sat"),
    (re.compile(r"(วันอาทิตย์|อาทิตย์|อา\.)"), "Sun"),
    (re.compile(r"\bMon(?:day)?\b", re.IGNORECASE), "Mon"),
    (re.compile(r"\bTue(?:sday)?\b", re.IGNORECASE), "Tue"),
    (re.compile(r"\bWed(?:nesday)?\b", re.IGNORECASE), "Wed"),
    (re.compile(r"\bThu(?:rsday)?\b", re.IGNORECASE), "Thu"),
    (re.compile(r"\bFri(?:day)?\b", re.IGNORECASE), "Fri"),
    (re.compile(r"\bSat(?:urday)?\b", re.IGNORECASE), "Sat"),
    (re.compile(r"\bSun(?:day)?\b", re.IGNORECASE), "Sun"),
]

THAI_MONTH_MAP = {
    "ม.ค": "Jan",
    "ก.พ": "Feb",
    "มี.ค": "Mar",
    "เม.ย": "Apr",
    "พ.ค": "May",
    "มิ.ย": "Jun",
    "ก.ค": "Jul",
    "ส.ค": "Aug",
    "ก.ย": "Sep",
    "ต.ค": "Oct",
    "พ.ย": "Nov",
    "ธ.ค": "Dec",
}

FUZZY_DAY_LABELS = [
    ("วันนี้", "Today"),
    ("เมื่อวาน", "Yesterday"),
    ("จ", "Mon"),
    ("จันทร์", "Mon"),
    ("วันจันทร์", "Mon"),
    ("อ", "Tue"),
    ("อังคาร", "Tue"),
    ("วันอังคาร", "Tue"),
    ("พ", "Wed"),
    ("พุธ", "Wed"),
    ("วันพุธ", "Wed"),
    ("พฤ", "Thu"),
    ("พฤหัส", "Thu"),
    ("พฤหัสบดี", "Thu"),
    ("วันพฤหัสบดี", "Thu"),
    ("ศ", "Fri"),
    ("ศุกร์", "Fri"),
    ("วันศุกร์", "Fri"),
    ("ส", "Sat"),
    ("เสาร์", "Sat"),
    ("วันเสาร์", "Sat"),
    ("อา", "Sun"),
    ("อาทิตย์", "Sun"),
    ("วันอาทิตย์", "Sun"),
]


def _normalize_timestamp_text(text: str) -> str:
    if not text:
        return ""

    normalized = str(text).translate(THAI_DIGITS)
    normalized = normalized.replace("เวลา", " เวลา ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_ocr_confusables(text: str) -> str:
    return text.translate(OCR_CONFUSABLES)


def _compact_token(text: str) -> str:
    return re.sub(r"[\s\.:/\\,\-]+", "", text)


def _distance_with_tolerance(value: str, target: str, tolerance: int) -> bool:
    if not value or not target:
        return False

    if abs(len(value) - len(target)) > tolerance:
        return False

    mismatches = abs(len(value) - len(target))
    for a, b in zip(value, target):
        if a != b:
            mismatches += 1
            if mismatches > tolerance:
                return False

    return mismatches <= tolerance


def _fuzzy_day_label(text: str) -> str:
    compact = _compact_token(_normalize_ocr_confusables(text))
    if not compact:
        return ""

    parts = re.split(r"\s+", _normalize_timestamp_text(text))
    candidates = [compact]
    candidates.extend(_compact_token(part) for part in parts if part)

    for candidate in candidates:
        if not candidate:
            continue
        for label, mapped in FUZZY_DAY_LABELS:
            target = _compact_token(label)
            if candidate == target or _distance_with_tolerance(candidate, target, TIMESTAMP_FORMAT_TOLERANCE):
                return mapped

    return ""


def _extract_time(text: str) -> str:
    normalized = _normalize_ocr_confusables(text)
    match = TIME_PATTERN.search(normalized)
    if not match:
        compact_match = re.search(r"\b(\d{3,4})\b", normalized)
        if not compact_match:
            return ""

        digits = compact_match.group(1)
        if len(digits) == 3:
            hour = int(digits[0])
            minute = int(digits[1:])
        else:
            hour = int(digits[:2])
            minute = int(digits[2:])
        if hour > 23 or minute > 59:
            return ""
        return f"{hour:02d}:{minute:02d}"

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return ""

    return f"{hour:02d}:{minute:02d}"


def _extract_relative_day(text: str) -> str:
    for pattern, label in RELATIVE_DAY_PATTERNS:
        if pattern.search(text):
            return label
    fuzzy = _fuzzy_day_label(text)
    if fuzzy in {"Today", "Yesterday"}:
        return fuzzy
    return ""


def _extract_day_of_week(text: str) -> str:
    for pattern, label in DAY_PATTERNS:
        if pattern.search(text):
            return label
    fuzzy = _fuzzy_day_label(text)
    if fuzzy not in {"", "Today", "Yesterday"}:
        return fuzzy
    return ""


def _extract_date(text: str) -> str:
    match = NUMERIC_DATE.search(text)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year = match.group(3)
        if 1 <= day <= 31 and 1 <= month <= 12:
            if year:
                return f"{day:02d}/{month:02d}/{year}"
            return f"{day:02d}/{month:02d}"

    match = THAI_TEXT_DATE.search(text)
    if match:
        day = int(match.group(1))
        month = THAI_MONTH_MAP.get(match.group(2), match.group(2))
        year = match.group(3)
        if year:
            return f"{day} {month} {year}"
        return f"{day} {month}"

    return ""


def _split_text_and_time(text: str, time_text: str):
    if not text or not time_text:
        return "", ""

    normalized = _normalize_ocr_confusables(text)
    match = TIME_PATTERN.search(normalized)
    if match:
        start, end = match.span()
        prefix = text[:start].strip(" -,:./\\")
        suffix = text[end:].strip(" -,:./\\")
        return prefix, suffix

    compact_match = re.search(r"\b(\d{3,4})\b", normalized)
    if compact_match:
        start, end = compact_match.span()
        prefix = text[:start].strip(" -,:./\\")
        suffix = text[end:].strip(" -,:./\\")
        return prefix, suffix

    return "", ""


def parse_timestamp_text(text: str):
    normalized = _normalize_timestamp_text(text)
    if not normalized or len(normalized) > 60:
        return None

    time_text = _extract_time(normalized)
    day_text = _extract_relative_day(normalized) or _extract_day_of_week(normalized)
    date_text = _extract_date(normalized)
    has_numeric_sequence = bool(
        time_text
        and re.search(r"\d(?:[\s:./\\,\-]*\d)+", _normalize_ocr_confusables(normalized))
    )
    has_timestamp_keyword = (
        "เวลา" in normalized
        or "น." in normalized
        or bool(_extract_relative_day(normalized))
        or bool(_extract_day_of_week(normalized))
    )

    if not any((time_text, day_text, date_text)):
        return None

    prefix_text, suffix_text = _split_text_and_time(normalized, time_text)
    text_part = " ".join(part for part in (prefix_text, suffix_text) if part).strip()
    format_type = "parsed"

    if time_text and has_numeric_sequence:
        format_type = "text_time"
    elif time_text and has_timestamp_keyword:
        pass
    elif time_text and (date_text or day_text):
        pass
    elif time_text and len(normalized) <= 12:
        pass
    elif date_text and day_text:
        pass
    else:
        return None

    display_parts = []
    if date_text:
        display_parts.append(date_text)
    elif day_text:
        display_parts.append(day_text)
    if time_text:
        display_parts.append(time_text)

    return {
        "raw_text": normalized,
        "day_text": day_text,
        "date_text": date_text,
        "time_text": time_text,
        "text_part": text_part,
        "format_type": format_type,
        "display_text": " ".join(display_parts).strip() or normalized,
    }


def is_timestamp(text: str) -> bool:
    if not text:
        return False

    parsed = parse_timestamp_text(text)
    if parsed and parsed["display_text"]:
        return True

    t = _normalize_timestamp_text(text)
    if len(t) > 60:
        return False

    score = 0

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

    if WEAK_TIME.search(t):
        score += 1
    if WEAK_YEAR.search(t):
        score += 1

    if any(m in t for m in THAI_MONTH_HINT):
        score += 1
    if any(m in t for m in EN_MONTH_HINT):
        score += 1

    if "น." in t or "เวลา" in t:
        score += 2

    return score >= 2