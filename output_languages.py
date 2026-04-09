"""Supported final output languages (Latin / Romance + Germanic families only).

Each :class:`OutputLanguage` value is the ``target`` code accepted by
``deep_translator.GoogleTranslator`` (Google Translate). Verified against that
library's supported set (see ``tests/test_output_languages_google.py``).

The Gemini pipeline still resolves in English; these codes are used only for the
post-step ``translate_en_to`` localization of the main JSON/PNG.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

# ---------------------------------------------------------------------------
# Enum: Romance (Latin-derived) + Germanic (incl. English / North & West Germanic)
# Excludes Slavic, Greek, non-European, etc.
# ---------------------------------------------------------------------------


class OutputLanguage(str, Enum):
    """Google Translate target codes for Latin-script Romance + Germanic languages."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    DANISH = "da"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    ICELANDIC = "is"
    AFRIKAANS = "af"
    ROMANIAN = "ro"
    LUXEMBOURGISH = "lb"


def google_translate_target_code(target_code: str) -> str:
    """Map a stored / CLI language code to the ``target=`` value for ``GoogleTranslator``.

    All :class:`OutputLanguage` values are two-letter ISO 639-1 codes accepted by
    ``deep_translator`` as lowercase (validated in ``tests/test_output_languages_google.py``).
    """
    raw = (target_code or "en").strip()
    low = raw.lower()
    if low in ("en", "en-us", "en-gb"):
        return "en"
    if len(raw) == 2:
        return low
    return raw


# Aliases (CLI / API) → enum member. Keys are lowercased in lookup.
_OUTPUT_LANG_ALIASES: dict[str, OutputLanguage] = {}
for _m in OutputLanguage:
    _OUTPUT_LANG_ALIASES[_m.value.lower()] = _m
    _OUTPUT_LANG_ALIASES[_m.name.lower()] = _m

_EXTRA_ALIASES: dict[str, OutputLanguage] = {
    "eng": OutputLanguage.ENGLISH,
    "ingles": OutputLanguage.ENGLISH,
    "castilian": OutputLanguage.SPANISH,
    "esp": OutputLanguage.SPANISH,
    "fra": OutputLanguage.FRENCH,
    "francais": OutputLanguage.FRENCH,
    "français": OutputLanguage.FRENCH,
    "deu": OutputLanguage.GERMAN,
    "deutsch": OutputLanguage.GERMAN,
    "ita": OutputLanguage.ITALIAN,
    "italiano": OutputLanguage.ITALIAN,
    "por": OutputLanguage.PORTUGUESE,
    "portugues": OutputLanguage.PORTUGUESE,
    "português": OutputLanguage.PORTUGUESE,
    "brazilian": OutputLanguage.PORTUGUESE,
    "nld": OutputLanguage.DUTCH,
    "nederlands": OutputLanguage.DUTCH,
    "flemish": OutputLanguage.DUTCH,
    "swe": OutputLanguage.SWEDISH,
    "svenska": OutputLanguage.SWEDISH,
    "dan": OutputLanguage.DANISH,
    "dansk": OutputLanguage.DANISH,
    "nob": OutputLanguage.NORWEGIAN,
    "norsk": OutputLanguage.NORWEGIAN,
    "norwegian_bokmal": OutputLanguage.NORWEGIAN,
    "bokmal": OutputLanguage.NORWEGIAN,
    "ron": OutputLanguage.ROMANIAN,
    "rum": OutputLanguage.ROMANIAN,
    "afr": OutputLanguage.AFRIKAANS,
    # Catalan / Galician / Frisian are not separate targets; map to Spanish / Dutch.
    "ca": OutputLanguage.SPANISH,
    "catalan": OutputLanguage.SPANISH,
    "cat": OutputLanguage.SPANISH,
    "català": OutputLanguage.SPANISH,
    "gl": OutputLanguage.SPANISH,
    "galician": OutputLanguage.SPANISH,
    "glg": OutputLanguage.SPANISH,
    "galego": OutputLanguage.SPANISH,
    "fy": OutputLanguage.DUTCH,
    "frisian": OutputLanguage.DUTCH,
    "fry": OutputLanguage.DUTCH,
    "ltz": OutputLanguage.LUXEMBOURGISH,
    "lux": OutputLanguage.LUXEMBOURGISH,
    "isl": OutputLanguage.ICELANDIC,
    "íslenska": OutputLanguage.ICELANDIC,
}
for _k, _v in _EXTRA_ALIASES.items():
    _OUTPUT_LANG_ALIASES[_k.lower()] = _v


def parse_output_language(value: str | None) -> OutputLanguage | None:
    """Resolve a user string to :class:`OutputLanguage`, or ``None`` if invalid."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_")
    return _OUTPUT_LANG_ALIASES.get(key)


def supported_output_language_codes() -> tuple[str, ...]:
    """Distinct Google target codes (enum values), sorted."""
    seen: set[str] = set()
    ordered: list[str] = []
    for m in OutputLanguage:
        if m.value not in seen:
            seen.add(m.value)
            ordered.append(m.value)
    return tuple(sorted(ordered))


def all_supported_output_languages() -> Iterable[OutputLanguage]:
    """Every enum member (for iteration / validation)."""
    return iter(OutputLanguage)
