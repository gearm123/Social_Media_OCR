"""Supported final output languages (Google Translate target codes).

The Gemini pipeline still resolves and glosses in English; strings are then
machine-translated to the selected target for ``translated_conversation.json``
and the main rendered image. Per-pass debug PNGs stay English.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class OutputLanguage(str, Enum):
    """Google Translate ``target`` codes known to work well for chat UI text."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    UKRAINIAN = "uk"
    CZECH = "cs"
    ROMANIAN = "ro"
    GREEK = "el"
    TURKISH = "tr"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"
    BENGALI = "bn"
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    FILIPINO = "tl"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    SWEDISH = "sv"
    DANISH = "da"
    NORWEGIAN = "no"
    FINNISH = "fi"
    HUNGARIAN = "hu"
    BULGARIAN = "bg"
    CROATIAN = "hr"
    SERBIAN = "sr"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    LITHUANIAN = "lt"
    LATVIAN = "lv"
    ESTONIAN = "et"
    PERSIAN = "fa"
    URDU = "ur"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    SWAHILI = "sw"
    AFRIKAANS = "af"
    CATALAN = "ca"
    ICELANDIC = "is"


# Aliases (CLI / API) → enum member. Keys are lowercased in lookup.
_OUTPUT_LANG_ALIASES: dict[str, OutputLanguage] = {}
for _m in OutputLanguage:
    _OUTPUT_LANG_ALIASES[_m.value.lower()] = _m
    _OUTPUT_LANG_ALIASES[_m.name.lower()] = _m

_EXTRA_ALIASES = {
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
    "pol": OutputLanguage.POLISH,
    "polski": OutputLanguage.POLISH,
    "rus": OutputLanguage.RUSSIAN,
    "russian": OutputLanguage.RUSSIAN,
    "українська": OutputLanguage.UKRAINIAN,
    "ces": OutputLanguage.CZECH,
    "cze": OutputLanguage.CZECH,
    "cesky": OutputLanguage.CZECH,
    "ron": OutputLanguage.ROMANIAN,
    "ell": OutputLanguage.GREEK,
    "tur": OutputLanguage.TURKISH,
    "ara": OutputLanguage.ARABIC,
    "heb": OutputLanguage.HEBREW,
    "hin": OutputLanguage.HINDI,
    "ben": OutputLanguage.BENGALI,
    "tha": OutputLanguage.THAI,
    "vie": OutputLanguage.VIETNAMESE,
    "ind": OutputLanguage.INDONESIAN,
    "indonesian": OutputLanguage.INDONESIAN,
    "msa": OutputLanguage.MALAY,
    "tagalog": OutputLanguage.FILIPINO,
    "fil": OutputLanguage.FILIPINO,
    "jpn": OutputLanguage.JAPANESE,
    "kor": OutputLanguage.KOREAN,
    "zh": OutputLanguage.CHINESE_SIMPLIFIED,
    "zh_cn": OutputLanguage.CHINESE_SIMPLIFIED,
    "zh-cn": OutputLanguage.CHINESE_SIMPLIFIED,
    "zho": OutputLanguage.CHINESE_SIMPLIFIED,
    "mandarin": OutputLanguage.CHINESE_SIMPLIFIED,
    "simplified_chinese": OutputLanguage.CHINESE_SIMPLIFIED,
    "chinese": OutputLanguage.CHINESE_SIMPLIFIED,
    "zh_tw": OutputLanguage.CHINESE_TRADITIONAL,
    "zh-tw": OutputLanguage.CHINESE_TRADITIONAL,
    "traditional_chinese": OutputLanguage.CHINESE_TRADITIONAL,
    "swe": OutputLanguage.SWEDISH,
    "svenska": OutputLanguage.SWEDISH,
    "dan": OutputLanguage.DANISH,
    "dansk": OutputLanguage.DANISH,
    "nob": OutputLanguage.NORWEGIAN,
    "norwegian": OutputLanguage.NORWEGIAN,
    "fin": OutputLanguage.FINNISH,
    "hun": OutputLanguage.HUNGARIAN,
    "magyar": OutputLanguage.HUNGARIAN,
    "bul": OutputLanguage.BULGARIAN,
    "hrv": OutputLanguage.CROATIAN,
    "srp": OutputLanguage.SERBIAN,
    "slk": OutputLanguage.SLOVAK,
    "slv": OutputLanguage.SLOVENIAN,
    "lit": OutputLanguage.LITHUANIAN,
    "lav": OutputLanguage.LATVIAN,
    "est": OutputLanguage.ESTONIAN,
    "fas": OutputLanguage.PERSIAN,
    "farsi": OutputLanguage.PERSIAN,
    "urd": OutputLanguage.URDU,
    "tam": OutputLanguage.TAMIL,
    "tel": OutputLanguage.TELUGU,
    "mar": OutputLanguage.MARATHI,
    "guj": OutputLanguage.GUJARATI,
    "kan": OutputLanguage.KANNADA,
    "swa": OutputLanguage.SWAHILI,
    "afr": OutputLanguage.AFRIKAANS,
    "cat": OutputLanguage.CATALAN,
    "català": OutputLanguage.CATALAN,
    "isl": OutputLanguage.ICELANDIC,
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
