"""Ensure every :class:`OutputLanguage` maps to a ``deep_translator`` Google target."""

import os
import sys
import unittest

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestOutputLanguagesGoogle(unittest.TestCase):
    def test_enum_values_in_google_translator(self):
        from deep_translator import GoogleTranslator
        from output_languages import OutputLanguage, google_translate_target_code

        t = GoogleTranslator(source="en", target="en")
        supported = set(t.get_supported_languages(as_dict=True).values())

        for m in OutputLanguage:
            resolved = google_translate_target_code(m.value)
            self.assertIn(
                resolved,
                supported,
                f"{m.name} value={m.value!r} resolves to {resolved!r} not in GoogleTranslator targets",
            )

    def test_google_translate_constructor_accepts_each_target(self):
        from deep_translator import GoogleTranslator
        from output_languages import OutputLanguage, google_translate_target_code

        for m in OutputLanguage:
            resolved = google_translate_target_code(m.value)
            try:
                GoogleTranslator(source="en", target=resolved)
            except Exception as exc:
                self.fail(f"GoogleTranslator(target={resolved!r}) for {m.name}: {exc}")


if __name__ == "__main__":
    unittest.main()
