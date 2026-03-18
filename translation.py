from deep_translator import GoogleTranslator

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