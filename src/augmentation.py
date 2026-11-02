"""Back-translation augmentation (program section 4.2)."""
from deep_translator import GoogleTranslator


def back_translate(text, src="id", mid="en"):
    """Translate Indonesian text to an intermediate language and back.

    If translation fails, the original text is returned, matching the source.
    """
    try:
        translated = GoogleTranslator(source=src, target=mid).translate(text)
        return GoogleTranslator(source=mid, target=src).translate(translated)
    except Exception as exc:
        print(f"Error translating text: {text}, {exc}")
        return text
