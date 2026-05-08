"""
translator.py
-------------
Multi-language translation using deep-translator (Google Translate backend).
100% free — no API key required.

Supported language codes used in app.py
----------------------------------------
en  English  (no translation needed)
ta  Tamil
hi  Hindi
fr  French
es  Spanish
de  German
ar  Arabic
"""

from __future__ import annotations


def translate_summary(text: str, target_lang: str) -> str:
    """
    Translate *text* to *target_lang* using deep-translator.

    Falls back to original text if translation fails for any reason,
    so the app never crashes due to a network or package issue.

    Parameters
    ----------
    text        : English summary text.
    target_lang : ISO 639-1 language code (e.g. 'ta', 'hi', 'fr').

    Returns
    -------
    str : Translated text (or original if translation unavailable).
    """
    if not text or not text.strip():
        return text

    if target_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        return translated if translated else text
    except ImportError:
        return text + f"\n\n[Translation to '{target_lang}' requires: pip install deep-translator]"
    except Exception as e:
        return text + f"\n\n[Translation unavailable: {e}]"
