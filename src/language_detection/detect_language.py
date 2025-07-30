import pymupdf
import logging
import re
from fasttext.FastText import _FastText
import os

model_path = os.getenv("FASTTEXT_MODEL_PATH")
detector = _FastText(model_path)

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en"]
DEFAULT_LANGUAGE = "de"
METADATA_THRESHOLD = 0.7
CLASSIFICATION_THRESHOLD = 0.4


def extract_text_from_page(page: pymupdf.Page) -> tuple[str, int]:
    """
    Extracts text from a PDF page for language detection.
        - Filters out short lines and noisy tokens.
        - Flattens the layout into a single string.
        - Removes digits and non-alphabetic garbage tokens.
    Args:
        page: The PDF page from which to extract text.
    Returns:
        Tuple[str, int]: A cleaned string for language detection,
                         and the count of non-trivial words (>4 letters).
    """
    raw_text = page.get_text()
    word_count_not_short = len(re.findall(r"[^\W\d_]{5,}", raw_text))

    lines = [line for line in raw_text.split("\n") if sum(char.isalpha() for char in line) > 4]
    text = " ".join(lines)

    tokens = re.split(r"\s+", text)
    clean_tokens = [
        token
        for token in tokens
        if len(token) > 1  # skip single-char words
        and not re.search(r"(^|\s)\S*[0-9]\S*(?=\s|$)", token)  # skip tokens with digits
        and not re.search(r"(^|\s)[^a-zA-Zéàèöäüç]+(?=\s|$)", token)  # must contain regular letters.
    ]
    text_for_detection = " ".join(clean_tokens)

    return text_for_detection, word_count_not_short


def detect_language(
    page: pymupdf.Page, supported_languages=None, default_language=DEFAULT_LANGUAGE
) -> tuple[str, str | None]:
    """
    Detects the language of a PDF page using FastText.
    - Returns a classification language (even on low confidence),
    - and a metadata language only if confidence > 0.7.

    Args:
        page: The PDF page to analyze.
        supported_languages: Allowed language codes. Defaults to SUPPORTED_LANGUAGES.
        default_language: Fallback language.

    Returns:
        (classification_language, metadata_language or None)
    """

    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES

    text, word_count = extract_text_from_page(page)

    if word_count < 4 or not text.strip():
        logger.info(f"Insufficient text for language detection. Using default: {default_language} for classification.")
        return default_language, None

    try:
        labels, scores = detector.predict(text.lower(), k=5)
        if not labels or not scores:
            logger.info(f"Empty prediction result. Using default: {default_language} for classification.")
            return default_language, None

        score = scores[0]
        language_code = labels[0].replace("__label__", "")

        if language_code in supported_languages and score >= CLASSIFICATION_THRESHOLD:
            classification_language = language_code
        else:
            logger.info(
                f"Language: '{language_code}' too low confidence or not supported. Falling back to: '{default_language}' for classification"
            )
            classification_language = default_language

        metadata_language = language_code if score >= METADATA_THRESHOLD else None
        return classification_language, metadata_language

    except Exception as e:
        logger.error(
            f"Error occurred during language detection: {e}. Falling back to default language: {default_language} for classification,"
        )
        return default_language, None
