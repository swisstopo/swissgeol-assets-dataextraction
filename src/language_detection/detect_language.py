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


def extract_cleaned_text(page: pymupdf.Page) -> tuple[str, int]:
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


def predict_language(text: str) -> list[tuple[str, float]]:
    """Returns list of (language_code, score) tuples from FastText."""
    if not text.strip():
        return []

    try:
        labels, scores = detector.predict(text.lower(), k=5)
        return [(label.replace("__label__", ""), score) for label, score in zip(labels, scores)]
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return []


def select_language(
    predictions: list[tuple[str, float]],
    word_count: int,
    mode: str = "classification",
    supported_languages: list[str] = None,
) -> str | None:
    """
    Selects the best matching language for the given mode.

    Modes:
    - "classification": requires language to be supported; fallback if none pass
    - "metadata": accepts any language if score >= threshold; returns None if none pass

    Returns:
        Language code (str) or fallback/None
    """
    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES

    if mode == "classification":
        threshold = CLASSIFICATION_THRESHOLD
        fallback = DEFAULT_LANGUAGE
        require_supported = True
    elif mode == "metadata":
        threshold = METADATA_THRESHOLD
        fallback = None
        require_supported = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if word_count < 4:
        logger.info(f"Too few words for detection. Returning fallback: {fallback}")
        return fallback

    for lang, score in predictions:
        if score >= threshold:
            if not require_supported or lang in supported_languages:
                return lang

    logger.info(f"No valid language found for mode='{mode}'. Returning fallback: {fallback}")
    return fallback
