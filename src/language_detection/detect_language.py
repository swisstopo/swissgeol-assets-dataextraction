import logging
import math
import os
import re

import fasttext
import pymupdf
from dotenv import load_dotenv

load_dotenv()

model_path = os.getenv("FASTTEXT_MODEL_PATH")
if not model_path or not os.path.isfile(model_path):
    raise FileNotFoundError(f"FASTTEXT model path is invalid or missing: {model_path}")
detector = fasttext.load_model(model_path)

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en"]
DEFAULT_LANGUAGE = "de"
METADATA_THRESHOLD = 0.7
CLASSIFICATION_THRESHOLD = 0.4


def extract_cleaned_text(page: pymupdf.Page) -> tuple[str, int]:
    """Extracts text from a PDF page for language detection.

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
        return [(label.replace("__label__", ""), score) for label, score in zip(labels, scores, strict=False)]
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return []


def select_classification_language(
    predictions: list[tuple[str, float]], word_count: int, supported_languages: list[str] = None
) -> str:
    """Returns the best classification language, falling back to default if no valid match is found.

    Args:
        predictions: List of (language_code, score)
        word_count: Non-trivial word count on the page
        supported_languages: Allowed language codes (defaults to SUPPORTED_LANGUAGES)

    Returns:
        Language code (str)
    """
    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES

    threshold = CLASSIFICATION_THRESHOLD
    fallback = DEFAULT_LANGUAGE

    if word_count < 4:
        logger.info(f"[Classification] Too few words ({word_count}). Fallback to '{fallback}'.")
        return fallback

    for lang, score in predictions:
        if score >= threshold and lang in supported_languages:
            return lang

    logger.info(f"[Classification] No valid prediction above threshold {threshold}. Fallback to '{fallback}'.")
    return fallback


def select_metadata_language(
    predictions: list[tuple[str, float]],
    word_count: int,
    is_frontpage: bool,
    page_number: int,
    scores: dict[str, float],
    long_counts: dict[str, int],
) -> str | None:
    """Selects metadata language and updates aggregated score trackers.

    Args:
        predictions: List of (language, score)
        word_count: Count of non-trivial words
        is_frontpage: Whether the page is a Belegblatt/front page
        page_number: Page index (1-based)
        scores: Aggregated log(word_count)/page_number for each language
        long_counts: Count of pages > 50 words per language

    Returns:
        Metadata language (or None if no confident prediction)
    """
    threshold = METADATA_THRESHOLD

    if word_count < 4:
        logger.info(f"[Metadata] Too few words ({word_count}). Returning None.")
        return None

    for lang, score in predictions:
        if score >= threshold:
            if not is_frontpage:
                scores[lang] += math.log(word_count) / page_number
                if word_count > 50:
                    long_counts[lang] += 1
            return lang

    logger.info(f"[Metadata] No language above threshold {threshold}. Returning None.")
    return None


def summarize_language_metadata(scores: dict[str, float], long_counts: dict[str, int], page_count: int) -> dict:
    """Summarizes detected languages for the PDF.

    - Selects the language with the highest aggregated score (based on weighted word counts).
    - Adds additional languages if they appear in at least 2 long pages (>50 words).

    Returns:
        A dictionary with page_count and list of dominant languages.
    """
    if scores:
        best = max(scores, key=scores.get)
        languages = [best] + [lang for lang, count in long_counts.items() if count >= 2 and lang != best]
    else:
        languages = []

    return {"page_count": page_count, "languages": languages}
