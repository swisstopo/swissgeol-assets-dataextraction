import logging
import math
import re

import pymupdf
import swissgeol_doc_processing as swiss
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en"]
DEFAULT_LANGUAGE = "de"
MIN_WORDS_PER_LANG = 4
MIN_WORDS_PER_PAGE = 50


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


def predict_language(text: str | None) -> str | None:
    """Detected language code.

    Args:
        text (str | None): Text used for detection.

    Returns:
        str | None: Detected language.
    """
    if not text or not text.strip():
        return None

    try:
        # If detected language is none of supported, return none
        return swiss.language_detection.detect_language_of_text(
            text=text.lower(), default_language=None, supported_languages=SUPPORTED_LANGUAGES
        )
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return None


def select_classification_language(
    prediction: str | None, word_count: int, supported_languages: list[str] = None
) -> str:
    """Returns the best classification language, falling back to default if no valid match is found.

    Args:
        prediction (str | None): Predicted language.
        word_count (int): Non-trivial word count on the page.
        supported_languages (list[str]): Allowed language codes (defaults to SUPPORTED_LANGUAGES).

    Returns:
        str: Language code
    """
    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES

    fallback = DEFAULT_LANGUAGE

    if word_count < MIN_WORDS_PER_LANG:
        logger.info(f"[Classification] Too few words ({word_count}). Fallback to '{fallback}'.")
        return fallback

    if prediction in supported_languages:
        return prediction
    else:
        logger.info(f"[Classification] No valid prediction. Fallback to '{fallback}'.")
        return fallback


def track_metadata_language(
    lang: str,
    word_count: int,
    is_frontpage: bool,
    page_number: int,
    scores: dict[str, float],
    long_counts: dict[str, int],
) -> None:
    """Track metadata language and updates aggregated score trackers.

    Args:
        lang (str): Language
        word_count (int): Count of non-trivial words
        is_frontpage (bool): Whether the page is a Belegblatt/front page
        page_number (int): Page index (1-based)
        scores (dict[str, float]): Aggregated log(word_count)/page_number for each language
        long_counts (dict[str, int]): Count of pages > MIN_WORDS_PER_PAGE words per language
    """
    if word_count < MIN_WORDS_PER_LANG:
        logger.info(f"[Metadata] Too few words ({word_count}).")
        return

    if not is_frontpage:
        scores[lang] += math.log(word_count) / page_number
        if word_count > MIN_WORDS_PER_PAGE:
            long_counts[lang] += 1


def summarize_language_metadata(scores: dict[str, float], long_counts: dict[str, int], page_count: int) -> dict:
    """Summarizes detected languages for the PDF.

    - Selects the language with the highest aggregated score (based on weighted word counts).
    - Adds additional languages if they appear in at least 2 long pages (>MIN_WORDS_PER_PAGE words).

    Returns:
        A dictionary with page_count and list of dominant languages.
    """
    if scores:
        best = max(scores, key=scores.get)
        languages = [best] + [lang for lang, count in long_counts.items() if count >= 2 and lang != best]
    else:
        languages = []

    return {"page_count": page_count, "languages": languages}
