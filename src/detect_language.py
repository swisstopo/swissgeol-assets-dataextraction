import pymupdf
import logging
import re
from fasttext.FastText import _FastText

detector = _FastText("models/FastText/lid.176.bin")

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en"]
DEFAULT_LANGUAGE = "de"


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

    # Keep lines that contain at least 5 alphabetic characters.
    lines = [line for line in raw_text.split("\n") if sum(char.isalpha() for char in line) > 4]
    text = " ".join(lines)

    # Clean up Tokens
    tokens = re.split(r"\s+", text)
    tokens = [
        token
        for token in tokens
        if len(token) > 1  # skip single-char words
        and not re.search(r"(^|\s)\S*[0-9]\S*(?=\s|$)", token)  # skip tokens with digits
        and not re.search(r"(^|\s)[^a-zA-Zéàèöäüç]+(?=\s|$)", token)  # must contain regular letters.
    ]
    text_for_detection = " ".join(tokens)

    return text_for_detection, word_count_not_short


def detect_language_of_page(page: pymupdf.Page, supported_languages=None, default_language=DEFAULT_LANGUAGE) -> str:
    """Detects the language of a PDF page using FastText.
    Args:
        page (pymupdf.Page): The PDF page from which to detect the language.
        supported_languages (list, optional): A list of language codes that are supported. Defaults to SUPPORTED_LANGUAGES.
        default_language (str, optional): The default language code to return if detection fails. Defaults to DEFAULT_LANGUAGE.
    Returns:
        str: The detected language code or the default language code if detection fails.
    """

    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES

    text, word_count = extract_text_from_page(page)

    if word_count < 4:
        logger.info(
            f"Too few words detected for language detection. Falling back to default language: {default_language}."
        )
        return default_language

    # The Fasttext language identification model does not work well with all-uppercase text
    labels, scores = detector.predict(text.lower(), k=5)

    try:
        if len(labels) and len(scores):
            language_code = labels[0].replace("__label__", "")
            logger.info(language_code)
            if language_code not in supported_languages:
                logger.info(f"Unsupported language detected. Falling back to default language: {default_language}.")
                return default_language
            else:
                return language_code
        else:
            logger.info(f"Failed to detect language, Falling back to default language: {default_language}.")
            return default_language
    except Exception as e:
        logger.error(
            f"Error occurred during language detection: {e}. Falling back to default language: {default_language}"
        )
        return default_language
