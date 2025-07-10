import pymupdf
import logging

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["de", "fr", "it", "en"]
DEFAULT_LANGUAGE= "de"

def extract_text_from_page(page: pymupdf.Page) -> str:
    text = page.get_text().replace("\n", " ")

    # remove all numbers and special characters from text
    return "".join(e for e in text if (e.isalnum() or e.isspace()) and not e.isdigit())


def detect_language_of_page(page: pymupdf.Page,
                            supported_languages=None,
                            default_language=DEFAULT_LANGUAGE) -> str:

    if supported_languages is None:
        supported_languages = SUPPORTED_LANGUAGES
    text = extract_text_from_page(page)
    try:
        language = detect(text)
        if language not in supported_languages:
            logging.warning(f"Language '{language}' not supported. Using default: '{default_language}'")
            language = default_language
    except LangDetectException:
        logging.warning(f"Language detection failed. Using default: '{default_language}'")
        language = default_language

    return language
