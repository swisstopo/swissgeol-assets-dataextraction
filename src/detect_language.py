import pymupdf
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def extract_text_from_page(page: pymupdf.Page) -> str:
    
    text = page.get_text().replace("\n", " ")

    # remove all numbers and special characters from text
    return "".join(e for e in text if (e.isalnum() or e.isspace()) and not e.isdigit())

def detect_language_of_page(page: pymupdf.Page) -> str:
  
    text = extract_text_from_page(page)
    try:
        language = detect(text)
    except LangDetectException:
        language = "Language Not Found"

    return language