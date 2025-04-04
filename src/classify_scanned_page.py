import pymupdf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .text import extract_words, create_text_lines, create_text_blocks
from .title_page import sparse_title_page
from .detect_language import detect_language_of_page
from .bounding_box import merge_bounding_boxes
from .identifiers.map import identify_map
from .identifiers.boreprofile import identify_boreprofile
from .page_classes import PageClasses
from .page_structure import PageAnalysis, PageContext, compute_text_features


def classify_page(page:pymupdf.Page, page_number: int, matching_params: dict, language: str) -> PageAnalysis:
    """classifies single pages into Text-, Boreprofile-, Map- or Unknown Page.
        Args:
            page: page that get classified
            page_number: page number in report
            matching_params: dictionary holding including and excluding expressions for classes in supported languages
            language: language of page content
        Returns:
            PageAnalysis object with classification and features,
    """
    analysis = PageAnalysis(page_number)

    words = extract_words(page, page_number)
    if not words:
        analysis.set_class(PageClasses.UNKNOWN)
        return analysis

    lines = create_text_lines(page, page_number)
    text_blocks = create_text_blocks(lines)
    page_text_rect = merge_bounding_boxes([line.rect for line in lines]) if lines else page.rect

    context = PageContext(
        lines=lines,
        words=words,
        text_blocks=text_blocks,
        language=language,
        page_rect=page_text_rect
    )
    analysis.features = compute_text_features(context.lines, context.text_blocks)

    if analysis.features["word_density"] > 1 and analysis.features["mean_words_per_line"] > 3:
        analysis.set_class(PageClasses.TEXT)
    elif identify_boreprofile(context, matching_params):
        analysis.set_class(PageClasses.BOREPROFILE)
    elif identify_map(context, matching_params):
        analysis.set_class(PageClasses.MAP)
    elif sparse_title_page(context.lines):
        analysis.set_class(PageClasses.TITLE_PAGE)
    else:
        analysis.set_class(PageClasses.UNKNOWN)

    return analysis

def classify_pdf(file_path: Path, matching_params: dict)-> dict:
    """Processes a pdf File, classifies each page"""

    if not file_path.is_file() or file_path.suffix.lower() != '.pdf':
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return {}

    classification = []

    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start = 1):
            
            language = detect_language_of_page(page)

            if language not in matching_params["material_description"]:
                logging.warning(f"Language '{language}' not supported. Using default german language.")
                language = "de"

            page_classification = classify_page(page,
                                                page_number,
                                                matching_params,
                                                language)

            classification.append(page_classification.to_classification_dict())
    return {"filename": file_path.name,
            "classification": classification}