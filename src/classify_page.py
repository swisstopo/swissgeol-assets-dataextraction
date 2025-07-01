import logging
from pathlib import Path

import pymupdf

from classifiers.baseline_classifier import DigitalPageClassifier, ScannedPageClassifier
from classifiers.pixtral_classifier import PixtralPDFClassifier
from src.bounding_box import get_page_bbox, merge_bounding_boxes
from src.detect_language import detect_language_of_page
from src.page_graphics import extract_page_graphics, get_page_bytes
from src.page_structure import PageAnalysis, PageContext
from src.text_objects import create_text_blocks, create_text_lines, extract_words
from src.utils import is_digitally_born

logger = logging.getLogger(__name__)


def classify_page(
    page: pymupdf.Page, page_number: int, matching_params: dict, language: str, classifier_name: str = "baseline"
) -> PageAnalysis:
    """classifies single pages into Text-, Boreprofile-, Map-, Title- or Unknown Page.
    Args:
        page: page that get classified
        page_number: page number in report
        matching_params: dictionary holding including and excluding expressions for classes in supported languages
        language: language of page content
        classifier_name: classifier name used for classification
    Returns:
        PageAnalysis object with classification and features,
    """
    analysis = PageAnalysis(page_number)

    is_digital = is_digitally_born(page)

    words = extract_words(page, page_number)
    lines = create_text_lines(page, page_number)
    text_blocks = create_text_blocks(lines)
    drawings, image_rects = extract_page_graphics(page, is_digital)
    page_rect = get_page_bbox(page)
    text_rect = merge_bounding_boxes([line.rect for line in lines]) if lines else page_rect

    context = PageContext(
        lines=lines,
        words=words,
        text_blocks=text_blocks,
        language=language,
        page_rect=page_rect,
        text_rect=text_rect,
        geometric_lines=[],
        is_digital=is_digital,
        drawings=drawings,
        image_rects=image_rects,
    )

    if classifier_name == "pixtral":
        max_doc_size = PixtralPDFClassifier.MAX_DOCUMENT_SIZE_MB - PixtralPDFClassifier.SLACK_SIZE_MB
        page_bytes = get_page_bytes(page, page_number, max_mb=max_doc_size)
        fallback = DigitalPageClassifier() if is_digital else ScannedPageClassifier()

        classifier = PixtralPDFClassifier(fallback_classifier=fallback)
        fallback_args = {
            "page": page,
            "context": context,
            "matching_params": matching_params,
        }

        page_class = classifier.determine_class(
            page_bytes, page_name=f"page_{page_number}", fallback_args=fallback_args
        )
    else:
        classifier = DigitalPageClassifier() if is_digital else ScannedPageClassifier()
        page_class = classifier.determine_class(page, context, matching_params)

    analysis.set_class(page_class)

    return analysis


def classify_pdf(file_path: Path, classifier: str, matching_params: dict) -> dict:
    """Processes a pdf File, classifies each page"""

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return {}

    classification = []

    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            language = detect_language_of_page(page)

            if language not in matching_params["material_description"]:
                logging.warning(f"Language '{language}' not supported. Using default german language.")
                language = "de"

            page_classification = classify_page(page, page_number, matching_params, language, classifier)

            classification.append(page_classification.to_classification_dict())
    return {"filename": file_path.name, "classification": classification}
