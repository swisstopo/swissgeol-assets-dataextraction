import logging
from pathlib import Path

import pymupdf

from src.classifiers.classifier_types import ClassifierTypes
from src.bounding_box import get_page_bbox, merge_bounding_boxes
from src.detect_language import detect_language_of_page
from src.page_graphics import extract_page_graphics, get_page_image_bytes
from src.page_structure import PageAnalysis, PageContext
from src.text_objects import create_text_blocks, create_text_lines, extract_words
from src.utils import is_digitally_born

logger = logging.getLogger(__name__)


def classify_page(
    page: pymupdf.Page, page_number: int, classifier, language: str, matching_params: dict
) -> PageAnalysis:
    """classifies single pages into Text-, Boreprofile-, Map-, Title- or Unknown Page.
    Args:
        page: page that get classified
        page_number: page number in report
        matching_params: dictionary holding including and excluding expressions for page classes in supported languages
        language: language of page content
        classifier: classifier used for classification
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

    if classifier.type == ClassifierTypes.PIXTRAL:
        max_doc_size = classifier.config["max_document_size_mb"] - classifier.config["slack_size_mb"]
        image_bytes = get_page_image_bytes(page, page_number, max_mb=max_doc_size)

        fallback_args = {
            "page": page,
            "context": context,
            "matching_params": matching_params,
        }

        page_class = classifier.determine_class(
            image_bytes=image_bytes,
            fallback_args=fallback_args
        )
    elif classifier.type == ClassifierTypes.LAYOUTLMV3:
        page_class = classifier.determine_class(page)

    elif classifier.type == ClassifierTypes.BASELINE:
        page_class = classifier.determine_class(page, context, matching_params)

    else:
        raise ValueError(f"Unsupported classifier type: {classifier.type}")

    analysis.set_class(page_class)

    return analysis

def classify_pdf(file_path: Path, classifier, matching_params: dict) -> dict:
    """
    Classify each page of a PDF file.

    Args:
        file_path: Path to the PDF file.
        classifier: Classifier object with a `determine_class` method.
        matching_params: dictionary holding including and excluding expressions for page classes in supported languages
    Returns:
        dict: Classification results per page.
    """

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return {}

    classification = []

    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            language = detect_language_of_page(page)

            page_classification = classify_page(page, page_number, classifier, language, matching_params)

            classification.append(page_classification.to_classification_dict())
    return {"filename": file_path.name, "classification": classification}
