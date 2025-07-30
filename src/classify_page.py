import logging
from pathlib import Path

import pymupdf

from src.classifiers.classifier_types import Classifier
from src.bounding_box import get_page_bbox, merge_bounding_boxes
from language_detection.detect_language import detect_language
from src.page_graphics import extract_page_graphics
from src.page_structure import PageAnalysis, PageContext
from src.text_objects import create_text_blocks, create_text_lines, extract_words
from src.utils import is_digitally_born

logger = logging.getLogger(__name__)


def classify_page(
    page: pymupdf.Page,
    page_number: int,
    classifier: Classifier,
    language: str,
) -> PageAnalysis:
    """classifies single pages into Text-, Boreprofile-, Map-, Title- or Unknown Page.
    Args:
        page: page that get classified
        page_number: page number in report
        classifier: classifier used for classification
        language: language of page content

    Returns:
        PageAnalysis object with page classification.
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

    page_class = classifier.determine_class(page=page, context=context, page_number=page_number)

    analysis.set_class(page_class)

    return analysis


def classify_pdf(file_path: Path, classifier: Classifier) -> dict:
    """
    Classify each page of a PDF file.

    Args:
        file_path: Path to the PDF file.
        classifier: Classifier object with a `determine_class` method.
    Returns:
        dict: Classification results per page.
    """

    if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
        logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
        return {}

    pages = []

    with pymupdf.Document(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            classification_language, metadata_language = detect_language(page)
            classification = classify_page(page, page_number, classifier, classification_language)

            pages.append(
                {
                    "page": page_number,
                    "classification": classification.to_classification_dict(),
                    "metadata": {"language": metadata_language},
                }
            )

    overall_language = []
    return {"filename": file_path.name, "metadata": {"languages": overall_language}, "pages": pages}
