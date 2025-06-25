import pymupdf
import logging
from pathlib import Path

from .utils import is_digitally_born

from .text_objects import extract_words, create_text_lines, create_text_blocks
from .detect_language import detect_language_of_page
from .bounding_box import merge_bounding_boxes
from .page_graphics import extract_page_graphics
from .page_structure import PageAnalysis, PageContext, compute_text_features
from .page_classifier import DigitalPageClassifier,ScannedPageClassifier
from .pixtral_classifier import PixtralPDFClassifier

logger = logging.getLogger(__name__)

def classify_page(page:pymupdf.Page,
                  page_number: int,
                  matching_params: dict,
                  language: str,
                  classifier_name: str = "baseline") -> PageAnalysis:
    """classifies single pages into Text-, Boreprofile-, Map-, Title- or Unknown Page.
        Args:
            page: page that get classified
            page_number: page number in report
            matching_params: dictionary holding including and excluding expressions for classes in supported languages
            language: language of page content
        Returns:
            PageAnalysis object with classification and features,
    """
    analysis = PageAnalysis(page_number)

    is_digital = is_digitally_born(page)
    if classifier_name == "pixtral":
        classifier = PixtralPDFClassifier()

    elif classifier_name == "baseline":
        classifier = DigitalPageClassifier() if is_digital else ScannedPageClassifier

    words = extract_words(page, page_number)
    lines = create_text_lines(page, page_number)
    text_blocks = create_text_blocks(lines)
    drawings, image_rects = extract_page_graphics(page, is_digital)
    page_text_rect = merge_bounding_boxes([line.rect for line in lines]) if lines else page.rect


    context = PageContext(
        lines=lines,
        words=words,
        text_blocks=text_blocks,
        language=language,
        page_rect=page_text_rect,
        geometric_lines = [],
        is_digital=is_digital,
        drawings = drawings,
        image_rects = image_rects,
    )

    analysis.features = compute_text_features(context.lines, context.text_blocks)

    if classifier_name == "pixtral":
        single_page_pdf = pymupdf.open()
        single_page_pdf.insert_pdf(page.parent, from_page=page_number, to_page=page_number)
        page_bytes = single_page_pdf.tobytes()

        page_class = classifier.determine_class(page_bytes, page_name=f"page_{page_number}")
    else:
        page_class = classifier.determine_class(page, context, matching_params, analysis.features)

    analysis.set_class(page_class)

    return analysis

def classify_pdf(file_path: Path,classifier: str, matching_params: dict)-> dict:
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
                                                language,
                                                classifier)

            classification.append(page_classification.to_classification_dict())
    return {"filename": file_path.name,
            "classification": classification}