import logging
from collections import defaultdict
from pathlib import Path

import pymupdf
from tqdm import tqdm

from src.bounding_box import get_page_bbox, merge_bounding_boxes
from src.classifiers.classifier_types import Classifier
from src.language_detection.detect_language import (
    extract_cleaned_text,
    predict_language,
    select_classification_language,
    select_metadata_language,
    summarize_language_metadata,
)
from src.language_detection.pages_to_ignore import is_belegblatt
from src.page_graphics import extract_page_graphics
from src.page_structure import PageAnalysis, PageContext
from src.text_objects import create_text_blocks, create_text_lines, extract_words
from src.utils import is_digitally_born

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Class to process PDF files and classify their pages into PageClasses.

    It uses a classifier to determine the class of each page based on its content and structure.
    Extracts metadata such as language on file and page level.

    Args:
        classifier: An instance of a classifier that implements the `determine_class` method.
    """

    def __init__(self, classifier: Classifier):
        self.classifier = classifier

    @staticmethod
    def build_full_context(page: pymupdf.Page, page_number: int, language: str) -> PageContext:
        is_digital = is_digitally_born(page)
        words = extract_words(page, page_number)
        lines = create_text_lines(page, page_number)
        text_blocks = create_text_blocks(lines)
        drawings, image_rects = extract_page_graphics(page, is_digital)
        page_rect = get_page_bbox(page)
        text_rect = merge_bounding_boxes([line.rect for line in lines]) if lines else page_rect

        return PageContext(
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

    def classify_page(self, page: pymupdf.Page, page_number: int, language: str) -> PageAnalysis:
        """Classifies single pages into available PageClasses (Text, Boreprofile, Map, Title Page or Unknown).

        Args:
            page: page that get classified
            page_number: page number in report (starting with 1)
            language: language of page content

        Returns:
            PageAnalysis object with page classification.
        """
        analysis = PageAnalysis(page_number)

        def ctx_builder():
            return self.build_full_context(page=page, page_number=page_number, language=language)

        page_class = self.classifier.determine_class(page=page, page_number=page_number, context_builder=ctx_builder)

        analysis.set_class(page_class)
        return analysis

    def process(self, file_path: Path) -> dict:
        """Process each page of a PDF file, returning classification and metadata.

        Args:
            file_path: Path to the PDF file to be processed.

        Returns:
            A dictionary containing the filename, metadata, and a list of classified pages and their metadata.
        """
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            logging.error(f"Invalid file path: {file_path}. Must be a valid PDF file.")
            return {}

        pages = []
        language_scores = defaultdict(float)
        long_page_counts = defaultdict(int)

        with pymupdf.Document(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                clean_text, word_count = extract_cleaned_text(page)
                is_frontpage = is_belegblatt(page.get_text())
                language_prediction = predict_language(clean_text)

                metadata_language = select_metadata_language(
                    predictions=language_prediction,
                    word_count=word_count,
                    is_frontpage=is_frontpage,
                    page_number=page_number,
                    scores=language_scores,
                    long_counts=long_page_counts,
                )

                classification_language = select_classification_language(language_prediction, word_count)
                classification = self.classify_page(page, page_number, classification_language)

                pages.append(
                    {
                        "page": page_number,
                        "classification": classification.to_classification_dict(),
                        "metadata": {"language": metadata_language, "is_frontpage": is_frontpage},
                    }
                )

        metadata = summarize_language_metadata(language_scores, long_page_counts, len(pages))

        return {"filename": file_path.name, "metadata": metadata, "pages": pages}

    def process_batch(self, pdf_files: list[Path]) -> list[dict]:
        """Process a batch of PDF files and return their classifications and metadata."""
        results = []
        with tqdm(total=len(pdf_files)) as pbar:
            for pdf in pdf_files:
                pbar.set_description(f"Processing {pdf.name}")
                result = self.process(pdf)
                if result:
                    results.append(result)
                pbar.update(1)
        return results
