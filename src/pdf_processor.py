import logging
import math
from collections import defaultdict
from pathlib import Path

import pymupdf
from tqdm import tqdm

from src.language_detection.pages_to_ignore import is_belegblatt
from src.bounding_box import get_page_bbox, merge_bounding_boxes
from src.language_detection.detect_language import select_language, extract_cleaned_text, predict_language
from src.page_graphics import extract_page_graphics
from src.page_structure import PageAnalysis, PageContext
from src.text_objects import create_text_blocks, create_text_lines, extract_words
from src.utils import is_digitally_born

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, classifier):
        self.classifier = classifier

    def classify_page(
        self,
        page: pymupdf.Page,
        page_number: int,
        language: str,
    ) -> PageAnalysis:
        """classifies single pages into Text-, Boreprofile-, Map-, Title- or Unknown Page.
        Args:
            page: page that get classified
            page_number: page number in report
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

        page_class = self.classifier.determine_class(page=page, context=context, page_number=page_number)
        analysis.set_class(page_class)

        return analysis

    def process(self, file_path: Path) -> dict:
        """
        Process each page of a PDF file, returning classification and metadata.
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

                metadata_language = self._update_language_score(
                    language_prediction, word_count, is_frontpage, page_number, language_scores, long_page_counts
                )

                classification_language = select_language(language_prediction, word_count)
                classification = self.classify_page(page, page_number, classification_language)

                pages.append(
                    {
                        "page": page_number,
                        "classification": classification.to_classification_dict(),
                        "metadata": {"language": metadata_language, "is_frontpage": is_frontpage},
                    }
                )

        metadata = self._summarize_language_metadata(language_scores, long_page_counts, len(pages))

        return {"filename": file_path.name, "metadata": metadata, "pages": pages}

    def process_batch(self, pdf_files: list[Path]) -> list[dict]:
        results = []
        with tqdm(total=len(pdf_files)) as pbar:
            for pdf in pdf_files:
                pbar.set_description(f"Processing {pdf.name}")
                result = self.process(pdf)
                if result:
                    results.append(result)
                pbar.update(1)
        return results

    @staticmethod
    def _update_language_score(
        predictions, word_count, is_frontpage, page_number, scores: dict, long_counts: dict
    ) -> str | None:
        metadata_language = select_language(predictions, word_count, mode="metadata")
        if metadata_language and not is_frontpage:
            if word_count > 0:
                scores[metadata_language] += math.log(word_count) / page_number
            if word_count > 50:
                long_counts[metadata_language] += 1
        return metadata_language

    @staticmethod
    def _summarize_language_metadata(scores: dict[str, float], long_counts: dict[str, int], page_count: int) -> dict:
        if scores:
            best = max(scores, key=scores.get)
            languages = [best] + [lang for lang, count in long_counts.items() if count >= 2 and lang != best]
        else:
            languages = []

        return {"page_count": page_count, "languages": languages}
