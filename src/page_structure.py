from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pymupdf
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine, TextWord

from src.page_classes import PageClasses

if TYPE_CHECKING:
    from extraction.minimal_pipeline import ExtractionContext


@dataclass()
class PageContext:
    """Contains processed text content and information from a page."""

    lines: list[TextLine]
    words: list[TextWord]
    text_blocks: list[TextBlock]
    language: str
    page_rect: pymupdf.Rect
    text_rect: pymupdf.Rect
    geometric_lines: list[Line]
    is_digital: bool
    drawings: list
    image_rects: list
    color_proportion: Counter | None = None
    extraction_context: ExtractionContext | None = None


class PageAnalysis:
    """Stores the classification result for a single page."""

    def __init__(self, page_number: int):
        self.page_number = page_number
        self.classification: dict[PageClasses, int] = {cls: 0 for cls in PageClasses}

    def set_class(self, label: PageClasses):
        self.classification[label] = 1

    def to_classification_dict(self):
        """Only exports classification and page number to dict."""
        return {cls.value: val for cls, val in self.classification.items()}
