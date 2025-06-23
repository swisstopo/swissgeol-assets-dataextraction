import numpy as np
import pymupdf
from dataclasses import dataclass

from .page_classes import PageClasses
from .text_objects import TextLine, TextWord, TextBlock
from .geometric_objects import Line

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

class PageAnalysis:
    """Stores the classification result for a single page."""

    def __init__(self, page_number: int):
        self.page_number = page_number
        self.classification: dict[PageClasses, int] = {cls: 0 for cls in PageClasses}

    def set_class(self, label: PageClasses):
        self.classification[label] = 1

    def to_classification_dict(self):
        """Only exports classification and page number to dict"""
        return { "Page": self.page_number,
        **{cls.value: val for cls, val in self.classification.items()}
                 }