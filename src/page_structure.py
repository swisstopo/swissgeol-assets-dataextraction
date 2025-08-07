from dataclasses import dataclass

import pymupdf

from src.geometric_objects import Line
from src.page_classes import PageClasses
from src.text_objects import TextBlock, TextLine, TextWord


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
        return {cls.value: val for cls, val in self.classification.items()}
