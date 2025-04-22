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
    geometric_lines: list[Line]
    is_digital: bool
    drawings: list
    images: list

class PageAnalysis:
    """Stores the classification result and associated features for a single page."""

    def __init__(self, page_number: int):
        self.page_number = page_number
        self.classification: dict[PageClasses, int] = {cls: 0 for cls in PageClasses}
        self.features = {}

    def set_class(self, label: PageClasses):
        self.classification[label] = 1

    def to_classification_dict(self):
        """Only exports classification and page number to dict"""
        return { "Page": self.page_number,
        **{cls.value: val for cls, val in self.classification.items()}
                 }

def compute_text_features(lines, text_blocks) -> dict:
    words_per_line = [len(line.words) for line in lines]
    mean_words_per_line = np.mean(words_per_line) if words_per_line else 0

    block_area = sum(block.rect.get_area() for block in text_blocks)
    word_area = sum(
        word.rect.get_area()
        for block in text_blocks
        for line in block.lines
        for word in line.words if len(line.words) > 1
    )

    return {
        "mean_words_per_line": mean_words_per_line,
        "block_area": block_area,
        "word_area": word_area,
        "word_density": word_area / block_area if block_area else 0
    }
