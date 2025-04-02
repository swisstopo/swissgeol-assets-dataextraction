import numpy as np
import pymupdf
from dataclasses import dataclass
from .text import TextLine, TextWord, TextBlock

@dataclass()
class PageFeatures:
    lines: list[TextLine]
    words: list[TextWord]
    text_blocks: list[TextBlock]
    language: str
    page_rect: pymupdf.Rect

class PageAnalysis:
    def __init__(self, page_number: int):
        self.page_number = page_number
        self.classification = {
            "Boreprofile": 0,
            "Maps": 0,
            "Text": 0,
            "Title_Page": 0,
            "Unknown": 0
        }
        self.features = {}

    def set_class(self, label: str):
        for key in self.classification:
            if key == label:
                self.classification[key] = 1
            else:
                self.classification[key] = 0

    def to_dict(self):
        return {
            "Page": self.page_number,
            "Classification": self.classification,
            "Features": self.features
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
