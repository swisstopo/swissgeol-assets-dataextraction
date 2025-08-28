"""Very similar to code in swissgeol-boreholes-dataextraction repo.

https://github.com/swisstopo/swissgeol-boreholes-dataextraction
The same including and excluding matching parameters used.
"""

import pymupdf

from src.text_objects import TextLine, TextWord, cluster_text_elements
from src.utils import is_description


class MaterialDescription:
    """Stores information about material description block."""

    def __init__(
        self, text_lines: list[TextLine], all_words: list[TextWord]
    ):  ## maybe possible only use words or lines here...
        self.text_lines = text_lines
        self.rect = self.compute_bbox()
        self.noise = self.compute_noise(all_words)

    def __repr__(self):
        return (
            f"MaterialDescription( lines = {[line.line_text() for line in self.text_lines]}, "
            f"rect = {self.rect}, noise={self.noise} )"
        )

    def compute_bbox(self):
        """Calculate the bounding box of the material description."""
        start_line = min(self.text_lines, key=lambda line: line.rect.y0)
        end_line = max(self.text_lines, key=lambda line: line.rect.y1)

        return pymupdf.Rect(start_line.rect.x0, start_line.rect.y0, end_line.rect.x1, end_line.rect.y1)

    def compute_noise(self, all_words: list[TextWord]):
        """For bounding box compute noise of words not containing to bbox entries."""
        description_words = [word for line in self.text_lines for word in line.words]
        noise_words = [word for word in all_words if self.rect.contains(word.rect) and word not in description_words]

        return len(noise_words) / len(description_words) if description_words else float("inf")

    def is_valid(self):
        return self.noise < 1.5 and len(self.text_lines) < 3


def detect_material_description(
    lines: list[TextLine], words: list[TextWord], material_description: dict
) -> list[MaterialDescription]:
    """Detects material descriptions in Textlines and returns List of MaterialDescriptions."""
    material_lines = [line for line in lines if is_description(line, material_description)]

    line_clusters = cluster_text_elements(material_lines, key_fn=lambda line: line.rect.x0)

    if not line_clusters:
        return []

    descriptions = [MaterialDescription(cluster, words) for cluster in line_clusters]
    return descriptions
