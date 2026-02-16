import logging
import re
from dataclasses import dataclass

import pymupdf
from swissgeol_doc_processing.text.textline import TextWord

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    """Represents a single entry for diagram detection."""

    rect: pymupdf.Rect
    value: float

    def __repr__(self):
        return f"{self.value}"


def detect_entries(words: list[TextWord]) -> list[Entry]:
    regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]*)?)[müMN\\.]*$")
    entries = []
    for word in sorted(words, key=lambda w: w.rect.y0):
        cleaned = word.text.strip().replace(",", ".")
        match = regex.match(cleaned)
        if match:
            try:
                entries.append(Entry(word.rect, float(match.group(1))))
            except ValueError:
                continue
    return entries


def is_mostly_increasing(entries: list[Entry], tolerance: float = 0.2) -> bool:
    """Check if a sequence is mostly increasing.

    Args:
        entries: list of Entry objects with a .value attribute.
        tolerance: fraction of allowed violations (default: 0.2 = 20%).

    Returns:
        True if the sequence is increasing within tolerance.
    """
    values = [e.value for e in entries]
    violations = sum(values[i] >= values[i + 1] for i in range(len(values) - 1))
    return violations <= tolerance * (len(values) - 1)
