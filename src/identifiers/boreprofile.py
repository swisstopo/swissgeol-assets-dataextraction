import logging
import re
import pymupdf
from dataclasses import dataclass

from ..text import TextWord
from ..page_structure import PageContext
from ..material_description import detect_material_description
from ..utils import cluster_text_elements
from .map import split_lines_by_orientation

logger = logging.getLogger(__name__)

@dataclass
class Entry:
    rect: pymupdf.Rect
    value: float

    def __repr__(self):
        return f"{self.value}"

def is_strictly_increasing(entries: list[Entry]) -> bool:
    return all(entries[i].value < entries[i + 1].value for i in range(len(entries) - 1))


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


def create_sidebars(words: list[TextWord]) -> list[list[Entry]]:
    """Create Sidebars from potential entries"""
    entries = detect_entries(words)
    clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.x0, tolerance=10)
    return [c for c in clusters if len(c) >= 3 and is_strictly_increasing(c)]


def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """
    Identifies whether a page contains a boreprofile.
    Requires a valid material description and boosts confidence with:
      - Sidebar length
      - Gridlike lines
      - Boreprofile-related keywords
    """
    descriptions = detect_material_description(
        ctx.lines, ctx.words,
        matching_params["material_description"].get(ctx.language, {})
    )

    grid_lines, _ = split_lines_by_orientation(ctx.geometric_lines)
    long_line_length = [length for length in (grid_lines or []) if length > ctx.page_rect.height / 3]

    sidebars = create_sidebars(ctx.words)
    longest_sidebar_len = len(max(sidebars, key=len)) if sidebars else 0

    keywords_found = any(
        w.text.lower() in matching_params["boreprofile"].get(ctx.language, {})
        for w in ctx.words
    )

    best_score = 0

    for desc in descriptions:
        if not desc.is_valid(ctx.page_rect, sidebar=sidebars):
            continue

        score = 1.0
        score += min(len(desc.text_lines) / 30, 1.0) * 0.5
        score += min(longest_sidebar_len / 30, 1.0) * 0.5
        score += min(len(long_line_length) / 10, 1.0) * 0.5
        score += max(0, (1.75 - desc.noise) / 1.75) * 0.5
        score += 0.1 if keywords_found else 0

        best_score = max(best_score, score)

    return best_score >= 1.5