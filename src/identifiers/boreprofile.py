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

def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of a valid material description in given language"""

    ## a) find material description, must exist for boreprofiles!
    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    ## b) find long lines, increases chance to be a boreprofile TODO: filter for only gridlike lines
    grid, non_grid = split_lines_by_orientation(ctx.geometric_lines)
    long_geometric_lines = [length for length in (grid or []) if length > ctx.page_rect.height /3]

    ## c) find longest sidebar -> increases chance to be boreprofile
    sidebar_columns = create_sidebar_columns(ctx.words)
    length_sidebar = len(sorted(sidebar_columns, key=len, reverse=True)[0]) if sidebar_columns else 0

    best_score = 0

    for description in material_descriptions:
        if not description.is_valid(ctx.page_rect, long_geometric_lines):
            continue

        num_lines = len(description.text_lines)
        noise = description.noise

        # Score components
        score = 1.0  # base for valid material description
        score += min(num_lines / 30, 1.0) * 0.5  # max 0.5 bonus
        score += min(length_sidebar / 30, 1.0) * 0.5  # max 0.5 bonus
        score += min(len(long_geometric_lines) / 10, 1.0) * 0.5  # max 0.5 bonus
        score += max(0, (1.75 - noise) / 1.75) * 0.5  # inverse noise, max 0.5 bonus

        logger.info({
            "score": round(score, 2),
            "lines": num_lines,
            "sidebar_len": length_sidebar,
            "long_lines": len(long_geometric_lines),
            "noise": round(noise, 2),
            "rect": description.rect
        })
        best_score = max(best_score, score)

    return best_score >= 1.5

def detect_entries(words: list[TextWord]) -> list[Entry]:
    """identifies potential entries"""
    entries = []
    regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]*)?)[müMN\\.]*$")
    for word in sorted(words, key=lambda word: word.rect.y0):
        try:
            input_string = word.text.strip().replace(",", ".")
            match = regex.match(input_string)
            if match:
                entries.append(Entry(word.rect, float(match.group(1))))
        except ValueError:
            pass
    return entries

def is_strictly_increasing(column) -> bool:
    return all(column[i].value < column[i + 1].value for i in range(len(column) - 1))


def create_sidebar_columns(words: list[TextWord]) -> list[Entry]:

    sidebar_entries = detect_entries(words)
    clusters =  cluster_text_elements(sidebar_entries, key_fn = lambda entries:entries.rect.x0, tolerance = 10)
    valid_sidebars =[]
    for cluster in clusters:
        if len(cluster) >= 3 and is_strictly_increasing(cluster):
            valid_sidebars.append(cluster)

    return valid_sidebars