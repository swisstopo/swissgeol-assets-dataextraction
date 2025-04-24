from ..text import TextWord
from ..page_structure import PageContext
from ..material_description import detect_material_description
import logging
import re
from ..utils import cluster_text_elements
from dataclasses import dataclass
import pymupdf

@dataclass
class Entry:
    rect: pymupdf.Rect
    value: float

    def __repr__(self):
        return f"{self.value}"

logger = logging.getLogger(__name__)

def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of a valid material description in given language"""
    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    if ctx.geometric_lines:
        long_geometric_lines = [line for line in ctx.geometric_lines if line.length > ctx.page_rect.height / 3]
    else:
        long_geometric_lines = []

    sidebar_columns = create_sidebar_columns(ctx.words)
    sidebar_columns_sorted = sorted(sidebar_columns, key=len, reverse=True)
    logger.info(sidebar_columns_sorted)

    return any(description.is_valid(ctx.page_rect, long_geometric_lines) for description in material_descriptions)

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