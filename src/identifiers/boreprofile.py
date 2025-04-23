from ..text import TextWord
from ..page_structure import PageContext
from ..material_description import detect_material_description
import logging
import re
from ..utils import cluster_text_elements
from dataclasses import dataclass
import pymupdf

logger = logging.getLogger(__name__)

def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """Identifies whether a page contains a boreprofile based on presence of a valid material description in given language"""
    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    if ctx.geometric_lines:
        long_geometric_lines = [line for line in ctx.geometric_lines if line.length > ctx.page_rect.height / 3]
    else:
        long_geometric_lines = []

    sidebar_entries = detect_entries(ctx.words)
    logger.info([entry[1] for entry in sidebar_entries])
    #sidebar_columns = create_sidebar_columns(sidebar_entries)

    return any(description.is_valid(ctx.page_rect, long_geometric_lines) for description in material_descriptions)

def detect_entries(words: list[TextWord]) -> list[tuple]:
    """identifies potential entries"""
    entries = []
    regex = re.compile(r"^-?\.?([0-9]+(\.[0-9]*)?)[müMN\\.]*$")
    for word in sorted(words, key=lambda word: word.rect.y0):
        try:
            input_string = word.text.strip().replace(",", ".")
            match = regex.match(input_string)
            if match:
                entries.append((word.rect, match.group(1)))
        except ValueError:
            pass
    return entries

# def create_sidebar_columns(entries):
#     clusters =  cluster_text_elements(entries, key_fn = lambda entries:entries[0])

