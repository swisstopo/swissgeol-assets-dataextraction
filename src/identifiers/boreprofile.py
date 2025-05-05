import logging
import re
import pymupdf
from dataclasses import dataclass

from ..text_objects import TextWord
from ..page_structure import PageContext
from ..material_description import detect_material_description
from ..utils import cluster_text_elements

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
import logging
import re
logger = logging.getLogger(__name__)

def identify_boreprofile(ctx: PageContext, matching_params) -> bool:
    """
    Determines whether a page contains a boreprofile.

    A boreprofile is detected if:
    - At least one valid material description is present
    - The description text accounts for a significant portion of the page (>= 30% with boosts)
    Boosts:
    - +0.2 if a valid sidebar is found
    - +0.1 if boreprofile-related keywords are present
    """
    if ctx.is_digital and not (ctx.drawings or ctx.images):
        return False

    descriptions = detect_material_description(
        ctx.lines, ctx.words,
        matching_params["material_description"].get(ctx.language, {})
    )

    if ctx.is_digital and ctx.images:
        return True if keywords_in_figure_description(ctx, matching_params) else False

    material_descriptions = detect_material_description(ctx.lines, ctx.words, matching_params["material_description"].get(ctx.language, {}))

    # Find sidebars
    sidebars = create_sidebars(ctx.words)
    has_sidebar = bool(sidebars)

    valid_descriptions = [
        desc for desc in descriptions if desc.is_valid(ctx.page_rect, sidebars)
    ]
    if not valid_descriptions:
        return False

    # Calculate ratio between material description and total page words
    total_words = sum(1 for word in ctx.words if word.text.isalpha())
    material_words = sum(
        len(line.words)
        for desc in valid_descriptions
        for line in desc.text_lines
    )
    ratio = material_words / total_words if total_words else 0.0

    # Keyword match
    keyword_set = matching_params["boreprofile"].get(ctx.language, {})
    has_keyword = any(word.text.lower() in keyword_set for word in ctx.words)

    # Apply boosts
    ratio += 0.2 if has_sidebar else 0.0
    ratio += 0.1 if has_keyword else 0.0

    return ratio > 0.3

def keywords_in_figure_description(ctx,matching_params):

    figure_patterns = [r"\b\d{1,2}(?:\.\d{1,2}){0,3}\b"]

    boreprofile_keywords = matching_params["boreprofile"].get(ctx.language, {})
    relevant_lines = []

    def is_close_to_image(line_rect, image_rect):
        image_y0, image_y1 = image_rect[1], image_rect[3]
        return (
                abs(line_rect.y1 - image_y0) < 20 or  # directly above
                abs(line_rect.y0 - image_y1) < 20  # directly below
        )
    for line in ctx.lines:
        for image in ctx.images:
            if is_close_to_image(line.rect, image["bbox"]):
                relevant_lines.append(line)

    figure_description_lines = []

    for line in ctx.lines:
        line_text = line.line_text()
        for pattern in figure_patterns:
            if re.search(pattern, line_text):
                logger.info(f"Matched figure pattern in line: {line_text}")
                figure_description_lines.append(line_text.lower())
                break

    boreprofile_lines = [
        line for line in figure_description_lines
        if any(keyword in line.lower() for keyword in boreprofile_keywords)
    ]

    logger.info(boreprofile_lines)
    return boreprofile_lines