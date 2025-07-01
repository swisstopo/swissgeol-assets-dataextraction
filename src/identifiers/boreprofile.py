import logging
import re
from dataclasses import dataclass

import pymupdf

from src.keyword_finding import find_figure_description
from src.material_description import detect_material_description
from src.page_structure import PageContext
from src.text_objects import TextWord, cluster_text_elements

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


def identify_boreprofile(ctx: PageContext, matching_params: dict) -> bool:
    """
    Determines whether a page contains a boreprofile.

    A boreprofile is detected if:
    - At least one valid material description is present
    - The description text accounts for a some portion of the page (>= 30% with boosts)
    Boosts:
    - +0.1 if a valid sidebar is found
    - +0.05 if boreprofile keywords are present
    """
    keywords = matching_params["material_description"].get(ctx.language, [])

    if not keywords:
        logger.warning(f"No keywords for language '{ctx.language}', falling back to 'de'")
        keywords = matching_params["material_description"].get("de", [])

    descriptions = detect_material_description(ctx.lines, ctx.words, keywords)

    # Find sidebars
    sidebars = create_sidebars(ctx.words)
    has_sidebar = bool(sidebars)

    valid_descriptions = [desc for desc in descriptions if desc.is_valid]
    if not valid_descriptions:
        return False

    # Calculate ratio between material description and total page words
    total_words = sum(1 for word in ctx.words if word.text.isalpha())
    material_words = sum(len(line.words) for desc in valid_descriptions for line in desc.text_lines)
    ratio = material_words / total_words if total_words else 0.0

    # Keyword match
    keyword_set = matching_params["boreprofile"].get(ctx.language, [])
    has_keyword = any(keyword in word.text.lower() for word in ctx.words for keyword in keyword_set)

    # Apply boosts
    ratio += 0.1 if has_sidebar else 0.0
    ratio += 0.05 if has_keyword else 0.0

    return ratio > 0.3


def keywords_in_figure_description(ctx: PageContext, matching_params) -> list[str]:
    caption_lines = find_figure_description(ctx)
    keyword_groups = (
        matching_params["caption_description"]["boreprofile"].get(ctx.language, []).get("must_contain", [])
    )

    if len(keyword_groups) < 2:
        logger.warning(
            f"Need 2 keyword groups (profile and borehole keywords) in figure_description.boreprofile, but got: {keyword_groups}"
        )
        return []

    matched_lines = []
    for line in caption_lines:
        text = line.line_text().lower()

        if all(any(keyword in text for keyword in group) for group in keyword_groups):
            matched_lines.append(line)

    return matched_lines
