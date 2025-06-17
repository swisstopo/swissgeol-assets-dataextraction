import regex
import logging
import re

from .bounding_box import is_line_below_box
from .page_structure import PageContext
from .text_objects import TextWord, TextLine
logger = logging.getLogger(__name__)

figure_pattern = re.compile(
    r"^(?:"                                               # Start of line + non-capturing group
    r"(?:fig(?:ure)?|abb(?:ildung)?|tab(?:le)?)\.?\s*[:.]?\s*"
    r")?"                                                        # Optional label
    r"\d{1,2}(?:[.:]\d{1,2}){0,3}"                               # Number + optional decimal/subsection
    r"\b",                                                       # Word boundary after pattern
    flags=re.IGNORECASE
)

def find_keyword(word: TextWord, keywords: list[str]) -> TextWord:
    for keyword in keywords:
        pattern = regex.compile(r"(\b" + regex.escape(keyword) + r"\b)", flags=regex.IGNORECASE)
        match = pattern.search(word.text)
        if match:
            return match.group(1)
    return None

def find_figure_description(ctx:PageContext) -> list[TextLine]:
    """
       Identifies lines near images that likely contain figure, table, or illustration captions,
        based on if line appears below any image and if it matches known figure/table patterns.
       Args:
           ctx (PageContext): The page context containing text lines and images.
       Returns:
           list[TextLine]: A list of lines matching the caption criteria.
       """

    relevant_lines = []
    added_lines = set()

    for line in ctx.lines:
        if id(line) in added_lines:
            continue

        for image_rect in ctx.image_rects:
            if is_line_below_box(line.rect, image_rect.rect):
                relevant_lines.append(line)
                added_lines.add(id(line))
                break

    figure_description_lines = []
    for line in relevant_lines:
        line_text = line.line_text()
        if figure_pattern.match(line_text):
            figure_description_lines.append(line)

    return figure_description_lines