import logging
import re
from typing import Optional

from .bounding_box import is_line_below_box
from .page_structure import PageContext
from .text_objects import TextLine
logger = logging.getLogger(__name__)

FIGURE_PATTERNS = re.compile(
    r"^(?:"                                               # Start of line + non-capturing group
    r"(?:fig(?:ure)?|abb(?:ildung)?|tab(?:le)?)\.?\s*[:.]?\s*"
    r")?"                                                        # Optional label
    r"\d{1,2}(?:[.:]\d{1,2}){0,3}"                               # Number + optional decimal/subsection
    r"\b",                                                       # Word boundary after pattern
    flags=re.IGNORECASE
)

DATE_PATTERNS = [
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",  # e.g. January 2000
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",  # e.g. 01.02.2001 or 1-2-01
    r"\b(19[0-9]{2}|20[0-1][0-9]|202[0-5])\b"  # 4-digit year
]

PHONE_PATTERNS = [r"\b(?:tel\.?|telefon)\s*[:\-]?\s*\+?\d[\d\s/().-]{8,}\b",
                  r"\b(?:0041|\+41|0)[\s]?\d{2}[\s]?\d{3}[\s]?\d{2}[\s]?\d{2}\b"
                  ]


def find_pattern(line: TextLine, patterns: list[str]) -> Optional[str]:
    """
        Searches for a match of any given regex pattern in the text of a line.

        Args:
            line: A TextLine object with a .line_text() method.
            patterns: List of regex strings to search for.

        Returns:
            The first matching string if found, otherwise None.
        """
    text = line.line_text().lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
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
        if FIGURE_PATTERNS.match(line_text):
            figure_description_lines.append(line)

    return figure_description_lines