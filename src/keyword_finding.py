import regex
import logging
import re
import pymupdf

from .page_structure import PageContext
from .text_objects import TextWord, TextLine
logger = logging.getLogger(__name__)


def find_keyword(word: TextWord, keywords: list[str]) -> TextWord:
    for keyword in keywords:
        pattern = regex.compile(r"(\b" + regex.escape(keyword) + r"\b)", flags=regex.IGNORECASE)
        match = pattern.search(word.text)
        if match:
            return match.group(1)
    return None

def find_keywords_in_lines(text_lines: list[TextLine], keywords : list[str]):
    found_keywords =[]

    for line in text_lines:
        for word in line.words:
            matched_keyword =find_keyword(word, keywords)
            if matched_keyword:
                found_keywords.append({"key": matched_keyword,
                                       "word":word, 
                                       "line": line})
    
    return found_keywords


def is_aligned_below(line_rect: pymupdf.Rect, image_rect: pymupdf.Rect) -> bool:
    """
      Determines whether a text line is directly below an image and horizontally aligned.
      Args:
          line_rect (pymupdf.Rect): Bounding box of the text line.
          image_rect (pymupdf.Rect): Bounding box of the image (transformed according to page rotation).
      Returns:
          bool: True if the line is well aligned else False
      """
    if image_rect.y1 - line_rect.y0 > image_rect.height * 0.25:
        return False

    max_offset = image_rect.width * 0.2
    left_within = line_rect.x0 >= image_rect.x0 - max_offset
    right_within = line_rect.x1 <= image_rect.y1 + max_offset

    return left_within and right_within

def find_figure_description(ctx:PageContext) -> list[TextLine]:
    """
       Identifies lines near images that likely contain figure, table, or illustration captions,
        based on if line appears below any image and if it matches known figure/table patterns.
       Args:
           ctx (PageContext): The page context containing text lines and images.
       Returns:
           list[TextLine]: A list of lines matching the caption criteria.
       """
    figure_pattern = re.compile(
        r"^(?:"                                               # Start of line + non-capturing group
        r"(?:fig(?:ure)?|abb(?:ildung)?|tab(?:le)?)\.?\s*[:.]?\s*"
        r")?"                                                        # Optional label
        r"\d{1,2}(?:[.:]\d{1,2}){0,3}"                               # Number + optional decimal/subsection
        r"\b",                                                       # Word boundary after pattern
        flags=re.IGNORECASE
    )

    relevant_lines = []
    added_lines = set()

    for line in ctx.lines:
        if id(line) in added_lines:
            continue

        for image_rect in ctx.image_rects:
            if is_aligned_below(line.rect, image_rect.rect):
                relevant_lines.append(line)
                added_lines.add(id(line))
                break

    figure_description_lines = []
    for line in relevant_lines:
        line_text = line.line_text()
        if figure_pattern.match(line_text):
            logger.info(f"Matched figure pattern: {line_text}")
            figure_description_lines.append(line)

    return figure_description_lines