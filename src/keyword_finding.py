import regex
import logging
import re

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


def is_aligned_below(line_rect, image_rect):

    image_x0, image_x1 = image_rect[0], image_rect[2]
    image_y0, image_y1 = image_rect[1], image_rect[3]
    image_width  = image_x1 - image_x0
    image_height = image_y1 - image_x0

    # Line must be below image TODO: check why for some images y1 is off...
    if line_rect.y0 < image_y1 or abs(line_rect.y0 - image_y1) > image_height * 0.25:
        return False

    max_offset = image_width * 0.1
    left_within = line_rect.x0 >= image_x0 - max_offset
    right_within = line_rect.x1 <= image_x1 + max_offset

    return left_within and right_within

def find_figure_description(ctx):

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

        for image in ctx.images:
            if is_aligned_below(line.rect, image["bbox"]):
                relevant_lines.append(line)
                added_lines.add(id(line))
                break

    figure_description_lines = []
    for line in relevant_lines:
        line_text = line.line_text()
        if figure_pattern.search(line_text):
            logger.info(f"Matched figure pattern: {line_text}")
            figure_description_lines.append(line)

    return figure_description_lines