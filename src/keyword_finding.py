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

def find_keywords_in_lines(text_lines: list[TextLine],keywords : list[str]):
    found_keywords =[]

    for line in text_lines:
        for word in line.words:
            matched_keyword =find_keyword(word, keywords)
            if matched_keyword:
                found_keywords.append({"key": matched_keyword,
                                       "word":word, 
                                       "line": line})
    
    return found_keywords


def is_close_to_image(line_rect, image_rect):
    image_y0, image_y1 = image_rect[1], image_rect[3]
    return (
            abs(line_rect.y1 - image_y0) < 20 or  # directly above
            abs(line_rect.y0 - image_y1) < 20  # directly below
    )

def find_figure_description(ctx):

    figure_patterns = [r"\b\d{1,2}(?:\.\d{1,2}){0,3}\b"]

    potential_lines = ctx.lines.copy()
    relevant_lines = []

    for line in potential_lines:
        for image in ctx.images:
            if is_close_to_image(line.rect, image["bbox"]):
                potential_lines.remove(line)
                relevant_lines.append(line)
                break

    figure_description_lines = []

    for line in relevant_lines:
        line_text = line.line_text()
        for pattern in figure_patterns:
            if re.search(pattern, line_text):
                logger.info(f"Matched figure pattern in line: {line_text}")
                figure_description_lines.append(line_text)
                break

    return figure_description_lines