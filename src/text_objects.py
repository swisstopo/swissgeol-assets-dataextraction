"""
Most of the code is copied from:
- the swissgeol-ocr repo (https://github.com/swisstopo/swissgeol-ocr)
- the swissgeol-boreholes-dataextraction repo (https://github.com/swisstopo/swissgeol-boreholes-dataextraction)
"""
import pymupdf
from collections import defaultdict
from typing import Callable, TypeVar

from .bounding_box import merge_bounding_boxes

T = TypeVar('T')

class TextWord:

    def __init__(self, rect: pymupdf.Rect, text: str, page: int):
        self.rect = rect
        self.text = text
        self.page_number = page

    def __repr__(self) -> str:
        return f"TextWord({self.rect}, {self.text})"


def extract_words(page, page_number):
    words= []
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        text_word = TextWord(rect=rect, text=word, page=page_number)
        words.append(text_word)
    return words        

class TextLine:

    def __init__(self, words: list[TextWord]):
        
        if not words:
            raise ValueError("Cannot create an empty TextLine.")

        self.rect = words[0].rect
        for word in words[1:]:
            self.rect.include_rect(word.rect)
        self.words = words
        self.page_number = words[0].page_number
        self.font_size = self.compute_font_size()

    
    def __repr__(self) -> str:
        return f"TextLine({self.rect},{self.line_text()})"

    def line_text(self):
        return ' '.join([word.text for word in self.words])
    
    def compute_font_size(self):
        return abs(self.rect.y1 - self.rect.y0)

def create_text_lines(page, page_number) -> list[TextLine]:

    words =[]
    words_by_line = defaultdict(list)

    for x0, y0, x1, y1, word, block_no, line_no, _word_no in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        text_word = TextWord(rect=rect, text=word, page=page_number)
        words.append(text_word)


        key = f"{block_no}_{line_no}"
        words_by_line[key].append(text_word)

    text_lines = [TextLine(words) for words in words_by_line.values() if words]
    return merge_text_lines(text_lines)


def merge_text_lines(naive_lines: list[TextLine]) -> list[TextLine]:
    """
    Merges raw lines into logical lines if PyMuPDF splits them unnecessarily.
    """
    merged_lines = []
    current_words = []

    for naive_line in naive_lines:
        for word in naive_line.words:
            if current_words:
                previous_word = current_words[-1]
                if not is_same_line(word, previous_word):
                    merged_lines.append(TextLine(current_words))
                    current_words = []

            current_words.append(word)

    if current_words:
        merged_lines.append(TextLine(current_words))

    return merged_lines


def is_same_line(previous_word: TextWord, current_word: TextWord) -> bool:
    """
    Determines whether two words belong to the same line based on their y-coordinates.
    """
    return abs(previous_word.rect.y0 - current_word.rect.y0) <= 2.0


class TextBlock:
    def __init__(self, lines: list[TextLine]):

        self.lines = lines
        self.rect = merge_bounding_boxes([line.rect for line in self.lines])

def overlaps(line, line2) -> bool:
    vertical_margin = 15
    ref_rect = pymupdf.Rect(line.rect.x0, line.rect.y0 - vertical_margin, line.rect.x1, line.rect.y1 + vertical_margin)
    return ref_rect.intersects(line2.rect)


def adjacent_lines(lines: list[TextLine]) -> list[set[int]]:
    result = [set() for _ in lines]
    for index, line in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index2 > index:
                if overlaps(line, line2):
                    result[index].add(index2)
                    result[index2].add(index)
    return result


def apply_transitive_closure(data: list[set[int]]) -> bool:
    found_new_relation = False
    for index, adjacent_indices in enumerate(data):
        new_adjacent_indices = set()
        for adjacent_index in adjacent_indices:
            new_adjacent_indices.update(
                new_index
                for new_index in data[adjacent_index]
                if new_index not in data[index]
            )

        for new_adjacent_index in new_adjacent_indices:
            data[index].add(new_adjacent_index)
            data[new_adjacent_index].add(index)
            found_new_relation = True
    return found_new_relation


def create_text_blocks(text_lines: list[TextLine]) -> list[TextBlock]:
    """sort lines into TextBlocks"""
    data = adjacent_lines(text_lines)

    while apply_transitive_closure(data):
        pass

    blocks: list[TextBlock] = []
    remaining_indices = {index for index, _ in enumerate(data)}
    for index, adjacent_indices in enumerate(data):
        if index in remaining_indices:
            selected_indices = adjacent_indices
            selected_indices.add(index)
            blocks.append(TextBlock(
                [text_lines[selected_index] for selected_index in sorted(list(selected_indices))]
            ))
            remaining_indices.difference_update(selected_indices)

    return blocks

def cluster_text_elements(
        elements: list[T],
        key_fn: Callable[[T], float],
        tolerance: int = 10
) -> list[list[T]]:
    """Cluster text elements based on coordinates of bounding box.

    Args:
        elements: List of object containing a `rect` attribute
        key_fn: Function that extracts a float from each element (e.g. lambda obj: obj.rect.y0)
        tolerance: max allowed difference between entries and a cluster key
    """

    if not elements:
        return []

    # Dictionary to hold clusters, keys are representative attribute values
    grouped = defaultdict(list)

    for element in elements:
        attribute = key_fn(element)
        matched_key = None

        # Check if attribute is within tolerance of an existing cluster
        for existing_key in grouped:
            if abs(existing_key - attribute) <= tolerance:
                matched_key = existing_key
                break

        # Add to an existing cluster or create a new one
        if matched_key is not None:
            grouped[matched_key].append(element)
        else:
            grouped[attribute].append(element)

    clusters = list(grouped.values())

    return clusters