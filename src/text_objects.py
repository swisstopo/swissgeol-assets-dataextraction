"""Most of the code is copied.

From:
- the swissgeol-ocr repo (https://github.com/swisstopo/swissgeol-ocr)
- the swissgeol-boreholes-dataextraction repo (https://github.com/swisstopo/swissgeol-boreholes-dataextraction)
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import pymupdf

from src.bounding_box import _y_center, merge_bounding_boxes

T = TypeVar("T")


class TextWord:
    """Represents a word in a PDF document with its bounding box and text content."""

    def __init__(self, rect: pymupdf.Rect, text: str, page: int):
        self.rect = rect
        self.text = text
        self.page_number = page

    def __repr__(self) -> str:
        return f"TextWord({self.text},{self.rect},)"


def extract_words(page, page_number):
    words = []
    for x0, y0, x1, y1, word, _block_no, _line_no, _word_no in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        text_word = TextWord(rect=rect, text=word, page=page_number)
        words.append(text_word)
    return words


class TextLine:
    """Represents a line of text composed of multiple words."""

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
        return " ".join([word.text for word in self.words])

    def compute_font_size(self):
        return abs(self.rect.y1 - self.rect.y0)


def create_text_lines(page, page_number) -> list[TextLine]:
    words = []
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
    """Merges raw lines into logical lines if PyMuPDF splits them unnecessarily."""
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
    """Determines whether two words belong to the same line based on their y-coordinates."""
    return abs(previous_word.rect.y0 - current_word.rect.y0) <= 2.0


class TextBlock:
    """Represents a block of text composed of multiple lines."""

    def __init__(self, lines: list[TextLine]):
        self.lines = lines
        self.rect = merge_bounding_boxes([line.rect for line in self.lines])


def overlaps(line, line2) -> bool:
    vertical_margin = 15
    ref_rect = pymupdf.Rect(
        line.rect.x0,
        line.rect.y0 - vertical_margin,
        line.rect.x1,
        line.rect.y1 + vertical_margin,
    )
    return ref_rect.intersects(line2.rect)


def adjacent_lines(lines: list[TextLine]) -> list[set[int]]:
    result = [set() for _ in lines]
    for index, line in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index2 > index and overlaps(line, line2):
                result[index].add(index2)
                result[index2].add(index)
    return result


def apply_transitive_closure(data: list[set[int]]) -> bool:
    found_new_relation = False
    for index, adjacent_indices in enumerate(data):
        new_adjacent_indices = set()
        for adjacent_index in adjacent_indices:
            new_adjacent_indices.update(
                new_index for new_index in data[adjacent_index] if new_index not in data[index]
            )

        for new_adjacent_index in new_adjacent_indices:
            data[index].add(new_adjacent_index)
            data[new_adjacent_index].add(index)
            found_new_relation = True
    return found_new_relation


def create_text_blocks(text_lines: list[TextLine]) -> list[TextBlock]:
    """Sort TextLines into TextBlocks."""
    data = adjacent_lines(text_lines)

    while apply_transitive_closure(data):
        pass

    blocks: list[TextBlock] = []
    remaining_indices = {index for index, _ in enumerate(data)}
    for index, adjacent_indices in enumerate(data):
        if index in remaining_indices:
            selected_indices = adjacent_indices
            selected_indices.add(index)
            blocks.append(TextBlock([text_lines[selected_index] for selected_index in sorted(list(selected_indices))]))
            remaining_indices.difference_update(selected_indices)

    return blocks


def cluster_text_elements(elements: list[T], key_fn: Callable[[T], float], tolerance: float = 10.0) -> list[list[T]]:
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


def cluster_connected_components(items: list[T], is_connected: Callable[[T, T], bool]) -> list[list[T]]:
    """Generic BFS clustering of items into connected components.

    Each item is connected to others if `is_connected(a, b)` returns True.
    Items that can be reached transitively form one cluster.

    Args:
        items: List of objects to cluster.
        is_connected: Predicate that decides whether two items are connected.

    Returns:
        List of clusters, where each cluster is a list of connected items.
    """
    n = len(items)
    visited = [False] * n
    components: list[list[T]] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        component = [items[i]]
        while queue:
            u = queue.pop()
            for v in range(n):
                if visited[v] or v == u:
                    continue
                if is_connected(items[u], items[v]):
                    visited[v] = True
                    queue.append(v)
                    component.append(items[v])
        components.append(component)

    return components


@dataclass
class TextColumn:
    """A vertical column of text, composed of multiple TextWords."""

    words: list[TextWord]

    @property
    def rect(self):
        return merge_bounding_boxes([w.rect for w in self.words])

    def __repr__(self):
        return f"TextColumn({[word.text for word in self.words]},{self.rect},)"

    def noise(self, all_words) -> float:
        """Metric that shows how noisy a column is.

        It's the ratio between actual column entries and non-column words overlapping with the column bounding box.
        Best noise = 1 (intersecting words = self.words). More intersecting words will lead to a higher ratio.
        """
        column_bbox = self.rect
        intersecting_words = [word for word in all_words if column_bbox.intersects(word.rect)]
        ratio = len(intersecting_words) / len(self.words)
        return ratio


@dataclass
class TextTable:
    """A table composed of multiple aligned TextColumns."""

    columns: list[TextColumn]
    words: list[TextWord] = field(init=False)

    def __post_init__(self):
        self.words = [word for col in self.columns for word in col.words]

    @property
    def rect(self) -> pymupdf.Rect:
        """Computes bounding box of text table."""
        return merge_bounding_boxes([c.rect for c in self.columns if c.rect is not None])

    def height_coverage(self, page_height: float) -> float:
        """Fraction of page height covered by text tables bounding box."""
        return self.rect.height / page_height

    def text_coverage(self, all_words: list[TextWord]) -> float:
        """Fraction of words belonging to the table relative to all words on the page."""
        if not all_words:
            return 0.0
        coverage = sum(len(col.words) for col in self.columns) / len(all_words)
        return coverage

    @property
    def confidence(self):
        """Confidence based on row alignment across columns.

        Steps:
        - Collect row centers from all columns.
        - Merge centers within a tolerance (rows aligning across columns).
        - Confidence = 1 -  (#merged rows) / (total entries).
        """
        if not self.columns:
            return 0.0

        def _same_row(w1: TextWord, w2: TextWord, tolerance: float = 3.0):
            w1_center, w2_center = _y_center(w1.rect), _y_center(w2.rect)
            return abs(w1_center - w2_center) <= tolerance

        merged_rows = cluster_connected_components(self.words, _same_row)

        total_entries = len(self.words)
        if total_entries == 0:
            return 0.0

        # fewer merged rows relative to total entries -> better alignment
        merged_count = len(merged_rows)
        q_rows = 1.0 - (merged_count / total_entries)

        return q_rows
