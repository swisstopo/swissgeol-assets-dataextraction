"""Text clustering utilities for grouping text elements."""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

import pymupdf
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine, TextWord

T = TypeVar("T")


def cluster_text_elements(elements: list[T], key_fn: Callable[[T], float], tolerance: float = 10.0) -> list[list[T]]:
    """Cluster text elements based on coordinates of bounding box.

    Args:
        elements: List of objects containing a `rect` attribute
        key_fn: Function that extracts a float from each element (e.g. lambda obj: obj.rect.y0)
        tolerance: Max allowed difference between entries and a cluster key

    Returns:
        List of clusters, where each cluster is a list of elements

    Example:
        # Cluster lines by y-coordinate
        clusters = cluster_text_elements(lines, lambda line: line.rect.y0, tolerance=5.0)
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
        items: List of objects to cluster
        is_connected: Predicate that decides whether two items are connected

    Returns:
        List of clusters, where each cluster is a list of connected items

    Example:
        # Cluster lines that are vertically close
        def are_close(line1, line2):
            return abs(line1.rect.y0 - line2.rect.y0) < 10

        clusters = cluster_connected_components(lines, are_close)
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


def is_same_line(word1: TextWord, word2: TextWord) -> bool:
    """Determine whether two words belong to the same line based on y-coordinates.

    Args:
        word1: First word to compare
        word2: Second word to compare

    Returns:
        True if words are on the same line (y-coordinates within 2.0 pixels)
    """
    return abs(word1.rect.y0 - word2.rect.y0) <= 2.0


def overlaps(line: TextLine, line2: TextLine, vertical_margin: float = 15.0) -> bool:
    """Check if two text lines overlap vertically with a margin.

    Args:
        line: First text line
        line2: Second text line
        vertical_margin: Margin in pixels to add above and below each line

    Returns:
        True if lines overlap within the margin
    """
    ref_rect = pymupdf.Rect(
        line.rect.x0,
        line.rect.y0 - vertical_margin,
        line.rect.x1,
        line.rect.y1 + vertical_margin,
    )
    return ref_rect.intersects(line2.rect)


def adjacent_lines(lines: list[TextLine]) -> list[set[int]]:
    """Find adjacent lines that overlap vertically.

    Args:
        lines: List of text lines to analyze

    Returns:
        List of sets, where each set contains indices of adjacent lines
    """
    result = [set() for _ in lines]
    for index, line in enumerate(lines):
        for index2, line2 in enumerate(lines):
            if index2 > index and overlaps(line, line2):
                result[index].add(index2)
                result[index2].add(index)
    return result


def apply_transitive_closure(data: list[set[int]]) -> bool:
    """Apply transitive closure to adjacency relationships.

    If A is adjacent to B and B is adjacent to C, then A is adjacent to C.

    Args:
        data: List of sets representing adjacency relationships

    Returns:
        True if new relationships were found, False otherwise
    """
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
    """Sort TextLines into TextBlocks.

    Groups text lines into blocks based on vertical overlap and transitivity.
    Lines that overlap (with margin) are grouped together, and transitive
    relationships are applied to form complete blocks.

    Args:
        text_lines: List of text lines to group into blocks

    Returns:
        List of text blocks, where each block contains adjacent lines

    Note:
        Uses swissgeol_doc_processing TextBlock class.
    """
    data = adjacent_lines(text_lines)

    # Apply transitive closure until no new relationships found
    while apply_transitive_closure(data):
        pass

    blocks: list[TextBlock] = []
    remaining_indices = {index for index, _ in enumerate(data)}

    for index, adjacent_indices in enumerate(data):
        if index in remaining_indices:
            selected_indices = adjacent_indices
            selected_indices.add(index)

            # Get the selected lines sorted by index
            selected_lines = [text_lines[selected_index] for selected_index in sorted(list(selected_indices))]

            # TextBlock from swissgeol_doc_processing automatically computes rect and page_number from lines
            block = TextBlock(lines=selected_lines)
            blocks.append(block)

            remaining_indices.difference_update(selected_indices)

    return blocks
