import pymupdf
from collections import defaultdict
from typing import Callable

from .text_objects import TextLine

def is_digitally_born(page: pymupdf.Page) -> bool:
    bboxes = page.get_bboxlog()

    for boxType, rectangle in bboxes:
        if (boxType == "fill-text" or boxType == "stroke-text") and not pymupdf.Rect(rectangle).is_empty:
            return True
    return False

def is_description(line: TextLine, matching_params: dict):
    """Check if the words in line matches with matching parameters."""
    line_text = line.line_text().lower()
    return any(
        line_text.find(word) > -1 for word in matching_params["including_expressions"]
    ) and not any(line_text.find(word) > -1 for word in matching_params["excluding_expressions"])

def cluster_text_elements(elements, key_fn = Callable[[pymupdf.Rect], float], tolerance: int = 10):
    """ cluster text elements based on coordinates of bounding box
    Args:
        elements: List of object containing a `rect` attribute
        key_fn: Function that extracts a float from each element (e.g. lambda obj: obj.rect.y0)
        tolerance: max allowed difference between entries and a cluster key"""

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