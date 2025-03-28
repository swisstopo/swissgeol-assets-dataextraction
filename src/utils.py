import pymupdf
import numpy as np
from collections import defaultdict
from typing import Callable

from .text import TextWord, TextLine


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

def classify_wordpos(words: list[TextWord]):
    """Classifies text structure on page based on distribution."""

    if not words:
        print( "Unknown")
        return

    # Extract Y-axis positions and widths
    y_positions = np.array([word.rect.y0 for word in words])
    x_positions = np.array([word.rect.x0 for word in words])
    widths = np.array([word.rect.x1 - word.rect.x0 for word in words])
    heights = np.array([word.rect.y1 - word.rect.y0 for word in words])

    # Compute spacing bewtween word to next word
    y_spacing = np.diff(np.sort(y_positions))
    x_spacing = np.diff(np.sort(x_positions))

    mean_y_spacing = float(np.mean(y_spacing)) if len(y_spacing) > 0 else 0
    median_x_spacing = float(np.median(x_spacing)) if len(x_spacing) > 0 else 0
    width_std = np.std(widths)
    height_std = np.std(heights)

    return {
        "mean_y_spacing": mean_y_spacing,
        "median_x_spacing": median_x_spacing,
        "median width": float(np.median(widths)),
        "width_std": float(width_std),
        "height_std":float(height_std)
    }

def calculate_distance(word1, word2):
    """Calculate Euclidean distance between two TextWord objects based on x0 and y0"""
    return word1.rect.top_left.distance_to(word2.rect.top_left)

def closest_word_distances(words):
    """Calculate distances between each word and its closest neighbor"""
    if not words or len(words) < 2:
        return []

    distances = []
    for i, word in enumerate(words):
        other_words = words[:i] + words[i+1:]  # Exclude current word
        closest_word = min(other_words, key=lambda w: calculate_distance(word, w))
        distances.append(calculate_distance(word, closest_word))

    return distances

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