import pymupdf
import numpy as np
from collections import defaultdict
from typing import Callable

from .text import TextWord, TextLine
from .bounding_box import merge_bounding_boxes

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

def calculate_distance(word1, word2):
    """Calculate Euclidean distance between two TextWord objects based on x0 and y0"""
    return word1.rect.top_left.distance_to(word2.rect.top_left)

def closest_word_distances(words): #not in use, might come in handy
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

def is_valid(cluster: list[TextLine], all_lines: list[TextLine], max_noise_ratio = 0.5, max_gap_factor = 2) -> bool: #not in use, might come in handy
    """ cluster in clusters is valid if:
    - more than 1 entry in cluster
    - noise within rectangle is small (less words that intersect with rectangle than entries cluster has)
    - distance between entries in cluster not too large ( in comparison to medium distance between lines on page"""
    if len(cluster) <2:
        return False

    cluster_bbox = merge_bounding_boxes([line.rect for line in cluster])

    noise_lines = [
        line for line in all_lines
        if line not in cluster and cluster_bbox.intersects(line.rect)
    ]

    if len(noise_lines) > len(cluster) * max_noise_ratio:

        return False

    # ys = sorted([line.rect.y0 for line in cluster])
    # gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    # if not gaps:
    #
    #     return False
    #
    # avg_cluster_gap = sum(gaps) / len(gaps)
    #
    # # Estimate global line spacing
    # all_ys = sorted([line.rect.y0 for line in all_lines])
    # global_gaps = [all_ys[i + 1] - all_ys[i] for i in range(len(all_ys) - 1) if all_ys[i + 1] > all_ys[i]]
    # global_avg_gap = sum(global_gaps) / len(global_gaps) if global_gaps else avg_cluster_gap
    #
    # if avg_cluster_gap > global_avg_gap * max_gap_factor:
    #     return False

    return True