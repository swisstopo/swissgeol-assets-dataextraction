import pymupdf
import numpy as np
import math
from collections import defaultdict

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

def classify_text_density(words, page_size):
    if not words:
        return {
            "classification": "No text",
            "text_density": 0,
            "text_area": 0,
            "avg_word_height": 0,
            "std_word_height": 0
        }

    page_area = page_size[0] * page_size[1]
    text_density = len(words) / page_area

    text_area = sum(word.rect.width * word.rect.height for word in words) / page_area

    word_heights = [word.rect.height for word in words]
    avg_word_height = float(np.mean(word_heights))
    std_word_height = float(np.std(word_heights))

    return {
        "text_density": text_density,
        "text_area": text_area,
        "avg_word_height": avg_word_height,
        "std_word_height": std_word_height
    }

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

    # # Compute pairwise Euclidean distances
    # dist_matrix = squareform(pdist(y_positions.reshape(-1, 1))) #instead use boundingbox?
    # threshold = np.percentile(dist_matrix, 20)
    # graph_matrix = (dist_matrix < threshold).astype(int)
    # lap_matrix = laplacian(graph_matrix, normed=True)

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
    x_dist = word1.rect.x0 - word2.rect.x0
    y_dist = word1.rect.y0 - word2.rect.y0
    return math.sqrt(x_dist**2 + y_dist**2)

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

def cluster_text_elements(elements, key="y0", tolerance: int = 10):
    """ cluster text elements based on coordinates of bounding box

    Args:
        elements: List of object containing a `rect` attribute with x0 or y0 etc
        key: attribute clustering is based on (y0 or x0)
        tolerance: max allowed difference between entries and a cluster key"""

    if not elements:
        return []

    ## make sure that key is acutally input we allow
    if not isinstance(key, str):
        raise TypeError(f"Expected 'key' to be a string, got {type(key)} instead.")
    valid_keys = {"y0", "x0"}

    if key not in valid_keys:
        raise ValueError(f"Invalid key '{key}'. Must be one of {valid_keys}.")

    # Dictionary to hold clusters, keys are representative attribute values
    grouped = defaultdict(list)

    for element in elements:
        attribute = getattr(element.rect, key)
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