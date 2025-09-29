import re
from collections.abc import Callable

import numpy as np
import pymupdf

from identifiers.table import detect_text_table
from src.geometric_objects import Line
from src.identifiers.boreprofile import Entry, create_sidebars, detect_entries, is_mostly_increasing
from src.identifiers.map import compute_angle_entropy, find_map_scales, map_lines_score, split_lines_by_orientation
from src.language_detection.detect_language import (
    DEFAULT_LANGUAGE,
    extract_cleaned_text,
    predict_language,
    select_classification_language,
)
from src.line_detection import extract_geometric_lines
from src.material_description import detect_material_description
from src.page_structure import PageContext
from src.text_objects import TextBlock, TextLine, cluster_text_elements, create_text_blocks, create_text_lines
from src.utils import is_description


def get_features(page: pymupdf.Page, page_number: int, matching_params: dict) -> list[float]:
    """Extracts numerical features from a  PDF page for training a classifier.

    This function is used during training, where language, text lines,
    text blocks, and geometric lines are all extracted from the page.

    Args:
        page (pymupdf.Page): The PDF page object.
        page_number (int): The page number within the document (starting form 1).
        matching_params (dict): Parameters for keyword matching.

    Returns:
        list[float]: A list of 17 computed features used for training tree-based classifiers.
    """
    ## detect language
    clean_text, word_count = extract_cleaned_text(page)
    language_prediction = predict_language(clean_text)
    language = select_classification_language(language_prediction, word_count)

    ## construct text features
    lines = create_text_lines(page, page_number)
    geometric_lines = extract_geometric_lines(page)
    text_blocks = create_text_blocks(lines)

    features = compute_text_features(lines, text_blocks, language, geometric_lines, matching_params)
    return features


def get_features_from_page(page: pymupdf.Page, ctx: PageContext, matching_params: dict) -> list[float]:
    """Computes features for an already processed page using its PageContext.

    It is used during page classification,
     where preprocessing has already been performed and stored in the PageContext.

    Args:
        page (pymupdf.Page): The PDF page object.
        ctx (PageContext): A pre-populated PageContext object containing lines, language, text blocks, etc.
        matching_params (dict): Parameters for keyword matching.

    Returns:
        list[float]: A list of 17 computed features used for classification.
    """
    ctx.geometric_lines = extract_geometric_lines(page)
    features = compute_text_features(ctx.lines, ctx.text_blocks, ctx.language, ctx.geometric_lines, matching_params)

    return features


def compute_text_features(
    lines: list[TextLine],
    text_blocks: list[TextBlock],
    language: str,
    geometric_lines: list[Line],
    matching_params: dict,
) -> list[float]:
    """Computes 17 numerical features used for tree-based page classification models.

    (e.g., Random Forest, XGBoost) based on extracted text and geometric lines.

    The features are derived from:
    - Text lines (e.g., line length, punctuation, capitalization)
    - Text block geometry (e.g., density, indentation)
    - Language-specific heuristics
    - Geometric lines on the page
    - Domain-specific keyword and structure matching

    Args:
        lines: List of detected text lines on the page.
        text_blocks: Grouped lines forming text blocks.
        language: Detected language of the text (e.g., "de", "fr", "it").
        geometric_lines: Detected graphical line elements on the page.
        matching_params: Configuration dictionary for keyword and pattern matching.

    Returns:
        list: A list of X computed feature values for the page. If no text lines are found, returns a zero vector.
    """
    if not lines:
        return [0.0] * 19  # Handle empty pages

    (
        line_count,
        word_per_line,
        word_density,
        mean_left,
        text_width,
        indent_std,
        capital_ratio,
    ) = get_word_features(lines, text_blocks)

    num_valid_descriptions, has_sidebar, has_bh_keyword = get_borehole_features(lines, language, matching_params)

    num_map_keyword_lines = get_map_features(lines, language, matching_params)

    grid_length_sum, non_grid_length_sum, angle_entropy, line_score = get_geom_line_features(geometric_lines)

    num_geo_profile_keywords = get_geo_profile_feature(lines, language, matching_params)

    num_unit, y_ok, x_ok = get_diagram_features(lines, matching_params)

    # num_tables = get_table_features(lines) # continue with this

    return [
        float(n)
        for n in [
            word_per_line,
            word_density,
            mean_left,
            text_width,
            line_count,
            indent_std,
            capital_ratio,
            has_sidebar,
            has_bh_keyword,
            num_valid_descriptions,
            num_map_keyword_lines,
            grid_length_sum,
            non_grid_length_sum,
            angle_entropy,
            line_score,
            num_geo_profile_keywords,
            num_unit,
            y_ok,
            x_ok,
        ]
    ]


def get_table_features(lines: list[TextLine], min_conf: float = 0.6, min_coverage: float = 0.3):
    words = [word for line in lines for word in line.words]
    text_table = detect_text_table(words)
    if not text_table:
        return 0

    good_text_tables = [table for table in text_table if table.confidence >= min_conf]

    return len([table.text_coverage(words) > min_coverage for table in good_text_tables])


def get_diagram_features(lines: list[TextLine], matching_params: dict):
    keywords_unit = matching_params["units"]

    num_unit = sum(
        bool(re.search(r"[\(\[]\s*" + re.escape(u) + r"\s*[\)\]]", line.line_text.lower()))
        for u in keywords_unit
        for line in lines
    )
    words = [word for line in lines for word in line.words]
    depths_entries = detect_entries(words)  # TODO should include

    vertical_clusters = cluster_text_elements(depths_entries, key_fn=lambda e: e.rect.x0, tolerance=10)
    horizontal_clusters = cluster_text_elements(depths_entries, key_fn=lambda e: e.rect.y0, tolerance=10)

    def normalize_direction(values: list[Entry]) -> list[Entry]:
        """Ensure values of entries go ascending; reverse if descending, leave otherwise."""
        if len(values) < 2:
            return values
        return values[::-1] if values[0].value > values[-1].value else values

    def is_true_axis(clusters: list[list[Entry]], key: Callable) -> bool:
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            axis = sorted(cluster, key=key)
            if is_mostly_increasing(normalize_direction(axis)):
                return True
        return False

    y_ok = is_true_axis(vertical_clusters, key=lambda e: e.rect.y0)
    x_ok = is_true_axis(horizontal_clusters, key=lambda e: e.rect.x0)
    return num_unit, y_ok, x_ok


def get_geo_profile_feature(lines: list[TextLine], language: str, matching_params: dict):
    geo_profile_key_words = matching_params["geo_profile"].get(language, DEFAULT_LANGUAGE)
    num_geo_profile_key_words = sum(kw in line.line_text.lower() for kw in geo_profile_key_words for line in lines)
    return num_geo_profile_key_words


def get_map_features(lines: list[TextLine], language: str, matching_params: dict):
    keywords = matching_params["map_terms"].get(language, {})
    num_map_keyword_lines = (
        len([line for line in lines if is_description(line, keywords) or find_map_scales(line)]) if keywords else 0
    )

    return num_map_keyword_lines


def get_geom_line_features(geometric_lines: list[Line]):
    angles = [line.line_angle for line in geometric_lines]
    grid_lengths, non_grid_lengths = split_lines_by_orientation(geometric_lines)
    grid_length_sum = sum(grid_lengths)
    non_grid_length_sum = sum(non_grid_lengths)
    angle_entropy = compute_angle_entropy(angles)
    lines_score = map_lines_score(geometric_lines)
    return float(grid_length_sum), float(non_grid_length_sum), float(angle_entropy), lines_score


def get_borehole_features(lines: list[TextLine], language: str, matching_params: dict):
    words = [word for line in lines for word in line.words]

    keywords = matching_params["material_description"].get(language, {})
    descriptions = detect_material_description(lines, words, keywords) if keywords else []
    num_valid_descriptions = len([desc for desc in descriptions if desc.is_valid])

    sidebars = create_sidebars(words)
    has_sidebar = int(bool(sidebars))

    keyword_set = matching_params["boreprofile"].get(language, {})
    has_bh_keyword = int(any(keyword in word.text.lower() for word in words for keyword in keyword_set))
    return num_valid_descriptions, has_sidebar, has_bh_keyword


def get_word_features(lines: list[TextLine], text_blocks: list[TextBlock]):
    lefts, rights, line_lengths = [], [], []
    punct_count = capital_chars = total_chars = word_count = 0

    for line in lines:
        x0, x1 = line.rect.x0, line.rect.x1
        words = line.words
        word_count += len(words)
        text = " ".join(word.text for word in words)

        lefts.append(x0)
        rights.append(x1)
        line_lengths.append(len(words))
        punct_count += len(re.findall(r"[.,!?;:()\"\']", text))
        capital_chars += sum(1 for c in text if c.isupper())
        total_chars += len(re.sub(r"\s", "", text))

    line_count = len(lines)
    word_per_line = word_count / line_count if line_count else 0
    word_area = sum(
        word.rect.get_area()
        for block in text_blocks
        for line in block.lines
        for word in line.words
        if len(line.words) > 1
    )
    tot_area = pymupdf.Rect(
        min(lefts) if lefts else 0,
        min(line.rect.y0 for line in lines) if lines else 0,
        max(rights) if rights else 0,
        max(line.rect.y1 for line in lines) if lines else 0,
    ).get_area()

    # Calculate word density as the ratio of word area to total area
    word_density = word_area / tot_area if tot_area > 0 else 0
    mean_left = np.mean(lefts)
    text_width = np.mean([r - left for r, left in zip(rights, lefts, strict=False)])
    indent_std = np.std(lefts)
    capital_ratio = capital_chars / total_chars if total_chars else 0
    return (
        line_count,
        word_per_line,
        word_density,
        mean_left,
        text_width,
        indent_std,
        capital_ratio,
    )
