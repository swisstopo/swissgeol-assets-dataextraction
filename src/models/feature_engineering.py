import re
from collections.abc import Callable

import numpy as np
import pymupdf
from extraction.minimal_pipeline import ExtractionContext, extract_page_features
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine
from swissgeol_doc_processing.utils.file_utils import read_params

from src.identifiers.boreprofile import Entry, detect_entries, is_mostly_increasing
from src.identifiers.map import compute_angle_entropy, find_map_scales, map_lines_score, split_lines_by_orientation
from src.language_detection.detect_language import (
    DEFAULT_LANGUAGE,
    extract_cleaned_text,
    predict_language,
    select_classification_language,
)
from src.page_structure import PageContext
from src.utils.text_clustering import cluster_text_elements, create_text_blocks
from src.utils.utility import is_description


def extract_and_cache_page_data(page: pymupdf.Page) -> ExtractionContext:
    """Extract and cache all page data using ExtractionContext.

    Returns:
        ExtractionContext: Context object with cached extraction results
    """
    line_detection_params = read_params("line_detection_params.yml")
    striplog_detection_params = read_params("striplog_detection_params.yml")
    table_detection_params = read_params("table_detection_params.yml")

    return ExtractionContext.from_page(page, line_detection_params, striplog_detection_params, table_detection_params)


def get_features(
    page: pymupdf.Page,
    page_number: int,
    matching_params: dict,
    borehole_matching_params: dict,
    ctx: PageContext | None = None,
) -> list[float]:
    """Extracts numerical features from a PDF page for classification.

    Unified function for both training and inference paths:
    - Training: Call without ctx, extracts everything fresh
    - Inference: Call with ctx, reuses cached ExtractionContext

    Args:
        page: The PDF page object.
        page_number: The page number within the document (starting from 1).
        matching_params: Parameters for keyword matching.
        borehole_matching_params: Parameters for borehole-specific matching.
        ctx: Optional pre-built PageContext with cached extraction data.

    Returns:
        list[float]: A list of 23 computed features.
    """
    if ctx is not None and ctx.extraction_context is not None:
        # INFERENCE PATH: Reuse cached data from PageContext
        extraction_context = ctx.extraction_context
        lines = ctx.lines
        text_blocks = ctx.text_blocks
        language = ctx.language
        geometric_lines = ctx.geometric_lines
    else:
        # TRAINING PATH: Extract everything fresh
        clean_text, word_count = extract_cleaned_text(page)
        language_prediction = predict_language(clean_text)
        language = select_classification_language(language_prediction, word_count)

        extraction_context = extract_and_cache_page_data(page)
        lines = extraction_context.text_lines
        geometric_lines = extraction_context.all_geometric_lines
        text_blocks = create_text_blocks(lines)

    features = compute_text_features(
        page,
        page_number - 1,
        lines,
        text_blocks,
        language,
        geometric_lines,
        matching_params,
        borehole_matching_params,
        extraction_context,
    )

    return features


def compute_text_features(
    page: pymupdf.Page,
    page_index: int,
    lines: list[TextLine],
    text_blocks: list[TextBlock],
    language: str,
    geometric_lines: list[Line],
    matching_params: dict,
    borehole_matching_params: dict,
    extraction_context: ExtractionContext | None = None,
) -> list[float]:
    """Computes 23 numerical features used for XGBoost page classification models.

    The features are derived from:
    - Text/word features (6): words per line, density, position, width, indentation, capitalization
    - Map features (5): keyword lines, grid/non-grid line lengths, angle entropy, line score
    - Geo profile and diagram features (4): keywords, units, axis scales
    - Borehole features (8): descriptions, strip logs, tables, boreholes, sidebars, geometric lines

    Args:
        page: The PDF page object.
        page_index: Zero-based page index within the document.
        lines: List of detected text lines on the page.
        text_blocks: Grouped lines forming text blocks.
        language: Detected language of the text (e.g., "de", "fr", "it").
        geometric_lines: Detected graphical line elements on the page.
        matching_params: Configuration dictionary for keyword and pattern matching.
        borehole_matching_params: Configuration for borehole-specific matching.
        extraction_context: Optional ExtractionContext for caching intermediate results.

    Returns:
        list: A list of 23 computed feature values for the page. If no text lines are found, returns a zero vector.
    """
    if not lines:
        return [0.0] * 23  # Handle empty pages (15 base + 8 borehole features)

    (
        word_per_line,
        word_density,
        mean_left,
        text_width,
        indent_std,
        capital_ratio,
    ) = get_word_features(lines, text_blocks)

    borehole_feature_list = get_borehole_feature_list(
        page, page_index, language, borehole_matching_params, extraction_context=extraction_context
    )

    num_map_keyword_lines = get_map_features(lines, language, matching_params)

    grid_length_sum, non_grid_length_sum, angle_entropy, line_score = get_geom_line_features(geometric_lines)

    num_geo_profile_keywords = get_geo_profile_feature(lines, language, matching_params)

    num_unit, y_ok, x_ok = get_diagram_features(lines, matching_params)

    features = [
        word_per_line,
        word_density,
        mean_left,
        text_width,
        indent_std,
        capital_ratio,
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

    features.extend(borehole_feature_list)

    return list(map(float, features))


def get_borehole_feature_list(
    page: pymupdf.Page,
    page_index: int,
    language,
    borehole_matching_params: dict,
    line_detection_params: dict | None = None,
    name_detection_params: dict | None = None,
    extraction_context: ExtractionContext | None = None,
) -> list[float]:
    """Extract borehole-related features from a page as a list for classification.

    Combines page feature extraction with conversion to a feature list suitable
    for tree-based classifiers.

    Args:
        page: The PDF page object.
        page_index: Zero-based page index within the document.
        language: Detected language of the text (e.g., "de", "fr", "it").
        borehole_matching_params: Parameters for keyword matching.
        line_detection_params: Optional parameters for line detection.
        name_detection_params: Optional parameters for name detection.
        extraction_context: Optional ExtractionContext for caching intermediate results.

    Returns:
        List of 8 feature values in consistent order for classification.
    """
    if line_detection_params is None:
        line_detection_params = read_params("line_detection_params.yml")
    if name_detection_params is None:
        name_detection_params = read_params("name_detection_params.yml")

    borehole_features = extract_page_features(
        page,
        page_index,
        language,
        borehole_matching_params,
        line_detection_params,
        name_detection_params,
        extraction_context,
        extract_boreholes=True,
    )
    sidebar_info = borehole_features.sidebar_information

    return [
        float(borehole_features.number_of_valid_borehole_descriptions),
        float(borehole_features.number_of_strip_logs),
        float(borehole_features.number_of_tables),
        float(borehole_features.number_of_boreholes),
        float(sidebar_info.number_of_good_sidebars),
        float(sidebar_info.best_sidebar_score),
        float(borehole_features.number_long_or_horizontal_lines),
        float(borehole_features.text_line_count),
    ]


def get_diagram_features(lines: list[TextLine], matching_params: dict):
    keywords_unit = matching_params["units"]

    num_unit = sum(
        bool(re.search(r"[\(\[]\s*" + re.escape(u) + r"\s*[\)\]]", line.text.lower()))
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
    num_geo_profile_key_words = sum(kw in line.text.lower() for kw in geo_profile_key_words for line in lines)
    return num_geo_profile_key_words


def get_map_features(lines: list[TextLine], language: str, matching_params: dict):
    keywords = matching_params["map_terms"].get(language, {})
    num_map_keyword_lines = (
        len([line for line in lines if is_description(line, keywords) or find_map_scales(line)]) if keywords else 0
    )

    return num_map_keyword_lines


def get_geom_line_features(geometric_lines: list[Line]):
    angles = [line.angle for line in geometric_lines]
    grid_lengths, non_grid_lengths = split_lines_by_orientation(geometric_lines)
    grid_length_sum = sum(grid_lengths)
    non_grid_length_sum = sum(non_grid_lengths)
    angle_entropy = compute_angle_entropy(angles)
    lines_score = map_lines_score(geometric_lines)
    return float(grid_length_sum), float(non_grid_length_sum), float(angle_entropy), lines_score


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
        word_per_line,
        word_density,
        mean_left,
        text_width,
        indent_std,
        capital_ratio,
    )
