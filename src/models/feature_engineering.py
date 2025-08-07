import re

import numpy as np
import pymupdf

from src.geometric_objects import Line
from src.identifiers.boreprofile import create_sidebars
from src.identifiers.map import compute_angle_entropy, find_map_scales, split_lines_by_orientation
from src.language_detection.detect_language import (
    extract_cleaned_text,
    predict_language,
    select_classification_language,
)
from src.line_detection import extract_geometric_lines
from src.material_description import detect_material_description
from src.page_structure import PageContext
from src.text_objects import TextBlock, TextLine, create_text_blocks, create_text_lines
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
        list: A list of 17 computed feature values for the page. If no text lines are found, returns a zero vector.
    """
    if not lines:
        return [0.0] * 17  # Handle empty pages

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
    wpl = word_count / line_count if line_count else 0
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
    mean_right = np.mean(rights)
    text_width = np.mean([r - left for r, left in zip(rights, lefts, strict=False)])
    line_len_var = np.var(line_lengths)
    indent_std = np.std(lefts)
    punct_density = punct_count / line_count if line_count else 0
    capital_ratio = capital_chars / total_chars if total_chars else 0

    words = [word for line in lines for word in line.words]

    keywords = matching_params["material_description"].get(language, {})
    descriptions = detect_material_description(lines, words, keywords) if keywords else []
    num_valid_descriptions = len([desc for desc in descriptions if desc.is_valid])

    sidebars = create_sidebars(words)
    has_sidebar = int(bool(sidebars))

    keyword_set = matching_params["boreprofile"].get(language, {})
    has_bh_keyword = int(any(keyword in word.text.lower() for word in words for keyword in keyword_set))

    keywords = matching_params["map_terms"].get(language, {})
    if keywords:
        num_map_keyword_lines = len(
            [line for line in lines if is_description(line, keywords) or find_map_scales(line)]
        )
    else:
        num_map_keyword_lines = 0

    angles = [line.line_angle for line in geometric_lines]
    grid_lengths, non_grid_lengths = split_lines_by_orientation(geometric_lines)
    grid_length_sum = sum(grid_lengths)
    non_grid_length_sum = sum(non_grid_lengths)
    angle_entropy = compute_angle_entropy(angles)

    return [
        float(n)
        for n in [
            wpl,
            word_density,
            mean_left,
            mean_right,
            text_width,
            line_count,
            line_len_var,
            indent_std,
            punct_density,
            capital_ratio,
            has_sidebar,
            has_bh_keyword,
            num_valid_descriptions,
            num_map_keyword_lines,
            float(grid_length_sum),
            float(non_grid_length_sum),
            float(angle_entropy),
        ]
    ]
