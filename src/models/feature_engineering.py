import pymupdf
import numpy as np
import re

from page_structure import PageContext
from text_objects import create_text_blocks, create_text_lines

from src.detect_language import detect_language_of_page
from identifiers.boreprofile import create_sidebars
from identifiers.map import find_map_scales, split_lines_by_orientation
from line_detection import extract_geometric_lines
from material_description import detect_material_description
from scipy.stats import entropy
from utils import is_description


def compute_text_features_chat(lines, text_blocks):
    if not lines:
        return [0.0] * 10  # Handle empty pages

    lefts = []
    rights = []
    line_lengths = []
    punct_count = 0
    total_chars = 0
    capital_chars = 0
    word_count = 0

    for line in lines:
        x0 = line.rect.x0
        x1 = line.rect.x1
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
    text_width = np.mean([r - l for r, l in zip(rights, lefts)])
    line_len_var = np.var(line_lengths)
    indent_std = np.std(lefts)
    punct_density = punct_count / line_count if line_count else 0
    capital_ratio = capital_chars / total_chars if total_chars else 0

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
        ]
    ]

def extract_more_features(lines, geometric_lines, language, matching_params):
    words = [word for line in lines for word in line.words]

    keywords = matching_params["material_description"].get(language, [])
    if keywords:
        descriptions = detect_material_description(lines, words, keywords)
    else:
        descriptions = []
    num_valid_descriptions = len([desc for desc in descriptions if desc.is_valid])

    sidebars = create_sidebars(words)
    has_sidebar = int(bool(sidebars))

    keyword_set = matching_params["boreprofile"].get(language, [])
    has_bh_keyword = int(any(keyword in word.text.lower() for word in words for keyword in keyword_set))

    keywords = matching_params["map_terms"].get(language, [])
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
    angle_hist = np.histogram(angles, bins=36, range=(0, 180))[0]
    angle_entropy = entropy(angle_hist) / np.log2(36)

    return [
        has_sidebar,
        has_bh_keyword,
        num_valid_descriptions,
        num_map_keyword_lines,
        float(grid_length_sum),
        float(non_grid_length_sum),
        float(angle_entropy),
    ]


def get_features(paths, matching_params):
    all_features = []
    for file_path in paths:
        with pymupdf.Document(file_path) as doc:
            print(f"Processing {file_path}", end="\r")
            if len(doc) > 1:
                raise ValueError(f"Expected single-page PDF, but found {len(doc)} pages in {file_path}")

            page_number = 1
            page = doc[page_number - 1]
            lines = create_text_lines(page, page_number)
            language = detect_language_of_page(page)
            geometric_lines = extract_geometric_lines(page)
            text_blocks = create_text_blocks(lines)
            feat = compute_text_features_chat(lines, text_blocks)
            feat.extend(extract_more_features(lines, geometric_lines, language, matching_params))
            all_features.append(feat)
    return all_features


def get_features_from_page(page:pymupdf, ctx:PageContext, matching_params:dict):
    features = compute_text_features_chat(ctx.lines, ctx.text_blocks)
    ctx.geometric_lines = extract_geometric_lines(page)

    features.extend(extract_more_features(ctx.lines, ctx.geometric_lines, ctx.language, matching_params))

    return features