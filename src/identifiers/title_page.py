import logging
from statistics import stdev
from typing import Callable, Sequence
from ..keyword_finding import find_pattern, date_patterns, phone_patterns
from ..page_structure import PageContext
from ..text_objects import TextLine

logger = logging.getLogger(__name__)

def identify_title_page(ctx: PageContext, matching_params) -> bool:
    """
       Identifies whether a page is likely a title page based on a combination of:
       - Line count
       - Centered or consistently aligned layout
       - Content clues (e.g., dates, keywords, phone info)
       - Large or varied fonts
       """

    if not (3 <= len(ctx.lines) <= 35):
        return False

    vertical_distances = vertical_spacing(ctx.lines)
    mean_gaps = sum(vertical_distances) / len(vertical_distances)
    if mean_gaps <= 25:
        return False
    if has_centered_layout(ctx):
        return True

    if has_aligned_layout(ctx):
        return True

    if has_large_font_layout(ctx) and contains_content_clues(ctx, matching_params):
        return True

    return False

def has_centered_layout(ctx: PageContext)-> bool:
    """
    Checks if the majority of text is in clusters that are horizontally centered.
    Filters out 'centered' clusters that are actually right-aligned based on x0 spread.
    """
    page_width = ctx.page_rect.width
    centered_clusters = find_aligned_clusters(ctx.lines,
                                              key_func = lambda line: line.rect.x0 + 0.5 * line.rect.width,
                                              threshold = 0.05 * page_width)

    valid_clusters = []
    for cluster in centered_clusters:
        x0_values = [line.rect.x0 for line in cluster]
        filtered_x0 = remove_outlier_if_needed(x0_values) #  removes a single line if this decreases x0 variability drastically
        if stdev(filtered_x0) >= 0.05 * page_width:
            valid_clusters.append(cluster)

    if not valid_clusters:
        return False

    cluster_words = sum(len(line.words)
                        for cluster in valid_clusters for line in cluster)

    return cluster_words / len(ctx.words) > 0.75


def has_aligned_layout(ctx: PageContext) -> bool:
    """
    Checks if the majority of lines belong to clusters that are left-, right- or center-aligned.
    """
    total_words = len(ctx.words)
    if not total_words:
        return False

    clusters = get_all_layout_clusters(ctx.lines, ctx.page_rect.width)
    words = sum(len(line.words) for cluster in clusters for line in cluster)

    return words / len(ctx.words) > 0.75

def has_large_font_layout(ctx: PageContext) -> bool:
    """
    Returns True if the page has at least one large font size and high font variety.
    """
    font_sizes = [line.font_size for line in ctx.lines]
    return len(set(font_sizes)) > 5 and max(font_sizes, default=0) > 20

def contains_content_clues(ctx: PageContext, matching_params) -> bool:
    """
     Returns True if the page contains at least 2 out of 3 indicators:
    - title keywords (language-dependent)
    - a date
    - a phone number
    """
    title_keywords = matching_params["title_page"].get(ctx.language, [])
    has_title_keyword = any(keyword in word.text.lower() for word in ctx.words for keyword in title_keywords)

    has_date = any(find_pattern(line, date_patterns) for line in ctx.lines)
    has_phone = any(find_pattern(line, phone_patterns) for line in ctx.lines)

    hits = sum([has_title_keyword, has_date, has_phone])
    return hits >= 2

def remove_outlier_if_needed(
    values: list[float],
    threshold: float = 0.6,
    removable_indices: Sequence[int] | None = None
) -> list[float]:
    """
    Removes one value (from allowed positions) if doing so significantly reduces the standard deviation.

    Args:
        values: List of floats.
        threshold: Reduction ratio threshold for std dev.
        removable_indices: Indices allowed to be removed (e.g. [0, 1, -1, -2]). If None, all indices are considered.

    Returns:
        A filtered list with at most one value removed, or the original list if no improvement found.
    """
    if len(values) < 3 or stdev(values) == 0:
        return values

    original_std = stdev(values)
    best_filtered = values
    best_ratio = 1.0

    if removable_indices is None:
        candidate_indices = range(len(values))
    else:
        candidate_indices = [(i if i >= 0 else len(values) + i) for i in removable_indices]
        candidate_indices = [i for i in candidate_indices if 0 <= i < len(values)]

    for i in candidate_indices:
        trial = values[:i] + values[i + 1:]
        if len(trial) < 2:
            continue
        trial_std = stdev(trial)
        ratio = trial_std / original_std

        if ratio < best_ratio and ratio < threshold:
            best_ratio = ratio
            best_filtered = trial

    return best_filtered

def get_all_layout_clusters(lines: list[TextLine], page_width: float) -> list[list[TextLine]]:
    """
    Returns all text line clusters aligned either left, right, or center.
    A line can belong to only one cluster (first match).
    """
    remaining_lines = set(lines)
    all_clusters = []

    layout_funcs = [
        lambda line: line.rect.x0,  # left-aligned
        lambda line: line.rect.x1,  # right-aligned
        lambda line: line.rect.x0 + 0.5 * line.rect.width  # center-aligned
    ]

    for layout in layout_funcs:
        unassigned_lines = list(remaining_lines)
        clusters = find_aligned_clusters(unassigned_lines, layout, threshold = 0.05 * page_width)

        for cluster in clusters:
            all_clusters.append(cluster)
            remaining_lines.difference_update(cluster)

    return all_clusters

def find_aligned_clusters(
    lines: list[TextLine],
    key_func: Callable[[TextLine], float],
    threshold: float
) -> list[list[TextLine]]:
    """
        Finds clusters of lines aligned by a given key (e.g. x0, x1, center).
        Considers only lines with vertical proximity to last line of current cluster.
        """
    remaining_lines = set(lines)
    clusters = []

    while remaining_lines:
        current_line = remaining_lines.pop()
        cluster = [current_line]
        cluster_key = key_func(current_line)
        font_size = current_line.font_size

        close_lines = {
            line for line in remaining_lines
            if abs(key_func(line) - cluster_key) < threshold and
               abs(line.rect.y0 - current_line.rect.y0) < 5.0 * font_size
        }

        cluster.extend(close_lines)
        remaining_lines -= close_lines

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def vertical_spacing(lines:list[TextLine]) -> list[float]:
    sorted_lines = sorted(lines, key=lambda line : (line.rect.y0, line.rect.x0))
    distances = []
    for line in sorted_lines:
        line.rect

    for i in range(len(sorted_lines) - 1):
        distance = sorted_lines[i+1].rect.y0 - sorted_lines[i].rect.y0
        distances.append(distance)


    filtered = remove_outlier_if_needed(distances, threshold=0.6, removable_indices=[0, 1, -2, -1])

    return filtered