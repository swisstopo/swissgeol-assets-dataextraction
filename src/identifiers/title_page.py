import logging
import re
from statistics import stdev
from typing import Callable

from ..page_structure import PageContext
from ..text_objects import TextLine

logger = logging.getLogger(__name__)

def identify_title_page(ctx: PageContext) -> bool:
    """
       Identifies whether a page is likely a title page based on a combination of:
       - Line count
       - Centered or consistently aligned layout
       - Content clues (e.g., dates, keywords, phone info)
       - Large or varied fonts
       """

    if not (3 <= len(ctx.lines) <= 35):
        return False

    if has_centered_layout(ctx):
        return True

    if has_aligned_layout(ctx):
        return True

    if contains_content_clues(ctx.lines):
        return True

    return False

def contains_content_clues(lines: list[TextLine]) -> bool:
    """
    Returns True if the page contains at least 3 typical title-page indicators.
    """
    title_keywords = ["bericht", "rapport", "projekt", "projet"]
    date_patterns = [r"\b\d{4}\b", r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"]
    phone_keywords = ["tel", "telefon"]
    phone_pattern = r"\b(?:tel\.?|telefon)\s*[:\-]?\s*\d{9,}"

    hits = 0
    for line in lines:
        text = line.line_text().lower()

        if any(kw in text for kw in title_keywords):
            hits += 1
        if any(re.search(pat, text) for pat in date_patterns):
            hits += 1
        if any(kw in text for kw in phone_keywords) or re.search(phone_pattern, text):
            hits += 1

    return hits >= 3

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
        filtered_x0 = remove_outliers_if_needed(x0_values) #  removes a single line if this decreases x0 variability drastically
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

def remove_outliers_if_needed(x0s:list,threshold: float = 0.6)->list[float]:
    """
    Removes a single outlier from x0 values if it significantly reduces the standard deviation.
    """

    if len(x0s) < 3 or stdev(x0s) == 0:
        return x0s

    original_std = stdev(x0s)
    best_filtered = x0s
    best_ratio = 1.0

    for i in range(len(x0s)):
        trial = x0s[:i] + x0s[i + 1:]
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