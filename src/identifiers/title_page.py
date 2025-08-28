import logging
from collections.abc import Callable, Sequence
from statistics import stdev

from src.keyword_finding import DATE_PATTERNS, PHONE_PATTERNS, find_pattern
from src.page_structure import PageContext
from src.text_objects import TextLine

logger = logging.getLogger(__name__)

ALIGNED_WORD_RATIO_THRESHOLD = 0.8
VERTICAL_SPACING_FACTOR = 5.0


def identify_title_page(ctx: PageContext, matching_params: dict) -> bool:
    """Identifies whether a page is likely a title page based on a combination of factors.

    Factors include:
    - Line count
    - Centered or consistently aligned layout
    - Content clues (e.g., dates, keywords, phone info)
    - Large or varied fonts
    """
    if not (3 <= len(ctx.lines) <= 35):
        return False

    vertical_distances = vertical_spacing(ctx.lines)
    if vertical_distances:
        mean_gaps = sum(vertical_distances) / len(vertical_distances)
        if mean_gaps <= 35:
            return False

    if has_centered_layout(ctx):
        return True

    if has_aligned_layout(ctx):
        return True

    return has_large_font_layout(ctx) and contains_content_clues(ctx, matching_params)


def has_centered_layout(ctx: PageContext) -> bool:
    """Checks if the majority of text is in clusters that are horizontally centered.

    Filters out 'centered' clusters that are actually right-aligned based on x0 spread.
    """
    page_width = ctx.page_rect.width
    centered_clusters = find_aligned_clusters(
        ctx.lines, key_func=lambda line: line.rect.x0 + 0.5 * line.rect.width, threshold=0.05 * page_width
    )

    valid_clusters = []
    for cluster in centered_clusters:
        x0_values = [line.rect.x0 for line in cluster]
        filtered_x0 = remove_outlier_if_needed(
            x0_values
        )  #  removes a single line if this decreases x0 variability drastically
        if stdev(filtered_x0) >= 0.05 * page_width:
            valid_clusters.append(cluster)

    if not valid_clusters:
        return False

    cluster_words = sum(len(line.words) for cluster in valid_clusters for line in cluster)

    return cluster_words / len(ctx.words) > 0.75


def has_aligned_layout(ctx: PageContext) -> bool:
    """Checks if the majority of lines belong to clusters that are left-, right- or center-aligned."""
    total_words = len(ctx.words)
    if not total_words:
        return False

    clusters = get_all_layout_clusters(ctx.lines, ctx.page_rect.width)
    words = sum(len(line.words) for cluster in clusters for line in cluster)

    return words / len(ctx.words) > ALIGNED_WORD_RATIO_THRESHOLD


def has_large_font_layout(ctx: PageContext) -> bool:
    """Returns True if the page has at least one large font size and high font variety."""
    font_sizes = [line.font_size for line in ctx.lines]
    return len(set(font_sizes)) > 5 and max(font_sizes, default=0) > 20


def contains_content_clues(ctx: PageContext, matching_params) -> bool:
    """Returns True if the page contains at least 2 out of 3 indicators.

    - title keywords (language-dependent)
    - a date
    - a phone number
    """
    title_keywords = matching_params["title_page"].get(ctx.language, [])
    has_title_keyword = any(keyword in word.text.lower() for word in ctx.words for keyword in title_keywords)

    has_date = any(find_pattern(line, DATE_PATTERNS) for line in ctx.lines)
    has_phone = any(find_pattern(line, PHONE_PATTERNS) for line in ctx.lines)

    hits = sum([has_title_keyword, has_date, has_phone])
    return hits >= 2


def remove_outlier_if_needed(
    values: list[float], threshold: float = 0.6, removable_indices: Sequence[int] | None = None
) -> list[float]:
    """Removes one value (from allowed positions) if doing so significantly reduces the standard deviation.

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
        trial = values[:i] + values[i + 1 :]
        if len(trial) < 2:
            continue
        trial_std = stdev(trial)
        ratio = trial_std / original_std

        if ratio < best_ratio and ratio < threshold:
            best_ratio = ratio
            best_filtered = trial

    return best_filtered


def get_all_layout_clusters(lines: list[TextLine], page_width: float) -> list[list[TextLine]]:
    """Returns all text line clusters aligned either left, right, or center.

    A line can belong to only one cluster (first match).
    """
    remaining_lines = set(lines)
    all_clusters = []

    layout_funcs = [
        lambda line: line.rect.x0,  # left-aligned
        lambda line: line.rect.x1,  # right-aligned
        lambda line: line.rect.x0 + 0.5 * line.rect.width,  # center-aligned
    ]

    for layout in layout_funcs:
        unassigned_lines = list(remaining_lines)
        clusters = find_aligned_clusters(unassigned_lines, layout, threshold=0.05 * page_width)

        for cluster in clusters:
            all_clusters.append(cluster)
            remaining_lines.difference_update(cluster)

    return all_clusters


def find_aligned_clusters(
    lines: list[TextLine], key_func: Callable[[TextLine], float], threshold: float
) -> list[list[TextLine]]:
    """Groups text lines into alignment clusters based on a key function (e.g. x0-position).

    This function finds clusters of visually aligned lines by comparing each line’s alignment key
    (e.g. x0, center, x1) and vertical position.

       Parameters:
        lines (list[TextLine]): List of text lines to cluster.
        key_func (Callable): Function to extract the alignment key from a line (e.g. lambda l: l.rect.x0).
        threshold (float): Maximal misalignment between lines for them to be considered aligned.

    Returns:
        list[list[TextLine]]: List of clusters, each a list of aligned TextLines.
    """
    remaining_lines = set(lines)
    clusters = []

    for current_line in sorted(lines, key=lambda line: line.rect.y0):
        if current_line not in remaining_lines:
            continue  # already clustered

        remaining_lines.remove(current_line)
        cluster = []
        line_queue = [current_line]

        while line_queue:
            line = line_queue.pop()
            cluster_key = key_func(line)
            cluster.append(line)
            font_size = line.font_size

            close_lines = {
                other
                for other in remaining_lines
                if abs(key_func(other) - cluster_key) < threshold  # limit for how much misaligned other line can be
                and abs(other.rect.y0 - line.rect.y0)
                < VERTICAL_SPACING_FACTOR * font_size  # limit for how far below other line can lie
            }

            line_queue.extend(close_lines)  # add close lines into queue
            remaining_lines -= close_lines  # remove all close lines

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def vertical_spacing(lines: list[TextLine]) -> list[float]:
    """Compute vertical distances between vertically non-overlapping text lines.

    Filters out first or last line if they drastically decrease mean gaps.
    """
    merged_lines = merge_y_overlapping_lines(lines)
    # compute vertical spacing between merged lines
    distances = [below.rect.y0 - above.rect.y0 for above, below in zip(merged_lines, merged_lines[1:], strict=False)]
    # remove potential header / footnote
    filtered_distances = remove_outlier_if_needed(distances, threshold=0.6, removable_indices=[0, -1])

    return filtered_distances


def merge_y_overlapping_lines(lines: list[TextLine]) -> list[TextLine]:
    # sort lines by y0 (top) and x0 (left)
    sorted_lines = sorted(lines, key=lambda line: (line.rect.y0, line.rect.x0))

    # merge vertically overlapping lines
    merged_lines = []
    current_group = [sorted_lines[0]]

    def vertically_overlap(line1, line2):
        return not (line1.rect.y1 < line2.rect.y0 or line2.rect.y1 < line1.rect.y0)

    for i in range(1, len(sorted_lines)):
        if vertically_overlap(current_group[-1], sorted_lines[i]):
            current_group.append(sorted_lines[i])
        else:
            merged_words = [word for line in current_group for word in line.words]
            merged_lines.append(TextLine(merged_words))
            current_group = [sorted_lines[i]]

    # Add last group
    if current_group:
        merged_words = [word for line in current_group for word in line.words]
        merged_lines.append(TextLine(words=merged_words))

    return merged_lines
