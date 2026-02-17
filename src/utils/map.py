import logging

import numpy as np
import regex
from scipy.stats import entropy
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textline import TextLine

logger = logging.getLogger(__name__)
pattern_maps = [
    regex.compile(r"1\s*:\s*[125](25|5)?000+"),
    regex.compile(r"1\s*:\s*[125]((0{1,2})?([',]000)+)"),
]


def find_map_scales(line: TextLine) -> regex.Match | None:
    return next(
        (match for pattern in pattern_maps for word in line.words if (match := pattern.search(word.text))),
        None,
    )


def is_grid_angle(angle: float, tolerance: float = 2.0) -> bool:
    """Check if angle is approximately horizontal or vertical."""
    return any(abs(angle - degree) < tolerance for degree in (0, 90, 180))


def split_lines_by_orientation(geometric_lines: list[Line]):
    """Return length of geometric lines in grid and non grid lists."""
    grid, non_grid = [], []

    for line in geometric_lines:
        if is_grid_angle(line.angle, tolerance=2.0):
            grid.append(line.length)
        else:
            non_grid.append(line.length)

    return grid, non_grid


def compute_angle_entropy(angles, angle_bin_count: int = 36):
    """Compute normalized entropy over the angle histogram.

    We compute the Shannon entropy H(p), measuring the uncertainty in the angle distribution.
    Angles are binned into `angle_bin_count` bins (default = 36), i.e., 5° intervals over [0, 180).
    The entropy is normalized by dividing by log2(angle_bin_count),
    the maximum possible entropy for a uniform distribution.
    This scales entropy to the range [0, 1].
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    """
    angle_hist = np.histogram(angles, bins=angle_bin_count, range=(0, 180))[0]
    return entropy(angle_hist) / np.log2(angle_bin_count)


def map_lines_score(geometric_lines: list[Line]) -> float:
    """Returns a score (0.0 to 1.0) indicating whether the page contains map-like line structure.

     A high score suggests the presence of:
    - Diverse angles (curved or non-orthogonal features, like contour lines)
    - Sum of non-grid line lengths higher than sum of to grid line lengths
    """
    if not geometric_lines:
        return 0.0

    angles = [line.angle for line in geometric_lines]

    # Grid/non-grid splitting of lines
    grid_lengths, non_grid_lengths = split_lines_by_orientation(geometric_lines)
    grid_length_sum = sum(grid_lengths)
    non_grid_length_sum = sum(non_grid_lengths)

    non_grid_length_ratio = non_grid_length_sum / (grid_length_sum + 1)  # avoid division by zero

    angle_entropy = compute_angle_entropy(angles)

    score = 0.5 * angle_entropy + 0.4 * min(non_grid_length_ratio / 10, 1.0)

    return score
