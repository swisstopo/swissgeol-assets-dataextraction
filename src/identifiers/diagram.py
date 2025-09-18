import logging
import math
import re
import statistics
from collections.abc import Callable

from src.identifiers.boreprofile import Entry, detect_entries, is_mostly_increasing
from src.page_structure import PageContext
from src.text_objects import cluster_text_elements

logger = logging.getLogger(__name__)


def has_units(ctx: PageContext, units: list[str]) -> bool:
    """Check if TextWords of PageContext contain units.

    We allow for (unit), [unit], with optional spaces inside
    """
    for word in ctx.words:
        text = word.text.lower()
        for u in units:
            pattern = r"[\(\[]\s*" + re.escape(u) + r"\s*[\)\]]"
            if re.search(pattern, text):
                return True
    return False


def identify_diagram(ctx: PageContext, matching_params: dict, axis_tolerance: int = 10) -> bool:
    """Determines whether a page contains a diagram based on keywords, units, and axis detection.

    Detection logic:
    - If a diagram keyword (language-specific) and a unit are present on the page, returns True.
    - Otherwise, attempts to detect both x- and y-axes by clustering numbers based on their positions
    - Returns True if both axes appear to be ordered scales, or if at least one axis shows numeric progression.
    """
    keywords = (matching_params.get("diagram", {}) or {}).get(ctx.language, []) or []
    units_cfg = matching_params.get("units", [])

    words_lower = (word.text.lower() for word in ctx.words)
    has_keyword = any(key in word for word in words_lower for key in keywords)
    if has_keyword and has_units(ctx, units_cfg):
        return True

    entries = detect_entries(ctx.words)

    vertical_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.x0, tolerance=axis_tolerance)
    horizontal_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.y0, tolerance=axis_tolerance)

    y_mono, y_prog = axis_checks(vertical_clusters, sort_key=lambda e: e.rect.y0)
    x_mono, x_prog = axis_checks(horizontal_clusters, sort_key=lambda e: e.rect.x0)

    if y_mono and x_mono:
        return True  #  both axes look like ordered scales
    return y_prog or x_prog  # at least one axis shows numeric progression


def axis_checks(clusters: list, sort_key: Callable) -> tuple[bool, bool]:
    """Checks clusters of values for monotonicity and numeric progression.

    Args:
        clusters: A list of clusters, where each cluster is a sequence of values.
        sort_key: Function used to sort the values within each cluster.

    Returns:
        tuple[bool, bool]:
            - The first element is True if any cluster is mostly increasing.
            - The second element is True if any cluster shows a valid numeric progression.
    """
    any_monotone = False
    any_progression = False

    for c in clusters:
        if len(c) < 3:
            continue
        axis = sorted(c, key=sort_key)

        mono = is_mostly_increasing(normalize_direction(axis))
        prog = detect_progression(axis)

        any_monotone |= mono
        any_progression |= prog

        if any_monotone and any_progression:
            break

    return any_monotone, any_progression


def normalize_direction(values: list[Entry]) -> list:
    """Ensure values of entries go ascending; reverse if descending, leave otherwise."""
    if len(values) < 2:
        return values
    return values[::-1] if values[0].value > values[-1].value else values


def detect_progression(entries: list[Entry]) -> bool:
    """Check if entries follow an arithmetic or log progression."""
    values = normalize_direction(entries)  # flip list if descending
    values = [value.value for value in values]

    return is_arithmetic_progression(values) or is_log_progression(values)


def is_arithmetic_progression(
    values: list[float],
    frac_ok: float = 0.8,
    abs_tol: float = 0.25,
) -> bool:
    """Checks if values approximately follow almost an arithmetic progression.

    values: values of the potential arithmetic progression.
    frac_ok: fraction of steps that must match the median (e.g. 0.8 allows some OCR noise).
    abs_tol: minimum absolute tolerance so small steps survive rounding/jitter. .
    """
    if len(values) <= 2:
        return False
    diffs = [b - a for a, b in zip(values, values[1:], strict=False)]
    step = round(statistics.median(diffs), 2)
    # has to be a meaningful positive step
    if step <= 0:
        return False

    same_steps = sum(1 for diff in diffs if math.isclose(diff, step, abs_tol=abs_tol))
    return same_steps >= frac_ok * len(diffs)


def is_log_progression(values: list, tol: float = 0.1) -> bool:
    """Checks if values are a log10 based progression."""
    if any(v <= 0 for v in values) or len(values) <= 2:  ## log never negative
        return False
    log_vals = [math.log10(v) for v in values]
    diffs = [b - a for a, b in zip(log_vals, log_vals[1:], strict=False)]
    common_steps = (math.log10(2), math.log10(3), math.log10(5), 1.0)
    good = 0
    for diff in diffs:
        if any(abs(diff - common_step) <= tol for common_step in common_steps):
            good += 1
    return good >= 0.8 * len(diffs)
