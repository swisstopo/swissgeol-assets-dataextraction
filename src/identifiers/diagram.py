import logging
import math
import re
import statistics

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


def identify_diagram(ctx: PageContext, matching_params: dict) -> bool:
    """Detect if page contains diagram.

    Check if Page contains horizontal  and vertical numeric scale which has to be almost increasing or decreasing,
    Checks if Page contains keywords and units (f.e kg/h)
    """
    keywords = matching_params["diagram"].get(ctx.language, [])
    keywords_unit = matching_params["units"]

    has_keyword = any(keyword in word.text.lower() for word in ctx.words for keyword in keywords)
    has_unit = has_units(ctx, keywords_unit)

    if has_keyword and has_unit:
        return True

    entries = detect_entries(ctx.words)

    vertical_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.x0, tolerance=10)
    horizontal_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.y0, tolerance=10)

    y_axis = [c for c in vertical_clusters if len(c) >= 3]
    x_axis = [c for c in horizontal_clusters if len(c) >= 3]

    def sorted_axis(axis, key):
        return [sorted(c, key=key) for c in axis]

    y_axis_sorted = sorted_axis(y_axis, key=lambda e: e.rect.y0)
    x_axis_sorted = sorted_axis(x_axis, key=lambda e: e.rect.x0)

    y_ok = any(is_mostly_increasing(normalize_direction(axis)) for axis in y_axis_sorted)
    x_ok = any(is_mostly_increasing(normalize_direction(axis)) for axis in x_axis_sorted)

    return y_ok and x_ok


def normalize_direction(values: list[Entry]) -> list:
    """Ensure values of entries go ascending; reverse if descending, leave otherwise."""
    if len(values) < 2:
        return values
    return values[::-1] if values[0].value > values[-1].value else values


##################tried but worse F1 ####################
def detect_progression(entries: list[Entry]) -> bool:
    """Check if entries follow an arithmetic progression."""
    values = sorted(
        entry.value for entry in entries
    )  # probably smarter not sort but check if ascending or descending...

    if is_arithmetic_progression(values):
        return True
    return is_log_progression(values)


def is_arithmetic_progression(values: list) -> bool:
    if len(values) <= 2:
        return False
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    step = round(statistics.median(diffs), 2)
    if step > 0:
        first, last = values[0], values[-1]
        arithmetic_progression = {round(n * step, 2) for n in range(int(first / step), int(last / step) + 1)}
        score = sum(val in arithmetic_progression for val in values)
        if score >= 0.8 * len(values):  # tolerate 20% OCR noise
            return True
    return False


def is_log_progression(values, tol=0.1):
    if any(v <= 0 for v in values) or len(values) <= 2:
        return False
    log_vals = sorted(math.log10(v) for v in values)
    diffs = [round(log_vals[i + 1] - log_vals[i], 1) for i in range(len(log_vals) - 1)]

    common_steps = [round(math.log10(s), 2) for s in (2, 3, 5, 10)]  # ≈0.3, 0.48, 0.7, 1.0 log10 base

    good = 0
    for d in diffs:
        if any(abs(d - cs) <= tol for cs in common_steps):
            good += 1
    return good >= 0.8 * len(diffs)
