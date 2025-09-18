import logging
import re
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


def identify_diagram(ctx: PageContext, matching_params: dict) -> bool:
    """Detect if page contains diagram.

    Check if Page contains horizontal and vertical numeric scale which has to be almost increasing or decreasing,
    Checks if Page contains keywords and units (e.g. kg/h)
    """
    keywords = matching_params["diagram"].get(ctx.language, [])
    keywords_unit = matching_params["units"]

    has_keyword = any(keyword in word.text.lower() for word in ctx.words for keyword in keywords)
    has_unit = has_units(ctx, keywords_unit)

    entries = detect_entries(ctx.words)

    vertical_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.x0, tolerance=10)
    horizontal_clusters = cluster_text_elements(entries, key_fn=lambda e: e.rect.y0, tolerance=10)

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

    votes = sum([has_keyword, has_unit, y_ok, x_ok])
    return votes >= 2


def normalize_direction(values: list[Entry]) -> list[Entry]:
    """Ensure values of entries go ascending; reverse if descending, leave otherwise."""
    if len(values) < 2:
        return values
    return values[::-1] if values[0].value > values[-1].value else values
