import numpy as np
import pymupdf

from src.bounding_box import _x_center
from src.page_structure import PageContext
from src.text_objects import TextColumn, TextTable, TextWord, cluster_connected_components


def identify_table(ctx: PageContext, min_conf: float = 0.6, min_coverage: float = 0.3) -> bool:
    """Identifies whether a page is likely to contain a table based on text alignment.

    Factors include:
    - Presence of text tables
    - Confidence of detected text tables based on row alignment
    - Text coverage of detected tables in relation to all text on the page
    """
    text_table = detect_text_table(ctx)
    if not text_table:
        return False

    good_text_tables = [table for table in text_table if table.confidence >= min_conf]

    return any(table.text_coverage(ctx.words) > min_coverage for table in good_text_tables)


def detect_text_table(
    ctx: PageContext, gap_factor: float = 4.0, x_tol: float = 2.0, min_cols: int = 3, max_noise: float = 1.5
) -> list[TextTable]:
    """Detects and returns text tables with minimum 3 columns on a page based on aligned text columns.

    Args:
        ctx: PageContext of the page to analyze.
        gap_factor: Factor by which mean vertical word gap will be multiplied
            To get maximal vertical gap with which column will be constructed.
            This allows adapting to different line spacings on the page.
        x_tol: Tolerance for x0 alignment of words to form columns (in px).
        min_cols: Minimum number of columns for a valid table.
        max_noise: Maximal noise (overlapping non-column words) a column is allowed to have to be considered valid.

    Returns:
        List of detected TextTables.
    """
    # adaptive vertical cap from page content
    word_gap = _median_line_gap(ctx.words)
    max_vert_gap = gap_factor * word_gap

    cols = make_text_columns(ctx.words, x_tol=x_tol, max_vertical_gap=max_vert_gap)
    if len(cols) < min_cols:
        return []

    valid_cols = [col for col in cols if col.noise(ctx.words) < max_noise]
    tables = make_text_tables(valid_cols)

    return [table for table in tables if len(table.columns) >= min_cols]


def make_text_columns(
    words: list[TextWord],
    max_vertical_gap: float,
    x_tol: float = 2.0,
    min_words: int = 3,
    overlap_min: float = 0.8,
) -> list[TextColumn]:
    """Build TextColumns based on x0 alignment of TextWords.

    Args:
        words: List of TextWords on the page.
        max_vertical_gap: max vertical distance allowed between word within one column
        x_tol: Tolerance for x0 alignment (in px).
        min_words: Minimum number of words in a column to be considered valid.
        overlap_min: Minimum horizontal overlap ratio of the words' bounding boxes.

    Returns:
        List of detected TextColumns.
    """
    if not words:
        return []

    def pred(a, b):
        return _belongs_to_same_column(
            a,
            b,
            x_tol=x_tol,
            overlap_min=overlap_min,
            max_vert_gap=max_vertical_gap,
        )

    clusters = cluster_connected_components(words, pred)
    return [TextColumn(col) for col in clusters if len(col) >= min_words]


def _belongs_to_same_column(
    w1: TextWord,
    w2: TextWord,
    x_tol: float,
    overlap_min: float,
    max_vert_gap: float,
) -> bool:
    """Checks if two words can belong to the sam TextColumn.

    Args:
        w1: First TextWord.
        w2: Second TextWord.
        x_tol: x distance tolerance between the x centers of the words.
        overlap_min: minimum horizontal overlap ratio of the words' bounding boxes.
        max_vert_gap: maximum vertical distance between the words.

    Returns:
        True if the two words belong to the same column.

    Words belong to the same column:
    - if their x centers are closer together than the x tolerance threshold.
    - if their bounding boxes overlap more than overlap_min.
    - if their vertical distance to each other is smaller than max_vert_gap
    """
    r1, r2 = w1.rect, w2.rect

    # 1) Similar horizontal anchor
    x_close = round(abs(_x_center(r1) - _x_center(r2))) <= x_tol
    x_overlap_ok = _hproj_overlap_ratio(r1, r2) >= overlap_min
    x_ok = x_close and x_overlap_ok

    if not x_ok:
        return False

    # 2) Vertically adjacent
    vertical_gap = _vertical_gap(r1, r2)
    return vertical_gap <= max_vert_gap


def make_text_tables(cols: list[TextColumn]) -> list[TextTable]:
    """Groups TextColumns into TextTables based on alignment.

    Args:
        cols: List of TextColumns to group into tables.

    Returns:
        List of detected TextTables.
    """
    components = cluster_connected_components(cols, _columns_align)
    return [TextTable(comp) for comp in components if len(comp) > 1]


def _columns_align(
    col1: TextColumn,
    col2: TextColumn,
    min_vert_overlap: float = 0.80,
    max_horizontal_overlap: float = 0.10,
    edge_align_tol: float = 6.0,
) -> bool:
    """Checks if two text columns align to form a table structure.

    Conditions:
    - Top or bottom edges roughly aligned  OR sufficient vertical overlap
    - have no more than a small horizontal overlap
    Args:
        col1: First TextColumn.
        col2: Second TextColumn.
        min_vert_overlap: Minimum vertical overlap ratio to consider aligned if edges not aligned.
        max_horizontal_overlap: Maximum horizontal overlap ratio to consider aligned.
        edge_align_tol: Tolerance in px for top/bottom edge alignment.

    Returns:
        True if columns align, False otherwise.
    """
    rect1, rect2 = col1.rect, col2.rect

    # Check vertical alignment, if not: check if vertical overlap ratio smaller than allowed
    if abs(rect1.y0 - rect2.y0) > edge_align_tol or abs(rect1.y1 - rect2.y1) > edge_align_tol:
        y0, y1 = max(rect1.y0, rect2.y0), min(rect1.y1, rect2.y1)
        v_inter = max(0.0, y1 - y0)
        v_ratio = v_inter / max(1e-6, min(rect1.height, rect2.height))
        if v_ratio < min_vert_overlap:
            return False

    # Check if horizontal overlap ratio smaller than allowed
    x0, x1 = max(rect1.x0, rect2.x0), min(rect1.x1, rect2.x1)
    h_inter = max(0.0, x1 - x0)
    h_ratio = h_inter / max(1e-6, min(rect1.width, rect2.width))
    return h_ratio <= max_horizontal_overlap


def _median_line_gap(words: list[TextWord]) -> float:
    """Rough line spacing estimate from nearest vertical neighbors."""
    if len(words) < 2:
        return 20.0
    ys = sorted((w.rect.y0 + w.rect.y1) / 2.0 for w in words)  # baselines
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] > ys[i]]
    return max(10.0, float(np.median(gaps))) if gaps else 20.0


def _hproj_overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Horizontal overlap ratio relative to the smaller width."""
    left = max(a.x0, b.x0)
    right = min(a.x1, b.x1)
    overlap = max(0.0, right - left)
    denom = max(1e-6, min(a.x1 - a.x0, b.x1 - b.x0))
    return overlap / denom


def _vertical_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Positive vertical distance between boxes if they don't overlap; 0 if they overlap."""
    if a.y1 < b.y0:
        return b.y0 - a.y1
    if b.y1 < a.y0:
        return a.y0 - b.y1
    return 0.0
