from src.page_structure import PageContext
from src.text_objects import TextColumn, TextTable, TextWord, cluster_text_elements


def identify_table(ctx: PageContext, min_conf: float = 0.6, min_coverage: float = 0.5) -> bool:
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


def detect_text_table(ctx: PageContext, x_tol: int = 2, min_cols: int = 2) -> list[TextTable]:
    """Detects and returns text tables with minimum 3 columns on a page based on aligned text columns.

    Args:
        ctx: PageContext of the page to analyze.
        x_tol: Tolerance for x0 alignment of words to form columns (in px).
        min_cols: Minimum number of columns for a valid table.

    Returns:
        List of detected TextTables.
    """
    cols = make_text_columns(ctx.words, x_tol=x_tol)
    if len(cols) < min_cols:
        return []

    tables = make_text_tables(cols)
    return [table for table in tables if len(table.columns) > min_cols]


def make_text_columns(words: list[TextWord], x_tol: int = 2, min_words: int = 3) -> list[TextColumn]:
    """Build TextColumns based on x0 alignment of TextWords.

    Args:
        words: List of TextWords on the page.
        x_tol: Tolerance for x0 alignment (in px).
        min_words: Minimum number of words in a column to be considered valid.

    Returns:
        List of detected TextColumns.
    """
    clusters = cluster_text_elements(words, key_fn=lambda w: w.rect.x0, tolerance=x_tol)
    return [TextColumn(col) for col in clusters if len(col) >= min_words]


def make_text_tables(cols: list[TextColumn]) -> list[TextTable]:
    """Groups TextColumns into TextTables based on alignment.

    Uses a simple BFS approach.

    Args:
        cols: List of TextColumns to group into tables.

    Returns:
        List of detected TextTables.
    """
    n = len(cols)
    visited = [False] * n
    tables: list[TextTable] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        table_columns = [cols[i]]
        while queue:
            u = queue.pop()
            for v in range(n):
                if visited[v] or v == u:
                    continue
                if _columns_align(cols[u], cols[v]):
                    visited[v] = True
                    queue.append(v)
                    table_columns.append(cols[v])
        table = TextTable(table_columns)
        tables.append(table)

    return tables


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
