"""Custom table detection structures.

- TextColumn: A vertical column of text
- TextTable: A table composed of multiple aligned columns

Note: These classes use swissgeol_doc_processing.text.textline.TextWord
"""

from dataclasses import dataclass, field

import pymupdf
from swissgeol_doc_processing.text.textline import TextWord

from src.bounding_box import _y_center, merge_bounding_boxes
from src.utils.text_clustering import cluster_connected_components


@dataclass
class TextColumn:
    """A vertical column of text, composed of multiple TextWords.

    This class represents a detected column in table-like structures.
    It provides methods to compute bounding boxes and assess column quality.

    Attributes:
        words: List of TextWord objects that make up the column
    """

    words: list[TextWord]

    @property
    def rect(self) -> pymupdf.Rect:
        """Compute the bounding box of the entire column.

        Returns:
            Bounding rectangle encompassing all words in the column
        """
        return merge_bounding_boxes([w.rect for w in self.words])

    def __repr__(self) -> str:
        return f"TextColumn({[word.text for word in self.words]},{self.rect},)"

    def noise(self, all_words: list[TextWord]) -> float:
        """Calculate noise metric for the column.

        Metric showing how noisy a column is. It's the ratio between
        actual column entries and non-column words overlapping with
        the column bounding box.

        Args:
            all_words: All words on the page

        Returns:
            Noise ratio. Best noise = 1.0 (intersecting words = self.words).
            More intersecting words will lead to a higher ratio.

        Example:
            If column has 5 words but 10 words intersect the column bbox,
            noise = 10 / 5 = 2.0 (indicating high noise)
        """
        column_bbox = self.rect
        intersecting_words = [word for word in all_words if column_bbox.intersects(word.rect)]
        ratio = len(intersecting_words) / len(self.words) if self.words else 0.0
        return ratio


@dataclass
class TextTable:
    """A table composed of multiple aligned TextColumns.

    This class represents a detected table structure with multiple columns.
    It provides metrics for table quality assessment including coverage
    and confidence based on row alignment.

    Attributes:
        columns: List of TextColumn objects that make up the table
        words: All words in the table (computed from columns)
    """

    columns: list[TextColumn]
    words: list[TextWord] = field(init=False)

    def __post_init__(self):
        """Initialize words list from all columns."""
        self.words = [word for col in self.columns for word in col.words]

    @property
    def rect(self) -> pymupdf.Rect:
        """Compute bounding box of the entire table.

        Returns:
            Bounding rectangle encompassing all columns
        """
        return merge_bounding_boxes([c.rect for c in self.columns if c.rect is not None])

    def height_coverage(self, page_height: float) -> float:
        """Calculate fraction of page height covered by table.

        Args:
            page_height: Height of the page in points

        Returns:
            Fraction of page height (0.0 to 1.0)
        """
        return self.rect.height / page_height if page_height > 0 else 0.0

    def text_coverage(self, all_words: list[TextWord]) -> float:
        """Calculate fraction of page words belonging to the table.

        Args:
            all_words: All words on the page

        Returns:
            Fraction of words in table (0.0 to 1.0)

        Example:
            If page has 100 words and table contains 30 words, coverage = 0.3
        """
        if not all_words:
            return 0.0
        coverage = sum(len(col.words) for col in self.columns) / len(all_words)
        return coverage

    @property
    def confidence(self) -> float:
        """Calculate confidence score based on row alignment across columns.

        Confidence is based on how well words align into rows across columns.
        Better alignment (more words on same horizontal level) means higher
        confidence.

        Algorithm:
        1. Collect all words from all columns
        2. Cluster words that are on the same horizontal level (same row)
        3. Confidence = 1 - (number of merged rows) / (total entries)

        Returns:
            Confidence score (0.0 to 1.0). Higher is better.
            - 1.0 = Perfect alignment (each word is its own row)
            - 0.0 = Poor alignment (all words merged into one row)

        Example:
            If table has 3 columns with 5 words each (15 total),
            and they align into 5 rows perfectly, merged_count = 5,
            confidence = 1 - (5 / 15) = 0.67
        """
        if not self.columns:
            return 0.0

        def _same_row(w1: TextWord, w2: TextWord, tolerance: float = 3.0) -> bool:
            """Check if two words are on the same horizontal row.

            Args:
                w1: First word
                w2: Second word
                tolerance: Maximum vertical difference in pixels

            Returns:
                True if words are vertically aligned within tolerance
            """
            w1_center, w2_center = _y_center(w1.rect), _y_center(w2.rect)
            return abs(w1_center - w2_center) <= tolerance

        # Cluster words into rows based on vertical alignment
        merged_rows = cluster_connected_components(self.words, _same_row)

        total_entries = len(self.words)
        if total_entries == 0:
            return 0.0

        # Fewer merged rows relative to total entries indicates better alignment
        # If each word is its own row (perfect table), merged_count = total_entries
        # If all words merge into one row (bad table), merged_count = 1
        merged_count = len(merged_rows)
        q_rows = 1.0 - (merged_count / total_entries)

        return q_rows
