"""Convert title / section document to processed entries."""

import re
from dataclasses import dataclass

import pymupdf
from pymupdf import Rect
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.textblock import TextBlock

from src.utils.text_clustering import create_text_blocks

_INSTITUTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "école",
        "ecole",
        "bundesamt",
        "bundesanstalt",
        "universität",
        "université",
        "universite",
        "hochschule",
        "ag",
        "gmbh",
        "sàrl",
        "département",
        "departement",
        "kantonales",
        "prof",
        "dr",
    }
)


@dataclass
class TitleCandidateTextBlock:
    """A scale-invariant text block candidate for title detection."""

    text: str
    line_count: int
    rect: pymupdf.Rect
    font_size: float
    font_consistency: float
    isolation: float

    def __init__(
        self,
        text_block: TextBlock,
        rect: Rect,
        font_size: float = 0.0,
        font_consistency: float = 1.0,
    ):
        """Create a scale invariant text block.

        The normalized text block is contained in a fictive [0, 0, 1, 1] rect.

        Args:
            text_block (TextBlock): Input text block.
            rect (Rect): Size of the page linked to text block.
            font_size (float): Median span font size in points for this block.
            font_consistency (float): Pre-computed font uniformity score; 1.0 if all
                spans share the same font size within 1 pt, 0.5 otherwise.
        """
        self.text = text_block.text
        self.line_count = text_block.line_count
        self.rect = pymupdf.Rect(
            text_block.rect.x0 / rect.width,
            text_block.rect.y0 / rect.height,
            text_block.rect.x1 / rect.width,
            text_block.rect.y1 / rect.height,
        )
        self.font_size = font_size
        self.font_consistency = font_consistency
        self.isolation = 1.0  # assigned externally by _assign_isolation_scores

    @property
    def length(self) -> float:
        """Return True if the text contains more than 5 characters."""
        return float(len(self.text) > 5)

    @property
    def horizontality(self) -> float:
        """Return True if the block starts in the left 40% of the page width."""
        return float(self.rect.x0 < 0.4)

    @property
    def verticality(self) -> float:
        """Return True if the block ends in the upper 75% of the page height."""
        return float(self.rect.y1 < 0.75)

    @property
    def non_numericality(self) -> float:
        """Return the fraction of non-digit characters in the text.

        Returns:
            float: Value in [0, 1]; 1.0 means no digits, 0.0 means all digits.
        """
        n_digits = len(re.findall(r"\d", self.text))
        n_total = len(self.text)
        return 1 - (n_digits / max(n_total, 1))

    @property
    def font(self) -> float:
        """Return an approximate normalised font size (block height per line), squared.

        Squaring amplifies the advantage of larger-font blocks over smaller ones.
        """
        return (self.rect.height / max(self.line_count, 1)) ** 2

    @property
    def highness(self) -> float:
        """Return a score favouring blocks near the top of the page."""
        return 1 - self.rect.y0

    @property
    def all_caps(self) -> float:
        """Return a boost factor when every alphabetic character in the text is uppercase.

        Returns:
            float: 1.5 if all alphabetic characters are uppercase, 1.0 otherwise.
        """
        alpha = [c for c in self.text if c.isalpha()]
        return 1.5 if alpha and all(c.isupper() for c in alpha) else 1.0

    @property
    def no_institution(self) -> float:
        """Return a penalty factor when institution or author keywords are detected.

        Checks for words typical of company names, government bodies, or research
        institutions (e.g. 'AG', 'GmbH', 'Bundesamt', 'école').

        Returns:
            float: 0.2 if an institution keyword is found as a whole word, 1.0 otherwise.
        """
        words = set(re.split(r"[\s,;.:()\[\]/]+", self.text.lower()))
        return 0.2 if words & _INSTITUTION_KEYWORDS else 1.0

    @property
    def score(self) -> float:
        """Return a composite title-likelihood score.

        Multiplies all heuristic signals: font size, horizontal position,
        vertical position, text length, non-numericality, highness,
        all-caps boost, institution keyword penalty, block isolation,
        and font-size consistency.
        A higher score indicates a stronger title candidate.

        Returns:
            float: Non-negative composite score; 0 if any binary signal is False/zero.
        """
        return (
            self.font
            # * self.horizontality
            * self.verticality
            * self.length
            * self.non_numericality
            * self.highness
            * self.all_caps
            * self.no_institution
            * self.isolation
            # * self.font_consistency
        )


def _assign_isolation_scores(candidates: list[TitleCandidateTextBlock]) -> None:
    """Set isolation on each candidate based on vertical gap to its nearest neighbour.

    Scores range from 0.5 (block immediately adjacent to another) to 1.0
    (block separated by ≥10 % of the normalised page height).
    Candidates are modified in-place.

    Args:
        candidates: All scored candidates for the page.
    """
    if len(candidates) <= 1:
        return
    for i, cand in enumerate(candidates):
        min_gap = min(
            max(0.0, max(cand.rect.y0, other.rect.y0) - min(cand.rect.y1, other.rect.y1))
            for j, other in enumerate(candidates)
            if i != j
        )
        # A gap of 0.1 (10 % of page height) is treated as fully isolated.
        cand.isolation = 0.5 + 0.5 * min(min_gap / 0.1, 1.0)


def _block_font_metrics(page_dict: dict, block_rect: pymupdf.Rect) -> tuple[float, float]:
    """Return the median font size and uniformity score for spans overlapping the block.

    Args:
        page_dict: Output of ``page.get_text("dict")``.
        block_rect: Unnormalized bounding box of the text block (page coordinates).

    Returns:
        tuple[float, float]: ``(median_font_size, consistency)`` where ``consistency``
            is 1.0 if all overlapping spans share the same font size within 1 pt,
            0.5 otherwise. ``median_font_size`` is 0.0 when no spans are found.
    """
    sizes = [
        span["size"]
        for block in page_dict.get("blocks", [])
        if block.get("type") == 0  # text blocks only
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip() and pymupdf.Rect(span["bbox"]).intersects(block_rect)
    ]
    if not sizes:
        return 0.0, 1.0
    sizes_sorted = sorted(sizes)
    mid = len(sizes_sorted) // 2
    median = (sizes_sorted[mid] + sizes_sorted[~mid]) / 2
    consistency = 1.0 if (max(sizes) - min(sizes)) < 1.0 else 0.5
    return median, consistency


def _merge_title_continuation(
    top: TitleCandidateTextBlock,
    candidates: list[TitleCandidateTextBlock],
) -> str:
    """Extend the top candidate downward by merging vertically adjacent continuation blocks.

    A block is appended when it immediately follows the current bottom edge (gap ≤ 3×
    the top block's line height), shares a similar font size (within 1 pt), and does not
    trigger the institution keyword penalty — which stops the merge at author or
    affiliation lines.

    Args:
        top: The highest-scoring title candidate.
        candidates: All candidates for the page.

    Returns:
        The full merged title text.
    """
    top_line_height = top.rect.height / max(top.line_count, 1)
    max_gap = max(top_line_height * 3, 0.02)

    below = sorted(
        (c for c in candidates if c is not top and c.rect.y0 >= top.rect.y1),
        key=lambda c: c.rect.y0,
    )

    parts = [top.text]
    current_bottom = top.rect.y1

    for candidate in below:
        if candidate.rect.y0 - current_bottom > max_gap:
            break
        if top.font_size > 0 and candidate.font_size > 0 and abs(candidate.font_size - top.font_size) > 1.0:
            break
        if candidate.no_institution < 1.0:
            break
        parts.append(candidate.text)
        current_bottom = candidate.rect.y1

    return " ".join(parts)


def extract_title_from_page(page: pymupdf.Page) -> str:
    """Extract the most likely title string from a single PDF page.

    Builds text blocks from the page's text lines, wraps them as
    scale-invariant blocks, scores them by title-likelihood, and returns
    the text of the highest-scoring candidate.

    Args:
        page (pymupdf.Page): The PDF page to analyse.

    Returns:
        str: Detected title for the page.
    """
    # Extract text segments from page and convert to text blocks
    lines = extract_text_lines(page)
    text_blocks = create_text_blocks(lines)

    page_dict = page.get_text("dict")
    title_candidates = []
    for text_block in text_blocks:
        font_size, font_consistency = _block_font_metrics(page_dict, text_block.rect)
        title_candidates.append(
            TitleCandidateTextBlock(
                text_block=text_block,
                rect=page.rect,
                font_size=font_size,
                font_consistency=font_consistency,
            )
        )
    _assign_isolation_scores(title_candidates)
    title_candidates = sorted(title_candidates, key=lambda x: x.score, reverse=True)
    return title_candidates[0].text if title_candidates else ""
    # if not title_candidates:
    #     return ""
    # return _merge_title_continuation(title_candidates[0], title_candidates)
