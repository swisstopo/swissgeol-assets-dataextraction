"""Title extraction from PDF section-header pages."""

import re
from dataclasses import dataclass

import pymupdf
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
    """A normalized text block candidate for title scoring.

    All positional attributes are expressed in page-relative coordinates
    so that scores are comparable across pages of different sizes.
    """

    text: str
    line_count: int
    rect: pymupdf.Rect
    font_size: float
    isolation: float

    def __init__(
        self,
        text_block: TextBlock,
        page_rect: pymupdf.Rect,
        font_size: float = 0.0,
    ):
        """Create a normalized title candidate from a raw text block.

        Coordinates are scaled to the unit square [0, 0, 1, 1] relative to the page.

        Args:
            text_block: Raw text block from the page.
            page_rect: Bounding rectangle of the page (page coordinates).
            font_size: Median span font size in points for this block.
        """
        self.text = text_block.text
        self.line_count = text_block.line_count
        self.rect = pymupdf.Rect(
            text_block.rect.x0 / page_rect.width,
            text_block.rect.y0 / page_rect.height,
            text_block.rect.x1 / page_rect.width,
            text_block.rect.y1 / page_rect.height,
        )
        self.font_size = font_size
        self.isolation = 1.0  # assigned externally by _assign_isolation_scores

    @property
    def length(self) -> float:
        """Return 1.0 if the text contains more than 5 characters, 0.0 otherwise."""
        return float(len(self.text) > 5)

    @property
    def verticality(self) -> float:
        """Return 1.0 if the block ends in the upper 75 % of the page, 0.0 otherwise."""
        return float(self.rect.y1 < 0.75)

    @property
    def non_numericality(self) -> float:
        """Return the fraction of non-digit characters in the text.

        Returns:
            float: Value in [0, 1]; 1.0 means no digits, 0.0 means all digits.
        """
        n_digits = len(re.findall(r"\d", self.text))
        return 1 - (n_digits / max(len(self.text), 1))

    @property
    def font(self) -> float:
        """Return a normalized font-size proxy (block height per line), squared.

        Uses the block's normalized height divided by its line count as a proxy for
        font size. Squaring amplifies the advantage of larger-font blocks over smaller ones.
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

        Multiplies heuristic signals: font-size proxy, vertical position, text length,
        non-numericality, page height position, all-caps boost, institution keyword
        penalty, and block isolation. A higher score indicates a stronger title candidate.

        Returns:
            float: Non-negative composite score; 0 if any binary signal is zero.
        """
        return (
            self.font
            * self.verticality
            * self.length
            * self.non_numericality
            * self.highness
            * self.all_caps
            * self.no_institution
            * self.isolation
        )


def _assign_isolation_scores(candidates: list[TitleCandidateTextBlock]) -> None:
    """Set isolation on each candidate based on vertical gap to its nearest neighbour.

    Scores range from 0.5 (block immediately adjacent to another) to 1.0
    (block separated by ≥ 10 % of the normalized page height).
    Candidates are modified in-place.

    Args:
        candidates: All title candidates for the page.
    """
    if len(candidates) <= 1:
        return
    for cand in candidates:
        min_gap = min(
            max(0.0, max(cand.rect.y0, other.rect.y0) - min(cand.rect.y1, other.rect.y1))
            for other in candidates
            if other is not cand
        )
        # A gap of 0.1 (10 % of page height) is treated as fully isolated.
        cand.isolation = 0.5 + 0.5 * min(min_gap / 0.1, 1.0)


def _median_block_font_size(page_dict: dict, block_rect: pymupdf.Rect) -> float:
    """Return the median font size across all text spans overlapping the block.

    Args:
        page_dict: Output of ``page.get_text("dict")``.
        block_rect: Bounding box of the text block in page coordinates (not normalized).

    Returns:
        Median span font size in points, or 0.0 if no spans overlap the block.
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
        return 0.0
    sizes_sorted = sorted(sizes)
    mid = len(sizes_sorted) // 2
    return (sizes_sorted[mid] + sizes_sorted[~mid]) / 2


def extract_title_from_page(page: pymupdf.Page) -> str:
    """Extract the most likely title string from a single PDF page.

    Builds text blocks from the page's text lines, scores them by title-likelihood,
    and returns the text of the highest-scoring candidate.

    Args:
        page: The PDF page to analyse.

    Returns:
        Detected title text, or an empty string if no candidates are found.
    """
    lines = extract_text_lines(page)
    text_blocks = create_text_blocks(lines)

    page_dict = page.get_text("dict")
    title_candidates = [
        TitleCandidateTextBlock(
            text_block=text_block,
            page_rect=page.rect,
            font_size=_median_block_font_size(page_dict, text_block.rect),
        )
        for text_block in text_blocks
    ]
    _assign_isolation_scores(title_candidates)
    title_candidates.sort(key=lambda x: x.score, reverse=True)
    return title_candidates[0].text if title_candidates else ""
