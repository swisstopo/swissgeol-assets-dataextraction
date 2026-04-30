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

    def __init__(self, text_block: TextBlock, rect: pymupdf.Rect):
        """Create a normalized title candidate from a raw text block.

        Coordinates are scaled to the unit square [0, 0, 1, 1] relative to the page.

        Args:
            text_block: Raw text block from the page.
            rect: Bounding rectangle of the page (page coordinates).
        """
        self.text = text_block.text
        self.line_count = text_block.line_count
        self.rect = pymupdf.Rect(
            text_block.rect.x0 / rect.width,
            text_block.rect.y0 / rect.height,
            text_block.rect.x1 / rect.width,
            text_block.rect.y1 / rect.height,
        )
        self._isolation: float = 1.0

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
    def text_area(self) -> float:
        """Return the area (width × height) of the text block in page-relative coordinates.

        Rewards blocks that occupy real page area rather than just per-line height,
        so a narrow single-word stamp is not unfairly boosted over a wide title.
        """
        return self.rect.width * self.rect.height / max(self.line_count, 1)

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
    def isolation(self) -> float:
        """Return the isolation score set by assign_isolation."""
        return self._isolation

    def assign_isolation(self, candidates: list["TitleCandidateTextBlock"]) -> None:
        """Set isolation based on the 2D bounding-box distance to the nearest other candidate.

        Uses both horizontal and vertical gaps so that side-by-side blocks in a
        two-column layout are not treated as adjacent to the title.
        Scores range from 0.5 (touching or overlapping) to 1.0 (distance ≥ 10 % of
        the normalised page diagonal).

        Args:
            candidates: All title candidates for the page, including self.
        """
        others = [other for other in candidates if other is not self]
        if not others:
            return

        def _bbox_dist(other: "TitleCandidateTextBlock") -> float:
            dx = max(0.0, max(self.rect.x0, other.rect.x0) - min(self.rect.x1, other.rect.x1))
            dy = max(0.0, max(self.rect.y0, other.rect.y0) - min(self.rect.y1, other.rect.y1))
            return (dx**2 + dy**2) ** 0.5

        min_dist = min(_bbox_dist(other) for other in others)
        self._isolation = 0.5 + 0.5 * min(min_dist / 0.1, 1.0)

    @property
    def no_institution(self) -> float:
        """Return a penalty factor when institution or author keywords are detected.

        Checks for words typical of company names, government bodies, or research
        institutions (e.g. 'AG', 'GmbH', 'Bundesamt', 'école').

        Returns:
            float: 0.2 if an institution keyword is found as a whole word, 1.0 otherwise.
        """
        words = set(re.findall(r"\w+", self.text.lower()))
        return 0.2 if words & _INSTITUTION_KEYWORDS else 1.0

    @property
    def score(self) -> float:
        """Return a composite title-likelihood score.

        Multiplies heuristic signals: text area proxy, vertical position, text length,
        non-numericality, page height position, all-caps boost, institution keyword
        penalty, and block isolation. A higher score indicates a stronger title candidate.

        Returns:
            float: Non-negative composite score; 0 if any binary signal is zero.
        """
        return (
            self.text_area
            * self.verticality
            * self.length
            * self.non_numericality
            * self.highness
            * self.all_caps
            * self.no_institution
            * self.isolation
        )


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

    title_candidates = [TitleCandidateTextBlock(text_block=text_block, rect=page.rect) for text_block in text_blocks]
    for cand in title_candidates:
        cand.assign_isolation(title_candidates)
    title_candidates.sort(key=lambda x: x.score, reverse=True)
    return title_candidates[0].text if title_candidates else ""
