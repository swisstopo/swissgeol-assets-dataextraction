"""Convert title / section document to processed entries."""

import re
from dataclasses import dataclass

import pymupdf
from pymupdf import Rect
from swissgeol_doc_processing.text.extract_text import extract_text_lines
from swissgeol_doc_processing.text.textblock import TextBlock

from src.utils.text_clustering import create_text_blocks


@dataclass
class TitleCandidateTextBlock:
    """A scale-invariant text block candidate for title detection."""

    text: str
    line_count: int
    rect: pymupdf.Rect

    def __init__(self, text_block: TextBlock, rect: Rect):
        """Create a scale invariant text block.

        The normalized text block is contained in a fictive [0, 0, 1, 1] rect.

        Args:
            text_block (TextBlock): Input text block.
            rect (Rect): Size of the page linked to text block.
        """
        self.text = text_block.text
        self.line_count = text_block.line_count
        self.rect = pymupdf.Rect(
            text_block.rect.x0 / rect.width,
            text_block.rect.y0 / rect.height,
            text_block.rect.x1 / rect.width,
            text_block.rect.y1 / rect.height,
        )

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
        """Return an approximate normalised font size (block height per line)."""
        return self.rect.height / max(self.line_count, 1)

    @property
    def highness(self) -> float:
        """Return a score favouring blocks near the top of the page."""
        return 1 - self.rect.y0

    @property
    def score(self) -> float:
        """Return a composite title-likelihood score.

        Multiplies all heuristic signals: font size, horizontal position,
        vertical position, text length, non-numericality, and highness.
        A higher score indicates a stronger title candidate.

        Returns:
            float: Non-negative composite score; 0 if any signal is False/zero.
        """
        return self.font * self.horizontality * self.verticality * self.length * self.non_numericality * self.highness


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

    # Create list of text candidates and return best
    title_candidates = [TitleCandidateTextBlock(text_block=text_block, rect=page.rect) for text_block in text_blocks]
    title_candidates = sorted(title_candidates, key=lambda x: x.score, reverse=True)

    return title_candidates[0].text if title_candidates else ""
