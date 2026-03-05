"""Convert title / section document to processed entries."""

from dataclasses import dataclass

import pymupdf
from pymupdf import Rect
from swissgeol_doc_processing.text.textblock import TextBlock

from src.models.feature_engineering import extract_and_cache_page_data
from src.utils.text_clustering import create_text_blocks
from src.utils.utility import standardize_text


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
    def contains_keywords(self) -> int:
        """Score item if it contains a keyword.

        Returns:
            int: 1 if keywords found, 0 otherwise.
        """
        std_text = standardize_text(self.text)
        return int(any([keyword in std_text for keyword in ["bericht", "etude"]]))

    @property
    def horizontal_centrality(self) -> float:
        """Horizontal centrality of the block.

        Returns:
            float: Score in [0, 1] where 1 means the block is perfectly horizontally centered.
        """
        return 1 - 2 * abs(0.5 - (self.rect.x1 + self.rect.x0) / 2)

    @property
    def horizontal_leftness(self) -> float:
        """Horizontal leftness score of the block.

        Returns:
            float: Score in [0, 1] where higher values indicate left position.
        """
        return min(1, 2 - (self.rect.x1 + self.rect.x0))

    @property
    def font(self) -> float:
        """Normalized font size proxy.

        Returns:
            float: Normalized line height in [0, 1] coordinate space.
        """
        return self.rect.height / max(self.line_count, 1)

    @property
    def highness(self) -> float:
        """Vertical position score.

        Higher values for blocks closer to the top of the page.

        Returns:
            float: Score in [0, 1] where 1 means the block starts at the very top of the page.
        """
        return 1 - self.rect.y0

    @property
    def score(self) -> float:
        """Combined title-likelihood score.

        The metric is based on horizontal centrality, font size, and vertical position.

        Returns:
            float: Estimated title-likelihood score. Higher means more likely a title.
        """
        # TODO improve metric
        # return (self.horizontal_centrality * self.font * self.highness) + self.contains_keywords
        # return self.horizontal_centrality * self.font * self.highness
        return self.font


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
    # Extract text block from page
    extraction_context = extract_and_cache_page_data(page)
    lines = extraction_context.text_lines
    text_blocks = create_text_blocks(lines)

    # Create list of text candidates and return best
    title_candidates = [TitleCandidateTextBlock(text_block=text_block, rect=page.rect) for text_block in text_blocks]
    title_candidates = sorted(title_candidates, key=lambda x: x.score, reverse=True)

    return title_candidates[0].text if title_candidates else ""
