"""Convert title / section document to processed entries."""

from pathlib import Path

import pymupdf
from pymupdf import Rect
from swissgeol_doc_processing.text.textblock import TextBlock

from src.entity.utils import select_pages
from src.models.feature_engineering import extract_and_cache_page_data
from src.page_classes import PageClasses
from src.page_structure import ProcessedEntities
from src.utils.text_clustering import create_text_blocks


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
    def horizontal_centrality(self) -> float:
        """Horizontal centrality of the block.

        Returns:
            float: Score in [0, 1] where 1 means the block is perfectly horizontally centered.
        """
        return 1 - 2 * abs(0.5 - (self.rect.x1 + self.rect.x0) / 2)

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
        return self.horizontal_centrality * self.font * self.highness


def _extract_title_from_page(page: pymupdf.Page) -> str:
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


def document_to_titlepages(
    pdf_file: Path, classification: PageClasses, page_start: int, page_end: int, lang: str | None
) -> list[ProcessedEntities]:
    """Extract title or section-header entities from a consecutive page range in a PDF.

    Each page is processed individually and yields one ProcessedEntities entry whose `title` field
    contains detected title.

    Args:
        pdf_file (Path): Path to the source PDF file.
        classification (PageClasses): Page class label to assign.
        page_start (int): First page index of the group (1-based).
        page_end (int): Last page index of the group (1-based).
        lang (str | None): Language code for the page group, or None if unknown.

    Returns:
        list[ProcessedEntities]: One ProcessedEntities per page, each with its `title`
            field set to the highest-scoring title candidate extracted from that page.
    """
    # Open the PDF file, select pages and save
    with pymupdf.Document(pdf_file) as doc:
        pdf_document_select = select_pages(doc, page_start, page_end)
        return [
            ProcessedEntities(
                classification=classification,
                page_start=page_start + page_id,
                page_end=page_start + page_id,
                language=lang,
                title=_extract_title_from_page(page=page),
            )
            for page_id, page in enumerate(pdf_document_select.pages())
        ]
