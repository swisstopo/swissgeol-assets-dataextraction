"""Base utils for entity extraction."""

import pymupdf
from pymupdf import Document


def select_pages(pdf_document: Document, page_start: int, page_end: int) -> Document:
    """Select pages from PDF.

    Args:
        pdf_document (Document): PDF to split.
        page_start (int): Start page (1-based).
        page_end (int): End page (1-based).

    Returns:
        Document: Selected subset.
    """
    # Create a new PDF for the selected pages
    select_pdf = pymupdf.open()

    page_numbers = list(range(page_start, page_end + 1))
    for page_number in page_numbers:
        # Insert the page into the new PDF
        select_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)

    return select_pdf
