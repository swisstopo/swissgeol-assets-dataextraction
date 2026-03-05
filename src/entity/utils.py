"""Base utils for entity extraction."""

from io import BytesIO

import pymupdf
from pymupdf import Document


def pages_to_bytes(pdf_document: Document, page_start: int, page_end: int) -> BytesIO:
    """Select pages from PDF.

    Args:
        pdf_document (Document): PDF to split.
        page_start (int): Start page (1-based).
        page_end (int): End page (1-based).

    Returns:
        BytesIO: Selected subset of pages as bytes.
    """
    # Create a new PDF for the selected pages
    select_pdf = pymupdf.open()

    for page_number in range(page_start, page_end + 1):
        # Insert the page into the new PDF
        select_pdf.insert_pdf(pdf_document, from_page=page_number - 1, to_page=page_number - 1)

    # Extarct bytes and close document
    select_pdf_bytes = BytesIO(select_pdf.tobytes())
    select_pdf.close()

    return select_pdf_bytes
