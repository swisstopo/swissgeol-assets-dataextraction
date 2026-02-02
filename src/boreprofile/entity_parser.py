"""COnvert boreprofile document to processed entries."""

import tempfile
from pathlib import Path

import fitz
from extraction.main import start_pipeline
from fitz import Document

from src.page_classes import PageClasses
from src.page_structure import ProcessedEntities


def _select_pages(pdf_document: Document, pages_id: list[int]) -> Document:
    """Select pages from PDF.

    Args:
        pdf_document (Document): PDF to split.
        pages_id (list[int]): List of pages to extract.

    Returns:
        Document: Selected subset.
    """
    # Create a new PDF for the selected pages
    select_pdf = fitz.open()

    for page_id in pages_id:
        # Insert the page into the new PDF
        select_pdf.insert_pdf(pdf_document, from_page=page_id, to_page=page_id)

    return select_pdf


def document_to_boreprofiles(pdf_file: Path, page_start: int, page_end: int, lang: str) -> list[ProcessedEntities]:
    """Convert documents pages to boreprofile entities.

    Args:
        pdf_file (Path): Path to pdf file.
        page_start (int): Starting page.
        page_end (int): Ending page.
        lang (str): Detected language.

    Returns:
        list[ProcessedEntities]: List of boreprofile as entities.
    """
    # Write file to temp location for finference
    with tempfile.TemporaryDirectory() as tmpdir:
        # Open the PDF file
        pdf_document = fitz.open(pdf_file)
        # Get subset of pages
        pdf_document_select = _select_pages(pdf_document, list(range(page_start, page_end + 1)))
        # Write as temporary
        path_document_select = Path(tmpdir) / pdf_file.name
        pdf_document_select.save(path_document_select)

        # Process pipeline
        # TODO add path pred / meta / out
        start_pipeline(
            input_directory=path_document_select,
            ground_truth_path=None,
            out_directory="",
            predictions_path="",
            metadata_path="",
            skip_draw_predictions=True,
            part="notall",
        )
        # Read back prediction file
        # TODO read file

    return [
        ProcessedEntities(
            classification=PageClasses.BOREPROFILE,
            page_start=page_start,
            page_end=page_end,
            language=lang,
            title="BS1",
        )
    ]
