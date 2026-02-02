"""Convert boreprofile document to processed entries."""

import json
import logging
import tempfile
from itertools import chain
from pathlib import Path

import fitz
from extraction.features.predictions.overall_file_predictions import OverallFilePredictions
from extraction.main import start_pipeline
from fitz import Document

from src.page_classes import PageClasses
from src.page_structure import ProcessedEntities

logger = logging.getLogger(__name__)


def _select_pages(pdf_document: Document, pages_id: list[int]) -> Document:
    """Select pages from PDF.

    Args:
        pdf_document (Document): PDF to split.
        pages_id (list[int]): List of pages to extract (1-based).

    Returns:
        Document: Selected subset.
    """
    # Create a new PDF for the selected pages
    select_pdf = fitz.open()

    for page_id in pages_id:
        # Insert the page into the new PDF
        select_pdf.insert_pdf(pdf_document, from_page=page_id - 1, to_page=page_id - 1)

    return select_pdf


def _find_undetected_pages(
    entities: list[ProcessedEntities],
    pages_id: list[int],
) -> list[int]:
    """Look for undetected pages in entities.

    Some pages fed to the borehole detection pipeline might have not been linked to any borehole. The function
    identifies the pages in pages_id that are not linked to any entity.

    Args:
        entities (list[ProcessedEntities]): List of detected entities.
        pages_id (list[int]): Pages to match.

    Returns:
        list[int]: List of undetected pages.
    """
    pages_covered = [list(range(entity.page_start, entity.page_end + 1)) for entity in entities]
    pages_covered_flattened = list(chain.from_iterable(pages_covered))
    return list(set(pages_id) - set(pages_covered_flattened))


def document_to_boreprofiles(
    pdf_file: Path, page_start: int, page_end: int, lang: str | None
) -> list[ProcessedEntities]:
    """Convert documents pages to boreprofile entities.

    Args:
        pdf_file (Path): Path to pdf file.
        page_start (int): Starting page (1-based).
        page_end (int): Ending page (1-based).
        lang (str): Detected language.

    Returns:
        list[ProcessedEntities]: List of boreprofile as entities.
    """
    # Define page range
    pages_id = list(range(page_start, page_end + 1))

    # Write file to temp location for inference
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create related files
        out_directory = Path(tmpdir)
        path_input = Path(out_directory) / pdf_file.name
        path_prediction = Path(out_directory) / (pdf_file.name + ".pred.json")
        path_metadata = Path(out_directory) / (pdf_file.name + ".meta.json")

        # Open the PDF file, select pages and save
        pdf_document = fitz.open(pdf_file)
        pdf_document_select = _select_pages(pdf_document, pages_id)
        pdf_document_select.save(path_input)

        start_pipeline(
            input_directory=path_input,
            ground_truth_path=None,
            out_directory=out_directory,
            predictions_path=path_prediction,
            metadata_path=path_metadata,
            skip_draw_predictions=True,
            part="all",
        )
        # Read back prediction file
        with open(path_prediction, encoding="utf8") as f:
            prediction = OverallFilePredictions.from_json(json.load(f))

        # Check that single prediction and correct id
        if (
            len(prediction.file_predictions_list) != 1
            or prediction.file_predictions_list[0].file_name != pdf_file.name
        ):
            logger.error(f"Unable to process predictions for {pdf_file.name}")
            return []

    # Parse to processed entities
    entities = [
        ProcessedEntities(
            classification=PageClasses.BOREPROFILE,
            page_start=min([page_start + (bbox.page - 1) for bbox in borehole.bounding_boxes]),
            page_end=max([page_start + (bbox.page - 1) for bbox in borehole.bounding_boxes]),
            language=lang,
            title=borehole.metadata.name.feature.name if borehole.metadata.name else None,
        )
        for borehole in prediction.file_predictions_list[0].borehole_predictions_list
    ]

    # Add dummy pages if any missed
    pages_missed_id = _find_undetected_pages(entities, pages_id)

    entities_missed = [
        ProcessedEntities(
            classification=PageClasses.BOREPROFILE,
            page_start=page_id,
            page_end=page_id,
            language=lang,
            title=None,
        )
        for page_id in pages_missed_id
    ]

    # Return detected borehole and missed pages
    return entities + entities_missed
