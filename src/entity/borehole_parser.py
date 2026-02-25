"""Convert boreprofile document to processed entries."""

import logging
from io import BytesIO
from pathlib import Path

import pymupdf
from extraction.runner import extract

from src.entity.utils import select_pages
from src.page_classes import PageClasses
from src.page_structure import ProcessedEntities

logger = logging.getLogger(__name__)


def _find_undetected_pages(
    entities: list[ProcessedEntities],
    page_numbers: list[int],
) -> list[int]:
    """Look for undetected pages in entities.

    Some pages fed to the borehole detection pipeline might have not been linked to any borehole. The function
    identifies the pages in page_numbers that are not linked to any entity.

    Args:
        entities (list[ProcessedEntities]): List of detected entities.
        page_numbers (list[int]): Pages to match.

    Returns:
        list[int]: List of undetected pages.
    """
    pages_covered = [
        page_number
        # Iterate over all entities
        for entity in entities
        # And pages range
        for page_number in range(entity.page_start, entity.page_end + 1)
    ]
    return list(set(page_numbers) - set(pages_covered))


def _assign_trailing_pages(
    entities: list[ProcessedEntities], page_numbers_missed: list[int]
) -> tuple[list[ProcessedEntities], list[int]]:
    """Assign undetected pages to existing entities if they directly follow them.

    Args:
        entities (list[ProcessedEntities]): List of detected borehole entities to extend.
        page_numbers_missed (list[int]): List of page numbers (1-based) that were not assigned
            to any entity during detection.

    Returns:
        tuple[list[ProcessedEntities], list[int]]: A tuple containing:
            - The updated list of entities with extended page ranges where applicable.
            - The sorted list of page numbers that could not be matched to any entity.
    """
    # Keep track of pages that were not matched
    page_numbers_not_matched: list[int] = []
    for page_number in page_numbers_missed:
        # Iterate over boreholes
        assigned: bool = False

        for entity in entities:
            # Check if current page can be assigned to existing
            if entity.page_end + 1 == page_number:
                entity.page_end = page_number
                assigned = True

        # Not able to match page
        if not assigned:
            page_numbers_not_matched.append(page_number)

    return entities, sorted(page_numbers_not_matched)


def document_to_boreprofiles(
    pdf_file: Path, page_start: int, page_end: int, lang: str | None
) -> list[ProcessedEntities]:
    """Convert documents pages to boreprofile entities.

    Entities are sorted first based on starting page. If two entities start
    on the same page, give priority to the one that ends before.

    Args:
        pdf_file (Path): Path to pdf file.
        page_start (int): Starting page (1-based).
        page_end (int): Ending page (1-based).
        lang (str | None): Detected language.

    Returns:
        list[ProcessedEntities]: List of boreprofile as entities.
    """
    # Define page range
    page_numbers = list(range(page_start, page_end + 1))

    # Open the PDF file, select pages and save
    with pymupdf.Document(pdf_file) as doc:
        pdf_document_select = select_pages(doc, page_start, page_end)
        bytes_document_select = BytesIO(pdf_document_select.tobytes())

    # Write file to temp location for inference
    prediction = extract(
        file=bytes_document_select,
        filename=pdf_file.name,
    )

    # Parse to processed entities
    entities = [
        ProcessedEntities(
            classification=PageClasses.BOREPROFILE,
            page_start=min([page_start + (bbox.page - 1) for bbox in borehole.bounding_boxes]),
            page_end=max([page_start + (bbox.page - 1) for bbox in borehole.bounding_boxes]),
            language=lang,
            title=borehole.metadata.name.feature.name if borehole.metadata.name else None,
        )
        for borehole in prediction.borehole_predictions_list
    ]

    # Add dummy pages if any missed
    page_numbers_missed = _find_undetected_pages(entities, page_numbers)
    entities, page_numbers_missed = _assign_trailing_pages(entities, page_numbers_missed)

    # Add an individual entity per page if there are still missing pages
    entities_missed = [
        ProcessedEntities(
            classification=PageClasses.BOREPROFILE,
            page_start=page_number,
            page_end=page_number,
            language=lang,
            title=None,
        )
        for page_number in page_numbers_missed
    ]

    # Return page sorted entities
    all_entities = entities + entities_missed

    # Sort based on page_start, then page_end
    return sorted(all_entities, key=lambda x: (x.page_start, x.page_end))
