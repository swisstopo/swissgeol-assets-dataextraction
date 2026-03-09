"""End-to-end test for document classification and entity grouping."""

import pytest

from main import main as script
from src.constants import DEFAULT_TREEBASED_MODEL_PATH


@pytest.fixture
def reference_document() -> str:
    """File that contains all classes once.

    Returns:
        str: Path to file.
    """
    return "examples/reference_document.pdf"


def test_end_to_end(reference_document: str) -> None:
    """Test main pipeline end to end.

    Args:
        reference_document (str): Reference document to classify.
    """
    # Infer reference document and check output exists
    documents_pages, documents_entities = script(
        input_path=reference_document,
        classifier_name="treebased",
        model_path=DEFAULT_TREEBASED_MODEL_PATH,
        write_result=False,
        return_entities=True,
    )
    assert documents_pages and len(documents_pages) == 1
    assert documents_entities and len(documents_entities) == 1

    # Unpack single-document results
    document_pages = documents_pages[0]
    document_entities = documents_entities[0]

    # ---- Check pages output
    n_pages = document_pages.metadata.page_count
    # All pages appear exactly once
    assert len(document_pages.pages) == n_pages
    assert len(set(page.page for page in document_pages.pages)) == n_pages

    # ---- Check entity output
    # Check that all pages are within range and in order
    assert all(
        0 < entity.page_start <= n_pages and entity.page_start <= entity.page_end
        for entity in document_entities.entities
    )

    # ---- Check coherence page - entity output
    # Test 1: Same number of pages
    assert document_entities.page_count == n_pages
    # Test 2: Test that detected classes are the same after processing
    assert set(page.classification for page in document_pages.pages) == set(
        entity.classification for entity in document_entities.entities
    )
