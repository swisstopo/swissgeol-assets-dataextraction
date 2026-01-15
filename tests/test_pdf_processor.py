"""Test function for document processing."""

import pytest

from src.pdf_processor import PDFProcessorDocument


@pytest.mark.parametrize(
    "payload",
    [
        # Supported languages
        (
            {
                "filename": "foo.pdf",
                "metadata": {"page_count": 2, "languages": ["fr", "de"]},
                "pages": [
                    {
                        "page": 1,
                        "classification": {
                            "text": 0,
                            "boreprofile": 1,
                            "map": 0,
                            "geo_profile": 0,
                            "title_page": 0,
                            "diagram": 0,
                            "table": 0,
                            "unknown": 0,
                        },
                        "metadata": {"is_frontpage": True, "language": None},
                    },
                    {
                        "page": 2,
                        "classification": {
                            "text": 0,
                            "boreprofile": 0,
                            "map": 0,
                            "geo_profile": 0,
                            "title_page": 0,
                            "diagram": 0,
                            "table": 0,
                            "unknown": 1,
                        },
                        "metadata": {"is_frontpage": False, "language": "de"},
                    },
                ],
            }
        ),
    ],
)
def test_document_schema(payload: dict) -> None:
    """Test document parsing model.

    Args:
        payload (dict): Payload to parse.
    """
    doc = PDFProcessorDocument.model_validate(payload)
    assert len(doc.pages) == doc.metadata.page_count
