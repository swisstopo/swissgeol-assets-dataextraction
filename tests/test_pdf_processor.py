"""Test function for document processing."""

import pytest

from src.page_structure import ProcessorDocument


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
                        "classification": "boreprofile",
                        "metadata": {"is_frontpage": True, "language": None},
                    },
                    {
                        "page": 2,
                        "classification": "text",
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
    doc = ProcessorDocument.model_validate(payload)
    assert len(doc.pages) == doc.metadata.page_count
