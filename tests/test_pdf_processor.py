"""Test function for document processing."""

from src.page_structure import ProcessorDocument, ProcessorDocumentEntities


def test_document_schema() -> None:
    """Test document parsing models."""
    example = ProcessorDocument.model_json_schema().get("example", None)
    assert ProcessorDocument.model_validate(example)

    example = ProcessorDocumentEntities.model_json_schema().get("example", None)
    assert ProcessorDocumentEntities.model_validate(example)
