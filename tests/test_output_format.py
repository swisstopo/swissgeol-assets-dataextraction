from pathlib import Path

import pytest

from main import main as pipeline_main
from src.predictions.compat import STABLE_LABELS

PDF = Path("examples/example_pdf.pdf")


@pytest.mark.integration
def test_pipeline_output_structure_stable(monkeypatch):
    if not PDF.exists():
        pytest.skip("Fixture PDF not found (examples/example_pdf.pdf).")

    monkeypatch.setenv("PREDICTION_PROFILE", "stable")
    result = pipeline_main(input_path=str(PDF.parent), classifier_name="baseline", write_result=False)

    doc = result[0]
    # Check format on document level
    assert isinstance(doc, dict)
    assert isinstance(doc.get("filename"), str)
    assert isinstance(doc.get("metadata"), dict)
    assert isinstance(doc.get("pages"), list) and len(doc["pages"]) >= 1

    # Check format of document metadata
    metadata = doc["metadata"]
    assert isinstance(metadata.get("page_count"), int) and metadata["page_count"] >= 1
    assert isinstance(metadata.get("languages"), list) and len(metadata["languages"]) >= 1

    # Pages length of document should be equal to page_count
    assert len(doc["pages"]) == metadata["page_count"]

    # Check format on page level
    for page in doc["pages"]:
        assert isinstance(page.get("page"), int) and page["page"] >= 1
        assert isinstance(page.get("classification"), dict)
        assert isinstance(page.get("metadata"), dict)

        cls = page["classification"]
        # Only stable keys in stable profile
        assert set(cls.keys()) == set(STABLE_LABELS)

        lang = page["metadata"].get("language")
        assert (lang is None) or isinstance(lang, str)
        assert isinstance(page["metadata"].get("is_frontpage"), bool)
