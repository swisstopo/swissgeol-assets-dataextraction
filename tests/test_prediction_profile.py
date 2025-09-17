from main import _apply_profile
from src.predictions.compat import STABLE_LABELS

classification_dev = {
    "text": 0,
    "boreprofile": 0,
    "map": 0,
    "geo_profile": 1,
    "title_page": 0,
    "diagram": 0,
    "table": 0,
    "unknown": 0,
}


def make_doc(cls):
    return [
        {
            "filename": "sample.pdf",
            "metadata": {"page_count": 1, "languages": ["de"]},
            "pages": [
                {
                    "page": 1,
                    "classification": cls,
                    "metadata": {"language": "de", "is_frontpage": False},
                }
            ],
        }
    ]


def test_stable_profile_mapping():
    output = make_doc(cls=classification_dev)
    output_mapped = _apply_profile(output, "stable")
    doc = output_mapped[0]
    cls = doc["pages"][0]["classification"]

    # Only stable labels are present
    assert set(cls.keys()) == set(STABLE_LABELS)

    # Currently only 1 class possible per page
    assert sum(int(v) for v in cls.values()) == 1

    # geo_profile should be remapped to unknown
    assert cls["unknown"] == 1

    # no profile version in stable version
    assert doc.get("profile_version", None) is None


def test_dev_profile_mapping():
    output = make_doc(cls=classification_dev)
    output_mapped = _apply_profile(output, "dev")
    doc = output_mapped[0]
    cls = doc["pages"][0]["classification"]

    # Only stable labels are present
    assert set(cls.keys()) != set(STABLE_LABELS)

    # Classification should stay as is
    assert cls["geo_profile"] == 1
    assert sum(int(v) for v in cls.values()) == 1

    assert "diagram" in cls
    assert "table" in cls

    # Profile version present in dev version
    assert doc.get("profile_version", None).endswith("dev")
