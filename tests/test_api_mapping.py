from api.utils.mapping import APP_LABELS, map_labels_for_app


def make_doc(cls=None):
    return {
        "filename": "sample.pdf",
        "metadata": {"page_count": 1, "languages": ["de"]},
        "pages": [
            {
                "page": 1,
                "classification": cls or {"text": 1, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0},
                "metadata": {"language": "de", "is_frontpage": False},
            }
        ],
    }


def test_mapping_to_stable_labels():
    doc = make_doc(cls={"text": 0, "boreprofile": 1, "map": 0, "title_page": 0, "unknown": 0})
    out = map_labels_for_app(doc)
    page = out["pages"][0]
    # all app labels present and renamed
    assert set(page["classification"].keys()) == set(APP_LABELS.values())
    # value preserved on the renamed key
    assert page["classification"][APP_LABELS["boreprofile"]] == 1
    assert sum(page["classification"].values()) == 1
