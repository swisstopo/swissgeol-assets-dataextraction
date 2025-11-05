from src.predictions.compat import STABLE_LABELS, map_to_stable_labels

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


def test_stable_profile_mapping():
    cls = map_to_stable_labels(classification_dev)

    # Only stable labels are present
    assert set(cls.keys()) == set(STABLE_LABELS)

    # Currently only 1 class possible per page
    assert sum(int(v) for v in cls.values()) == 1

    # geo_profile should be remapped to unknown
    assert cls["unknown"] == 1
