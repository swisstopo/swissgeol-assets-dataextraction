import pytest

from api.app.v1.schemas import PascalPageClasses, predicted_class


@pytest.mark.parametrize(
    "classes,expected",
    [
        ({"text": 0, "boreprofile": 1, "map": 0, "title_page": 0, "unknown": 0}, PascalPageClasses.BOREPROFILE),
        ({"text": 0, "boreprofile": 0, "map": 1, "title_page": 0, "unknown": 0}, PascalPageClasses.MAP),
        ({"text": 0, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, PascalPageClasses.UNKNOWN),
        ({"TEXTTT": 1, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, PascalPageClasses.UNKNOWN),
    ],
)
def test_mapping_to_stable_labels(classes: dict[str, int], expected: PascalPageClasses):
    """Test label stable mapping.

    Args:
        classes (dict[str, int]): Predicted class map.
        expected (PascalPageClasses): Expected page class enum.
    """
    page_pred = predicted_class(classes)
    assert page_pred == expected
