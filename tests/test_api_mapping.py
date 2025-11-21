import pytest

from api.app.v1.schemas import predicted_class
from src.page_classes import PageClasses


@pytest.mark.parametrize(
    "classes,expected",
    [
        ({"text": 0, "boreprofile": 1, "map": 0, "title_page": 0, "unknown": 0}, PageClasses.BOREPROFILE),
        ({"text": 0, "boreprofile": 0, "map": 1, "title_page": 0, "unknown": 0}, PageClasses.MAP),
        ({"text": 0, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, PageClasses.UNKNOWN),
        ({"TEXTTT": 1, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, PageClasses.UNKNOWN),
    ],
)
def test_mapping_to_stable_labels(classes: dict[str, int], expected: PageClasses):
    """Test label stable mapping.

    Args:
        classes (dict[str, int]): Predicted class map.
        expected (PageClasses): Expected page class enum.
    """
    page_pred = predicted_class(classes)
    assert page_pred.name == expected.name
