import pytest

from api.app.v1.schemas import predicted_class


@pytest.mark.parametrize(
    "classes,expected",
    [
        ({"text": 0, "boreprofile": 1, "map": 0, "title_page": 0, "unknown": 0}, "Boreprofile"),
        ({"text": 0, "boreprofile": 0, "map": 1, "title_page": 0, "unknown": 0}, "Map"),
        ({"text": 0, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, "Unknown"),
        ({"TEXTTT": 1, "boreprofile": 0, "map": 0, "title_page": 0, "unknown": 0}, "Texttt"),
    ],
)
def test_predicted_class(classes, expected):
    page_pred = predicted_class(classes)
    assert page_pred == expected
