import pytest

from api.app.v1.schemas import PageClasses, PascalPageClasses, predicted_class


@pytest.mark.parametrize(
    "classes,expected",
    [
        (PageClasses.TEXT, PascalPageClasses.TEXT),
        (PageClasses.BOREPROFILE, PascalPageClasses.BOREPROFILE),
        (PageClasses.MAP, PascalPageClasses.MAP),
        (PageClasses.GEO_PROFILE, PascalPageClasses.GEO_PROFILE),
        (PageClasses.TITLE_PAGE, PascalPageClasses.TITLE_PAGE),
        (PageClasses.DIAGRAM, PascalPageClasses.DIAGRAM),
        (PageClasses.TABLE, PascalPageClasses.TABLE),
        (PageClasses.UNKNOWN, PascalPageClasses.UNKNOWN),
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
