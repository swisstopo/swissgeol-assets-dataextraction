import pytest

from api.app.shared.schemas import CollectPayload, StartPayload
from api.app.v1.schemas import CollectResponse as CollectResponseV1
from api.app.v1.schemas import PascalPageClasses, predicted_class
from api.app.v2.schemas import CollectResponse as CollectResponseV2
from src.page_classes import PageClasses


@pytest.mark.parametrize(
    "classes,expected",
    [
        (PageClasses.TEXT, PascalPageClasses.TEXT),
        (PageClasses.BOREPROFILE, PascalPageClasses.BOREPROFILE),
        (PageClasses.MAP, PascalPageClasses.MAP),
        (PageClasses.GEO_PROFILE, PascalPageClasses.GEO_PROFILE),
        (PageClasses.TITLE_PAGE, PascalPageClasses.TITLE_PAGE),
        (PageClasses.SECTION_HEADER, PascalPageClasses.TITLE_PAGE),
        (PageClasses.DIAGRAM, PascalPageClasses.DIAGRAM),
        (PageClasses.TABLE, PascalPageClasses.TABLE),
        (PageClasses.UNKNOWN, PascalPageClasses.UNKNOWN),
    ],
)
def test_mapping_to_stable_label(classes: PageClasses, expected: PascalPageClasses):
    """Test label stable mapping.

    Args:
        classes (PageClasses): Predicted class map.
        expected (PascalPageClasses): Expected page class enum.
    """
    page_pred = predicted_class(classes)
    assert page_pred == expected


def test_schemas_shared() -> None:
    """Test shared API schemas."""
    example = CollectPayload.model_json_schema().get("example", None)
    assert CollectPayload.model_validate(example)

    example = StartPayload.model_json_schema().get("example", None)
    assert StartPayload.model_validate(example)


def test_schemas_v1() -> None:
    example = CollectResponseV1.model_json_schema().get("example", None)
    assert CollectResponseV1.model_validate(example)


def test_schemas_v2() -> None:
    example = CollectResponseV2.model_json_schema().get("example", None)
    assert CollectResponseV2.model_validate(example)
