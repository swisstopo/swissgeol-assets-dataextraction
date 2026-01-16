from enum import Enum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal

from src.page_classes import PageClasses

# dynamically created Enum to expose PascalCase class names to the API.
PascalPageClasses = Enum(
    "PascalPageClasses",
    {name: to_pascal(value) for name, value in PageClasses.__members__.items()},
    type=str,
)
PascalPageClasses: TypeAlias = PascalPageClasses  # pyright: ignore[reportInvalidTypeForm]


class MetaDataSchema(BaseModel):
    """Schema for document-level metadata including page count and languages."""

    page_count: int
    languages: list[str]

    @classmethod
    def from_prediction(cls, metadata: dict):
        if not all(key in metadata for key in ["page_count", "languages"]):
            raise ValueError("Missing required metadata fields: page_count, languages")
        return cls(page_count=metadata["page_count"], languages=metadata["languages"])


class PageMetaDataSchema(BaseModel):
    """Schema for page-level metadata including language and frontpage status."""

    language: str | None
    is_frontpage: bool

    @classmethod
    def from_prediction(cls, metadata: dict):
        return cls(language=metadata["language"], is_frontpage=metadata["is_frontpage"])


class PagePrediction(BaseModel):
    """Schema for individual page prediction results including class, number and metadata."""

    predicted_class: PascalPageClasses
    page_number: int = Field(gt=0, description="Page number must be greater than 0")
    page_metadata: PageMetaDataSchema

    @classmethod
    def from_prediction(cls, prediction: dict):
        return cls(
            predicted_class=predicted_class(prediction["classification"]),
            page_number=prediction["page"],
            page_metadata=PageMetaDataSchema.from_prediction(prediction["metadata"]),
        )


class PredictionSchema(BaseModel):
    """Schema for the complete document prediction results including metadata and page predictions."""

    filename: str
    metadata: MetaDataSchema
    pages: list[PagePrediction]

    @classmethod
    def from_prediction(cls, prediction: dict[dict]):
        return cls(
            filename=prediction["filename"],
            metadata=MetaDataSchema.from_prediction(prediction["metadata"]),
            pages=[PagePrediction.from_prediction(page_pred) for page_pred in prediction["pages"]],
        )


class CollectResponse(BaseModel):
    """Response model for the collect request endpoint containing processing status and results."""

    has_finished: bool
    data: list[PredictionSchema] | None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_finished": True,
                "data": [
                    {
                        "filename": "input.pdf",
                        "metadata": {"page_count": 1, "languages": ["fr"]},
                        "pages": [
                            {
                                "predicted_class": "Map",
                                "page_number": 1,
                                "page_metadata": {"language": "fr", "is_frontpage": True},
                            }
                        ],
                    }
                ],
            }
        }
    )

    @classmethod
    def create_response(cls, predictions: list[dict]):
        """Currently the api only handles requests with one file, `predictions` is a list of only one element."""
        return cls(has_finished=True, data=[PredictionSchema.from_prediction(pred) for pred in predictions])


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str


def predicted_class(classification: PageClasses) -> PascalPageClasses:
    """Parse the predicted class from a one-hot encoded classification dictionary.

    The values of the dict are the sting representation of each class in the PageClasses enum.
    """
    try:
        return PascalPageClasses(to_pascal(classification))
    except ValueError:
        return PascalPageClasses.UNKNOWN
