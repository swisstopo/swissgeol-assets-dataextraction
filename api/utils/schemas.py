from pydantic import BaseModel, ConfigDict, Field

from api.utils.mapping import parse_predicted_class


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str


class StartPayload(BaseModel):
    """Payload model for initiating a new document processing task."""

    file: str = Field(min_length=5)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # This allows using non-standard types like Path
        json_schema_extra={
            "example": {"file": "10122_1.pdf"},
        },
    )


class StartResponse(BaseModel):
    """Response returned when a task has been successfully started."""

    message: str


class CollectPayload(BaseModel):
    """Payload model for retrieving results of a processed document."""

    file: str = Field(min_length=1)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # This allows using non-standard types like Path
        json_schema_extra={
            "example": {"file": "10122_1.pdf"},
        },
    )


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

    predicted_class: str
    page_number: int = Field(gt=0, description="Page number must be greater than 0")
    page_meta_data: PageMetaDataSchema

    @classmethod
    def from_prediction(cls, prediction: dict):
        return cls(
            predicted_class=parse_predicted_class(prediction["classification"]),
            page_number=prediction["page"],
            page_meta_data=PageMetaDataSchema.from_prediction(prediction["metadata"]),
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
                                "page_meta_data": {"language": "fr", "is_frontpage": True},
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
