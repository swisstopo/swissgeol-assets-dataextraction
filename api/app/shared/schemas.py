"""These models can be used in various versions of the API and therefore need to always be backward compatible."""

from pydantic import BaseModel, ConfigDict, Field


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


class CollectPayload(BaseModel):
    """Payload model for retrieving results of a processed document."""

    file: str = Field(min_length=1)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # This allows using non-standard types like Path
        json_schema_extra={
            "example": {"file": "10122_1.pdf"},
        },
    )
