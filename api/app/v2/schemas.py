from pydantic import BaseModel, ConfigDict

from src.page_structure import ProcessorDocumentEntities


class CollectResponse(BaseModel):
    """Response model for the collect request endpoint containing processing status and results."""

    has_finished: bool
    data: ProcessorDocumentEntities | None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_finished": True,
                "data": {
                    "filename": "input.pdf",
                    "page_count": 3,
                    "languages": ["de"],
                    "entities": [
                        {
                            "classification": "boreprofile",
                            "page_start": 1,
                            "page_end": 3,
                            "language": "de",
                        },
                    ],
                },
            }
        }
    )

    @classmethod
    def create_response(cls, prediction: dict):
        """Creates response from prediction."""
        return cls(has_finished=True, data=ProcessorDocumentEntities.model_validate(prediction))
