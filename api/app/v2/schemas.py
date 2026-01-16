from pydantic import BaseModel, ConfigDict

from src.page_structure import ProcessorDocumentEntities


class CollectResponse(BaseModel):
    """Response model for the collect request endpoint containing processing status and results."""

    has_finished: bool
    data: list[ProcessorDocumentEntities] | None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_finished": True,
                "data": [
                    {
                        "filename": "input.pdf",
                        "metadata": {"page_count": 3, "languages": ["de"]},
                        "entities": [
                            {
                                "start_page": 1,
                                "end_page": 3,
                                "lang": "de",
                                "classification": "boreprofile",
                                "data": None,
                            },
                        ],
                    }
                ],
            }
        }
    )

    @classmethod
    def create_response(cls, predictions: list[dict]):
        """Currently the api only handles requests with one file, `predictions` is a list of only one element."""
        return cls(has_finished=True, data=[ProcessorDocumentEntities.model_validate(pred) for pred in predictions])
