from pydantic import BaseModel, ConfigDict

from src.page_classes import PageClasses


class DocumentMetadata(BaseModel):
    """Document-level metadata extracted from a PDF."""

    page_count: int


class DocumentPage(BaseModel):
    """Classification annotation for a single page."""

    page: int
    title: str | None = None
    classification: dict[PageClasses, int]


class DocumentGroundTruth(BaseModel):
    """Ground-truth annotation for a complete PDF document."""

    filename: str
    metadata: DocumentMetadata
    pages: list[DocumentPage]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "1801_62.pdf",
                "metadata": {"page_count": 1},
                "pages": [
                    {
                        "page": 1,
                        "title": "Diagram exmaple",
                        "classification": {
                            "text": 0,
                            "boreprofile": 0,
                            "map": 0,
                            "geo_profile": 0,
                            "title_page": 0,
                            "diagram": 1,
                            "table": 0,
                            "unknown": 0,
                            "section_header": 0,
                        },
                    }
                ],
            }
        }
    )
