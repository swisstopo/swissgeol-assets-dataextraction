from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from itertools import groupby

import pymupdf
from pydantic import BaseModel, ConfigDict, FieldSerializationInfo, field_serializer

from src.geometric_objects import Line
from src.page_classes import PageClasses
from src.text_objects import TextBlock, TextLine, TextWord


@dataclass()
class PageContext:
    """Contains processed text content and information from a page."""

    lines: list[TextLine]
    words: list[TextWord]
    text_blocks: list[TextBlock]
    language: str
    page_rect: pymupdf.Rect
    text_rect: pymupdf.Rect
    geometric_lines: list[Line]
    is_digital: bool
    drawings: list
    image_rects: list
    color_proportion: Counter | None = None


class ProcessorPageMetadata(BaseModel):
    """Processed pagee metadata."""

    model_config = ConfigDict(extra="forbid")

    is_frontpage: bool
    language: str | None


class ProcessorDocumentMetadata(BaseModel):
    """Processed document metadata."""

    model_config = ConfigDict(extra="forbid")

    page_count: int
    languages: list[str]


class ProcessorPage(BaseModel):
    """Processed PDF page entity."""

    model_config = ConfigDict(extra="forbid")

    page: int
    classification: PageClasses
    metadata: ProcessorPageMetadata

    @field_serializer("classification")
    def classification_onehot(self, v: PageClasses, info: FieldSerializationInfo):
        """Change type of classification representation based on context.

        If legacy is provided throught context ({"legacy": True}), model returns onehoe encoding. This is
        done to support legacy of API.

        Args:
            v (PageClasses): Classification of object.
            info (FieldSerializationInfo): Context that should contain legacy tag.

        Returns:
            PageClasses | dict[PageClasses, int]: _description_
        """
        legacy = bool(info.context and info.context.get("legacy"))
        if legacy:
            return {p.value: int(p == v) for p in set(PageClasses)}
        else:
            return v


class ProcessorDocument(BaseModel):
    """PDF object structure."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    metadata: ProcessorDocumentMetadata
    pages: list[ProcessorPage]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "foo.pdf",
                "metadata": {"page_count": 2, "languages": ["fr", "de"]},
                "pages": [
                    {
                        "page": 1,
                        "classification": "boreprofile",
                        "metadata": {"is_frontpage": True, "language": None},
                    },
                    {
                        "page": 2,
                        "classification": "text",
                        "metadata": {"is_frontpage": False, "language": "de"},
                    },
                ],
            }
        },
    )

    def group_pages_by_type(
        self,
    ) -> Generator[tuple[tuple[PageClasses, str | None], list[ProcessorPage]], None, None]:
        """Group pages by classes and languages.

        Yields:
            Generator[tuple[tuple[PageClasses, str | None], list[ProcessorPage]], None, None]: Returns
                a generator with grouped pages par class and language along with corresponding tags.
        """

        # Get detected classes for each page
        def key_fn(x: PageClasses) -> tuple[PageClasses, str | None]:
            return x.classification, x.metadata.language

        for key, group in groupby(self.pages, key=key_fn):
            yield key, list(group)


class ProcessedEntitiesMetadata(BaseModel):
    """Processed page entities metadata."""

    page_start: int
    page_end: int
    language: str | None


class ProcessedEntities(BaseModel):
    """Processed page entities from PDF."""

    classification: PageClasses
    metadata: ProcessedEntitiesMetadata
    data: None


class ProcessorDocumentEntities(BaseModel):
    """Restructured document as entities."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    metadata: ProcessorDocumentMetadata
    entities: list[ProcessedEntities]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "foo.pdf",
                "metadata": {"page_count": 2, "languages": ["fr", "de"]},
                "entities": [
                    {
                        "classification": "boreprofile",
                        "metadata": {
                            "page_start": 1,
                            "page_end": 2,
                            "language": "de",
                        },
                        "data": None,
                    },
                    {
                        "classification": "boreprofile",
                        "metadata": {
                            "page_start": 3,
                            "page_end": 5,
                            "language": "fr",
                        },
                        "data": None,
                    },
                ],
            }
        },
    )
