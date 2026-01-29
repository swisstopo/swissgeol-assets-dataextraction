from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pymupdf
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine, TextWord
from itertools import groupby

import pymupdf
from pydantic import BaseModel, ConfigDict, FieldSerializationInfo, field_serializer

from src.page_classes import PageClasses

if TYPE_CHECKING:
    from extraction.minimal_pipeline import ExtractionContext


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
    extraction_context: ExtractionContext | None = None


class ProcessorPageMetadata(BaseModel):
    """Processed page metadata."""

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

        If legacy is provided through context ({"legacy": True}), model returns one-hot encoding. This is
        done to support the legacy of API.

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
        """Group pages by class and language.

        Yields:
            Generator[tuple[tuple[PageClasses, str | None], list[ProcessorPage]], None, None]: Returns
                a generator with grouped pages per class and language along with corresponding tags.
        """

        # Get the detected class for each page
        def key_fn(x: ProcessorPage) -> tuple[PageClasses, str | None]:
            return x.classification, x.metadata.language

        for key, group in groupby(self.pages, key=key_fn):
            yield key, list(group)


class ProcessedEntities(BaseModel):
    """Processed page entities from PDF."""

    classification: PageClasses
    page_start: int
    page_end: int
    language: str | None


class ProcessorDocumentEntities(BaseModel):
    """Restructured document as entities."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    page_count: int
    languages: list[str]
    entities: list[ProcessedEntities]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        },
    )
