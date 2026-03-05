from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import pymupdf
from extraction.minimal_pipeline import ExtractionContext
from pydantic import BaseModel, ConfigDict, Field, FieldSerializationInfo, field_serializer
from swissgeol_doc_processing.geometry.geometry_dataclasses import Line
from swissgeol_doc_processing.text.textblock import TextBlock
from swissgeol_doc_processing.text.textline import TextLine, TextWord

from src.page_classes import PageClasses
from src.schemas import DocumentGroundTruth, DocumentMetadata, DocumentPage


@dataclass()
class PageContext:
    """Contains processed text content and information from a page."""

    words: list[TextWord]
    text_blocks: list[TextBlock]
    language: str
    page_rect: pymupdf.Rect
    text_rect: pymupdf.Rect
    is_digital: bool
    drawings: list
    image_rects: list
    extraction_context: ExtractionContext
    color_proportion: Counter | None = None

    @property
    def lines(self) -> list[TextLine]:
        """Access text lines from extraction context."""
        return self.extraction_context.text_lines

    @property
    def geometric_lines(self) -> list[Line]:
        """Access geometric lines from extraction context."""
        return list(self.extraction_context.all_geometric_lines)


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
    title: str | None
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
            PageClasses | dict[PageClasses, int]: Converted page classes.
        """
        legacy = bool(info.context and info.context.get("legacy"))
        if legacy:
            return {p.value: int(p == v) for p in set(PageClasses)}
        else:
            return v


class ProcessorDocument(BaseModel):
    """PDF object structure."""

    filename: str
    path: Path = Field(exclude=True)
    metadata: ProcessorDocumentMetadata
    pages: list[ProcessorPage]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "filename": "foo.pdf",
                "path": "path/to/file/foo.pdf",
                "metadata": {"page_count": 2, "languages": ["fr", "de"]},
                "pages": [
                    {
                        "page": 1,
                        "title": None,
                        "classification": "boreprofile",
                        "metadata": {"is_frontpage": True, "language": None},
                    },
                    {
                        "page": 2,
                        "title": "Quartalbericht",
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

    def to_ground_truth(self) -> DocumentGroundTruth:
        """Convert document to ground truth format for evaluation.

        Returns:
            DocumentGroundTruth: Parsed document to ground truth format.
        """
        return DocumentGroundTruth(
            filename=self.filename,
            metadata=DocumentMetadata(page_count=self.metadata.page_count),
            pages=[
                DocumentPage(
                    page=page.page,
                    classification={page_cls: int(page_cls == page.classification) for page_cls in PageClasses},
                )
                for page in self.pages
            ],
        )


class ProcessedEntities(BaseModel):
    """Processed page entities from PDF."""

    classification: PageClasses
    language: str | None
    page_start: int
    page_end: int
    title: str | None = None


class ProcessorDocumentEntities(BaseModel):
    """Restructured document as entities."""

    filename: str
    page_count: int
    languages: list[str]
    entities: list[ProcessedEntities]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "filename": "input.pdf",
                "page_count": 3,
                "languages": ["de"],
                "entities": [
                    {
                        "classification": "boreprofile",
                        "language": "de",
                        "page_start": 1,
                        "page_end": 3,
                        "title": "BS1",
                    },
                    {
                        "classification": "map",
                        "language": "fr",
                        "page_start": 4,
                        "page_end": 4,
                        "title": None,
                    },
                ],
            }
        },
    )
