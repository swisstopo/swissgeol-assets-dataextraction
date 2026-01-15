from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from itertools import groupby

import pymupdf
from pydantic import BaseModel, ConfigDict

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


class ProcessorDocument(BaseModel):
    """PDF object structure."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    metadata: ProcessorDocumentMetadata
    pages: list[ProcessorPage]

    def group_pages_by_type(
        self,
    ) -> Generator[tuple[tuple[PageClasses, str | None], list[ProcessorPage]], None, None]:
        # Get detected classes for each page
        def key_fn(x: PageClasses) -> tuple[PageClasses, str | None]:
            return x.classification, x.metadata.language

        for key, group in groupby(self.pages, key=key_fn):
            yield key, list(group)


class ProcessedEntities(BaseModel):
    """Processed page entities from PDF."""

    start_page: int
    end_page: int
    lang: str | None
    classification: PageClasses
    data: None
