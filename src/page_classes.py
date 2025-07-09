"""page class module."""

from enum import Enum


class PageClasses(Enum):
    """Enum for classifying pages into page types."""

    TEXT = "Text"
    BOREPROFILE = "Boreprofile"
    MAP = "Maps"
    TITLE_PAGE = "Title_Page"
    UNKNOWN = "Unknown"

# Derived mappings
LABEL2ID = {cls.value: idx for idx, cls in enumerate(PageClasses)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ENUM2ID = {cls: idx for idx, cls in enumerate(PageClasses)}
ID2ENUM = {v: k for k, v in ENUM2ID.items()}

NUM_LABELS = len(PageClasses)