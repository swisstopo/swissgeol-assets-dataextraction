"""page class module."""

from enum import Enum


class PageClasses(Enum):
    """Enum for classifying pages into page types."""

    TEXT = "Text"
    BOREPROFILE = "Boreprofile"
    MAP = "Maps"
    TITLE_PAGE = "Title_Page"
    UNKNOWN = "Unknown"