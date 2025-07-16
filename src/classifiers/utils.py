import logging
import re

from src.page_classes import PageClasses

logger = logging.getLogger(__name__)


def clean_label(label: str) -> str:
    """
    Cleans a raw string returned by Pixtral and standardizes formatting.
    """
    label = label.strip().lower()
    label = re.sub(r"[`\"']", "", label)  # remove backticks, quotes
    label = re.sub(r"[.:\s]+$", "", label)  # remove trailing punctuation/spaces
    return label


def map_string_to_page_class(label: str) -> PageClasses:
    """Maps a string label to a PageClasses enum member."""
    label = label.strip().lower()

    match label:
        case "text":
            return PageClasses.TEXT
        case "boreprofile" | "borehole" | "boreholes":
            return PageClasses.BOREPROFILE
        case "map" | "maps":
            return PageClasses.MAP
        case "title page" | "title_page" | "title":
            return PageClasses.TITLE_PAGE
        case _:
            if label != "unknown":
                logger.warning(f"Unknown label: {label}, mapping it to unknown.")
            return PageClasses.UNKNOWN
