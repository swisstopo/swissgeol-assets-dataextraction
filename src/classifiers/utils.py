import re
from src.page_classes import PageClasses

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
        case "boreprofile":
            return PageClasses.BOREPROFILE
        case "map" | "maps":
            return PageClasses.MAP
        case "title page" | "title_page":
            return PageClasses.TITLE_PAGE
        case _:
            return PageClasses.UNKNOWN