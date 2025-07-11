"""page class module."""

from enum import Enum


class PageClasses(Enum):
    """Enum for classifying pages into page types."""

    TEXT = "Text"
    BOREPROFILE = "Boreprofile"
    MAP = "Maps"
    TITLE_PAGE = "Title_Page"
    UNKNOWN = "Unknown"


## ID mappings

label2id = {
    "Boreprofile": 0,
    "Maps": 1,
    "Text": 2,
    "Title_Page": 3,
    "Unknown": 4,
}

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

enum2id = {
    PageClasses.BOREPROFILE: label2id["Boreprofile"],
    PageClasses.MAP: label2id["Maps"],
    PageClasses.TEXT: label2id["Text"],
    PageClasses.TITLE_PAGE: label2id["Title_Page"],
    PageClasses.UNKNOWN: label2id["Unknown"],
}

id2enum = {v: k for k, v in enum2id.items()}