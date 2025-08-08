"""page class module."""

from enum import Enum


class PageClasses(Enum):
    """Enum for classifying pages into page types."""

    TEXT = "Text"
    BOREPROFILE = "Boreprofile"
    MAP = "Map"
    GEO_PROFILE = "Geo_Profile"
    TITLE_PAGE = "Title_Page"
    DIAGRAM = "Diagram"
    TABLE = "Table"
    UNKNOWN = "Unknown"


## ID mappings

label2id = {
    "Boreprofile": 0,
    "Map": 1,
    "Text": 2,
    "Geo_Profile": 3,
    "Title_Page": 4,
    "Diagram": 5,
    "Table": 6,
    "Unknown": 7,
}
id2label = {v: k for k, v in label2id.items()}
enum2id = {class_: label2id[class_.value] for class_ in PageClasses}
id2enum = {v: k for k, v in enum2id.items()}

num_labels = len(label2id)
