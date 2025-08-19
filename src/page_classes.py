"""page class module."""

from enum import Enum


class PageClasses(Enum):
    """Enum for classifying pages into page types."""

    TEXT = "text"
    BOREPROFILE = "boreprofile"
    MAP = "map"
    GEO_PROFILE = "geo_profile"
    TITLE_PAGE = "title_page"
    DIAGRAM = "diagram"
    TABLE = "table"
    UNKNOWN = "unknown"


## ID mappings
label2id = {cls.value: idx for idx, cls in enumerate(PageClasses)}
id2label = {idx: cls.value for idx, cls in enumerate(PageClasses)}
enum2id = {cls: label2id[cls.value] for cls in PageClasses}
id2enum = {v: k for k, v in enum2id.items()}

num_labels = len(label2id)

# Centralized mapping between string labels (aliases) and PageClasses
ALIASES = {
    "borehole": "boreprofile",
    "boreholes": "boreprofile",
    "geo profile": "geo_profile",
    "geological profile": "geo_profile",
    "geo_profile": "geo_profile",  # legacy Camel_Snake
    "geoprofile": "geo_profile",
    "title": "title_page",
    "title_page": "title_page",  # legacy Camel_Snake
    "maps": "map",
}

# mapping aliases to PageClasses
label_mappings = {k: PageClasses[v] for k, v in ALIASES.items()}

# Add canonical names (lowercased)
for cls in PageClasses:
    label_mappings[cls.value.lower()] = cls
