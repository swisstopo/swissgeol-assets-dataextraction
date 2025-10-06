# stable keys -> API keys
V1_APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Maps",
    "title_page": "Title_Page",
    "unknown": "Unknown",
}

V2_APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Map",
    "title_page": "TitlePage",
    "geo_profile": "GeoProfile",
    "table": "Table",
    "diagram": "Diagram",
}


def map_labels_for_app(doc: dict) -> dict:
    """Return a copy of the classification results with keys renamed to the app's labels.

    Kept for backward compatibility with v1. Once v2 is in use, this function can safely be deleted.
    """
    pages = []
    for p in doc.get("pages", []):
        cls = p.get("classification", {}) or {}
        cls = {V1_APP_LABELS.get(key, key): value for key, value in cls.items()}  # rename only
        pages.append({**p, "classification": cls})
    return {**doc, "pages": pages}


def parse_predicted_class(classification: dict) -> str:
    """Parse the predicted class from a one-hot encoded classification dictionary."""
    return next((V2_APP_LABELS.get(k, k) for k, v in classification.items() if v == 1), "Unknown")
