# stable keys -> API keys
from src.predictions.compat import map_to_stable_labels

V0_APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Maps",
    "title_page": "Title_Page",
    "unknown": "Unknown",
}

V1_APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Map",
    "title_page": "TitlePage",
    "geo_profile": "GeoProfile",
    "table": "Table",
    "diagram": "Diagram",
}


def map_labels_for_v0(doc: dict) -> dict:
    """Return a copy of the classification results with keys renamed to the app's labels.

    Kept for backward compatibility with v0. Once v1 is in use, this function can safely be deleted.
    """
    pages = []
    for p in doc.get("pages", []):
        cls = map_to_stable_labels(p.get("classification", {}) or {})
        cls = {V0_APP_LABELS.get(key, key): value for key, value in cls.items()}  # rename only
        pages.append({**p, "classification": cls})
    return {**doc, "pages": pages}


def parse_predicted_class(classification: dict) -> str:
    """Parse the predicted class from a one-hot encoded classification dictionary."""
    return next((V1_APP_LABELS.get(k, k) for k, v in classification.items() if v == 1), "Unknown")
