# stable keys -> API keys
from pydantic.alias_generators import to_pascal

from src.predictions.compat import map_to_stable_labels

V0_APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Maps",
    "title_page": "Title_Page",
    "unknown": "Unknown",
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


def predicted_class_v1(classification: dict) -> str:
    """Parse the predicted class from a one-hot encoded classification dictionary."""
    return next((to_pascal(k) for k, v in classification.items() if v == 1), "Unknown")
