# stable keys -> API keys
APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Maps",
    "title_page": "Title_Page",
    "unknown": "Unknown",
}

STABLE_KEYS = tuple(APP_LABELS.keys())


def map_labels_for_app(doc: dict) -> dict:
    """Return a copy of the classification results with classification keys renamed to the app’s labels."""
    pages = []
    for p in doc.get("pages", []):
        cls = p.get("classification", {}) or {}
        cls = {APP_LABELS.get(key, key): value for key, value in cls.items()}  # rename only
        pages.append({**p, "classification": cls})
    return {**doc, "pages": pages}
