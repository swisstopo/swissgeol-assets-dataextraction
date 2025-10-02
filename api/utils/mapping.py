# stable keys -> API keys
APP_LABELS: dict[str, str] = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Map",
    "title_page": "TitlePage",
    "geo_profile": "GeoProfile",
    "table": "Table",
    "diagram": "Diagram",
    "unknown": "Unknown",
}


def parse_predicted_class(classification: dict) -> str:
    """Parse the predicted class from a one-hot encoded classification dictionary."""
    return next((APP_LABELS.get(k, k) for k, v in classification.items() if v == 1), "Unknown")
