from collections import defaultdict
from collections.abc import Sequence

STABLE_LABELS: tuple = ("text", "boreprofile", "map", "title_page", "unknown")
STABLE_CLASS_MAPPING: dict = {"geo_profile": "unknown", "diagram": "unknown", "table": "unknown"}


def map_to_stable_labels(
    classification: dict[str, float | int],
    labels: Sequence[str] = STABLE_LABELS,
    class_mapping: dict[str, str] | None = None,
) -> dict[str, int]:
    """Adapt the (extended) per-page classification to an api-stable dictionary.

    - Any label in class_mapping is remapped (e.g., Diagram->Unknown, Geo_Profile->Unknown, Table->Unknown).
    - If a label is not in `class_mapping`, it is left unchanged.
    Labels not in the api-stable version are dropped from the output dictionary.
    """
    mapping = STABLE_CLASS_MAPPING if class_mapping is None else class_mapping
    labels = tuple(labels)

    # 1) Remap extended labels to targets
    remapped = defaultdict(int)
    for label, value in classification.items():
        target_label = mapping.get(label, label)
        if target_label not in labels:
            target_label = "unknown"
        remapped[target_label] += value

    # 2) Keep only stable labels and fill missing with 0.
    filtered = {label: int(remapped.get(label, 0)) for label in labels}

    return filtered
