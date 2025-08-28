from collections.abc import Iterable

V1_LABELS_DEFAULT = ("text", "boreprofile", "map", "title_page", "unknown")


def adapt_page_classification(
    classification: dict[str, float | int],
    labels_v1: Iterable[str] = V1_LABELS_DEFAULT,
    class_mapping: dict[str, str] | None = None,
) -> dict[str, int]:
    """Adapt a (possibly extended) per-page classification to a v1-compatible dictionary.

    - Any label in class_mapping is remapped (e.g., Diagram->Unknown, Geo_Profile->Unknown, Table->Unknown).
    - If a label is not in `class_mapping`, it is left unchanged.
    Non-v1 labels are dropped from the output dictionary.
    """
    labels_v1 = tuple(labels_v1)
    mapping = class_mapping or {}

    # 1) Remap extended labels to targets
    remapped: dict[str, int] = {}
    for label, value in classification.items():
        target_label = mapping.get(label, label)
        remapped[target_label] = value

    # 2) Keep only v1 labels and fill missing with 0.
    filtered = {label: remapped.get(label, 0) for label in labels_v1}

    return filtered
