import json
import logging
from pathlib import Path

from src.page_classes import label2id

logger = logging.getLogger(__name__)


def build_filename_to_label_map(gt_json_path: Path) -> dict[tuple[str, int], int]:
    """Build a map from filename to class ID based on the ground truth JSON."""
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    label_lookup = {}
    for entry in gt_data:
        filename = entry["filename"]
        for pages in entry["pages"]:
            page = pages["page"]
            for label_name, value in pages["classification"].items():
                if value == 1:
                    try:
                        label_id = label2id[label_name]
                        label_lookup[(filename, page)] = label_id
                    except KeyError as err:
                        raise ValueError(f"Unknown label: {label_name}") from err
    return label_lookup
