import json
from pathlib import Path

from src.page_classes import PageClasses

# --- Setup ---
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = REPO_ROOT / "data" / "single_pages/"
OUTPUT_PATH = REPO_ROOT / "data" / "gt_single_pages.json"

# --- Class Mapping ---
CLASSES = [cls.value for cls in PageClasses]

FOLDER_TO_CLASS = {
    "text": "Text",
    "boreprofile": "Boreprofile",
    "map": "Map",
    "geo_profile": "Geo_Profile",
    "title_page": "Title_Page",
    "diagram": "Diagram",
    "table": "Table",
    "unknown": "Unknown",
}


# --- Ground Truth Construction ---
ground_truth = []

for class_folder in INPUT_FOLDER.iterdir():
    if not class_folder.is_dir():
        continue

    folder_name = class_folder.name.lower()
    class_label = FOLDER_TO_CLASS.get(folder_name)

    if not class_label:
        print(f" Skipping unrecognized folder: {class_folder.name}")
        continue

    for pdf_file in class_folder.glob("*.pdf"):
        entry = {
            "filename": pdf_file.name,
            "metadata": {
                "page_count": None,
                "languages": [],
            },
            "pages": {
                "classification": [{cls: int(cls == class_label) for cls in CLASSES}],
                "metadata": {"language": None, "is_frontpage": None},
            },
        }
        ground_truth.append(entry)

# --- Output ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=4)

print(f" Saved {len(ground_truth)} entries to {OUTPUT_PATH}")
