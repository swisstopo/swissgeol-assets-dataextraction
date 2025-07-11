from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_text

from main import get_pdf_files
from src.models.feature_engineering import get_features
from src.page_classes import PageClasses
from src.utils import read_params
from src.page_classes import (
    id2label,
    enum2id
)

# === Paths ===
DATAPATH = Path("/home/lillemor/PycharmProjects/swissgeol-assets-dataextraction/data")
GROUND_TRUTH_FILE = DATAPATH / "gt_single_pages.json"
TRAIN_FOLDER = DATAPATH / "train"
VAL_FOLDER = DATAPATH / "val"
MODEL_PATH = "/home/lillemor/PycharmProjects/swissgeol-assets-dataextraction/models/random_forest/model.joblib"
MATCHING_PARAMS_PATH = "/home/lillemor/PycharmProjects/swissgeol-assets-dataextraction/config/matching_params.yml"
matching_params = read_params(MATCHING_PARAMS_PATH)

class_names = [label for _, label in sorted(id2label.items())]

# Feature names (must match get_features() output order)
feature_names = [
    "Words Per Line",
    "Text zone Density",
    "Mean Left",
    "Mean Right",
    "Text Width",
    "Line Count",
    "Line Length Variance",
    "Indent Std Dev",
    "Punctuation Density",
    "Capitalization Ratio",
    "has_sidebar",
    "has_bh_keyword",
    "num_valid_descriptions",
    "num_map_keyword_lines",
    "grid_length_sum",
    "non_grid_length_sum",
    "angle_entropy",
]

# === Functions ===
def build_filename_to_label_map(gt_json_path: Path) -> dict[str, int]:
    """Build a map from filename to class ID based on the ground truth JSON."""
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    label_lookup = {}
    for entry in gt_data:
        filename = entry["filename"]
        for classification in entry["classification"]:
            for label_name, value in classification.items():
                if label_name != "Page" and value == 1:
                    try:
                        label_enum = next(p for p in PageClasses if p.value == label_name)
                        label_id = enum2id[label_enum]
                        label_lookup[filename] = label_id
                    except KeyError:
                        raise ValueError(f"Unknown label: {label_name}")
    return label_lookup


def load_data_and_labels(folder_path: Path, label_map: dict[str, int]):
    """Extract features and labels for all PDF pages in a folder."""
    file_paths = get_pdf_files(folder_path)
    features = get_features(file_paths, matching_params)
    labels = []
    for f in file_paths:
        filename = os.path.basename(f)
        if filename not in label_map:
            raise ValueError(f"Missing label for file: {filename}")
        labels.append(label_map[filename])
    return features, labels


# === Load data ===
print("Building label lookup from ground truth")
label_lookup = build_filename_to_label_map(GROUND_TRUTH_FILE)

print("Loading training data")
X_train, y_train = load_data_and_labels(TRAIN_FOLDER, label_lookup)
print("Loading validation data")
X_val, y_val = load_data_and_labels(VAL_FOLDER, label_lookup)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# === Train model ===
print("Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
print("Evaluation on validation set:")
y_pred = clf.predict(X_val)
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# === Save model ===
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# === Feature importances ===
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Importance Ranking:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.ylabel("Importance")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# === Optional: Show one tree ===
print("Decision rules from one tree:")
tree_rules = export_text(clf.estimators_[0], feature_names=[f"f{i}" for i in range(X_train.shape[1])])
print(tree_rules)
