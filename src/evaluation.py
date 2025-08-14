import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.page_classes import PageClasses

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

if mlflow_tracking:
    import mlflow


logger = logging.getLogger(__name__)
LABELS = [cls.value for cls in PageClasses]


def load_predictions(predictions: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, int]]:
    """Normalizes predictions list into:
    { (filename, page_number): classification_dict }
    Works for both model predictions and ground-truth lists.
    """
    pred_dict: dict[tuple[str, int], dict[str, int]] = {}

    for entry in predictions:
        filename = entry.get("filename")
        pages = entry.get("pages", [])

        for page_entry in pages:
            page_number = page_entry.get("page")
            classification = page_entry.get("classification")

            key = (filename, page_number)
            if key in pred_dict:
                logger.warning(f"Duplicate entry for {key}; overwriting previous value.")
            pred_dict[key] = classification
    return pred_dict


def load_ground_truth(ground_truth_path: Path) -> dict | None:
    """Loads ground truth data from a JSON file."""
    try:
        with open(ground_truth_path) as f:
            gt_list = json.load(f)
            return load_predictions(gt_list)
    except Exception as e:
        logger.error(f"Invalid ground truth path or JSON: {e}")
        return None


def compute_confusion_stats(predictions: dict, ground_truth: dict) -> tuple[dict, int, int]:
    """Computes confusion matrix entries, total pages and files processed for evaluating classification results."""
    stats = {
        label: {"true_positives": 0, "false_negatives": 0, "false_positives": 0, "true_negatives": 0}
        for label in LABELS
    }

    pred_keys = set(predictions.keys())
    gt_keys = set(ground_truth.keys())

    # Evaluate on the intersection so we don't crash when pages are missing
    common_keys = pred_keys & gt_keys

    missing_in_pred = gt_keys - pred_keys
    missing_in_gt = pred_keys - gt_keys
    if missing_in_pred:
        logger.info(f"{len(missing_in_pred)} GT pages have no prediction (e.g., {next(iter(missing_in_pred))}).")
    if missing_in_gt:
        logger.info(f"{len(missing_in_gt)} predicted pages missing in GT (e.g., {next(iter(missing_in_gt))}).")

    total_pages = len(common_keys)
    total_files = len({fname for (fname, _page) in common_keys})

    for key in common_keys:
        pred_page = predictions.get(key, {})
        gt_page = ground_truth.get(key, {})
        for label in LABELS:
            pred = int(pred_page.get(label, 0))
            gt = int(gt_page.get(label, 0))

            if gt == 1 and pred == 1:
                stats[label]["true_positives"] += 1
            elif gt == 1 and pred == 0:
                stats[label]["false_negatives"] += 1
            elif gt == 0 and pred == 1:
                stats[label]["false_positives"] += 1
            else:
                stats[label]["true_negatives"] += 1

    return stats, total_files, total_pages


def save_confusion_stats(stats: dict, output_dir: Path) -> Path:
    """Saves confusion matrix to output directory."""
    csv_path = output_dir / "evaluation_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Class",
                "True_Positives",
                "False_Negatives",
                "False_Positives",
                "True_Negatives",
            ]
        )
        for label, s in stats.items():
            writer.writerow(
                [
                    label,
                    s["true_positives"],
                    s["false_negatives"],
                    s["false_positives"],
                    s["true_negatives"],
                ]
            )
    return csv_path


def log_metrics_to_mlflow(stats: dict, total_files: int, total_pages: int) -> None:
    """Calculates and logs F1, precision and recall to MLflow."""
    if not mlflow_tracking:
        return None

    precisions = []
    recalls = []
    f1_scores = []
    for label, s in stats.items():
        tp, fn, fp = s["true_positives"], s["false_negatives"], s["false_positives"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        mlflow.log_metric(f"F1 {label}", f1)
        mlflow.log_metric(f"{label}_precision", precision)
        mlflow.log_metric(f"{label}_recall", recall)

        logger.info(f"{label}: F1={f1:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    mlflow.log_metric("Macro Avg Precision", macro_precision)
    mlflow.log_metric("Macro Avg Recall", macro_recall)
    mlflow.log_metric("Macro Avg F1", macro_f1)

    logger.info(f"Macro Avg: F1={macro_f1:.2%}, Precision={macro_precision:.2%}, Recall={macro_recall:.2%}")

    mlflow.log_metric("total_pages", total_pages)
    mlflow.log_metric("total_files", total_files)


def create_page_comparison(pred_dict: dict, gt_dict: dict, output_dir: Path) -> pd.DataFrame:
    """Create a per-page comparison CSV/DF for pages present in both predictions and ground truth (intersection)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "per_page_comparison.csv"

    columns = (
        ["Filename", "Page"]
        + [f"{label}_pred" for label in LABELS]
        + [f"{label}_gt" for label in LABELS]
        + [f"{label}_match" for label in LABELS]
        + ["All_labels_match", "Status"]
    )

    pred_keys = set(pred_dict.keys())
    gt_keys = set(gt_dict.keys())

    # Only evaluate files/pages that are in predictions.
    keys = pred_keys & gt_keys

    rows = []
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for filename, page_num in sorted(keys, key=lambda k: (k[0], k[1])):
            pred_page = pred_dict[(filename, page_num)]
            gt_page = gt_dict[(filename, page_num)]

            preds = [int(pred_page.get(label, 0)) for label in LABELS]
            gts = [int(gt_page.get(label, 0)) for label in LABELS]
            matches = [int(p == g) for p, g in zip(preds, gts, strict=False)]
            all_match = int(all(matches))

            # Only keep misclassifications
            if not all_match:
                status = "mismatch"
                row = [filename, page_num] + preds + gts + matches + [all_match, status]
                writer.writerow(row)
                rows.append(row)

    if mlflow_tracking:
        mlflow.log_artifact(str(report_path))
    logger.info(f"Logged misclassifications to {report_path}")

    return pd.DataFrame(rows, columns=columns)


def save_misclassifications(df: pd.DataFrame, output_dir: Path) -> None:
    """Save misclassified pages and per-class CSVs."""

    def get_active_labels(row, suffix):
        return [label for label in LABELS if row[f"{label}_{suffix}"] == 1]

    df["Predicted_labels"] = df.apply(lambda row: get_active_labels(row, suffix="pred"), axis=1)
    df["Ground_truth_labels"] = df.apply(lambda row: get_active_labels(row, suffix="gt"), axis=1)

    misclassified = df[df["All_labels_match"] == 0][["Filename", "Page", "Ground_truth_labels", "Predicted_labels"]]

    mis_path = output_dir / "misclassifications.csv"
    misclassified.to_csv(mis_path, index=False)
    if mlflow_tracking:
        mlflow.log_artifact(str(mis_path))

    for true_class in LABELS:
        class_mis = misclassified[
            misclassified["Ground_truth_labels"].apply(lambda labels, cls=true_class: cls in labels)
        ]
        if not class_mis.empty:
            path = output_dir / f"misclassified_{true_class}.csv"
            class_mis.to_csv(path, index=False)
            if mlflow_tracking:
                mlflow.log_artifact(str(path))


def evaluate_results(
    predictions: list[dict], ground_truth_path: Path, output_dir: Path = Path("evaluation")
) -> dict | None:
    """Evaluate classification predictions against ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dict = load_ground_truth(ground_truth_path)
    if gt_dict is None:
        return None

    pred_dict = load_predictions(predictions)

    stats, total_files, total_pages = compute_confusion_stats(pred_dict, gt_dict)
    stats_path = save_confusion_stats(stats, output_dir)

    if mlflow_tracking:
        log_metrics_to_mlflow(stats, total_files, total_pages)
        mlflow.log_artifact(str(stats_path))
    comparison_data = create_page_comparison(pred_dict, gt_dict, output_dir)
    save_misclassifications(comparison_data, output_dir)

    return stats
