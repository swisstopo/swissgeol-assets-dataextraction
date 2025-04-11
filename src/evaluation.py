import mlflow
import json
import csv
import pandas as pd
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .page_classes import PageClasses

logger = logging.getLogger(__name__)
LABELS = [cls.value for cls in PageClasses]

def load_ground_truth(ground_truth_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(ground_truth_path, 'r') as f:
            return {entry["filename"]: entry["classification"] for entry in json.load(f)}
    except Exception as e:
        logger.error(f"Invalid ground truth path: {e}")
        return None


def get_label(row: pd.Series, suffix: str) -> str:
    return next((label for label in LABELS if row.get(f"{label}_{suffix}")), "None")


def compute_confusion_stats(predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> tuple[dict, int, int]:
    """Computes confusion matrix entries, total pages and files processed for evaluating classification results."""
    stats = {
        label: {
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 0
        } for label in LABELS
    }

    total_files, total_pages = 0, 0

    for filename, pred_pages in predictions.items():
        gt_pages = ground_truth.get(filename)
        if gt_pages is None:
            logger.info(f"No ground truth for {filename}. Skipping.")
            continue

        if len(pred_pages) != len(gt_pages):
            logger.info(f"Page count mismatch in {filename}. Skipping.")
            continue

        total_files += 1
        total_pages += len(pred_pages)

        for pred_page, gt_page in zip(pred_pages, gt_pages):
            for label in LABELS:
                pred = pred_page.get(label, 0)
                gt = gt_page.get(label, 0)

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
        writer.writerow(["Class", "True_Positives", "False_Negatives", "False_Positives", "True_Negatives"])
        for label, s in stats.items():
            writer.writerow([
                label,
                s["true_positives"],
                s["false_negatives"],
                s["false_positives"],
                s["true_negatives"]
            ])
    return csv_path


def log_metrics_to_mlflow(stats: dict, total_files: int, total_pages: int) -> None:
    """Calculates and logs F1, precision and recall to MLflow."""
    for label, s in stats.items():
        tp, fn, fp = s["true_positives"], s["false_negatives"], s["false_positives"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        mlflow.log_metric(f"F1 {label}", f1)
        mlflow.log_metric(f"{label}_precision", precision)
        mlflow.log_metric(f"{label}_recall", recall)

        logger.info(f"{label}: F1={f1:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")

    mlflow.log_metric("total_pages", total_pages)
    mlflow.log_metric("total_files", total_files)


def create_page_comparison(pred_dict: dict, gt_dict: dict, output_dir: Path) -> pd.DataFrame:
    """Creates and saves a per-page comparison DataFrame of prediction and ground truth labels."""
    report_path = output_dir / "per_page_comparison.csv"

    columns = (
        ["Filename", "Page"] +
        [f"{label}_pred" for label in LABELS] +
        [f"{label}_gt" for label in LABELS] +
        [f"{label}_match" for label in LABELS] +
        ["All_labels_match"]
    )
    rows = []

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for filename, pred_pages in pred_dict.items():
            gt_pages = gt_dict.get(filename)
            if gt_pages is None:
                logger.info(f"WARNING: {filename} not in ground truth, skipping.")
                continue

            if len(pred_pages) != len(gt_pages):
                logger.info(f"WARNING: Page length mismatch in {filename}, skipping.")
                continue

            for page_num in range(len(pred_pages)):
                pred_page = pred_pages[page_num]
                gt_page = gt_pages[page_num]

                preds = [int(pred_page.get(label, 0)) for label in LABELS]
                gts = [int(gt_page.get(label, 0)) for label in LABELS]
                matches = [int(preds[i] == gts[i]) for i in range(len(LABELS))]
                all_match = int(all(matches))

                row = [filename, page_num + 1] + preds + gts + matches + [all_match]
                writer.writerow(row)
                rows.append(row)

    mlflow.log_artifact(str(report_path))
    logger.info(f"Logged page-by-page comparison to {report_path}")
    return pd.DataFrame(rows, columns=columns)


def save_misclassifications(df: pd.DataFrame, output_dir: Path) -> None:
    """Save misclassified pages and per-class CSVs."""
    df["Predicted_label"] = df.apply(get_label, axis=1, suffix="pred")
    df["Ground_truth_label"] = df.apply(get_label, axis=1, suffix="gt")

    misclassified = df[df["All_labels_match"] == 0][
        ["Filename", "Page", "Ground_truth_label", "Predicted_label"]
    ]

    mis_path = output_dir / "misclassifications.csv"
    misclassified.to_csv(mis_path, index=False)
    mlflow.log_artifact(str(mis_path))

    for true_class in LABELS:
        class_mis = misclassified[misclassified["Ground_truth_label"] == true_class]
        if not class_mis.empty:
            path = output_dir / f"misclassified_{true_class}.csv"
            class_mis.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

def evaluate_results(predictions: dict, ground_truth_path: Path, output_dir: Path = Path("evaluation")) -> Optional[dict]:
    """Main entry point for evaluating predictions against ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dict = load_ground_truth(ground_truth_path)
    if gt_dict is None:
        return None

    class_dict = {entry["filename"]: entry["classification"] for entry in predictions}

    stats, total_files, total_pages = compute_confusion_stats(class_dict, gt_dict)
    stats_path = save_confusion_stats(stats, output_dir)

    log_metrics_to_mlflow(stats, total_files, total_pages)
    mlflow.log_artifact(str(stats_path))
    comparison_data = create_page_comparison(class_dict, gt_dict, output_dir)
    save_misclassifications(comparison_data, output_dir)

    return stats
