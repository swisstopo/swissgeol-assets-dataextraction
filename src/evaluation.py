import mlflow
import json
import csv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

LABELS = ["Text", "Boreprofile", "Title_Page", "Maps", "Unknown"]

def load_ground_truth(ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        return {entry["filename"]: entry["classification"] for entry in json.load(f)}


def compute_confusion_stats(predictions, ground_truth):
    stats = {
        label: {
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 0
        } for label in LABELS
    }

    total_files = 0
    total_pages = 0

    for filename, pred_pages in predictions.items():
        if filename not in ground_truth:
            logger.info(f"No ground truth for {filename}. Skipping.")
            continue

        gt_pages = ground_truth[filename]
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
                elif gt == 0 and pred == 0:
                    stats[label]["true_negatives"] += 1

    return stats, total_files, total_pages


def save_confusion_stats(stats, output_dir):
    csv_path = Path(output_dir) / "evaluation_metrics.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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


def log_metrics_to_mlflow(stats, total_files, total_pages):
    for label, s in stats.items():
        tp = s["true_positives"]
        fn = s["false_negatives"]
        fp = s["false_positives"]
        tn = s["true_negatives"]

        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        mlflow.log_metric(f"{label}_accuracy", accuracy)
        mlflow.log_metric(f"{label}_precision", precision)
        mlflow.log_metric(f"{label}_recall", recall)

        logger.info(f"Class: {label}")
        logger.info(f" Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")

    mlflow.log_metric("total_pages", total_pages)
    mlflow.log_metric("total_files", total_files)


def create_page_comparison(pred_dict, gt_dict, output_dir="evaluation"):
    report_path = Path(output_dir) / "per_page_comparison.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = ["Text", "Boreprofile", "Title_Page", "Maps", "Unknown"]

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
                ["Filename", "Page"] +
                [f"{label}_pred" for label in labels] +
                [f"{label}_gt" for label in labels] +
                [f"{label}_match" for label in labels] +
                ["All_labels_match"]
        )

        writer.writerow(header)

        for filename, pred_pages in pred_dict.items():
            if filename not in gt_dict:
                continue

            gt_pages = gt_dict[filename]
            if len(pred_pages) != len(gt_pages):
                continue

            for i, (pred_page, gt_page) in enumerate(zip(pred_pages, gt_pages), start=1):
                row = [filename, i]
                matches = []

                for label in labels:
                    pred = pred_page.get(label, 0)
                    gt = gt_page.get(label, 0)
                    match = int(pred == gt)
                    row.extend([pred, gt, match])
                    matches.append(match)

                all_match = int(all(matches))
                row.append(all_match)

                writer.writerow(row)

    mlflow.log_artifact(str(report_path))
    print(f"Logged page-by-page comparison to {report_path}")
    return report_path


def evaluate_results(predictions, ground_truth_path, output_dir="evaluation"):
    gt_dict = load_ground_truth(ground_truth_path)
    class_dict = {entry["filename"]: entry["classification"] for entry in predictions}

    stats, total_files, total_pages = compute_confusion_stats(class_dict, gt_dict)
    stats_path = save_confusion_stats(stats, output_dir)
    log_metrics_to_mlflow(stats, total_files, total_pages)
    mlflow.log_artifact(str(stats_path))
    create_page_comparison(class_dict, gt_dict, output_dir)

    return stats


