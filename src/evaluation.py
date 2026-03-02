import csv
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter

from src.page_classes import PageClasses
from src.page_structure import ProcessorDocumentEntities
from src.schemas import DocumentGroundTruth, DocumentPage

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING").lower() == "true"

if mlflow_tracking:
    import mlflow


logger = logging.getLogger(__name__)
LABELS = [cls.value for cls in PageClasses]


def load_ground_truth(ground_truth_path: Path) -> list[DocumentGroundTruth] | None:
    """Loads ground truth data from a JSON file."""
    try:
        with open(ground_truth_path) as f:
            gt_list = json.load(f)
            gt_list = TypeAdapter(list[DocumentGroundTruth]).validate_python(gt_list)
            return gt_list
    except Exception as e:
        logger.error(f"Invalid ground truth path or JSON: {e}")
        return None


def groundtruth_doc_to_pages(documents: list[DocumentGroundTruth]) -> dict[str, DocumentPage]:
    """Convert list of documents to list of keyed pages.

    Args:
        documents (list[DocumentGroundTruth]): Documents with pages to flatten

    Returns:
        dict[str, DocumentPage]: Keyed pages.
    """
    return {f"{doc.filename}-{page.page}": page for doc in documents for page in doc.pages}


# def compute_title_metric(title_gt: str, title_pred: str) -> bool | None:
#     if title_gt is None:
#         return None
#     else:
#         return title_gt.lower().strip() == title_pred.lower().strip()


def compute_classification_stats(
    predictions: dict[str, DocumentGroundTruth], ground_truth: dict[str, DocumentGroundTruth]
) -> dict:
    stats = {label: {"true_positives": 0, "false_negatives": 0, "false_positives": 0} for label in LABELS}
    common_keys = predictions.keys() & ground_truth.keys()

    for key in common_keys:
        pred_page = predictions.get(key, {})
        gt_page = ground_truth.get(key, {})
        for label in LABELS:
            pred = int(pred_page.classification.get(label, 0))
            gt = int(gt_page.classification.get(label, 0))

            if gt == 1 and pred == 1:
                stats[label]["true_positives"] += 1
            elif gt == 1 and pred == 0:
                stats[label]["false_negatives"] += 1
            elif gt == 0 and pred == 1:
                stats[label]["false_positives"] += 1

    return stats


def compute_title_stats(
    predictions: dict[str, DocumentGroundTruth], ground_truth: dict[str, DocumentGroundTruth]
) -> dict:
    stats = {"true_positives": 0, "false_negatives": 0, "false_positives": 0}
    common_keys = predictions.keys() & ground_truth.keys()

    for key in common_keys:
        pred_title = predictions[key].title
        gt_title = ground_truth[key].title
        logger.info(f"{key}: {gt_title} == {pred_title}")
        if pred_title and gt_title and pred_title == gt_title:
            stats["true_positives"] += 1
        else:
            stats["false_negatives"] += 1
            stats["false_positives"] += 1

    return {"title": stats}


def compute_stats(
    predictions: list[DocumentGroundTruth], ground_truths: list[DocumentGroundTruth]
) -> tuple[dict, dict]:
    """Computes confusion matrix entries, total pages and files processed for evaluating classification results."""
    pred_keyed = groundtruth_doc_to_pages(predictions)
    gt_keyed = groundtruth_doc_to_pages(ground_truths)

    # Evaluate on the intersection so we don't crash when pages are missing
    pred_keys, gt_keys = set(pred_keyed.keys()), set(gt_keyed.keys())

    missing_in_pred = gt_keys - pred_keys
    missing_in_gt = pred_keys - gt_keys
    if missing_in_pred:
        logger.info(f"{len(missing_in_pred)} GT pages have no prediction (e.g., {next(iter(missing_in_pred))}).")
    if missing_in_gt:
        logger.info(f"{len(missing_in_gt)} predicted pages missing in GT (e.g., {next(iter(missing_in_gt))}).")

    classification_stats = compute_classification_stats(pred_keyed, gt_keyed)
    title_stats = compute_title_stats(pred_keyed, gt_keyed)

    return classification_stats, title_stats


def save_stats(stats_classification: list, csv_path: Path) -> Path:
    """Saves confusion matrix to output directory."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Label",
                "True_Positives",
                "False_Negatives",
                "False_Positives",
            ]
        )
        for label, s in stats_classification.items():
            writer.writerow(
                [
                    label,
                    s["true_positives"],
                    s["false_negatives"],
                    s["false_positives"],
                ]
            )
    return csv_path


def log_metrics_to_mlflow(stats_classification: dict, stats_title: dict) -> None:
    """Calculates and logs F1, precision and recall to MLflow."""
    if not mlflow_tracking:
        return None

    # Log metrics for title extraction
    tp, fp, fn = [stats_title["title"][label] for label in ["true_positives", "false_negatives", "false_positives"]]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    mlflow.log_metric("title/f1", f1)
    mlflow.log_metric("title/precision", precision)
    mlflow.log_metric("title/recall", recall)

    logger.info(f"Title: F1={f1:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")

    # Log metrics for classification output
    precisions = []
    recalls = []
    f1_scores = []
    for label, s in stats_classification.items():
        tp, fn, fp = s["true_positives"], s["false_negatives"], s["false_positives"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        mlflow.log_metric(f"classification/{label}_f1", f1)
        mlflow.log_metric(f"classification/{label}_precision", precision)
        mlflow.log_metric(f"classification/{label}_recall", recall)

        logger.info(f"{label}: F1={f1:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    mlflow.log_metric("classification/macro_precision", macro_precision)
    mlflow.log_metric("classification/macro_recall", macro_recall)
    mlflow.log_metric("classification/marco_f1", macro_f1)

    logger.info(f"Classification Macro: F1={macro_f1:.2%}, Precision={macro_precision:.2%}, Recall={macro_recall:.2%}")


def evaluate_results(
    predictions: list[ProcessorDocumentEntities], ground_truth_path: Path, output_dir: Path = Path("evaluation")
) -> tuple[Path, Path]:
    """Evaluate classification predictions against ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_list = load_ground_truth(ground_truth_path)
    pred_list = [pred.to_ground_truth() for pred in predictions]

    stats_classification, stats_title = compute_stats(pred_list, gt_list)
    stats_classification_path = save_stats(stats_classification, output_dir / "evaluation_metrics_classification.csv")
    stats_title_path = save_stats(stats_title, output_dir / "evaluation_metrics_title.csv")

    if mlflow_tracking:
        log_metrics_to_mlflow(stats_classification, stats_title)
        mlflow.log_artifact(str(stats_classification_path))
        mlflow.log_artifact(str(stats_title_path))

    return stats_classification, stats_title_path
