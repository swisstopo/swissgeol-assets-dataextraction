import csv
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from Levenshtein import distance
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
    """Load ground truth data from a JSON file.

    Args:
        ground_truth_path (Path): Path to the JSON file containing ground truth annotations.

    Returns:
        list[DocumentGroundTruth] | None: Parsed list of document ground truths, or None on error.
    """
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


def standardize_text(text: str) -> str:
    """Standardize text by removing new lines, double spaces and uppercaps.

    Args:
        text (str): Text to standardize.

    Returns:
        str: Standardized text.
    """
    # Remove new lines
    text = text.replace("\n", " ")
    # Remove double spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accents "ü" -> "u"
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    # Enforce lowercases
    return text.lower()


def are_texts_close(text_gt: str, text_pred: str, r_error: float = 0.25) -> bool:
    """Check if two texts are similar based on Levenshtein distance.

    Before matching the tiles are standardized.

    Args:
        text_gt (str): Ground truth text.
        text_pred (str): Predicted text.
        r_error (float, optional): Accepted relative error. Defaults to 1e-1.

    Returns:
        bool: True if both text are consifered close to eachothers.
    """
    text_gt = standardize_text(text_gt)
    text_pred = standardize_text(text_pred)
    return distance(text_gt, text_pred) / max(1, len(text_gt)) < r_error


def compute_classification_stats(predictions: dict[str, DocumentPage], ground_truth: dict[str, DocumentPage]) -> dict:
    """Compute per-label classification confusion statistics over matched page keys.

    Args:
        predictions (dict[str, DocumentPage]): Keyed predictions ('filename-page').
        ground_truth (dict[str, DocumentPage]): Keyed ground truth ('filename-page').

    Returns:
        dict: Per-label counts of true_positives, false_negatives, and false_positives.
    """
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


def compute_title_stats(predictions: dict[str, DocumentPage], ground_truth: dict[str, DocumentPage]) -> dict:
    """Compute title extraction confusion statistics over matched page keys.

    Only pages with a non-empty ground truth title are evaluated.

    Args:
        predictions (dict[str, DocumentPage]): Keyed predictions ('filename-page').
        ground_truth (dict[str, DocumentPage]): Keyed ground truth ('filename-page').

    Returns:
        dict: A dict with key "title" containing true_positives, false_negatives, false_positives.
    """
    stats = {"true_positives": 0, "false_negatives": 0, "false_positives": 0}
    common_keys = predictions.keys() & ground_truth.keys()

    for key in common_keys:
        pred_title = predictions[key].title
        gt_title = ground_truth[key].title
        # Check if GT exists
        if not gt_title:
            continue

        # Measure
        if pred_title and are_texts_close(gt_title, pred_title):
            stats["true_positives"] += 1
        else:
            # TODO: remove before final PR
            logger.info(f"{key}: {gt_title} == {pred_title}")
            stats["false_positives"] += 1
            stats["false_negatives"] += 1

    return {"title": stats}


def compute_stats(
    predictions: list[DocumentGroundTruth], ground_truths: list[DocumentGroundTruth]
) -> tuple[dict, dict]:
    """Compute classification and title extraction statistics against ground truth.

    Args:
        predictions (list[DocumentGroundTruth]): Predicted document annotations.
        ground_truths (list[DocumentGroundTruth]): Ground truth document annotations.

    Returns:
        tuple[dict, dict]: A tuple of (classification_stats, title_stats), each as per-label
            confusion dictionaries.
    """
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


def save_stats(stats_classification: dict, csv_path: Path) -> Path:
    """Save per-label confusion statistics to a CSV file.

    Args:
        stats_classification (dict): Per-label dict with true_positives, false_negatives, false_positives.
        csv_path (Path): Destination path for the output CSV file.

    Returns:
        Path: The path to the written CSV file.
    """
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
    """Calculate and log F1, precision, and recall metrics to MLflow.

    Args:
        stats_classification (dict): Per-label classification confusion stats.
        stats_title (dict): Title extraction confusion stats.
    """
    if not mlflow_tracking:
        return None

    # Log metrics for title extraction
    tp, fn, fp = [stats_title["title"][label] for label in ["true_positives", "false_negatives", "false_positives"]]
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
    mlflow.log_metric("classification/macro_f1", macro_f1)

    logger.info(f"Classification Macro: F1={macro_f1:.2%}, Precision={macro_precision:.2%}, Recall={macro_recall:.2%}")


def evaluate_results(
    predictions: list[ProcessorDocumentEntities], ground_truth_path: Path, output_dir: Path = Path("evaluation")
) -> tuple[Path | None, Path | None]:
    """Evaluate classification and title predictions against ground truth.

    Args:
        predictions (list[ProcessorDocumentEntities]): Model predictions to evaluate.
        ground_truth_path (Path): Path to the ground truth JSON file.
        output_dir (Path): Directory to write evaluation CSV files (default: "evaluation").

    Returns:
        tuple[Path | None, Path | None]: Paths to the classification and title evaluation CSV files,
            or (None, None) if ground truth or predictions could not be loaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_list = load_ground_truth(ground_truth_path)
    pred_list = [pred.to_ground_truth() for pred in predictions]

    if not gt_list or not pred_list:
        return None, None

    stats_classification, stats_title = compute_stats(pred_list, gt_list)
    stats_classification_path = save_stats(stats_classification, output_dir / "evaluation_metrics_classification.csv")
    stats_title_path = save_stats(stats_title, output_dir / "evaluation_metrics_title.csv")

    if mlflow_tracking:
        log_metrics_to_mlflow(stats_classification, stats_title)
        mlflow.log_artifact(str(stats_classification_path))
        mlflow.log_artifact(str(stats_title_path))

    return stats_classification_path, stats_title_path
