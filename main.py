import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.classifiers.classifier_factory import ClassifierTypes, create_classifier
from src.evaluation import evaluate_results
from src.pdf_processor import PDFProcessor
from src.predictions.compat import STABLE_CLASS_MAPPING, STABLE_LABELS, map_to_stable_labels
from src.utils import get_pdf_files, read_params

# Load .env and check MLFlow
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "true"
prediction_profile = os.getenv("PREDICTION_PROFILE") or "stable"

if mlflow_tracking:
    import mlflow
    import pygit2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _apply_profile(predictions: list[dict], profile: str):
    """Apply profile to a single document or list of documents."""

    def _apply_to_doc(doc: dict) -> dict:
        if profile == "stable":
            for page in doc.get("pages", []):
                page["classification"] = map_to_stable_labels(
                    page.get("classification", {}),
                    labels=STABLE_LABELS,
                    class_mapping=STABLE_CLASS_MAPPING,
                )

            doc.setdefault("profile_version", "page_classification:stable")
        elif profile == "dev":
            doc.setdefault("profile_version", "page_classification:dev")
        return doc

    return [_apply_to_doc(d) for d in predictions]


def setup_mlflow(
    input_path: Path, ground_truth_path: Path, model_path: str, matching_params: dict, classifier_name: str
):
    mlflow.set_experiment("PDF Page Classification")
    mlflow.start_run()

    mlflow.set_tag("input_path", str(input_path))

    if ground_truth_path:
        mlflow.set_tag("ground_truth_path", str(ground_truth_path))
    if model_path:
        mlflow.set_tag("model_path", str(model_path))
    if classifier_name:
        mlflow.set_tag("classifier_name", str(classifier_name))

    mlflow.log_params(flatten_dict(matching_params))

    try:
        repo = pygit2.Repository(".")
        commit = repo[repo.head.target]
        mlflow.set_tag("git_branch", repo.head.shorthand)
        mlflow.set_tag("git_commit", str(commit.id))
        mlflow.set_tag("git_message", commit.message.strip())
    except Exception as e:
        logger.warning(f"Could not attach Git metadata to MLflow: {e}")


def flatten_dict(d, parent_key="", sep=".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(
    input_path: str,
    ground_truth_path: str = None,
    model_path: str = None,
    classifier_name: str = "baseline",
    write_result: bool = False,
):
    """Run the page classification pipeline on input documents.

    Args:
        input_path (str): Path to directory with PDF pages or documents.
        ground_truth_path (str, optional): Path to ground truth JSON file for evaluation.
        model_path (str, optional): Path to pretrained LayoutLMv3 model.
        classifier_name (str, optional): Classifier to use ("baseline", "pixtral", etc.).
        write_result (bool): If True, writes results to prediction.json.

    Raises:
        ValueError: If an unsupported classifier is specified.
    """
    input_path = Path(input_path)
    ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
    pdf_files = get_pdf_files(input_path)
    if not pdf_files:
        logger.error("No valid PDFs found.")
        return

    matching_params = read_params("config/matching_params.yml")

    # Start MLFlow tracking
    if mlflow_tracking:
        setup_mlflow(input_path, ground_truth_path, model_path, matching_params, classifier_name)

    # Set up classifier
    classifier_type = ClassifierTypes.infer_type(classifier_name)
    classifier = create_classifier(classifier_type, model_path, matching_params)

    logger.info(f"Start classifying {len(pdf_files)} PDF files with {classifier.type.value} classifier")

    # Processed PDFs
    processor = PDFProcessor(classifier)
    results = processor.process_batch(pdf_files)

    if not results:
        logger.warning("No data to save.")
        return

    results = _apply_profile(results, prediction_profile)
    # Save to JSON
    if write_result:
        output_file = Path("data") / "prediction.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as json_file:
            json.dump(results, json_file, indent=4)

    if ground_truth_path:
        evaluate_results(results, ground_truth_path)

    if mlflow_tracking:
        mlflow.end_run()

    if not write_result:
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF page classification")

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input directory containing PDF files.",
    )

    parser.add_argument(
        "-g",
        "--ground_truth_path",
        type=str,
        required=False,
        help="(Optional) Path to the ground truth JSON file for evaluation.",
    )

    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        required=False,
        default="baseline",
        help="Specify which classifier to use for classification. Default set to baseline.",
    )

    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=False,
        help="Path to pretrained LayoutLMv3 or Tree Based model for classification.",
    )
    parser.add_argument(
        "-w",
        "--write-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Writes classification results to prediction.json file.",
    )
    args = parser.parse_args()

    # Check if model_path is required based on classifier
    if args.classifier.lower() in ["layoutlmv3", "treebased"] and not args.model_path:
        parser.error(f"--model_path is required when using classifier '{args.classifier}'")

    main(
        input_path=args.input_path,
        ground_truth_path=args.ground_truth_path,
        model_path=args.model_path,
        classifier_name=args.classifier,
        write_result=args.write_results,
    )
