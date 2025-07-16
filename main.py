import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.classifiers.classifier_factory import ClassifierTypes, create_classifier
from src.classify_page import classify_pdf
from src.evaluation import evaluate_results
from src.utils import read_params

# Load .env and check MLFlow
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

if mlflow_tracking:
    import mlflow
    import pygit2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_pdf_files(input_path: Path) -> list[Path]:
    """Returns a list of PDF files from a directory or a single file."""
    if input_path.is_dir():
        return [f for f in input_path.rglob("*.pdf")]
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    logging.error("Invalid input path: must be a PDF file or a directory containing PDFs.")
    return []


def setup_mlflow(input_path: Path, ground_truth_path: Path, matching_params: dict):
    mlflow.set_experiment("PDF Page Classification")
    mlflow.start_run()

    mlflow.set_tag("input_path", str(input_path))

    if ground_truth_path:
        mlflow.set_tag("ground_truth_path", str(ground_truth_path))

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


def process_pdfs(pdf_files: list[Path], classifier, **matching_params) -> list[dict]:
    results = []
    with tqdm(total=len(pdf_files)) as pbar:
        for pdf in pdf_files:
            pbar.set_description(f"Processing {pdf.name}")
            classification_data = classify_pdf(pdf, classifier, matching_params)
            if classification_data:
                results.append(classification_data)
            pbar.update(1)

    return results


def main(input_path: str, ground_truth_path: str = None, model_path: str = None, classifier_name: str = "baseline"):
    """
    Run the page classification pipeline on input documents.

    Args:
        input_path (str): Path to directory with PDF pages or documents.
        ground_truth_path (str, optional): Path to ground truth JSON file for evaluation.
        model_path (str, optional): Path to pretrained LayoutLMv3 model.
        classifier_name (str, optional): Classifier to use ("baseline", "pixtral", etc.).

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
        setup_mlflow(input_path, ground_truth_path, matching_params)

    # Set up classifier
    classifier_type = ClassifierTypes.infer_type(classifier_name)
    classifier = create_classifier(classifier_type, model_path, matching_params)

    logger.info(f"Start classifying {len(pdf_files)} PDF files with {classifier.type.value} classifier")

    # Processed PDFs
    results = process_pdfs(pdf_files, classifier)

    if not results:
        logger.warning("No data to save.")
        return

    # Save to JSON
    output_file = Path("data") / "prediction.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as json_file:
        json.dump(results, json_file, indent=4)

    # log raw outputs
    csv_path = "data/labels.csv"
    if mlflow_tracking:
        mlflow.log_artifact(csv_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    if ground_truth_path:
        evaluate_results(results, ground_truth_path)

    if mlflow_tracking:
        mlflow.end_run()


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
        "-p", "--model_path", type=str, required=False, help="Path to pretrained LayoutLMv3 model for classification."
    )
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        required=False,
        default="baseline",
        help="Specify which classifier to use for classification. Default set to baseline.",
    )
    args = parser.parse_args()
    main(args.input_path, args.ground_truth_path, args.model_path, args.classifier)
