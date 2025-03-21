import os
import logging
import argparse
import yaml
import json
from tqdm import tqdm
from dotenv import load_dotenv

from src.classify_scanned_page import classify_pdf
from src.evaluation import evaluate_results

# Load .env and check MLFlow
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

if mlflow_tracking:
    import mlflow
    import pygit2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_pdf_files(input_path: str) -> list:
    """Returns a list of PDF files from a directory or a single file."""
    if os.path.isdir(input_path):
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.pdf')]
    elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        return [input_path]
    
    logging.error("Invalid input path: must be a PDF file or a directory containing PDFs.")
    return []

def read_params(params_name: str) -> dict:
    with open(params_name) as f:
        return yaml.safe_load(f)

def setup_mlflow(input_path, output_path, ground_truth_path, matching_params):
    mlflow.set_experiment("PDF Page Classification")
    mlflow.start_run()

    mlflow.set_tag("input_path", str(input_path))
    mlflow.set_tag("output_path", str(output_path))
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

def flatten_dict(d, parent_key='', sep='.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_pdfs(pdf_files: list, **matching_params) -> list[dict]:
    results = []
    with tqdm(total=len(pdf_files)) as pbar:
        for pdf in pdf_files:
            pbar.set_description(f"Processing {os.path.basename(pdf)}")
            classification_data = classify_pdf(pdf, matching_params)
            if classification_data:
                results.append(classification_data)
            pbar.update(1)

    return results

def main(input_path: str, output_path: str, ground_truth_path: str = None):
    pdf_files = get_pdf_files(input_path)
    if not pdf_files:
        logger.error("No valid PDFs found.")
        return

    matching_params = read_params("matching_params.yml")

    # Start MLflow tracking
    if mlflow_tracking:
        setup_mlflow(input_path, output_path, ground_truth_path, matching_params)

    logger.info(f"Start classifying {len(pdf_files)} PDF files")
    
    # Processed PDFs
    results = process_pdfs(pdf_files, **matching_params)

    if not results:
        logger.warning("No data to save.")
        return

    # Save to JSON
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent = 4)

    if ground_truth_path:
        evaluate_results(results, ground_truth_path)

    if mlflow_tracking:
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF page classification")
    
    parser.add_argument(
        "-i", "--input_path", type=str, required=True,
        help="Path to the input directory containing PDF files."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to save classification results JSON."
    )
    parser.add_argument(
        "-g", "--ground_truth_path", type=str, required=False,
        help="(Optional) Path to the ground truth JSON file for evaluation."
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.ground_truth_path)