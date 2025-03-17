import os
from tabulate import tabulate
import pandas as pd
import logging
import argparse
import yaml
from src.classify_scanned_page import classify_pdf
from tqdm import tqdm


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
    """Read parameters from a yaml file.

    Args:
        params_name (str): Name of the params yaml file.
    """
    with open(params_name) as f:
        params = yaml.safe_load(f)

    return params

def process_pdfs(pdf_files: list, **matching_params) -> pd.DataFrame:
    """Processes a list of PDFs and returns a classification DataFrame."""
    results = []
    with tqdm(total=len(pdf_files)) as pbar:

        for pdf in pdf_files:
            pbar.set_description(f"Processing {os.path.basename(pdf)}")
            for entry in classify_pdf(pdf, matching_params):
                results.append(entry)
            pbar.update(1)
        return pd.DataFrame(results)

def main(input_path: str, output_path: str, ground_truth_path: str = None):
    """Runs PDF classification and saves results to CSV."""

    pdf_files = get_pdf_files(input_path)

    if not pdf_files:
        logging.error("No valid PDFs found.")
        return

    matching_params = read_params("matching_params.yml")

    logger.info(f"Start classifying {len(pdf_files)} PDF files")
    results_df = process_pdfs(pdf_files, **matching_params)

    if results_df.empty:
        logging.warning("No data to save.")
        return

    results_df.to_csv(output_path, index=False)
    logger.info(f"Classification results saved to: {output_path}")

    #dislay classification ratios
    calculate_classification_ratios(results_df)

    if ground_truth_path:
        evaluate_results(results_df, ground_truth_path)

def evaluate_results(results: pd.DataFrame, ground_truth_path: str):
    """Compares classification results with ground truth labels."""
    try:
        ground_truth = pd.read_csv(ground_truth_path)
        results = results.merge(ground_truth, on=["Filename", "Page Number"], how="left")
        results["Correct"] = results["Classification"] == results["True Label"]

        from sklearn.metrics import classification_report
        report = classification_report(results["True Label"], results["Classification"], zero_division=0)
        logger.info("\nClassification Report:\n" + report)

    except FileNotFoundError:
        logger.warning(f"\nGround truth file not found: {ground_truth_path}")

def calculate_classification_ratios(results_df: pd.DataFrame):
    """Calculates and logs classification ratios using tabulate."""
    if results_df.empty:
        logger.warning("No data to analyze.")
        return

    total_pages = len(results_df)
    classification_counts = results_df["Classification"].value_counts().to_dict()

    summary_data = [
        [category, count, f"{(count / total_pages) * 100:.2f}%"]
        for category, count in classification_counts.items()
    ]

    logger.info("\nClassification Summary:\n" + tabulate(summary_data, headers=["Category", "Count", "Percentage"], tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF page classification")
    
    parser.add_argument(
        "-i", "--input_path", type=str, required=True,
        help="Path to the input directory containing PDF files."
    )
    
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to save classification results CSV."
    )
    
    parser.add_argument(
        "-g", "--ground_truth_path", type=str, required=False,
        help="(Optional) Path to the ground truth CSV file for evaluation."
    )

    args = parser.parse_args()

    # Execute classification
    main(args.input_path, args.output_path, args.ground_truth_path)