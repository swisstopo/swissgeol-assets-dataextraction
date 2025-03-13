import os
import sys
import pandas as pd
import logging
import argparse
from classify_scanned_page import classify_pdf 

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(input_dir, output_dir,ground_truth_path = None):
    
    logging.info(f"Starting page classification for input folder: {input_dir}")

    results = classify_pdf(input_dir)
    results.to_csv(os.path.join(output_dir, "classification.csv"), index= False)
    logger.info(f"Classification results saved to: {os.path.join(output_dir, 'classification.csv')}")

    if ground_truth_path:
        try:
            ground_truth = pd.read_csv(ground_truth_path)
            results = results.merge(ground_truth, on=["Filename", "Page Number"], how="left")
            results["Correct"] = results["Classification"] == results["True Label"]

            from sklearn.metrics import classification_report
            report = classification_report(results["True Label"], results["Classification"], zero_division=0)
            logger.info("\nClassification Report:\n" + report)
        except FileNotFoundError:
            logger.warning(f"\nGround truth file not found: {ground_truth_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF page classification")
    
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the input directory containing PDF files."
    )
    
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to save classification results CSV."
    )
    
    parser.add_argument(
        "--ground_truth_path", type=str, required=False,
        help="(Optional) Path to the ground truth CSV file for evaluation."
    )

    args = parser.parse_args()

    # Execute classification
    main(args.input_dir, args.output_dir, args.ground_truth_path)