"""Parallelized version of train.py with multiprocessing for feature extraction."""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from swissgeol_doc_processing.utils.file_utils import read_params as swissgeol_read_params
from tqdm import tqdm

from src.models.feature_engineering import get_features
from src.models.treebased.basetrainer import XGBoostTrainer
from src.models.treebased.model_explanation import explain_model
from src.page_classes import label2id

# Reuse trainer classes from train.py to avoid duplication
from src.utils.utility import get_pdf_files, read_params

logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

if mlflow_tracking:
    import mlflow

MATCHING_PARAMS_PATH = "config/local_matching_params.yml"
matching_params = read_params(MATCHING_PARAMS_PATH)
borehole_matching_params = swissgeol_read_params("matching_params.yml")


def build_filename_to_label_map(gt_json_path: Path) -> dict[tuple[str, int], int]:
    """Build a map from filename to class ID based on the ground truth JSON."""
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    label_lookup = {}
    for entry in gt_data:
        filename = entry["filename"]
        for pages in entry["pages"]:
            page = pages["page"]
            for label_name, value in pages["classification"].items():
                if value == 1:
                    try:
                        label_id = label2id[label_name]
                        label_lookup[(filename, page)] = label_id
                    except KeyError as err:
                        raise ValueError(f"Unknown label: {label_name}") from err
    return label_lookup


def extract_features_from_page(args):
    """Extract features from a single page (used for multiprocessing).

    Args:
        args: Tuple of (file_path, page_number, matching_params, borehole_matching_params)

    Returns:
        Tuple of (filename, page_number, features) or None if error
    """
    file_path, page_number, matching_params, borehole_matching_params = args
    filename = os.path.basename(file_path)

    try:
        with pymupdf.Document(file_path) as doc:
            # Page numbers are 1-indexed for user, 0-indexed for pymupdf
            page = doc[page_number - 1]
            features = get_features(page, page_number, matching_params, borehole_matching_params)
            return (filename, page_number, features)
    except Exception as e:
        logger.exception(f"Error processing {filename} page {page_number}: {e}")
        return None


def load_data_and_labels_parallel(folder_path: Path, label_map: dict[tuple[str, int], int], max_workers: int = None):
    """Loads data and labels from PDF files using parallel processing.

    Args:
        folder_path: Path to the folder containing PDF files.
        label_map: Mapping from (filename, page_number) to label ID.
        max_workers: Maximum number of parallel workers. If None, uses CPU count.

    Returns:
        Tuple of (features_list, labels_list) in deterministic order.
    """
    file_paths = get_pdf_files(folder_path)

    # Build ordered list of tasks with keys to preserve order
    tasks = []
    task_keys = []  # Store (filename, page_number) in submission order
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        with pymupdf.Document(file_path) as doc:
            page_count = len(doc)
            for page_number in range(1, page_count + 1):
                if (filename, page_number) in label_map:
                    tasks.append((file_path, page_number, matching_params, borehole_matching_params))
                    task_keys.append((filename, page_number))

    print(f"Processing {len(tasks)} pages from {len(file_paths)} files...")

    # Process pages in parallel, collecting results by key
    results = {}  # Map (filename, page_number) -> features

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(extract_features_from_page, task): task for task in tasks}

        # Collect results with progress bar (order doesn't matter here, we'll sort later)
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Extracting features"):
            result = future.result()
            if result is not None:
                filename, page_number, features = result
                results[(filename, page_number)] = features

    # Reconstruct features and labels in ORIGINAL ORDER (deterministic)
    all_features = []
    labels = []
    for key in task_keys:
        if key in results:
            all_features.append(results[key])
            labels.append(label_map[key])
        else:
            logger.warning(f"Missing result for {key}, skipping")

    return all_features, labels


def load_data_and_labels_sequential(folder_path: Path, label_map: dict[tuple[str, int], int]):
    """Sequential version for comparison/fallback."""
    file_paths = get_pdf_files(folder_path)
    all_features = []
    labels = []

    for file_path in tqdm(file_paths, desc="Processing files"):
        filename = os.path.basename(file_path)

        with pymupdf.Document(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                key = (filename, page_number)
                if key not in label_map:
                    continue  # Skip pages without labels
                features = get_features(page, page_number, matching_params, borehole_matching_params)
                all_features.append(features)
                labels.append(label_map[key])

    return all_features, labels


def main(config_path: str, out_directory: str, tuning: bool = False, parallel: bool = True, max_workers: int = None):
    """Main function to train the Random Forest or XGBoost model.

    Args:
        config_path: Path to the YAML configuration file.
        out_directory: Directory where the trained model and logs will be saved.
        tuning: Whether to perform hyperparameter tuning. Default is False.
        parallel: Whether to use parallel feature extraction. Default is True.
        max_workers: Maximum number of parallel workers. If None, uses CPU count.
    """
    if not mlflow_tracking:
        raise RuntimeError("MLflow tracking is disabled. Set MLFLOW_TRACKING=True in .env to enable it.")

    mlflow.set_experiment("Classifier Training")

    config = read_params(config_path)
    train_folder = Path(config["train_folder_path"])
    val_folder = Path(config["val_folder_path"])
    ground_truth_path = Path(config["ground_truth_file_path"])
    trainer_name = config["model_type"]

    model_out_directory = Path(out_directory) / time.strftime("%Y%m%d-%H%M%S")

    # --- Step 1: Load dataset train and validation
    label_lookup = build_filename_to_label_map(ground_truth_path)

    print(f"\nFeature extraction mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    trainer = XGBoostTrainer(config, model_out_directory)
    trainer.prepare_model()
    # Choose loading strategy
    start_time = time.time()
    load_fn = (
        partial(load_data_and_labels_parallel, max_workers=max_workers)
        if parallel
        else load_data_and_labels_sequential
    )

    X_train, y_train = load_fn(train_folder, label_lookup)
    X_val, y_val = load_fn(val_folder, label_lookup)

    elapsed = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed:.1f}s ({len(X_train) + len(X_val)} pages)")

    # --- Step 2: Build model trained
    if trainer_name != "xgboost":
        raise ValueError(f"Unsupported trainer: '{trainer_name}'. Only 'xgboost' is supported.")

    trainer = XGBoostTrainer(config, model_out_directory)

    # --- Step 3: Train model
    with mlflow.start_run(run_name=trainer_name):
        trainer.load_data(X_train, y_train, X_val, y_val)
        # if tuning:
        #     # TODO check tuning ?
        #     search_params = config["tuning"]["param_grid"]
        #     n_iter = config["tuning"].get("n_iter", 20)
        #     scoring = config["tuning"].get("scoring", "f1_micro")
        #     cv = config["tuning"].get("cv", 3)

        #     best_params, best_score = trainer.tune_hyperparameters(
        #         param_dist=search_params,
        #         n_iter=n_iter,
        #         scoring=scoring,
        #         cv=cv,
        #     )
        #     trainer.config["hyperparameters"].update(best_params)
        #     trainer.prepare_model()  # with best params

        #     mlflow.log_params(best_params)
        #     mlflow.log_metric("best_cv_score", best_score)
        # else:
        trainer.prepare_model()

        trainer.train()
        explain_model(trainer.model, trainer.X_train, trainer.id2label)
        trainer.save_model()

        y_pred = trainer.model.predict(X_val)
        metrics = trainer.evaluate(y_pred)

        # Log to mlflow
        mlflow.log_params(trainer.config.get("hyperparameters", {}))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_out_directory))

        if trainer.feature_names:
            mlflow.log_dict({"features": trainer.feature_names}, "features.json")
            trainer.plot_and_log_feature_importance()

        # Log confusion matrix and classification report
        trainer.plot_and_log_confusion_matrix(y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-path", required=True, help="Path to YAML config file")
    parser.add_argument("--out-directory", required=True, help="Output directory root")
    parser.add_argument("--tuning", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel processing (use sequential)")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers")
    args = parser.parse_args()

    main(
        config_path=args.config_file_path,
        out_directory=args.out_directory,
        tuning=args.tuning,
        parallel=not args.sequential,
        max_workers=args.max_workers,
    )
