import os
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

import mlflow
from sklearn.ensemble import RandomForestClassifier

from src.models.basetrainer import BaseTrainer
from src.models.feature_engineering import get_features
from src.utils import read_params, get_pdf_files
from src.page_classes import id2label
from src.classifiers.pdf_dataset_builder import build_filename_to_label_map

logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

MATCHING_PARAMS_PATH = "config/matching_params.yml"
matching_params = read_params(MATCHING_PARAMS_PATH)

class RandomForestTrainer(BaseTrainer):
    def prepare_model(self):
        hyperparams = self.config.get("hyperparameters", {})
        self.model = RandomForestClassifier(**hyperparams)

class_names = [label for _, label in sorted(id2label.items())]

def load_data_and_labels(folder_path: Path, label_map: dict[str, int]):
    """Extract features and labels for all PDF pages in a folder."""
    file_paths = get_pdf_files(folder_path)
    features = get_features(file_paths, matching_params)
    labels = []
    for f in file_paths:
        filename = os.path.basename(f)
        if filename not in label_map:
            raise ValueError(f"Missing label for file: {filename}")
        labels.append(label_map[filename])
    return features, labels


def main(config_path: str, out_directory:str):

    config = read_params(config_path)
    train_folder = Path(config["train_folder_path"])
    val_folder = Path(config["val_folder_path"])
    ground_truth_path = Path(config["ground_truth_file_path"])

    model_out_directory = Path(out_directory) / time.strftime("%Y%m%d-%H%M%S")
    ## create dataset
    label_lookup = build_filename_to_label_map(ground_truth_path)
    X_train, y_train = load_data_and_labels(train_folder, label_lookup)
    X_val, y_val = load_data_and_labels(val_folder, label_lookup)

    mlflow.set_experiment("Classifier Training")

    ## create trainer
    trainer = RandomForestTrainer(config, model_out_directory)
    ## train model
    trainer.load_data(X_train, y_train, X_val, y_val)
    trainer.prepare_model()
    trainer.train()

    y_pred = trainer.model.predict(X_val)
    metrics = trainer.evaluate(y_pred)

    ##logg to mlflow
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_out_directory))

    if trainer.feature_names:
        mlflow.log_dict({"features": trainer.feature_names}, "features.json")
        trainer.plot_and_log_feature_importance()

        # Log confusion matrix and classification report
    y_pred = trainer.model.predict(trainer.X_val)
    trainer.plot_and_log_confusion_matrix(y_pred, class_names=class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--out", required=True, help="Output directory root")
    args = parser.parse_args()
    main(args.config, args.out)