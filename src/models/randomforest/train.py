import os
import time
import argparse
import logging
import pymupdf
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from pathlib import Path
import mlflow

from classifiers.pdf_dataset_builder import build_filename_to_label_map
from src.models.basetrainer import TreeBasedTrainer
from src.models.feature_engineering import get_features
from src.utils import read_params, get_pdf_files

logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"

MATCHING_PARAMS_PATH = "config/matching_params.yml"
matching_params = read_params(MATCHING_PARAMS_PATH)

class RandomForestTrainer(TreeBasedTrainer):
    """Trainer for Random Forest models.
    
    This class extends the TreeBasedTrainer to implement specific methods for training and evaluating
    Random Forest models using the provided configuration and data.
    """
    def prepare_model(self):
        """Prepares the Random Forest model for training."""
        hyperparams = self.config.get("hyperparameters", {})
        self.model = RandomForestClassifier(**hyperparams)

    def tune_hyperparameters(self,
                             param_dist: dict,
                             n_iter: int = 20,
                             cv: int = 3,
                             scoring: str = "f1_micro",
                             random_state: int = 42):
        """Runs RandomizedSearchCV to tune hyperparameters. 
        Args:
            param_dist (dict): Dictionary with parameters to search.
            n_iter (int): Number of parameter settings that are sampled.
            cv (int): Number of folds in cross-validation.
            scoring (str): Scoring method to use for evaluation.
            random_state (int): Random seed for reproducibility.
            
            Returns:
                best_params (dict): Best hyperparameters found during tuning.
                best_score (float): Best score achieved during tuning.
        """
        search = RandomizedSearchCV(
            RandomForestClassifier(),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            verbose=1,
            n_jobs=1,
        )
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        return search.best_params_, search.best_score_

class XGBoostTrainer(TreeBasedTrainer):
    """Trainer for XGBoost models.
    
    This class extends the TreeBasedTrainer to implement specific methods for training and evaluating
    XGBoost models using the provided configuration and data.
    """
    def prepare_model(self):
        """Prepares the XGBoost model for training."""
        hyperparams = self.config.get("hyperparameters", {})
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=self.num_labels,
            **hyperparams
        )

    def tune_hyperparameters(self, param_dist, n_iter=20, scoring="f1_micro", cv=3):
        """Runs RandomizedSearchCV to tune hyperparameters for XGBoost.
        Args:
            param_dist (dict): Dictionary with parameters to search.
            n_iter (int): Number of parameter settings that are sampled.
            scoring (str): Scoring method to use for evaluation.
            cv (int): Number of folds in cross-validation.
            
            Returns:
                best_params (dict): Best hyperparameters found during tuning.
                best_score (float): Best score achieved during tuning.
        """
        # Initialize XGBoost model with default parameters
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_labels,
            eval_metric="mlogloss"
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)
        return search.best_params_, search.best_score_

def load_data_and_labels(folder_path: Path, label_map: dict[tuple[str, int], int]):
    """Loads data and labels from PDF files in the specified folder.

    Args:
        folder_path (Path): Path to the folder containing PDF files.
        label_map (dict): Mapping from (filename, page_number) to label ID.

    Returns:
        tuple: A tuple containing a list of features and a list of labels.
    """
    file_paths = get_pdf_files(folder_path)
    all_features = []
    labels = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)

        with pymupdf.Document(file_path) as doc:
            print(f"Processing {filename}", end="\r")

            for page_number, page in enumerate(doc, start = 1):
                features = get_features(page,page_number,matching_params)
                all_features.append(features)

                key = (filename, page_number)
                if key not in label_map:
                    raise ValueError(f"Missing label for file: {key}")
                labels.append(label_map[key])

    return all_features, labels


def main(config_path: str, out_directory: str, tuning: bool = False):
    """Main function to train the Random Forest or XGBoost model based on the provided configuration.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        out_directory (str): Directory where the trained model and logs will be saved.
        tuning (bool): Whether to perform hyperparameter tuning. Default is False.
    """
    if not mlflow_tracking:
        print("MLflow tracking is disabled. Set MLFLOW_TRACKING=True in .env to enable it.")

    config = read_params(config_path)
    train_folder = Path(config["train_folder_path"])
    val_folder = Path(config["val_folder_path"])
    ground_truth_path = Path(config["ground_truth_file_path"])
    trainer_name = config["model_type"]

    model_out_directory = Path(out_directory) / time.strftime("%Y%m%d-%H%M%S")

    ## create dataset
    label_lookup = build_filename_to_label_map(ground_truth_path)
    X_train, y_train = load_data_and_labels(train_folder, label_lookup)
    X_val, y_val = load_data_and_labels(val_folder, label_lookup)

    mlflow.set_experiment("Classifier Training")

    if trainer_name == "random_forest":
        trainer = RandomForestTrainer(config, model_out_directory)
    elif trainer_name == "xgboost":
        trainer = XGBoostTrainer(config, model_out_directory)
    else:
        raise ValueError(f"Unsupported trainer: {trainer_name}")

    with mlflow.start_run(run_name=trainer_name):
        trainer.load_data(X_train, y_train, X_val, y_val)
        if tuning:
            search_params = config["tuning"]["param_grid"]
            n_iter = config["tuning"].get("n_iter", 20)
            scoring = config["tuning"].get("scoring", "f1_micro")
            cv = config["tuning"].get("cv", 3)

            best_params, best_score = trainer.tune_hyperparameters(
                param_dist=search_params,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
            )
            trainer.config["hyperparameters"].update(best_params)
            trainer.prepare_model()  # with best params

            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", best_score)
        else:
            trainer.prepare_model()

        trainer.train()
        trainer.save_model()

        y_pred = trainer.model.predict(X_val)
        metrics = trainer.evaluate(y_pred)

        ##logg to mlflow
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
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--out", required=True, help="Output directory root")
    args = parser.parse_args()
    main(args.config, args.out)