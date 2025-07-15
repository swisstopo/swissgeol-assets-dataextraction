import os
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

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

class_names = [label for _, label in sorted(id2label.items())]

class RandomForestTrainer(BaseTrainer):
    def prepare_model(self):
        hyperparams = self.config.get("hyperparameters", {})
        self.model = RandomForestClassifier(**hyperparams)

    def tune_hyperparameters(self,
                             param_dist: dict,
                             n_iter: int = 20,
                             cv: int = 3,
                             scoring: str = "f1_micro",
                             random_state: int = 42):
        """Run RandomizedSearchCV to tune hyperparameters."""
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

class XGBoostTrainer(BaseTrainer):
    def prepare_model(self):
        hyperparams = self.config.get("hyperparameters", {})
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=len(class_names),
            **hyperparams
        )

    def tune_hyperparameters(self, param_dist, n_iter=20, scoring="f1_micro", cv=3):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
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

def main(config_path: str, out_directory:str, tuning:bool = True):

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

    ## create trainer
    trainer.load_data(X_train, y_train, X_val, y_val)

    with mlflow.start_run(run_name=trainer_name):

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

        ## train model
        trainer.prepare_model()
        trainer.train()

        ##save model
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
        trainer.plot_and_log_confusion_matrix(y_pred, class_names=class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--out", required=True, help="Output directory root")
    args = parser.parse_args()
    main(args.config, args.out)