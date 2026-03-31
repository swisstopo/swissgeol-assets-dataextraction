import abc
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.evaluation import save_csv
from src.models.treebased.model import XGBOODClassifier
from src.page_classes import id2label, num_labels


class TreeBasedTrainer(abc.ABC):
    """Abstract base class for training models.

    This class defines the structure for model training workflows,
    including methods for data loading, training, evaluation, and model saving.

    Subclasses should implement the `prepare_model` method to initialize their specific model type.

    Attributes:
        id2label (dict): Mapping from IDs to label names.
        num_labels (int): Number of unique labels.
        config (dict): Configuration dictionary containing model parameters.
        model (object): The machine learning model to be trained.
        feature_names (list): List of feature names used in the model.
        model_dir (Path): Directory where the trained model will be saved.
    """

    def __init__(self, config: dict, output_path: Path):
        """Initializes the BaseTrainer with configuration and output path.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            output_path (Path): Directory where the trained model will be saved.
        """
        self.id2label = id2label
        self.num_labels = num_labels

        self.config = config
        self.model = None
        self.feature_names = config.get("feature_names")
        self.model_dir = output_path
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def prepare_model(self) -> None:
        """Prepares the model for training. This method should be implemented by subclasses."""
        pass

    def load_data(
        self,
        X_train: list[list[float]],
        y_train: list[int],
        k_train: list[tuple[str, int]],
        X_val: list[list[float]],
        y_val: list[int],
        k_val: list[tuple[str, int]],
    ) -> None:
        """Loads training and validation data into numpy arrays.

        Args:
            X_train (list[list[float]]): Training features.
            y_train (list[int]): Training labels.
            k_train (list[tuple[str, int]]): Keys (filename, page number) for each training sample.
            X_val (list[list[float]]): Validation features.
            y_val (list[int]): Validation labels.
            k_val (list[tuple[str, int]]): Keys (filename, page number) for each validation sample.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)
        self.k_train = k_train
        self.k_val = k_val

    def train(self) -> None:
        """Trains the model using the loaded training data."""
        if self.model is None:
            raise ValueError("Model is not prepared. Call prepare_model() before training.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, y_pred) -> dict:
        """Evaluates the model's performance on the validation set.

        Args:
            y_pred (list): Predicted labels for the validation set.

        Returns:
            dict: A dictionary containing precision, recall, and F1 score (micro, macro).
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average="micro", zero_division=0
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(self.y_val, y_pred, average="macro", zero_division=0)
        return {"precision_micro": precision, "recall_micro": recall, "f1_micro": f1, "f1_macro": f1_macro}

    def save_model(self, filename: Path = "model.joblib") -> Path:
        """Saves the trained model to the specified file."""
        path = self.model_dir / filename
        joblib.dump(self.model, path)
        signature = mlflow.models.infer_signature(self.X_train[:1], self.model.predict(self.X_train[:1]))
        mlflow.sklearn.log_model(self.model, name=self.model_name, signature=signature)
        return path

    def plot_and_log_feature_importance(self) -> None:
        """Plots and logs the feature importance of the trained model."""
        # Get feature importances and sort them
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature importances. Ensure it is a tree-based model.")

        if self.feature_names is None:
            raise ValueError("Feature names are not provided in the configuration.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_names = [self.feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), sorted_names, rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.tight_layout()
        fig_path = self.model_dir / "feature_importance.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(str(fig_path))

    def plot_and_log_confusion_matrix(self, y_pred: list) -> None:
        """Plots and logs the confusion matrix for the validation set predictions.

        Args:
            y_pred (list): Predicted labels for the validation set.
        """
        class_names = [self.id2label[i] for i in sorted(self.id2label)]
        cm = confusion_matrix(self.y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation="vertical")
        plt.tight_layout()
        fig_path = self.model_dir / "confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(str(fig_path))

        # Also log classification report as JSON
        report_dict = classification_report(self.y_val, y_pred, target_names=class_names, output_dict=True)
        report_path = self.model_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(str(report_path))

    def plot_and_file_predictions(self, y_pred: list) -> None:
        """Create a per-page classification report and save it as a CSV.

        Args:
            y_pred (list): List of predicted class IDs for the validation set.
        """
        class_names = [self.id2label[i] for i in sorted(self.id2label)]

        # Create output table structure
        output_table = [["Filename", "Page", "Ground truth", "Prediction"]]
        for y_gt, y_pr, key in zip(self.y_val, y_pred, self.k_val, strict=True):
            output_table.append([key[0], key[1], class_names[y_gt], class_names[y_pr]])

        # Log results to CSV file
        report_path = self.model_dir / "summary_files.csv"
        save_csv(output_table, report_path)

        # Save to mlflow
        mlflow.log_artifact(str(report_path))


class XGBoostTrainer(TreeBasedTrainer):
    """Trainer for XGBoost models.

    This class extends the TreeBasedTrainer to implement specific methods for training and evaluating
    XGBoost models using the provided configuration and data.
    """

    model_name = "xgboost_model"

    def __init__(self, config: dict, output_path: Path):
        super().__init__(config, output_path)
        # Extract configuration variables for model
        self.ood_use = config.get("ood_use", False)

        # Choose model backbone
        self.model_builder = XGBOODClassifier if self.ood_use else XGBClassifier

        # Set base parameters (add OOD if activated) - make copy to avoid overwritting
        self.hyperparams = dict(config.get("hyperparameters", {}))
        if self.ood_use:
            hyperparams_ood = self.config.get("hyperparameters_ood", {})
            self.hyperparams.update(hyperparams_ood)

    def prepare_model(self) -> None:
        """Prepares the XGBoost model for training."""
        self.model = self.model_builder(objective="multi:softprob", num_class=self.num_labels, **self.hyperparams)

    def tune_hyperparameters(self, param_dist: dict, scoring: str = "f1_micro", cv: int = 3) -> tuple[dict, float]:
        """Runs GridSearchCV to tune hyperparameters for XGBoost.

        Args:
            param_dist: Dictionary with parameters to search.
            scoring: Scoring method to use for evaluation.
            cv: Number of folds in cross-validation.

        Returns:
            tuple[dict, float]: A tuple of (best_params, best_score) where
                * best_params: Best hyperparameters found during tuning.
                * best_score: Best score achieved during tuning.
        """
        # Only consider params that are part of the hyperparameters set
        param_dist = {k: v for k, v in param_dist.items() if k in self.hyperparams}

        # Run grid search over the pre-initialized model.
        search = GridSearchCV(
            estimator=self.model,
            param_grid=param_dist,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )

        # Fit grid search and return best params
        search.fit(self.X_train, self.y_train)
        return search.best_params_, search.best_score_
