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

from src.page_classes import (
    enum2id,
    id2enum,
    id2label,
    label2id,
    num_labels,
)


class TreeBasedTrainer(abc.ABC):
    """Abstract base class for training models.

    This class defines the structure for model training workflows,
    including methods for data loading, training, evaluation, and model saving.

    Subclasses should implement the `prepare_model` method to initialize their specific model type.

    Attributes:
        label2id (dict): Mapping from label names to IDs.
        id2label (dict): Mapping from IDs to label names.
        enum2id (dict): Mapping from enum names to IDs.
        id2enum (dict): Mapping from IDs to enum names.
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
        self.label2id = label2id
        self.id2label = id2label
        self.enum2id = enum2id
        self.id2enum = id2enum
        self.num_labels = num_labels

        self.config = config
        self.model = None
        self.feature_names = config.get("feature_names")
        self.model_dir = output_path
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def prepare_model(self):
        """Prepares the model for training. This method should be implemented by subclasses."""
        pass

    def load_data(self, X_train, y_train, X_val, y_val):
        """Loads training and validation data into numpy arrays."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)

    def train(self):
        """Trains the model using the loaded training data."""
        if self.model is None:
            raise ValueError("Model is not prepared. Call prepare_model() before training.")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, y_pred):
        """Evaluates the model's performance on the validation set.

        Args:
            y_pred (list): Predicted labels for the validation set.

        Returns:
            dict: A dictionary containing precision, recall, and F1 score.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average="micro", zero_division=0
        )
        return {"precision_micro": precision, "recall_micro": recall, "f1_micro": f1}

    def save_model(self, filename: str = "model.joblib"):
        """Saves the trained model to the specified file."""
        path = self.model_dir / filename
        joblib.dump(self.model, path)
        return path

    def plot_and_log_feature_importance(self):
        """Plots and logs the feature importance of the trained model."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature importances. Ensure it is a tree-based model.")

        # Get feature importances and sort them
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

    def plot_and_log_confusion_matrix(self, y_pred: list):
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
