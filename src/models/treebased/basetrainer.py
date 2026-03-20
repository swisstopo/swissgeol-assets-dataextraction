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
from sklearn.model_selection import RandomizedSearchCV

from src.models.treebased.model import XGBClassifier, XGBOODClassifier
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
        _, _, f1_macro, _ = precision_recall_fscore_support(self.y_val, y_pred, average="macro", zero_division=0)
        return {"precision_micro": precision, "recall_micro": recall, "f1_micro": f1, "f1_macro": f1_macro}

    def save_model(self, filename: str = "model.joblib"):
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


class XGBoostTrainer(TreeBasedTrainer):
    """Trainer for XGBoost models.

    This class extends the TreeBasedTrainer to implement specific methods for training and evaluating
    XGBoost models using the provided configuration and data.
    """

    model_name = "xgboost_model"

    def prepare_model(self):
        """Prepares the XGBoost model for training."""
        hyperparams = self.config.get("hyperparameters", {})
        # self.model = XGBClassifier(objective="multi:softprob", num_class=self.num_labels, **hyperparams)
        self.model = XGBOODClassifier(objective="multi:softprob", num_class=self.num_labels, **hyperparams)

    def tune_hyperparameters(
        self, param_dist: dict, n_iter: int = 20, scoring: str = "f1_micro", cv: int = 3, random_state: int = 42
    ) -> tuple[dict, float]:
        """Runs RandomizedSearchCV to tune hyperparameters for XGBoost.

        Args:
            param_dist: Dictionary with parameters to search.
            n_iter: Number of parameter settings that are sampled.
            scoring: Scoring method to use for evaluation.
            cv: Number of folds in cross-validation.
            random_state (int): Random seed for reproducibility.

        Returns:
                best_params: Best hyperparameters found during tuning.
                best_score: Best score achieved during tuning.
        """
        # Initialize XGBoost model with default parameters
        model = XGBClassifier(objective="multi:softprob", num_class=self.num_labels, eval_metric="mlogloss")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)
        return search.best_params_, search.best_score_
