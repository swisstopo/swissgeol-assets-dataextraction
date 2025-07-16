import abc
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, ConfusionMatrixDisplay
import json
import mlflow
from src.page_classes import (
    label2id,
    id2label,
    enum2id,
    id2enum,
    num_labels,
)

class BaseTrainer(abc.ABC):
    def __init__(self, config: dict, output_path: Path):
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
        pass

    def load_data(self, X_train, y_train, X_val, y_val):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average="micro", zero_division=0
        )
        return {"precision_micro": precision, "recall_micro": recall, "f1_micro": f1}

    def save_model(self, filename: str = "model.joblib"):
        path = self.model_dir / filename
        joblib.dump(self.model, path)
        return path

    def plot_and_log_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_names = [self.feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), sorted_names, rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.tight_layout()
        fig_path = self.model_dir  / "feature_importance.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(str(fig_path))

    def plot_and_log_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=id2label)
        disp.plot(xticks_rotation="vertical")
        plt.tight_layout()
        fig_path = self.model_dir  / "confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(str(fig_path))

        # Also log classification report as JSON
        report_dict = classification_report(self.y_val, y_pred, target_names=id2label, output_dict=True)
        report_path = self.model_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        mlflow.log_artifact(str(report_path))
