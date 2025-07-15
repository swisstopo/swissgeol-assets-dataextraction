from pathlib import Path

import joblib
import logging

from src.page_classes import (
    label2id,
    id2label,
    enum2id,
    id2enum,
    num_labels,
)
logger = logging.getLogger(__name__)

class RandomForest:
    """Random Forest model for page classification.

        This class wraps the Random Forest model and provides methods for preprocessing,
        prediction, and training.
    """

    def __init__(self, model_path: str = None):

        self.label2id = label2id
        self.id2label = id2label
        self.enum2id = enum2id
        self.id2enum = id2enum
        self.num_labels = num_labels

        self.model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Failed to load Random Forest model from {model_path}.")

    def predict(self, x):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict(x)

    def load_model(self, model_path: str):
        self.model = joblib.load(Path(model_path))