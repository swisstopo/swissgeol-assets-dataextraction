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

class TreeBasedModel:
    """Tree-based model for page classification.
    This class includes loading the model from a file and making predictions based on input features.
    Attributes:
        model: The trained model used for predictions.
    Args:
        model_path (str): Path to the trained model file. If None, the model is not loaded.
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
            logger.warning(f"Failed to load Tree-based model from {model_path}.")

    def predict(self, x: list[float]) -> list[int]:
        """Predict the class labels for the input features.
        
        Args:
            x: Input features for prediction.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict(x)

    def load_model(self, model_path: str):
        """Load the model from the specified path.
        
        Args:
            model_path (str): Path to the model file.
        """
        self.model = joblib.load(Path(model_path))