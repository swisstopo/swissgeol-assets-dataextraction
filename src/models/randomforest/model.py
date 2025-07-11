from pathlib import Path

import joblib
import logging

from src.page_classes import PageClasses

logger = logging.getLogger(__name__)

class RandomForest:
    """Random Forest model for page classification.

        This class wraps the Random Forest model and provides methods for preprocessing,
        prediction, and training.
    """
    label2id = {"Boreprofile": 0, "Maps": 1, "Text": 2, "Title_Page": 3, "Unknown": 4}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    enum2id = {
        PageClasses.BOREPROFILE: 0,
        PageClasses.MAP: 1,
        PageClasses.TEXT: 2,
        PageClasses.TITLE_PAGE: 3,
        PageClasses.UNKNOWN: 4,
    }
    id2enum = {v: k for k, v in enum2id.items()}

    def __init__(self, model_path: str = None):
        self.model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("Failed to load Random Forest model.")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model.predict(X)

    def load_model(self, model_path: str):
        self.model = joblib.load(Path(model_path))

    def save_model(self, model_path: str):
        joblib.dump(self.model, Path(model_path))