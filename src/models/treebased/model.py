import logging
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import halfnorm
from xgboost import XGBClassifier

from src.page_classes import (
    PageClasses,
    enum2id,
    id2enum,
    id2label,
    label2id,
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


class XGBOODClassifier(XGBClassifier):
    """Out of distribution (OOD) XGBoost classifier."""

    def __init__(self, prob_ood: float = 0.95, **kwargs):
        """Initialisation of the XGBoostOOD classifier.

        Args:
            num_class (int): Number of classes (with OOD).
            prob_ood (float, optional): Probability for OOD detection. Defaults to 0.95.
            kwargs (dict): Additional parameters.
        """
        self.id_ood = label2id[PageClasses.UNKNOWN]
        self.prob_ood = prob_ood
        self.thresholds = np.zeros(kwargs.get("num_class") - 1, dtype=np.float64)
        # XGBoost's init can overwrite attrs set after it
        super().__init__(**kwargs)

    def _estimate_thresholds(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate thresholds for each classed in OOD detection.

        Args:
            X (NDArray[np.float64]): Data features.
            y (NDArray[np.float64]): Data classes.

        Returns:
            NDArray[np.float64]: _description_
        """
        thresholds = np.zeros_like(self.thresholds)
        for c in range(len(thresholds)):
            # Fit probability for given class to half normal distribution
            Xc_proba = super().predict_proba(X[y == c, :])
            _, sigma = halfnorm.fit(1 - Xc_proba[:, c], floc=0)
            # Define threshold as prob_ood confidence
            thresholds[c] = 1 - halfnorm.ppf(self.prob_ood, scale=sigma)
        return thresholds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs) -> Self:
        """Fit model.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training label.
            kwargs (dict): Additional parameters.

        Returns:
            Self: Fitted model
        """
        # Step 1: Fit XGBoost with all classes except OOD
        super().fit(X[self.id_ood != y, :], y[self.id_ood != y], **kwargs)
        # Step 2: Estimate OOD threhsiold based on class distribution
        self.thresholds = self._estimate_thresholds(X, y)
        return self

    def predict(self, X: NDArray[np.float64], **kwargs) -> NDArray[np.int64]:
        """Predict classes based on input features.

        Args:
            X (NDArray[np.float64]): Features to predict.
            kwargs (dict): Additional parameters.

        Returns:
            NDArray[np.int64]: Predicted classes
        """
        # Compute probability over all classes and get argmax/max
        y_proba = super().predict_proba(X, **kwargs)
        y_label = y_proba.argmax(axis=1)
        y_label_th = y_proba.max(axis=1)
        # Replace prediction where threshold is not meet
        y_label[self.thresholds[y_label] > y_label_th] = self.id_ood
        return y_label
