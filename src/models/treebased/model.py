import logging
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import halfnorm
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier as XGB

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


class XGBClassifier(BaseEstimator):
    """Base XGBoost classifier wrapped in base estimator."""

    def __init__(self, objective: str, num_class: int, **kwargs):
        """Initialisation of the XGBoost classifier.

        Args:
            objective (str): Objective function for XGBoost.
            num_class (int): Numebr of classes (with OOD).
            kwargs (dict): Additional parameters.
        """
        self.id2label = id2label
        self.model = XGB(objective=objective, num_class=num_class, **kwargs)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fit model using SKLearn BaseEstimator.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training label.

        Returns:
            Self: Fitted model
        """
        self.is_fitted_ = True
        self.model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict classes based on input features.

        Args:
            X (NDArray[np.float64]): Features to predict.

        Returns:
            NDArray[np.int64]: Predicted classes
        """
        return self.model.predict(X)

    def get_tree_model(self) -> XGB:
        """Get inner tree model.

        Returns:
            XGB: Returned model
        """
        return self.model

    def get_model_feature_importances(self) -> XGB:
        """Get inner tree model.

        Returns:
            XGB: Returned model
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature importances. Ensure it is a tree-based model.")
        return self.model.feature_importances_


class XGBOODClassifier(XGBClassifier):
    """Out of distribution (OOD) XGBoost classifier wrapped in base estimator."""

    def __init__(self, objective: str, num_class: int, prob_ood: float = 0.95, **kwargs):
        """Initialisation of the XGBoostOOD classifier.

        Args:
            objective (str): Objective function for XGBoost.
            num_class (int): Numebr of classes (with OOD).
            prob_ood (float, optional): Probability for OOD detection. Defaults to 0.95.
            kwargs (dict): Additional parameters.
        """
        super().__init__(objective, num_class, **kwargs)

        self.id_ood = label2id[PageClasses.UNKNOWN]
        self.prob_ood = prob_ood
        self.num_class = num_class - 1
        self.thresholds = np.zeros_like(self.num_class, dtype=np.float64)

    def _estimate_thresholds(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate thresholds for each classed in OOD detection.

        Args:
            X (NDArray[np.float64]): Data features.
            y (NDArray[np.float64]): Data classes.

        Returns:
            NDArray[np.float64]: _description_
        """
        thresholds = np.zeros(self.num_class, dtype=np.float64)
        for c in range(self.num_class):
            # Fit probability for given class to half normal distribution
            Xc_proba = self.model.predict_proba(X[y == c, :])
            _, sigma = halfnorm.fit(1 - Xc_proba[:, c], floc=0)
            # Define threshold as prob_ood confidence
            thresholds[c] = 1 - halfnorm.ppf(self.prob_ood, scale=sigma)
        return thresholds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fit model using SKLearn BaseEstimator.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training label.

        Returns:
            Self: Fitted model
        """
        self.is_fitted_ = True
        # Step 1: Fit XGBoost with all classes except OOD
        self.model.fit(X[self.id_ood != y, :], y[self.id_ood != y])
        # Step 2: Estimate OOD threhsiold based on class distribution
        self.thresholds = self._estimate_thresholds(X, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict classes based on input features.

        Args:
            X (NDArray[np.float64]): Features to predict.

        Returns:
            NDArray[np.int64]: Predicted classes
        """
        # Compute probability over all classes and get argmax/max
        y_proba = self.model.predict_proba(X)
        y_label = y_proba.argmax(axis=1)
        y_label_th = y_proba.max(axis=1)
        # Replace prediction where threshold is not meet
        y_label[self.thresholds[y_label] > y_label_th] = self.id_ood
        return y_label
