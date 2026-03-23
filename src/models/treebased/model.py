import logging
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import halfnorm
from sklearn.mixture import GaussianMixture
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
    """XGBoost classifier with out-of-distribution (OOD) detection.

    Extends `XGBClassifier` to filter low-confidence predictions.
    Any sample whose predicted-class probability falls below that class threshold is
    reassigned to the OOD class (`PageClasses.UNKNOWN`).

    """

    def __init__(self, **kwargs):
        """Initialisation of the XGBoostOOD classifier.

        Args:
            kwargs (dict): Additional parameters.
        """
        if kwargs.get("num_class") is None:
            raise ValueError("num_class must be provided")

        if kwargs.get("mode") is None:
            raise ValueError("mode must be provided")

        self.mode = kwargs.pop("mode")
        self.id_ood = label2id[PageClasses.UNKNOWN]
        self.thresholds = np.zeros(kwargs.get("num_class") - 1, dtype=np.float64)
        super().__init__(**kwargs)

    @staticmethod
    def _estimate_from_half_normal(p: NDArray[np.float64], conf: float = 0.95) -> float:
        """Estimate the OOD threshold for a class using a half-normal distribution fit.

        Fits a half-normal distribution to the complement of the class probabilities (1 - p1),
        then returns the threshold at the given confidence quantile.

        Args:
            p (NDArray[np.float64]): In-distribution class probabilities for samples of that class.
            conf (float, optional): Confidence quantile used to set the threshold. Defaults to 0.95.

        Returns:
            float: Probability threshold below which a sample is considered out-of-distribution.
        """
        _, sigma = halfnorm.fit(1 - p, floc=0)
        return 1 - halfnorm.ppf(conf, scale=sigma)

    @staticmethod
    def _estimate_from_gmm(
        p: NDArray[np.float64], p_ood: NDArray[np.float64], conf: float = 0.5, n_estimate: int = 1000
    ) -> float:
        """Estimate the OOD threshold between two distributions using a Gaussian Mixture Model.

        Fits a 2-component GMM to the combined probability scores of in-distribution (`p`) and
        out-of-distribution (`p_ood`) samples, then finds the decision boundary where both components
        have equal probability (i.e., the crossing point near 0.5 each).

        Args:
            p (NDArray[np.float64]): In-distribution class probabilities for samples of that class.
            p_ood (NDArray[np.float64]): OOD class probabilities for the OOD samples.
            conf (float, optional): Confidence quantile used to set the threshold. Defaults to 0.5.
            n_estimate (int, optional): Number of points used to sweep [0, 1] when searching for the
                decision boundary. Defaults to 1000.

        Returns:
            float: Probability threshold that best separates in-distribution from OOD samples.
        """
        # Fit GMM to estimate distribution (assume Gaussian, even if not really)
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(np.concatenate([p, p_ood]).reshape(-1, 1))
        id_class = np.argmax(gmm.means_)

        # Find threshold: i.e., where probability is conf for p mixtures
        x_sweep = np.linspace(0, 1, num=n_estimate, endpoint=False).reshape(-1, 1)
        p_sweep = gmm.predict_proba(x_sweep)
        err_sweep = (p_sweep[:, id_class] - conf) ** 2

        return x_sweep[err_sweep.argmin()].item()

    def _estimate_thresholds(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate thresholds for each class in OOD detection.

        Args:
            X (NDArray[np.float64]): Data features.
            y (NDArray[np.float64]): Data classes.

        Returns:
            NDArray[np.float64]: Per-class probability thresholds. Predictions whose max probability
                falls below the threshold for their predicted class are reassigned to OOD.
        """
        thresholds = np.zeros_like(self.thresholds)
        for c in range(len(thresholds)):
            # Extract probability for class and OOD
            p_xc = self.predict_proba(X[y == c, :])[:, c]
            p_ood = self.predict_proba(X[y == self.id_ood, :])[:, c]

            # Estimate threshold based on selected mode
            if self.mode == "gmm":
                threshold = self._estimate_from_gmm(p=p_xc, p_ood=p_ood)
            elif self.mode == "hnorm":
                threshold = self._estimate_from_half_normal(p=p_xc)
            else:
                raise NotImplementedError(f"Unknown mode {self.mode=}")

            thresholds[c] = threshold
        return thresholds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs) -> Self:
        """Fit model.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.float64]): Training labels (integer class IDs).
            **kwargs: Additional keyword arguments passed to
                `XGBClassifier.fit()`.

        Returns:
            Self: The fitted classifier.
        """
        # Step 1: Fit XGBoost with all classes except OOD
        super().fit(X[self.id_ood != y, :], y[self.id_ood != y], **kwargs)
        # Step 2: Estimate OOD threshold based on class distribution
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
        # Replace prediction where threshold is not met
        y_label[self.thresholds[y_label] > y_label_th] = self.id_ood
        return y_label
