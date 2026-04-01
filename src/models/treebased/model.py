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
        # Update number of classes to correct number
        self.ood_mode = kwargs.pop("ood_mode")
        self.ood_confidence = kwargs.pop("ood_confidence")
        self.id_ood = label2id[PageClasses.UNKNOWN]
        self.thresholds = np.zeros(kwargs["num_class"] - 1, dtype=np.float64)
        super().__init__(**kwargs)

    def get_xgb_params(self) -> dict:
        """Return XGBoost-native parameters, excluding OOD-specific keys.

        Overrides the parent method which does not recognise them and would raise a warning.

        Returns:
            dict: XGBoost parameters without OOD keys.
        """
        params = super().get_xgb_params()
        params.pop("ood_mode", None)
        params.pop("ood_confidence", None)
        return params

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for cross validation search.

        Args:
            deep (bool, optional): Deep copy. Defaults to True.

        Returns:
            dict: Updated parameters
        """
        params = super().get_params(deep=deep)
        params["ood_mode"] = self.ood_mode
        params["ood_confidence"] = self.ood_confidence
        return params

    def set_params(self, **params) -> Self:
        """Set parameters for cross validation search.

        Returns:
             Self: The updated classifier instance.
        """
        if "ood_mode" in params:
            self.ood_mode = params.pop("ood_mode")
        if "ood_confidence" in params:
            self.ood_confidence = params.pop("ood_confidence")
        return super().set_params(**params)

    @staticmethod
    def _estimate_from_half_normal(p: NDArray[np.float64], confidence: float = 0.95) -> float:
        """Estimate the OOD threshold for a class using a half-normal distribution fit.

        Fits a half-normal distribution to the complement of the class probabilities (1 - p1),
        then returns the threshold at the given confidence quantile.

        Args:
            p (NDArray[np.float64]): In-distribution class probabilities for samples of that class.
            confidence (float, optional): Confidence quantile used to set the threshold. Defaults to 0.95.

        Returns:
            float: Probability threshold below which a sample is considered out-of-distribution.
        """
        _, sigma = halfnorm.fit(1 - p, floc=0)
        return 1 - halfnorm.ppf(confidence, scale=sigma)

    @staticmethod
    def _estimate_from_gmm(
        p: NDArray[np.float64],
        p_ood: NDArray[np.float64],
        confidence: float = 0.05,
        n_estimate: int = 1000,
        random_state: int = 42,
    ) -> float:
        """Estimate the OOD threshold between two distributions using a Gaussian Mixture Model.

        Fits a 2-component GMM to the combined probability scores of in-distribution (`p`) and
        out-of-distribution (`p_ood`) samples, then finds the decision boundary where both components
        have equal probability (i.e., the crossing point near 0.5 each).

        Args:
            p (NDArray[np.float64]): In-distribution class probabilities for samples of that class.
            p_ood (NDArray[np.float64]): OOD class probabilities for the OOD samples.
            confidence (float, optional): Confidence quantile used to set the threshold. Defaults to 0.05.
            n_estimate (int, optional): Number of points used to sweep [0, 1] when searching for the
                decision boundary. Defaults to 1000.
            random_state (int, optional): Random state for reproducibility.

        Returns:
            float: Probability threshold that best separates in-distribution from OOD samples.
        """
        # Fit GMM to estimate distribution (assume Gaussian, even if not really)
        gmm = GaussianMixture(n_components=2, random_state=random_state)
        gmm.fit(np.concatenate([p, p_ood]).reshape(-1, 1))
        id_class = np.argmax(gmm.means_)

        # Find threshold: i.e., where probability is conf for p mixtures
        x_sweep = np.linspace(0, 1, num=n_estimate, endpoint=False).reshape(-1, 1)
        p_sweep = gmm.predict_proba(x_sweep)
        err_sweep = (p_sweep[:, id_class] - confidence) ** 2

        return x_sweep[err_sweep.argmin()].item()

    def _estimate_thresholds(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate thresholds for each class in OOD detection.

        Args:
            X (NDArray[np.float64]): Data features.
            y (NDArray[np.int64]): Data classes.

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
            if self.ood_mode == "gmm":
                threshold = self._estimate_from_gmm(p=p_xc, p_ood=p_ood, confidence=self.ood_confidence)
            elif self.ood_mode == "hnorm":
                threshold = self._estimate_from_half_normal(p=p_xc, confidence=self.ood_confidence)
            else:
                raise NotImplementedError(f"Unknown mode {self.ood_mode=}")

            thresholds[c] = threshold
        return thresholds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64], **kwargs) -> Self:
        """Fit model.

        Args:
            X (NDArray[np.float64]): Training features.
            y (NDArray[np.int64]): Training labels (integer class IDs).
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
        """Predict class labels with OOD reassignment.

        Args:
            X (NDArray[np.float64]): Input features.
            **kwargs: Additional keyword arguments passed to
                `XGBClassifier.predict_proba()`.

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
