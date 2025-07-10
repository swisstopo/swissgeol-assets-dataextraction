from enum import Enum
from abc import ABC, abstractmethod
from src.page_classes import PageClasses

class ClassifierTypes(Enum):
    """Enum for all available classifier types."""

    BASELINE = "baseline"
    PIXTRAL = "pixtral"
    LAYOUTLMV3 = "layoutlmv3"

    @classmethod
    def infer_type(cls, classifier_str: str) -> "ClassifierTypes":
        for classifier in cls:
            if classifier.value == classifier_str.lower():
                return classifier
        raise ValueError(
            f"Invalid classifier type: {classifier_str}. Choose from {[c.value for c in cls]}"
        )


class Classifier(ABC):
    """
    Abstract base class for all page classifiers.
    All classifiers must define a `type` and a `determine_class` method.
    """

    type: ClassifierTypes

    @abstractmethod
    def determine_class(self, **kwargs) -> PageClasses:
        """
        Determine the class of a page.

        Keyword Args:
            page: Page object.
            context: Preprocessed page context (e.g., text blocks, lines).
            matching_params (dict): Optional params for baseline classifier.

        Returns:
            PageClasses: Predicted class for the page.
        """
        pass