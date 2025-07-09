from enum import Enum


class ClassifierTypes(Enum):
    """Enum for all available classifier types."""

    BASELINE = "baseline"
    PIXTRAL = "pixtral"

    @classmethod
    def infer_type(cls, classifier_str: str) -> "ClassifierTypes":
        for classifier in cls:
            if classifier.value == classifier_str.lower():
                return classifier
        raise ValueError(
            f"Invalid classifier type: {classifier_str}. Choose from {[c.value for c in cls]}"
        )
