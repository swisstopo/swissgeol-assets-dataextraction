import logging
import os

from dotenv import load_dotenv

from src.classifiers.baseline_classifier import BaselineClassifier
from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.classifiers.layoutlmv3_classifier import LayoutLMv3Classifier
from src.classifiers.pixtral_classifier import PixtralClassifier
from src.classifiers.treebased_classifier import TreeBasedClassifier
from src.utils import get_aws_config, read_params

logger = logging.getLogger(__name__)
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING").lower() == "true"

if mlflow_tracking:
    import mlflow

PIXTRAL_CONFIG_FILE_PATH = "config/pixtral_config.yml"


def create_classifier(
    classifier_type: ClassifierTypes, model_path: str = None, matching_params: dict = None
) -> Classifier:
    """Create and return a classifier instance based on the given type.

    Args:
        classifier_type (ClassifierTypes): The type of classifier to initialize
            (e.g., BASELINE, PIXTRAL, LAYOUTLMV3).
        model_path: path to pretrained model if LayoutLMv3 is used.
        matching_params: Expressions used for identifying page classes in baseline classifiers.

    Returns:
        A classifier instance matching the specified type.
    """
    if classifier_type == ClassifierTypes.BASELINE:
        return BaselineClassifier(matching_params)

    elif classifier_type == ClassifierTypes.TREEBASED:
        return TreeBasedClassifier(matching_params=matching_params, model_path=model_path)

    elif classifier_type == ClassifierTypes.LAYOUTLMV3:
        return LayoutLMv3Classifier(model_path=model_path)

    elif classifier_type == ClassifierTypes.PIXTRAL:
        pixtral_config = read_params(PIXTRAL_CONFIG_FILE_PATH)
        if mlflow_tracking:
            mlflow.log_params(pixtral_config)

        if not pixtral_config:
            raise ValueError("Missing pixtral in pixtral_config.yml")
        aws_config = get_aws_config()
        fallback = BaselineClassifier(matching_params)

        return PixtralClassifier(config=pixtral_config, aws_config=aws_config, fallback_classifier=fallback)

    else:
        raise NotImplementedError(f"Classifier type '{classifier_type} is not supported.")
