import logging
import os

from dotenv import load_dotenv

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.utils.utility import get_aws_config, read_params

logger = logging.getLogger(__name__)
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING").lower() == "true"

if mlflow_tracking:
    import mlflow

PIXTRAL_CONFIG_FILE_PATH = "config/pixtral_config.yml"


def create_classifier(
    classifier_type: ClassifierTypes,
    model_path: str = None,
    matching_params: dict = None,
    borehole_matching_params: dict = None,
    explain_model: bool = False,
) -> Classifier:
    """Create and return a classifier instance based on the given type.

    Args:
        classifier_type (ClassifierTypes): The type of classifier to initialize
            (e.g., TREEBASED, PIXTRAL).
        model_path: path to pretrained model if TreeBased is used.
        matching_params: Expressions used for identifying page classes in classifiers.
        borehole_matching_params: Parameters for borehole matching.
        explain_model (bool): If True, generates plots to explain the model's choices.

    Returns:
        A classifier instance matching the specified type.
    """
    if explain_model and classifier_type != ClassifierTypes.TREEBASED:
        logger.warning(f"No model explanation available for type {classifier_type}")

    if classifier_type == ClassifierTypes.TREEBASED:
        from src.classifiers.treebased_classifier import TreeBasedClassifier

        return TreeBasedClassifier(
            matching_params=matching_params,
            borehole_matching_params=borehole_matching_params,
            model_path=model_path,
            explain_model=explain_model,
        )

    elif classifier_type == ClassifierTypes.PIXTRAL:
        from src.classifiers.pixtral_classifier import PixtralClassifier
        from src.classifiers.treebased_classifier import TreeBasedClassifier

        pixtral_config = read_params(PIXTRAL_CONFIG_FILE_PATH)
        if mlflow_tracking:
            mlflow.log_params(pixtral_config)

        if not pixtral_config:
            raise ValueError("Missing pixtral in pixtral_config.yml")
        aws_config = get_aws_config()

        # Create TreeBased classifier as fallback
        # Use stable model path if not provided
        fallback_model_path = model_path if model_path else "models/stable/model.joblib"
        fallback = TreeBasedClassifier(
            matching_params=matching_params,
            borehole_matching_params=borehole_matching_params,
            model_path=fallback_model_path,
            explain_model=False,
        )

        return PixtralClassifier(config=pixtral_config, aws_config=aws_config, fallback_classifier=fallback)

    else:
        raise NotImplementedError(f"Classifier type '{classifier_type} is not supported.")
