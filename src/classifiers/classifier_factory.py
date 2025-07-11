import logging

from classifiers.randomforest_classifier import RandomForestClassifier
from src.classifiers.classifier_types import ClassifierTypes, Classifier
from src.classifiers.baseline_classifier import BaselineClassifier
from src.classifiers.pixtral_classifier import PixtralClassifier
from src.classifiers.layoutlmv3_classifier import LayoutLMv3Classifier
from src.utils import read_params, get_aws_config

logger = logging.getLogger(__name__)

PIXTRAL_CONFIG_FILE_PATH = "config/pixtral_config.yml"


def create_classifier(classifier_type: ClassifierTypes,
                      model_path: str= None,
                      matching_params: dict= None) -> Classifier:
    """
        Create and return a classifier instance based on the given type.

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

    elif classifier_type == ClassifierTypes.RANDOMFOREST:
        return RandomForestClassifier(matching_params= matching_params,
                                      model_path=model_path)

    elif classifier_type == ClassifierTypes.LAYOUTLMV3:
        return LayoutLMv3Classifier(model_path= model_path)

    elif classifier_type == ClassifierTypes.PIXTRAL:
        pixtral_config = read_params(PIXTRAL_CONFIG_FILE_PATH)

        if not pixtral_config:
            raise ValueError("Missing pixtral in pixtral_config.yml")
        aws_config = get_aws_config()
        fallback = BaselineClassifier(matching_params)

        return PixtralClassifier(
            config=pixtral_config,
            aws_config=aws_config,
            fallback_classifier=fallback)

    else:
        raise NotImplementedError(f"Classifier type '{classifier_type} is not supported.")