from src.classifiers.classifier_type import ClassifierTypes
from src.classifiers.baseline_classifier import BaselineClassifier
from src.classifiers.pixtral_classifier import PixtralClassifier
from src.utils import read_params, get_aws_config

CONFIG_FILE_PATH = "config.yml"

def create_classifier(classifier_type: ClassifierTypes):
    if classifier_type == ClassifierTypes.BASELINE:
        return BaselineClassifier()

    elif classifier_type == ClassifierTypes.PIXTRAL:
        full_config = read_params(CONFIG_FILE_PATH)
        pixtral_config = full_config.get("pixtral")
        if not pixtral_config:
            raise ValueError("Missing pixtral in config.yml")
        aws_config = get_aws_config()
        fallback = BaselineClassifier()

        return PixtralClassifier(
            config=pixtral_config,
            aws_config= aws_config,
            fallback_classifier=fallback)

    else:
        raise NotImplementedError(f"Classifier type '{classifier_type} is not supported.")