import pymupdf

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.models.treebased.model import TreeBasedModel
from src.page_classes import PageClasses
from src.models.feature_engineering import get_features_from_page
from src.page_structure import PageContext


class TreeBasedClassifier(Classifier):
    """
    Tree-based page classifier based on extracted text and line features.
    This classifier uses a Tree-based model to classify pages into predefined classes.
    It requires a trained model to be specified at initialization.
    Attributes:
        type (ClassifierTypes): The type of classifier, set to TREEBASED.
        matching_params (dict): Parameters used for matching page classes.
        model (TreeBasedModel): The Tree-based model used for classification.
    Args:
        matching_params (dict): Parameters used for matching page classes.
        model_path (str): Path to the trained Tree-based model. A valid model path is required.
            If None, it raises a ValueError.

    """

    def __init__(self, matching_params: dict, model_path: str = None):
        """Initializes the Tree-based classifier with a trained model.

        Args:
            model_path (str): Path to the trained Tree-based model. A valid model path is required.
                If None, it raises a ValueError.
        """
        self.type = ClassifierTypes.TREEBASED
        self.matching_params = matching_params
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = TreeBasedModel(model_path=model_path)


    def determine_class(self, page: pymupdf.Page, context: PageContext, page_number: int, **kwargs) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content.

        Args:
            page (pymupdf.Page): The page to classify.
            page_number: Page number of page
            context: PageContext
        Returns:
            PageClasses: The predicted class of the page.
        """

        features = get_features_from_page(page= page, ctx=context ,matching_params= self.matching_params)

        predictions = self.model.predict([features])

        return self.model.id2enum[predictions[0]]