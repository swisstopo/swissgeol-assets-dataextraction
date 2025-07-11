import pymupdf

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.models.randomforest.model import RandomForest
from src.page_classes import PageClasses
from src.models.feature_engineering import get_features_from_page


class RandomForestClassifier(Classifier):
    """
     Random Forest page classifier
    """

    def __init__(self,matching_params: dict, model_path: str = None):
        """Initializes the Random Forest classifier with a trained model.

        Args:
            model_path (str): Path to the trained Random Forest model. A valid model path is required.
                If None, it raises a ValueError.
        """
        self.type = ClassifierTypes.RANDOMFOREST
        self.matching_params = matching_params
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = RandomForest(model_path=model_path)


    def determine_class(self, page: pymupdf.Page, page_number:int, **kwargs) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content.

        Args:
            page (pymupdf.Page): The page to classify.
            page_number: Page number of page
        Returns:
            PageClasses: The predicted class of the page.
        """

        features = get_features_from_page(page= page, page_number= page_number,matching_params= self.matching_params)

        predictions = self.model.predict([features])

        return self.model.id2enum[predictions[0]]