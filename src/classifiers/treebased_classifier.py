from collections.abc import Callable

import pymupdf

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.models.feature_engineering import get_features
from src.models.treebased.model import TreeBasedModel
from src.models.treebased.model_explanation import explain_prediction
from src.page_classes import PageClasses
from src.page_structure import PageContext


class TreeBasedClassifier(Classifier):
    """Tree-based page classifier based on extracted text and line features.

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

    def __init__(
        self,
        matching_params: dict,
        borehole_matching_params: dict,
        model_path: str = None,
        explain_model: bool = False,
    ):
        """Initializes the Tree-based classifier with a trained model.

        Args:
            matching_params (dict): Parameters used for matching page classes.
            borehole_matching_params (dict): Parameters for borehole matching.
            model_path (str): Path to the trained Tree-based model. A valid model path is required.
                If None, it raises a ValueError.
            explain_model (bool): If True, generates plots to explain the model's choices.
        """
        self.type = ClassifierTypes.TREEBASED
        self.matching_params = matching_params
        self.borehole_matchin_params = borehole_matching_params
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = TreeBasedModel(model_path=model_path)
        self.explain_model = explain_model

    def determine_class(
        self, page: pymupdf.Page, page_number: int, context_builder: Callable[[], PageContext], **kwargs
    ) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content.

        Args:
            page (pymupdf.Page): The page to classify.
            page_number: Page number of page
            context_builder: Lazy PageContext Builder
            kwargs: Additional keyword arguments (not used).

        Returns:
            PageClasses: The predicted class of the page.
        """
        context = context_builder()
        features = get_features(
            page=page,
            page_number=page_number,
            matching_params=self.matching_params,
            borehole_matching_params=self.borehole_matchin_params,
            ctx=context,
        )

        predictions = self.model.predict([features])

        if self.explain_model:
            page_name = page.parent.name.split("/")[-1].strip(".pdf") + f"_page_{page_number}"
            explain_prediction(self.model.model, features, predictions[0], page_name, self.model.id2label)

        return self.model.id2enum[predictions[0]]
