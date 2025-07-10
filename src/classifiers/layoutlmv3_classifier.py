"""Module to utilize the LayoutLMv3 model."""
import pymupdf
from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.classifiers.classifier_types import ClassifierTypes, Classifier
from src.classifiers.pdf_dataset_builder import build_dataset_from_page_list
from src.models.model import LayoutLMv3
from src.page_classes import PageClasses

class LayoutLMv3Classifier(Classifier):
    """
    Transformer-based page classifier using LayoutLMv3.
    """

    def __init__(self, model_path: str = None):
        """Initializes the LayoutLMv3PageClassifier with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained LayoutLMv3 model. A valid model path is required.
                If None, it raises a ValueError.
        """
        self.type = ClassifierTypes.LAYOUTLMV3
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = LayoutLMv3(model_name_or_path=model_path, device="cpu")

    def _prepare_data(self, page_list: list[pymupdf.Page], batch_size: int = 32) -> DataLoader:
        """Prepares the data for the LayoutLMv3 model.

        Args:
            page_list (list[pymupdf.Page]): List of pymupdf Page objects to be classified.
            batch_size (int): Batch size for the DataLoader.

        Returns:
            DataLoader: A DataLoader containing the processed data ready for classification.
        """
        data = build_dataset_from_page_list(page_list, ground_truth_map=None)

        processed_data = data.map(self.model.preprocess, remove_columns=["words", "bboxes", "image"])

        dataloader = DataLoader(processed_data, batch_size, collate_fn=default_data_collator)
        return dataloader

    def determine_class(self, page: pymupdf.Page) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content.

        Args:
            page (pymupdf.Page): The page to classify.
        Returns:
            PageClasses: The predicted class of the page.
        """
        dataloader = self._prepare_data([page])

        predictions, _ = self.model.predict_batch(dataloader)

        return self.model.id2enum[predictions[0]]
