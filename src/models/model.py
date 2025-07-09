from pathlib import Path
import logging
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
)
from src.page_classes import LABEL2ID, ID2LABEL, ENUM2ID, ID2ENUM, NUM_LABELS

logger = logging.getLogger(__name__)

class LayoutLMv3:
    """LayoutLMv3 model for page classification.

    This class wraps the LayoutLMv3ForSequenceClassification model and provides methods for preprocessing,
    prediction, and training. It supports freezing and unfreezing layers for fine-tuning.
    """

    label2id = LABEL2ID
    id2label = ID2LABEL
    enum2id = ENUM2ID
    id2enum = ID2ENUM
    num_labels = NUM_LABELS

    def __init__(self, model_name_or_path: str = "microsoft/layoutlmv3-base", device: str = None):
        """Initializes the LayoutLMv3 model.
        Args:
            model_name_or_path (str): Path to a fine-tuned LayoutLMv3 model checkpoint or a Hugging Face model name.
                If a local path is provided, it should point to a directory containing the model files.
            device (str): Device to run the model on, e.g., "cuda" or "cpu". If None, it defaults to "cuda" if available,
                otherwise "cpu".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Only use `apply_ocr=False` if using a base model name (not a saved checkpoint)
        if Path(model_name_or_path).exists():
            self.processor = LayoutLMv3Processor.from_pretrained(model_name_or_path)
        else:
            self.processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)

        self.hf_model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=self.num_labels
        )  # total number of params: 125'921'413 (503.7 MB)
        self.hf_model.to(self.device).eval()

    def preprocess(self, sample: dict) -> dict:
        """Preprocess a single sample for LayoutLMv3.

        Args:
            sample (dict): A dictionary containing the following keys:
                - "words": List of words in the page.
                - "bboxes": List of bounding boxes corresponding to the words.
                - "image": The image of the page as a PIL Image or numpy array.
                - "label": The label for the sample (optional).
        Returns:
            dict: A dictionary containing the processed inputs for the model, including:
                - "input_ids": Token IDs for the words.
                - "attention_mask": Attention mask for the input tokens.
                - "bbox": Bounding boxes for the words.
                - "pixel_values": Pixel values of the image.
                - "label": The label for the sample, if available.
        """
        encoding = self.processor(
            text=sample["words"],
            boxes=sample["bboxes"],
            images=sample["image"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
            "bbox": encoding.bbox[0],
            "pixel_values": encoding.pixel_values[0],
            "label": sample["label"] if "label" in sample else None,
        }

    def predict_batch(self, dataloader: DataLoader) -> tuple[list[int], list[list[float]]]:
        """Predicts classes for a batch of samples using the LayoutLMv3 model.

        Args:
            dataloader (DataLoader): A DataLoader containing the preprocessed samples.
        Returns:
            tuple: A tuple containing:
                - all_preds: List of predicted class IDs for each sample.
                - all_probs: List of predicted probabilities for each class for each sample.
        """
        all_preds = []
        all_probs = []

        self.hf_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.hf_model(**batch)
                logits = outputs.logits

                predicted_class = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)

                all_preds.extend(predicted_class.cpu().tolist())
                all_probs.extend(probabilities.cpu().tolist())

        return all_preds, all_probs

    def freeze_all_layers(self):
        """Freeze all layers of the model."""
        for name, param in self.hf_model.named_parameters():
            logger.debug(f"Freezing Param: {name}")
            param.requires_grad = False

    def unfreeze_list(self, unfreeze_list: list[str]):
        """Unfreeze a list of layers.

        Args:
            unfreeze_list (list[str]): A list of layers to unfreeze. Possible values are:
                - "classifier"
                - "rel_pos_encoder"
                - "layer_11"
                - "all" (unfreezes all layers)
        """
        if not unfreeze_list:
            logger.warning("No layer to unfreeze, the model will not be trained.")
        if "all" in unfreeze_list:
            logger.warning("Warning: Unfreezing all layers may consume excessive RAM and raise an error.")
            self.unfreeze_all_layers()
            return
        for layer in unfreeze_list:
            if layer == "classifier":
                self.unfreeze_classifier()
            elif layer == "rel_pos_encoder":
                self.unfreeze_rel_pos_encoder()
            elif layer == "layer_11":
                self.unfreeze_layer_11()
            else:
                raise ValueError(f"Unknown layer to unfreeze: {layer}.")

    def unfreeze_classifier(self):
        """Unfreeze all the classifier layers.

        This will put requires_grad=True for the following parameters:
            - classifier.weight
            - classifier.bias
            - classifier.out_proj.weight
            - classifier.out_proj.bias
        """
        for name, param in self.hf_model.named_parameters():
            if name.startswith("classifier."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_rel_pos_encoder(self):
        """Unfreeze all the classifier layers.

        This will put requires_grad=True for the following parameters:
            - layoutlmv3.encoder.rel_pos_bias.weight
            - layoutlmv3.encoder.rel_pos_x_bias.weight
            - layoutlmv3.encoder.rel_pos_y_bias.weight
        """
        for name, param in self.hf_model.named_parameters():
            if name.startswith("layoutlmv3.encoder.rel_pos_"):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_layer_11(self):
        """Unfreeze the last layer of the transformer encoder, the 11th layer.

        The 11th layer is a basic self-attention bloc and it has all the following parameters:
            - layoutlmv3.encoder.layer.11.attention.self.query.weight
            - layoutlmv3.encoder.layer.11.attention.self.query.bias
            - layoutlmv3.encoder.layer.11.attention.self.key.weight
            - layoutlmv3.encoder.layer.11.attention.self.key.bias
            - layoutlmv3.encoder.layer.11.attention.self.value.weight
            - layoutlmv3.encoder.layer.11.attention.self.value.bias
            - layoutlmv3.encoder.layer.11.attention.output.dense.weight
            - layoutlmv3.encoder.layer.11.attention.output.dense.bias
            - layoutlmv3.encoder.layer.11.attention.output.LayerNorm.weight
            - layoutlmv3.encoder.layer.11.attention.output.LayerNorm.bias
            - layoutlmv3.encoder.layer.11.intermediate.dense.weight
            - layoutlmv3.encoder.layer.11.intermediate.dense.bias
            - layoutlmv3.encoder.layer.11.output.dense.weight
            - layoutlmv3.encoder.layer.11.output.dense.bias
            - layoutlmv3.encoder.layer.11.output.LayerNorm.weight
            - layoutlmv3.encoder.layer.11.output.LayerNorm.bias
        """
        for name, param in self.hf_model.named_parameters():
            if name.startswith("layoutlmv3.encoder.layer.11."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_all_layers(self):
        """Unfreeze all layers (base model + classifier)."""
        for name, param in self.hf_model.named_parameters():
            logger.debug(f"Unfreezing Param: {name}")
            param.requires_grad = True
