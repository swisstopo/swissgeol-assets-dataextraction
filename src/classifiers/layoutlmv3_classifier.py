"""Module to utilize the LayoutLMv3 model."""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable

import click
import mlflow
import pymupdf
import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    EvalPrediction,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import TrainOutput

from classifiers.pdf_dataset_builder import (
    build_dataset_from_page_list,
    build_lazy_dataset,
)
from page_classes import PageClasses

if __name__ == "__main__":
    # Only configure logging if this script is run directly (e.g. training pipeline entrypoint)
    import os

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # remove warning

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )


logger = logging.getLogger(__name__)

load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING") == "True"


class LayoutLMv3PageClassifier:
    """
    Transformer-based page classifier using LayoutLMv3.
    """

    def __init__(self, model_path: str = None):
        """Initializes the LayoutLMv3PageClassifier with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained LayoutLMv3 model. A valid model path is required.
                If None, it raises a ValueError.
        """
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = LayoutLMv3(model_name_or_path=model_path)

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


class LayoutLMv3:
    """LayoutLMv3 model for page classification.

    This class wraps the LayoutLMv3ForSequenceClassification model and provides methods for preprocessing,
    prediction, and training. It supports freezing and unfreezing layers for fine-tuning.
    """

    label2id = {"Boreprofile": 0, "Maps": 1, "Text": 2, "Title_Page": 3, "Unknown": 4}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    enum2id = {
        PageClasses.BOREPROFILE: 0,
        PageClasses.MAP: 1,
        PageClasses.TEXT: 2,
        PageClasses.TITLE_PAGE: 3,
        PageClasses.UNKNOWN: 4,
    }
    id2enum = {v: k for k, v in enum2id.items()}

    def __init__(self, model_name_or_path="microsoft/layoutlmv3-base", device=None):
        """Initializes the LayoutLMv3 model.
        Args:
            model_name_or_path (str): Path to a finetuned LayoutLMv3 model checkpoint or a Hugging Face model name.
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
                # Move data to device if needed
                # batch = {k: v.to(self.device) for k, v in batch.items()}
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
                raise ValueError(f"Uknown layer to unfreeze: {layer}.")

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


class LayoutLMv3Trainer:
    """Trainer class for LayoutLMv3 model.

    This class handles the training and evaluation of the LayoutLMv3 model using the Hugging Face Trainer API.
    It initializes the model, loads the training and validation datasets, sets up the training arguments,
    and provides methods for training, saving the model, logging metrics, and saving the training state.
    """

    def __init__(self, model_config: dict, out_directory: Path, model_checkpoint: Path):
        """Initializes the LayoutLMv3Trainer with the model configuration and output directory.

        Args:
            model_config (dict): The configuration dictionary containing model parameters and paths.
            out_directory (Path): The directory where the trained model and logs will be saved.
            model_checkpoint (Path | None): Optional path to a pre-trained model checkpoint. If None, the model will
                be initialized from the hugging face librairy using the config file.
        """
        model_path = model_config["model_path"] if model_checkpoint is None else model_checkpoint
        self.model = LayoutLMv3(model_path)

        # Freeze all layers by default and unfreeze only the specified layers to fine-tune
        self.model.freeze_all_layers()
        self.model.unfreeze_list(model_config["unfreeze_layers"])

        self.out_directory = out_directory

        train_dataset, eval_dataset, tot_num_pages = self.load_data(model_config)

        self.training_arguments = self.setup_training_args(model_config, tot_num_pages)

        self.trainer = Trainer(
            model=self.model.hf_model,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.model.processor,
            compute_metrics=self.get_compute_metrics_func(),
        )

    def setup_training_args(self, model_config: dict, tot_num_pages: int) -> TrainingArguments:
        """Create a TrainingArgument object from the config file.

        Args:
            model_config (dict): The dictionary containing the model configuration.
            tot_num_pages (int): The total number of pages in the training dataset.

        Returns:
            TrainingArgument: the training arguments.
        """
        report_to = "mlflow" if mlflow_tracking else "none"

        # The total number of training steps is needed to setup the scheduler (because generator datasets are used)
        # This wrongly displays the current epoch in the logs, but it is not a big problem.
        train_steps = math.ceil(tot_num_pages / (model_config["batch_size"]))  # for one epoch
        total_steps = train_steps * model_config["num_epochs"]

        # Read hyperparameters from the config file
        training_args = TrainingArguments(
            output_dir=self.out_directory,
            logging_dir=self.out_directory / "logs",
            per_device_train_batch_size=model_config["batch_size"],
            per_device_eval_batch_size=model_config["batch_size"],
            num_train_epochs=model_config["num_epochs"],
            max_steps=total_steps,  #
            weight_decay=float(model_config["weight_decay"]),
            learning_rate=float(model_config["learning_rate"]),
            lr_scheduler_type=model_config["lr_scheduler_type"],
            warmup_ratio=float(model_config["warmup_ratio"]),
            max_grad_norm=float(model_config["max_grad_norm"]),
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to=report_to,
            # logging_first_step=True,
            save_total_limit=2,  # Limit checkpoints to save space, only keep best two
            dataloader_pin_memory=not torch.backends.mps.is_available(),  # Fix MPS pin_memory warning
        )

        return training_args

    def load_data(self, model_config: dict) -> tuple[Dataset, Dataset, int]:
        """Load training and validation datasets from the specified paths in the model configuration.

        Args:
            model_config (dict): The configuration dictionary containing paths to training and validation datasets,
                as well as the ground truth file.
        Returns:
            tuple[Dataset, Dataset, int]: A tuple containing:
                - train_dataset: The training dataset as a Dataset object.
                - val_dataset: The validation dataset as a Dataset object.
                - num_pages: The total number of pages in the training dataset.
        """
        ground_truth_file_path = Path(model_config["ground_truth_file_path"])
        ground_truth_map = self.build_ground_truth_map(ground_truth_file_path)

        training_data_path = Path(model_config["train_folder_path"])
        train_files = [p for p in training_data_path.iterdir() if p.name.lower().endswith(".pdf")]
        num_pages = self.count_pdf_pages(train_files)
        train_dataset = build_lazy_dataset(train_files, self.model.preprocess, ground_truth_map)

        val_data_path = Path(model_config["val_folder_path"])
        val_files = [p for p in val_data_path.iterdir() if p.name.lower().endswith(".pdf")]
        val_dataset = build_lazy_dataset(val_files, self.model.preprocess, ground_truth_map)
        return train_dataset, val_dataset, num_pages

    def build_ground_truth_map(self, ground_truth_file_path: Path) -> dict[tuple[str, int], int]:
        """Build a ground truth map from the JSON file.

        Args:
            ground_truth_file_path (Path): Path to the ground truth JSON file.
        Returns:
            dict[tuple[str, int], int]: A dictionary mapping (filename, page number) to label IDs.
                The keys are tuples of (filename, page number) and the values are label IDs inferred from the
                    ground truth data.
        """
        with open(ground_truth_file_path, "r") as f:
            gt_data = json.load(f)

        label_lookup = {}
        for entry in gt_data:
            filename = entry["filename"]
            for classification in entry["classification"]:  # I think all data curently only have one classification
                page = classification["Page"]
                for label_name, value in classification.items():
                    if label_name != "Page" and value == 1:
                        label_id = self.model.label2id[label_name]
                        label_lookup[(filename, page)] = label_id
        return label_lookup

    def count_pdf_pages(self, pdf_files: list[Path]) -> int:
        """Count the total number of pages in a list of PDF files.

        Args:
            pdf_files (list[Path]): List of Path objects representing PDF files.
        Returns:
            int: The total number of pages across all PDF files.
        """
        return sum(len(pymupdf.open(pdf)) for pdf in pdf_files)

    def get_compute_metrics_func(self) -> Callable[[EvalPrediction], dict[str, float]]:
        """Returns a function to compute metrics for the LayoutLMv3 model.

        This function computes micro-averaged precision, recall, and F1 score based on the model's predictions
        on the evaluation set and the true labels. It is only used for evaluation and will not be used during training.

        Returns:
            Callable: A function that takes an EvalPrediction object and returns a dictionary of metrics.
        """

        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)

            tp, fp, fn = 0, 0, 0
            classes = set(predictions) | set(labels.tolist())

            for cls in classes:
                tp += sum((pred == cls and lab == cls) for pred, lab in zip(predictions, labels))
                fp += sum((pred == cls and lab != cls) for pred, lab in zip(predictions, labels))
                fn += sum((pred != cls and lab == cls) for pred, lab in zip(predictions, labels))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                "micro_precision": round(precision, 4),
                "micro_recall": round(recall, 4),
                "micro_f1": round(f1, 4),
            }

        return compute_metrics

    def train(self) -> TrainOutput:
        """Train the LayoutLMv3 model using the Hugging Face Trainer.

        Returns:
            TrainOutput: The result of the training, including metrics and other information.
        """
        return self.trainer.train()

    def save_model(self):
        """Save the trained model and processor to the output directory."""
        self.trainer.save_model()

    def log_metrics(self, split, metrics):
        """Log metrics to the trainer's logging system.

        Args:
            split (str): The split of the dataset (e.g., "train", "eval").
            metrics (dict): A dictionary containing the metrics to log.
        """
        self.trainer.log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        """Save metrics to the trainer's output directory.

        Args:
            split (str): The split of the dataset (e.g., "train", "eval").
            metrics (dict): A dictionary containing the metrics to save.
        """
        self.trainer.save_metrics(split, metrics)

    def save_state(self):
        """Save the trainer's state to the output directory."""
        self.trainer.save_state()


def setup_mlflow_tracking(
    model_config: dict,
    out_directory: Path,
    experiment_name: str = "LayoutLMv3 training",
):
    """Set up MLFlow tracking.

    Args:
        model_config (dict): The configuration dictionary containing model parameters and paths.
        out_directory (Path): The directory where the trained model and logs will be saved.
        experiment_name (str): The name of the MLFlow experiment.
    """
    if mlflow.active_run():
        mlflow.end_run()  # Ensure the previous run is closed
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.set_tag("Training data directory", model_config.get("train_folder_path"))
    mlflow.set_tag("Eval data directory", model_config.get("val_folder_path"))
    mlflow.set_tag("Ground truth file", model_config.get("ground_truth_file_path"))
    mlflow.set_tag("out_directory", str(out_directory))
    mlflow.log_params(model_config)


def common_options(f):
    """Decorator to add common options to commands."""
    f = click.option(
        "-cf",
        "--config-file-path",
        required=True,
        type=str,
        help="Name (not path) of the configuration yml file inside the `config` folder.",
    )(f)
    f = click.option(
        "-c",
        "--model-checkpoint",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Path to a local folder containing an existing layoutlmv3 model (e.g. models/your_model_folder).",
    )(f)
    f = click.option(
        "-o",
        "--out-directory",
        type=click.Path(path_type=Path),
        default="models",
        help="Path to the output directory.",
    )(f)
    return f


@click.command()
@common_options
def train_model(
    config_file_path: Path,
    model_checkpoint: Path,
    out_directory: Path,
):
    """Train a LayoutLMv3 model using the specified datasets and configurations from the YAML config file.

    Args:
        config_file_path (Path): Path to the YAML configuration file containing model parameters and paths.
        model_checkpoint (Path): Optional path to a pre-trained model checkpoint. If None, the model will be initialized
            from the Hugging Face library using the config file.
        out_directory (Path): Path to the output directory where the trained model and logs will be saved.
    """

    with open(config_file_path) as f:
        model_config = yaml.safe_load(f)

    model_out_directory = out_directory / time.strftime("%Y%m%d-%H%M%S")

    if mlflow_tracking:
        logger.info("Logging to MLflow.")
        setup_mlflow_tracking(model_config, model_out_directory)

    # Initialize the trainer
    trainer = LayoutLMv3Trainer(model_config, model_out_directory, model_checkpoint)

    # Start training
    logger.info("Beginning the training.")
    train_result = trainer.train()

    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Training successfull.")


if __name__ == "__main__":
    train_model()
