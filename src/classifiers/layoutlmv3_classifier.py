"""Module to utilize the LayoutLMv3 model."""

import json
import logging
import os
import time
from pathlib import Path

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
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from classifiers.pdf_dataset_builder import (
    build_dataset_from_page_list,
    build_dataset_from_path_list,
)
from page_classes import PageClasses

if __name__ == "__main__":
    # Only configure logging if this script is run directly (e.g. training pipeline entrypoint)
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

    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("Model path should specify the path to a trained model.")
        self.model = LayoutLMv3(model_name_or_path=model_path)

    def _prepare_data(self, page_list: list[pymupdf.Page], batch_size=32) -> DataLoader:
        data = build_dataset_from_page_list(page_list, ground_truth_map=None)

        processed_data = data.map(self.model.preprocess, batched=True, batch_size=batch_size)
        processed_data = processed_data.remove_columns(["words", "bboxes", "image"])

        dataloader = DataLoader(processed_data, batch_size, collate_fn=default_data_collator)
        return dataloader

    def determine_class(self, page: pymupdf.Page) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content."""
        dataloader = self._prepare_data([page])

        predictions, _ = self.model.predict_batch(dataloader)

        return self.model.id2enum[predictions[0]]


class LayoutLMv3:
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

    def __init__(
        self,
        model_name_or_path="microsoft/layoutlmv3-base",
        device=None,
    ):
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

    def preprocess(self, sample):
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
        }

    def predict_batch(self, dataloader):
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
        """Freeze all layers (base bert model + classifier)."""
        for name, param in self.hf_model.named_parameters():
            logger.debug(f"Freezing Param: {name}")
            param.requires_grad = False

    def unfreeze_list(self, unfreeze_list: list[str]):
        """Unfreeze a list of layers.

        Args:
            unfreeze_list (list[str]): A list of layers to unfreeze. Possible values are:
                - "classifier"
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
            else:
                raise ValueError(f"Uknown layer to unfreeze: {layer}.")

    def unfreeze_classifier(self):
        """Unfreeze all the classifier layers.

        This will put requires_grad=True for the following parameters:
            - classifier.weight
            - classifier.bias
        """
        for name, param in self.hf_model.named_parameters():
            if name.startswith("classifier."):
                logger.debug(f"Unfreezing Param: {name}")
                param.requires_grad = True

    def unfreeze_all_layers(self):
        """Unfreeze all layers (base model + classifier)."""
        for name, param in self.hf_model.named_parameters():
            logger.debug(f"Unfreezing Param: {name}")
            param.requires_grad = True


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        # Save processor in the checkpoint folder
        self.processor.save_pretrained(args.output_dir)


class LayoutLMv3Trainer:
    def __init__(self, model_config, out_directory, model_checkpoint):
        model_path = model_config["model_path"] if model_checkpoint is None else model_checkpoint
        self.model = LayoutLMv3(model_path)
        self.model.freeze_all_layers()
        self.model.unfreeze_list(model_config["unfreeze_layers"])

        self.out_directory = out_directory
        self.training_arguments = self.setup_training_args(model_config)

        training_data_path = Path(model_config["train_folder_path"])
        val_data_path = Path(model_config["val_folder_path"])
        ground_truth_file_path = Path(model_config["ground_truth_file_path"])
        train_dataset, eval_dataset = self.load_data(training_data_path, val_data_path, ground_truth_file_path)

        self.trainer = Trainer(
            model=self.model.hf_model,
            args=self.training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.model.processor,  # ???
            # data_collator=DataCollatorWithPadding(tokenizer=self.model.processor),
            compute_metrics=self.get_compute_metrics_func(),
        )

    def setup_training_args(self, model_config: dict) -> TrainingArguments:
        """Create a TrainingArgument object from the config file.

        Args:
            model_config (dict): The dictionary containing the model configuration.

        Returns:
            TrainingArgument: the training arguments.
        """
        report_to = "mlflow" if mlflow_tracking else "none"
        # Read hyperparameters from the config file
        training_args = TrainingArguments(
            output_dir=self.out_directory,
            logging_dir=self.out_directory / "logs",
            per_device_train_batch_size=model_config["batch_size"],
            per_device_eval_batch_size=model_config["batch_size"],
            num_train_epochs=model_config["num_epochs"],
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
            save_total_limit=2,  # Limit checkpoints to save space, only keep best two
        )
        return training_args

    def load_data(
        self, training_data_path: Path, val_data_path: Path, ground_truth_file_path: Path
    ) -> tuple[Dataset, Dataset]:
        ground_truth_map = self.build_ground_truth_map(ground_truth_file_path)

        logger.info("Loading and preprocessing all files. This might take a while.")
        logger.info("Building training dataset.")
        train_data = build_dataset_from_path_list(
            list(training_data_path.iterdir()), ground_truth_map=ground_truth_map
        )
        train_data = train_data.map(self.model.preprocess, batched=True, batch_size=32).remove_columns(
            ["words", "bboxes", "image"]
        )
        logger.info("Building validation dataset.")
        val_data = build_dataset_from_path_list(list(val_data_path.iterdir()), ground_truth_map=ground_truth_map)
        val_data = val_data.map(self.model.preprocess, batched=True, batch_size=32).remove_columns(
            ["words", "bboxes", "image"]
        )
        return train_data, val_data

    def build_ground_truth_map(self, ground_truth_file_path: Path) -> dict[tuple[str, int], int]:
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

    def get_compute_metrics_func(self):
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

    def train(self):
        return self.trainer.train()

    def save_model(self):
        self.trainer.save_model()
        # self.model.processor.save_pretrained(self.out_directory)

    def log_metrics(self, split, metrics):
        self.trainer.log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        self.trainer.save_metrics(split, metrics)

    def save_state(self):
        self.trainer.save_state()


def setup_mlflow_tracking(
    model_config: dict,
    out_directory: Path,
    experiment_name: str = "LayoutLMv3 training",
):
    """Set up MLFlow tracking."""
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
        help="Path to a local folder containing an existing bert model (e.g. models/your_model_folder).",
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
    """Train a LayoutLMv3 model using the specified datasets and configurations from the YAML config file."""

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

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Training successfull.")


if __name__ == "__main__":
    train_model()
