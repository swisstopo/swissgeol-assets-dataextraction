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
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import TrainOutput

from classifiers.pdf_dataset_builder import (
    build_lazy_dataset, build_filename_to_label_map,
)
from src.models.layoutlmv3.model import LayoutLMv3

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
        self.model = LayoutLMv3(model_path, device="cpu")

        # Freeze all layers and unfreeze only the specified layers to fine tune
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

        # The total number of training steps is needed to setup the scheduler (because generator datasets are used and
        # the number of steps is not known in advance).
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
            max_steps=total_steps,
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
        ground_truth_map = build_filename_to_label_map(ground_truth_file_path)

        training_data_path = Path(model_config["train_folder_path"])
        train_files = [p for p in training_data_path.iterdir() if p.name.lower().endswith(".pdf")]
        num_pages = self.count_pdf_pages(train_files)
        train_dataset = build_lazy_dataset(train_files, self.model.preprocess, ground_truth_map)

        val_data_path = Path(model_config["val_folder_path"])
        val_files = [p for p in val_data_path.iterdir() if p.name.lower().endswith(".pdf")]
        val_dataset = build_lazy_dataset(val_files, self.model.preprocess, ground_truth_map)
        return train_dataset, val_dataset, num_pages

    def count_pdf_pages(self, pdf_files: list[Path]) -> int:
        """Count the total number of pages in a list of PDF files.

        Args:
            pdf_files (list[Path]): List of Path objects representing PDF files.
        Returns:
            int: The total number of pages across all PDF files.
        """
        total_pages = 0
        for pdf in pdf_files:
            if not pdf.name.lower().endswith(".pdf"):
                continue
            with pymupdf.open(pdf) as doc:
                total_pages += len(doc)
        return total_pages

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
    logger.info("Training successful.")


if __name__ == "__main__":
    train_model()
