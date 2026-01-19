import argparse
import json
import logging
import os
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv

from src.classifiers.classifier_factory import ClassifierTypes, create_classifier
from src.page_structure import (
    ProcessedEntities,
    ProcessedEntitiesMetadata,
    ProcessorDocument,
    ProcessorDocumentEntities,
)
from src.pdf_processor import PDFProcessor
from src.utils import get_pdf_files, read_params

# Load .env and check MLFlow
load_dotenv()
mlflow_tracking = os.getenv("MLFLOW_TRACKING").lower() == "true"

if mlflow_tracking:
    import mlflow
    import pygit2

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow(
    input_path: Path, ground_truth_path: Path, model_path: str, matching_params: dict, classifier_name: str
):
    mlflow.set_experiment("PDF Page Classification")
    mlflow.start_run()

    mlflow.set_tag("input_path", str(input_path))

    if ground_truth_path:
        mlflow.set_tag("ground_truth_path", str(ground_truth_path))
    if model_path:
        mlflow.set_tag("model_path", str(model_path))
    if classifier_name:
        mlflow.set_tag("classifier_name", str(classifier_name))

    mlflow.log_params(flatten_dict(matching_params))

    try:
        repo = pygit2.Repository(".")
        commit = repo[repo.head.target]
        mlflow.set_tag("git_branch", repo.head.shorthand)
        mlflow.set_tag("git_commit", str(commit.id))
        mlflow.set_tag("git_message", commit.message.strip())
    except Exception as e:
        logger.warning(f"Could not attach Git metadata to MLflow: {e}")


def flatten_dict(d, parent_key="", sep=".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def group_consecutive(values: list[int]) -> list[list[int]]:
    """Group sorted integers into consecutive sequences.

    Args:
        values: Sorted list of integers.

    Returns:
        A list of lists, where each sublist contains consecutive integers.
    """
    return [
        list(group)
        for _, group in groupby(
            values,
            key=lambda x, c=iter(range(len(values))): x - next(c),
        )
    ]


def forward_document(
    pdf_files: list[Path],
    matching_params: dict,
    model_path: str | None = None,
    classifier_name: str = "baseline",
    explain_model: bool = False,
) -> list[ProcessorDocument]:
    """Infer document classes.

    Args:
        pdf_files (list[Path]): List fo documents to classify.
        matching_params (dict): Dict of parameters for document processing.
        model_path (str, optional): Path to pretrained LayoutLMv3 model.
        classifier_name (str, optional): Classifier to use ("baseline", "pixtral", etc.).
        explain_model (bool): If True, generates plots to explain the model's choices.

    Returns:
        list[ProcessorDocument]: Classified documents.
    """
    # Set up classifier
    classifier_type = ClassifierTypes.infer_type(classifier_name)
    classifier = create_classifier(classifier_type, model_path, matching_params, explain_model)
    logger.info(f"Start classifying {len(pdf_files)} PDF files with {classifier.type.value} classifier")

    # Processed PDFs
    processor = PDFProcessor(classifier)
    return processor.process_batch(pdf_files)


def forward_document_entities(
    documents: list[ProcessorDocument],
) -> list[ProcessorDocumentEntities]:
    """Convert classified docuemnts pages to entities.

    Args:
        documents (list[ProcessorDocument]): List of documents to process.

    Returns:
       list[ProcessorDocumentEntities]: Processed documents entities
    """
    documents_entities: list[ProcessorDocumentEntities] = []
    for document in documents:
        # Reset list of entities for current document
        results_entities: list[ProcessedEntities] = []
        # Iterate over grouped entities types
        for (pages_type, lang), pages in document.group_pages_by_type():
            # Get pages sequences
            results_entities.extend(
                [
                    ProcessedEntities(
                        metadata=ProcessedEntitiesMetadata(
                            page_start=min(pages_group), page_end=max(pages_group), language=lang
                        ),
                        classification=pages_type,
                        data=None,
                    )
                    # Group consecutive [1,2,10] -> [1,2], [10]
                    for pages_group in group_consecutive([page.page for page in pages])
                ]
            )
        # Create document from filename, metadata, entities
        documents_entities.append(
            ProcessorDocumentEntities(
                filename=document.filename, metadata=document.metadata, entities=results_entities
            )
        )
    return documents_entities


def forward(
    input_path: str,
    matching_params: dict,
    model_path: str | None = None,
    classifier_name: str = "baseline",
    explain_model: bool = False,
) -> tuple[list[ProcessorDocument], list[ProcessorDocumentEntities]]:
    """Infer documents structures.

    Args:
        input_path (str): Path to directory with PDF pages or documents.
        matching_params (dict): Dict of parameters for document processing.
        model_path (str, optional): Path to pretrained LayoutLMv3 model.
        classifier_name (str, optional): Classifier to use ("baseline", "pixtral", etc.).
        explain_model (bool): If True, generates plots to explain the model's choices.

    Returns:
        tuple[list[ProcessorDocument], list[ProcessorDocumentEntities]]: Result of processed entities
            * List of documents with per page classification
            * List of documents with content parsed as (multi-)page entities.
    """
    # Load files
    pdf_files = get_pdf_files(input_path)
    if not pdf_files:
        logger.error("No valid PDFs found.")
        return [], []

    # Run individual page classification
    documents_pages = forward_document(
        pdf_files=pdf_files,
        matching_params=matching_params,
        model_path=model_path,
        classifier_name=classifier_name,
        explain_model=explain_model,
    )

    # Extract pages entities
    documents_entities = forward_document_entities(documents=documents_pages)
    return documents_pages, documents_entities


def main(
    input_path: str,
    ground_truth_path: str | None = None,
    model_path: str | None = None,
    classifier_name: str = "baseline",
    write_result: bool = False,
    explain_model: bool = False,
) -> tuple[list[ProcessorDocument], list[ProcessorDocumentEntities]]:
    """Run the page classification pipeline on input documents.

    Args:
        input_path (str): Path to directory with PDF pages or documents.
        ground_truth_path (str, optional): Path to ground truth JSON file for evaluation.
        model_path (str, optional): Path to pretrained LayoutLMv3 model.
        classifier_name (str, optional): Classifier to use ("baseline", "pixtral", etc.).
        write_result (bool): If True, writes results to prediction.json.
        explain_model (bool): If True, generates plots to explain the model's choices.

    Return:
        tuple[list[ProcessorDocument], list[ProcessorDocumentEntities]]: Result of processed entities
            * List of documents with per page classification
            * List of documents with content parsed as (multi-)page entities.

    Raises:
        ValueError: If an unsupported classifier is specified.
    """
    input_path = Path(input_path)
    ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
    matching_params = read_params("config/matching_params.yml")

    # Start MLFlow tracking
    if mlflow_tracking:
        setup_mlflow(input_path, ground_truth_path, model_path, matching_params, classifier_name)

    # Process pages
    documents_pages, documents_entities = forward(
        input_path=input_path,
        matching_params=matching_params,
        model_path=model_path,
        classifier_name=classifier_name,
        explain_model=explain_model,
    )

    # Check if data need to be saves
    if write_result:
        output_file = Path("data") / "prediction.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps([r.model_dump() for r in documents_pages], indent=4),
            encoding="utf-8",
        )

    # Check if GT need to be computed
    if ground_truth_path:
        from src.evaluation import evaluate_results

        evaluate_results(
            [result.model_dump(context={"legacy": True}) for result in documents_pages], ground_truth_path
        )

    # End mlflow tracking
    if mlflow_tracking:
        mlflow.end_run()

    return documents_pages, documents_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF page classification")

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input directory containing PDF files.",
    )

    parser.add_argument(
        "-g",
        "--ground_truth_path",
        type=str,
        required=False,
        help="(Optional) Path to the ground truth JSON file for evaluation.",
    )

    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        required=False,
        default="baseline",
        help="Specify which classifier to use for classification. Default set to baseline.",
    )

    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=False,
        help="Path to pretrained LayoutLMv3 or Tree Based model for classification.",
    )
    parser.add_argument(
        "-w",
        "--write-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Writes classification results to prediction.json file.",
    )
    parser.add_argument(
        "-x",
        "--explain-model",
        action="store_true",
        help="Generates explainability plots for the model's decisions.",
    )
    args = parser.parse_args()

    # Check if model_path is required based on classifier
    if args.classifier.lower() in ["layoutlmv3", "treebased"] and not args.model_path:
        parser.error(f"--model_path is required when using classifier '{args.classifier}'")

    main(
        input_path=args.input_path,
        ground_truth_path=args.ground_truth_path,
        model_path=args.model_path,
        classifier_name=args.classifier,
        write_result=args.write_results,
        explain_model=args.explain_model,
    )
