import argparse
import itertools
import json
import logging
import os
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from swissgeol_doc_processing.utils.file_utils import read_params as swissgeol_read_params

from src.boreprofile.entity_parser import document_to_boreprofiles
from src.classifiers.classifier_factory import ClassifierTypes, create_classifier
from src.constants import DEFAULT_TREEBASED_MODEL_PATH
from src.page_classes import PageClasses
from src.page_structure import (
    ProcessedEntities,
    ProcessorDocument,
    ProcessorDocumentEntities,
)
from src.pdf_processor import PDFProcessor
from src.utils.utility import get_pdf_files, read_params

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
    return [[v for _, v in group] for _, group in groupby(enumerate(values), key=lambda iv: iv[1] - iv[0])]


def forward_document(
    pdf_files: list[Path],
    matching_params: dict,
    borehole_matching_params: dict,
    model_path: str | None = None,
    classifier_name: str = "treebased",
    explain_model: bool = False,
) -> list[ProcessorDocument]:
    """Infer document classes.

    Args:
        pdf_files (list[Path]): List fo documents to classify.
        matching_params (dict): Dict of parameters for document processing.
        borehole_matching_params (dict): Dict of parameters for borehole matching.
        model_path (str, optional): Path to pretrained model.
        classifier_name (str, optional): Classifier to use ("treebased", "pixtral", etc.).
        explain_model (bool): If True, generates plots to explain the model's choices.

    Returns:
        list[ProcessorDocument]: Classified documents.
    """
    # Set up classifier
    classifier_type = ClassifierTypes.infer_type(classifier_name)
    classifier = create_classifier(
        classifier_type, model_path, matching_params, borehole_matching_params, explain_model
    )
    logger.info(f"Start classifying {len(pdf_files)} PDF files with {classifier.type.value} classifier")

    # Processed PDFs
    processor = PDFProcessor(classifier)
    return processor.process_batch(pdf_files)


def forward_document_entities_group(
    classification: PageClasses,
    page_start: int,
    page_end: int,
    language: str | None,
    pdf_file: Path,
) -> list[ProcessedEntities]:
    """Extract entities from a group of consecutive pages with the same classification.

    Args:
        classification (PageClasses): The classification type of the page group.
        page_start (int): First page index in the consecutive group (1-based).
        page_end (int): Last page index in the consecutive group (1-based).
        language (str): Detected language of the page group.
        pdf_file (Path): Path to the source PDF file.

    Returns:
        list[ProcessedEntities]: Extracted entities from the page group.
    """
    if classification == PageClasses.BOREPROFILE:
        return document_to_boreprofiles(pdf_file=pdf_file, page_start=page_start, page_end=page_end, lang=language)
    else:
        return [
            ProcessedEntities(
                classification=classification,
                page_start=page_start,
                page_end=page_end,
                language=language,
            )
        ]


def forward_document_entities(
    documents: list[ProcessorDocument],
    pdf_files: list[Path],
) -> list[ProcessorDocumentEntities]:
    """Convert classified documents pages to entities.

    Args:
        documents (list[ProcessorDocument]): List of documents to convert to entities.
        pdf_files (list[Path]): List of path to documents.

    Returns:
       list[ProcessorDocumentEntities]: Processed documents entities
    """
    documents_entities: list[ProcessorDocumentEntities] = []
    for document, pdf_file in zip(documents, pdf_files, strict=True):
        # Reset list of entities for current document
        results_entities: list[ProcessedEntities] = []
        # Iterate over grouped entities types
        for (pages_type, lang), pages in document.group_pages_by_type():
            # Get pages sequences
            page_numbers = sorted([page.page for page in pages])
            entities = [
                forward_document_entities_group(
                    classification=pages_type,
                    page_start=min(pages_group),
                    page_end=max(pages_group),
                    language=lang,
                    pdf_file=pdf_file,
                )
                # Group consecutive [1,2,10] -> [1,2], [10]
                for pages_group in group_consecutive(page_numbers)
            ]
            # Flatten list of lists
            entities = list(itertools.chain.from_iterable(entities))
            # Extend entry
            results_entities.extend(entities)
        # Create document from filename, metadata, entities
        documents_entities.append(
            ProcessorDocumentEntities(
                filename=document.filename,
                page_count=document.metadata.page_count,
                languages=document.metadata.languages,
                entities=results_entities,
            )
        )

    return documents_entities


def main(
    input_path: str,
    ground_truth_path: str | None = None,
    model_path: str | None = None,
    classifier_name: str = "treebased",
    write_result: bool = False,
    explain_model: bool = False,
    return_entities: bool = False,
) -> list[ProcessorDocument] | list[ProcessorDocumentEntities]:
    """Run the page classification pipeline on input documents.

    Args:
        input_path (str): Path to directory with PDF pages or documents.
        ground_truth_path (str, optional): Path to ground truth JSON file for evaluation.
        model_path (str, optional): Path to pretrained model.
        classifier_name (str, optional): Classifier to use ("treebased", "pixtral", etc.).
        write_result (bool): If True, writes results to prediction.json.
        explain_model (bool): If True, generates plots to explain the model's choices.
        return_entities (bool): If True, return grouped entities instead of per-page results.

    Return:
        list[ProcessorDocument] | list[ProcessorDocumentEntities]::
            * A list of `ProcessorDocument` containing per-page classifications, or
            * A list of `ProcessorDocumentEntities` containing grouped (multi-page) entities
            when `return_entities=True`.

    Raises:
        ValueError: If an unsupported classifier is specified.
    """
    input_path = Path(input_path)
    ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
    matching_params = read_params("config/local_matching_params.yml")
    borehole_matching_params = swissgeol_read_params("matching_params.yml")

    # Start MLFlow tracking
    if mlflow_tracking:
        setup_mlflow(input_path, ground_truth_path, model_path, matching_params, classifier_name)

    # Process pages
    pdf_files = get_pdf_files(input_path)
    if not pdf_files:
        logger.error("No valid PDFs found.")
        return []

    # Run individual page classification
    documents_pages = forward_document(
        pdf_files=pdf_files,
        matching_params=matching_params,
        borehole_matching_params=borehole_matching_params,
        model_path=model_path,
        classifier_name=classifier_name,
        explain_model=explain_model,
    )

    # Check if data need to be saved
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

    if not return_entities:
        return documents_pages
    else:
        return forward_document_entities(documents=documents_pages, pdf_files=pdf_files)


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
        default="treebased",
        help="Specify which classifier to use for classification. Default set to treebased.",
    )

    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=False,
        default=DEFAULT_TREEBASED_MODEL_PATH,
        help="Path to pretrained model for classification.",
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
    if args.classifier.lower() == "treebased" and not args.model_path:
        parser.error(f"--model_path is required when using classifier '{args.classifier}'")

    main(
        input_path=args.input_path,
        ground_truth_path=args.ground_truth_path,
        model_path=args.model_path,
        classifier_name=args.classifier,
        write_result=args.write_results,
        explain_model=args.explain_model,
        return_entities=True,
    )
