import json
import logging
from pathlib import Path

import click
import pymupdf
from pydantic import TypeAdapter

from src.classifiers.pixtral_classifier import PixtralFeatureExtraction
from src.schemas import DocumentGroundTruth
from src.utils.utility import get_aws_config, read_params

logger = logging.getLogger(__name__)


PIXTRAL_CONFIG = read_params("config/pixtral_config.yml")
AWS_CONFIG = get_aws_config()


def update_ground_truth(
    ground_truth: DocumentGroundTruth, document: Path, pixtral_interface: PixtralFeatureExtraction
) -> DocumentGroundTruth:
    """Runs Pixtral feature extraction on each page of a document and updates the ground truth in-place.

    Args:
        ground_truth (DocumentGroundTruth): Ground truth object whose pages will be updated.
        document (Path): Path to the PDF file to process.
        pixtral_interface (PixtralFeatureExtraction): Configured Pixtral extractor used to find
            the features on each page.

    Returns:
        DocumentGroundTruth: The same `ground_truth` object with fields populated for every page.
    """
    # Open document and iterate over gt pages
    with pymupdf.Document(document) as doc:
        for ground_truth_page in ground_truth.pages:
            # Load page
            page = doc.load_page(ground_truth_page.page - 1)
            # Extract feature (title)
            ground_truth_page.title = pixtral_interface.find(page=page)

    return ground_truth


@click.command(help="Path to documents to extract title from.")
@click.option(
    "-i",
    "--input-directory",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing input *.pdf files.",
)
@click.option(
    "-p",
    "--prompt",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prompt file.",
)
@click.option(
    "-v",
    "--prompt-version",
    type=str,
    required=True,
    help="System prompt version to use.",
)
@click.option(
    "-g",
    "--ground-truth",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ground truth JSON file.",
)
def extract_feature(input_directory: Path, prompt: Path, prompt_version: str, ground_truth: Path) -> None:
    """CLI command to extract a feature (e.g. title) from every page of every PDF in a directory.

    Iterates recursively over all *.pdf files in `input_directory`.

    Args:
        input_directory (Path): Directory to search recursively for *.pdf files.
        prompt (Path): Path to a YAML prompt file containing versioned prompt dicts.
        prompt_version (str): Key within the prompt YAML selecting the system prompt.
        ground_truth (Path): Path to ground truth file.
    """
    # Read input files recursively and check if any
    paths = [p for p in input_directory.rglob("*.pdf")]
    if not paths:
        logger.error(f"No PDF files found in {input_directory}")
        return

    prompt_dict = read_params(prompt).get(prompt_version, None)

    if not prompt_dict:
        logger.error(f"Prompt version not found: {prompt_version}")
        return

    # Create pixtral interface and apply to all documents
    pixtral_interface = PixtralFeatureExtraction(
        config=PIXTRAL_CONFIG,
        aws_config=AWS_CONFIG,
        system_prompt=prompt_dict["system_prompt"],
    )

    # Read ground truth and parse
    with open(ground_truth) as f:
        gt_list = json.load(f)
        gt_list = TypeAdapter(list[DocumentGroundTruth]).validate_python(gt_list)

    # Update GT if needed
    gt_list_new = []
    for gt in gt_list:
        # Look for file in path
        matched_files = list(filter(lambda x: x.name == gt.filename, paths))
        if matched_files:
            gt = update_ground_truth(gt, document=matched_files[0], pixtral_interface=pixtral_interface)

        else:
            gt_updated.append()
        # Compute new features

    # Write updated items
    with open(ground_truth, "w", encoding="utf-8") as f:
        json.dump([gt.model_dump(exclude_none=True) for gt in gt_list], f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    extract_feature()
