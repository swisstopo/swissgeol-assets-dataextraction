"""Common utility functions for document processing."""

import logging
import os
import re
import unicodedata
from pathlib import Path

import pymupdf
import yaml
from dotenv import load_dotenv
from swissgeol_doc_processing.text.textline import TextLine

load_dotenv()


def is_digitally_born(page: pymupdf.Page) -> bool:
    """Check if a page is digitally born (has embedded text).

    Args:
        page: PDF page to check

    Returns:
        True if page has embedded text, False otherwise
    """
    bboxes = page.get_bboxlog()

    for boxType, rectangle in bboxes:
        if (boxType == "fill-text" or boxType == "stroke-text") and not pymupdf.Rect(rectangle).is_empty:
            return True
    return False


def is_description(line: TextLine, matching_params: dict):
    """Check if the words in line matches with matching parameters.

    Args:
        line: Text line to check
        matching_params: Dict with "including_expressions" and "excluding_expressions"

    Returns:
        True if line matches criteria, False otherwise
    """
    line_text = line.text.lower()
    return any(line_text.find(word) > -1 for word in matching_params["including_expressions"]) and not any(
        line_text.find(word) > -1 for word in matching_params["excluding_expressions"]
    )


def read_params(params_name: str) -> dict:
    """Read parameters from YAML file.

    Args:
        params_name: Path to YAML config file

    Returns:
        Dictionary with config parameters
    """
    with open(params_name) as f:
        return yaml.safe_load(f)


def get_aws_config() -> dict:
    """Get AWS configuration from environment variables.

    Returns:
        Dictionary with AWS region and model_id
    """
    return {
        "region": os.environ.get("AWS_MODEL_REGION"),
        "model_id": os.environ.get("AWS_MODEL_ID"),
    }


def get_pdf_files(input_path: Path) -> list[Path]:
    """Returns a list of PDF files from a directory or a single file.

    Args:
        input_path: Path to PDF file or directory containing PDFs

    Returns:
        List of PDF file paths
    """
    if input_path.is_dir():
        return [f for f in input_path.rglob("*.pdf")]
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    logging.error("Invalid input path: must be a PDF file or a directory containing PDFs.")
    return []


def standardize_text(text: str) -> str:
    """Standardize text by removing new lines, double spaces and uppercaps.

    Args:
        text (str): Text to standardize.

    Returns:
        str: Standardized text.
    """
    # Remove new lines
    text = text.replace("\n", " ")
    # Remove double spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accents "ü" -> "u"
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    # Enforce lowercases
    return text.lower()
