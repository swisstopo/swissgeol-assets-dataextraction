import logging
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

from src.text_objects import TextLine


def is_digitally_born(page: pymupdf.Page) -> bool:
    bboxes = page.get_bboxlog()

    for boxType, rectangle in bboxes:
        if (boxType == "fill-text" or boxType == "stroke-text") and not pymupdf.Rect(rectangle).is_empty:
            return True
    return False


def is_description(line: TextLine, matching_params: dict):
    """Check if the words in line matches with matching parameters."""
    line_text = line.line_text().lower()
    return any(line_text.find(word) > -1 for word in matching_params["including_expressions"]) and not any(
        line_text.find(word) > -1 for word in matching_params["excluding_expressions"]
    )

def read_params(params_name: str) -> dict:
    with open(params_name) as f:
        return yaml.safe_load(f)

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        return f.read()

def get_aws_config() -> dict:
    return {
        "region": os.environ.get("AWS_MODEL_REGION"),
        "model_id": os.environ.get("AWS_MODEL_ID"),
    }


def get_pdf_files(input_path: Path) -> list[Path]:
    """Returns a list of PDF files from a directory or a single file."""
    if input_path.is_dir():
        return [f for f in input_path.rglob("*.pdf")]
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    logging.error("Invalid input path: must be a PDF file or a directory containing PDFs.")
    return []