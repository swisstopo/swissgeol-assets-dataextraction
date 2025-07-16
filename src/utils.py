import os

import pymupdf
import yaml
from dotenv import load_dotenv

from src.text_objects import TextLine

load_dotenv()


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


def get_aws_config() -> dict:
    return {
        "region": os.environ.get("AWS_MODEL_REGION"),
        "model_id": os.environ.get("AWS_MODEL_ID"),
    }
