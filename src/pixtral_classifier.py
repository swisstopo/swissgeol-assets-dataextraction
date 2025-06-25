import boto3
from botocore.exceptions import ClientError
from .page_classes import PageClasses
import logging
import re

logger = logging.getLogger(__name__)

class PixtralPDFClassifier:
    def __init__(self, region="eu-central-1", model_id="eu.mistral.pixtral-large-2502-v1:0", fallback_classifier=None):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.fallback_classifier = fallback_classifier

    def determine_class(self, page_bytes: bytes, page_name: str = "Page", fallback_args: dict = None) -> PageClasses:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"text":
                    "You are a document classification expert.\n\n"
                    "Carefully analyze the layout, formatting, and content of the following scanned PDF page.\n\n"
                    "Classify it into **exactly one** of the following categories:\n"
                    "- title page: first or early pages that contain report titles, author names, company names, dates, logos, or file metadata — usually sparse and centered.\n"
                    "- text: pages with flowing paragraphs or body text, typically structured in columns or full-width.\n"
                    "- boreprofile: pages with borehole diagrams, longitudinal profiles, or geotechnical probes.\n"
                    "- map: geological or topographic maps, often including scale bars or coordinates.\n"
                    "- unknown: any other layout such as tables, mixed content, charts, or pages you cannot confidently classify.\n\n"
                    "**Important**: Do not classify a page as 'text' unless it clearly contains continuous paragraphs.\n\n"
                    "Return only the category name, e.g., title page or map.\n\n"
                    "If unsure, return `unknown`."},
                    {
                        "document": {
                            "format": "pdf",
                            "name": page_name,
                            "source": {"bytes": page_bytes},
                        }
                    },
                ],
            }
        ]

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=conversation,
                inferenceConfig={"maxTokens": 200, "temperature": 0.2},
            )
            string_label = response["output"]["message"]["content"][0]["text"]
            cleaned_label = clean_label(string_label)
            category = map_string_to_page_class(cleaned_label)

            if category == PageClasses.UNKNOWN and cleaned_label not in ("unknown", ""):
                logger.warning(f"Pixtral returned malformed category: '{cleaned_label}' — falling back to baseline.")
                if self.fallback_classifier and fallback_args:
                    return self.fallback_classifier.determine_class(**fallback_args)

            return category
        except (ClientError, Exception) as e:
            logger.info(f"Pixtral classification failed: {e}. Fallback to baseline classification")
            if self.fallback_classifier and fallback_args:
                return self.fallback_classifier.determine_class(**fallback_args)
            return PageClasses.UNKNOWN


def clean_label(label: str) -> str:
    """Clean LLM output string to remove formatting, quotes, punctuation."""
    label = label.strip().lower()
    label = re.sub(r"[`\"']", "", label)  # remove backticks, quotes
    label = re.sub(r"[.:\s]+$", "", label)  # remove trailing punctuation/spaces
    return label

def map_string_to_page_class(label: str) -> PageClasses:
    """Maps a string label to a PageClasses enum member."""
    label = label.strip().lower()

    match label:
        case "text":
            return PageClasses.TEXT
        case "boreprofile":
            return PageClasses.BOREPROFILE
        case "map" | "maps":
            return PageClasses.MAP
        case "title page" | "title_page":
            return PageClasses.TITLE_PAGE
        case _:
            return PageClasses.UNKNOWN