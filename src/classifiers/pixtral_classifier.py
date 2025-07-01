import logging
import re

import boto3
from botocore.exceptions import ClientError

from page_classes import PageClasses

logger = logging.getLogger(__name__)


class PixtralPDFClassifier:
    def __init__(
        self,
        region="eu-central-1",
        model_id="eu.mistral.pixtral-large-2502-v1:0",
        fallback_classifier=None,
    ):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.fallback_classifier = fallback_classifier

    def determine_class(self, page_bytes: bytes, page_name: str = "Page", fallback_args: dict = None) -> PageClasses:
        """
        Determines the class of a document page using the Pixtral model.

        Args:
            page_bytes (bytes): The image content of the page as a byte string.
            page_name (str, optional): An optional name for the page (used in logging/debugging). Defaults to "Page".
            fallback_args (dict, optional): Arguments passed to a fallback classifier in case the Pixtral output is invalid or missing.

        Returns:
            PageClasses: The predicted page class.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "You are a document layout classification expert.\n\n"
                        "Carefully examine the **layout, visual structure, formatting cues, and content** of the following scanned PDF page.\n\n"
                        "Classify the page into **exactly one** of the following categories:\n"
                        "- **title page**: Early pages showing report titles, author names, company logos, dates, or metadata. These are typically sparse, center-aligned, and minimal in text.\n"
                        "- **text**: Pages with continuous body paragraphs, often arranged in full-width or multi-column layouts. Must clearly show flowing narrative text.\n"
                        "- **boreprofile**: Pages that contain borehole logs, geotechnical cross-sections, longitudinal profiles, or probe data — often with diagrams and axes.\n"
                        "- **map**: Pages with geological or topographic maps, typically containing scale bars, coordinates, legends, or terrain features.\n"
                        "- **unknown**: Anything else — such as pages with tables, charts, mixed layouts, or unclear structure. Also use if classification is uncertain.\n\n"
                        "**Critical Notes**:\n"
                        "- Only classify a page as **text** if there are clearly visible, uninterrupted paragraphs.\n"
                        "- Pay close attention to **visual layout and spatial positioning** of elements.\n"
                        "- Prioritize accuracy over guessing — if in doubt, return `unknown`.\n\n"
                        "Return only the category name: title page, text, boreprofile, map, or unknown."
                    },
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
    """Clean pixtral output string to remove formatting, quotes, punctuation."""
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
