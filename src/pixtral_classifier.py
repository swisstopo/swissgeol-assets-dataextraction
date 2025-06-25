import boto3
from botocore.exceptions import ClientError
from .page_classes import PageClasses
import logging

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
                    "Carefully analyze the layout and content of the following scanned PDF page.\n\n"
                    "Classify it into exactly one of the following categories:\n"
                    "- title page\n"
                    "- text\n"
                    "- boreprofile\n"
                    "- map\n"
                    "- unknown\n"
                    "Return **only** the category name — no explanation, no extra text.\n\n"
                    "If you're uncertain, choose 'unknown'."},
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
            string_label = response["output"]["message"]["content"][0]["text"].strip().lower()
            category = map_string_to_page_class(string_label)

            if category == PageClasses.UNKNOWN and string_label not in ("unknown", ""):
                logger.warning(f"Pixtral returned malformed category: '{string_label}' — falling back to baseline.")
                if self.fallback_classifier and fallback_args:
                    return self.fallback_classifier.determine_class(**fallback_args)

            return category
        except (ClientError, Exception) as e:
            logger.info(f"Pixtral classification failed: {e}. Fallback to baseline classification")
            if self.fallback_classifier and fallback_args:
                return self.fallback_classifier.determine_class(**fallback_args)
            return PageClasses.UNKNOWN

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