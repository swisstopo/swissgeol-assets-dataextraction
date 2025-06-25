import boto3
from botocore.exceptions import ClientError
from .page_classes import PageClasses

class PixtralPDFClassifier:
    def __init__(self, region="eu-central-1", model_id="eu.mistral.pixtral-large-2502-v1:0"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def determine_class(self, page_bytes: bytes, page_name: str = "Page") -> PageClasses:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"text": "Classify this page as one of: title page, table, boreprofile, map, text, other. Return only the category name."},
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
            return map_string_to_page_class(string_label)
        except (ClientError, Exception) as e:
            print(f"Pixtral classification failed: {e}")
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