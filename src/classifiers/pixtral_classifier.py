import logging

import boto3
import pymupdf
from botocore.exceptions import ClientError

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.classifiers.utils import clean_label, map_string_to_page_class, read_image_bytes
from src.page_classes import PageClasses
from src.page_graphics import get_page_image_bytes
from src.page_structure import PageContext
from src.utils import read_params

logger = logging.getLogger(__name__)


class PixtralClassifier(Classifier):
    """Page Classifier using Pixtral Large."""

    def __init__(
        self,
        config: dict,
        aws_config: dict,
        fallback_classifier=None,
    ):
        self.type = ClassifierTypes.PIXTRAL
        self.config = config
        self.prompts_dict = read_params(config["prompt_path"])[config["prompt_version"]]
        self.client = boto3.client("bedrock-runtime", region_name=aws_config["region"])
        self.fallback_classifier = fallback_classifier
        self.model_id = aws_config["model_id"]

        self.system_content = [{"text": self.prompts_dict["system_prompt"]}]
        borehole_bytes = read_image_bytes(config["borehole_img_path"])
        text_bytes = read_image_bytes(config["text_img_path"])
        maps_bytes = read_image_bytes(config["maps_img_path"])
        title_bytes = read_image_bytes(config["title_img_path"])
        unknown_bytes = read_image_bytes(config["unknown_img_path"])
        geo_profile_bytes = read_image_bytes(config["geo_profile_img_path"])
        table_bytes = read_image_bytes(config["table_img_path"])
        diagram_bytes = read_image_bytes(config["diagram_img_path"])
        self.examples_bytes = {
            "borehole": borehole_bytes,
            "text": text_bytes,
            "maps": maps_bytes,
            "title": title_bytes,
            "geo_profile": geo_profile_bytes,
            "diagram": diagram_bytes,
            "table": table_bytes,
            "unknown": unknown_bytes,
        }

    def determine_class(self, page: pymupdf.Page, context: PageContext, page_number: int, **kwargs) -> PageClasses:
        """Determines the class of a document page using the Pixtral model.

        Falls back to baseline classifier if output is malformed or ClientError.

        Args:
            page: The page of th document that should be classified
            context: Preprocessed page context (e.g., text blocks, lines).
            page_number: the Page number of the page that should be classified
            **kwargs: Additionally passed unused arguments

        Returns:
            PageClasses: The predicted page class.
        """
        max_doc_size = self.config["max_document_size_mb"] - self.config["slack_size_mb"]
        image_bytes = get_page_image_bytes(page, page_number, max_mb=max_doc_size)

        fallback_args = {"page": page, "context": context}

        conversation = self._build_conversation(image_bytes=image_bytes)

        try:
            response = self._send_conversation(conversation)
            raw_label = response["output"]["message"]["content"][0]["text"]

            label = clean_label(raw_label)
            category = map_string_to_page_class(label)

            if category == PageClasses.UNKNOWN and label not in ("unknown", ""):
                logger.warning("Falling back to baseline classification, due to malformed category.")
                if self.fallback_classifier and fallback_args:
                    return self.fallback_classifier.determine_class(**fallback_args)

            return category

        except ClientError as e:
            logger.info(f"Pixtral classification failed due to ClientError: {e}. Fallback to baseline classification")
            if self.fallback_classifier and fallback_args:
                return self.fallback_classifier.determine_class(**fallback_args)
            return PageClasses.UNKNOWN

        except Exception as e:
            logger.exception(f"Unexpected error during Pixtral classification: {e}")
            if self.fallback_classifier and fallback_args:
                return self.fallback_classifier.determine_class(**fallback_args)
            return PageClasses.UNKNOWN

    def _build_conversation(self, image_bytes: bytes) -> list[dict]:
        content = [
            {"image": {"format": "jpeg", "source": {"bytes": self.examples_bytes[text.strip("@")]}}}
            if text.startswith("@")  # @category encodes the image of the category and adds it to the content
            else {"text": text}
            for text in self.prompts_dict.get("examples_prompt", [])
        ]
        content.append({"text": self.prompts_dict["user_prompt"]})
        content.append({"image": {"format": "jpeg", "source": {"bytes": image_bytes}}})

        return [{"role": "user", "content": content}]

    def _send_conversation(self, conversation: list) -> dict:
        """Sends the conversation to Bedrock and returns the raw response."""
        return self.client.converse(
            modelId=self.model_id,
            messages=conversation,
            system=self.system_content,
            inferenceConfig={
                "maxTokens": self.config.get("max_tokens", 200),
                "temperature": self.config.get("temperature", 0.2),
            },
        )
