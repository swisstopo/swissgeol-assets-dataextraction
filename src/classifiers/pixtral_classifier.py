import logging
import boto3

from botocore.exceptions import ClientError

from src.classifiers.utils import clean_label, map_string_to_page_class
from src.page_classes import PageClasses
from src.classifiers.config_loader import load_prompt

logger = logging.getLogger(__name__)

class PixtralPDFClassifier:
    def __init__(
            self,
            config: dict,
            aws_config: dict,
            fallback_classifier=None,
    ):
        self.config = config
        self.prompt_text = load_prompt(config["prompt_path"])
        self.client = boto3.client("bedrock-runtime", region_name=aws_config["region"])
        self.fallback_classifier = fallback_classifier
        self.model_id = aws_config["model_id"]

    def determine_class(self,
                        image_bytes: bytes,
                        fallback_args: dict = None) -> PageClasses:
        """
        Determines the class of a document page using the Pixtral model.
        Falls back to baseline classifier if output is malformed or ClientError.
        Args:
            image_bytes (bytes): The image content of the page as a byte string.
            fallback_args (dict, optional): Arguments passed to a fallback classifier in case the Pixtral output is invalid or missing.

        Returns:
            PageClasses: The predicted page class.
        """
        conversation = self._build_conversation(image_bytes=image_bytes)

        try:
            response = self._send_conversation(conversation)
            raw_label = response["output"]["message"]["content"][0]["text"]
            label = clean_label(raw_label)
            category = map_string_to_page_class(label)

            if category == PageClasses.UNKNOWN and label not in ("unknown", ""):
                logger.warning(f"Pixtral returned malformed category: '{label}' —  Fallback to baseline classification.")

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


    def _build_conversation(self, image_bytes: bytes):
        """Constructs a conversation input for Pixtral."""
        content = {"text": self.prompt_text}

        image = {
            "image": {
                "format": "jpeg",
                "source": {"bytes": image_bytes}
            }
        }
        content_block = [content, image]
        return [{"role": "user", "content": content_block}]

    def _send_conversation(self, conversation: list) -> dict:
        """Sends the conversation to Bedrock and returns the raw response."""
        return self.client.converse(
            modelId=self.model_id,
            messages=conversation,
            inferenceConfig={
                "maxTokens": self.config.get("max_tokens", 200),
                "temperature": self.config.get("temperature", 0.2),
            },
        )