import logging
import random
import threading
import time
from collections.abc import Callable

import boto3
import pymupdf
from botocore.exceptions import ClientError

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.classifiers.utils import clean_label, map_string_to_page_class, read_image_bytes
from src.page_classes import PageClasses
from src.page_graphics import get_page_image_bytes
from src.utils import read_params

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket QPS limiter."""

    def __init__(self, qps: float):
        self.qps = max(0.1, qps)
        self.lock = threading.Lock()
        self.tokens = 0.0
        self.last = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens += (now - self.last) * self.qps
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.01)


def is_throttle_error(e) -> bool:
    try:
        code = e.response["Error"]["Code"]
        if code in {
            "ThrottlingException",
            "ProvisionedThroughputExceededException",
        }:
            return True
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status in (429, 500)
    except Exception:
        return False


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
        self.examples_bytes = {
            "borehole": read_image_bytes(config["borehole_img_path"]),
            "text": read_image_bytes(config["text_img_path"]),
            "map": read_image_bytes(config["map_img_path"]),
            "title": read_image_bytes(config["title_img_path"]),
            "geo_profile": read_image_bytes(config["geo_profile_img_path"]),
            "diagram": read_image_bytes(config["diagram_img_path"]),
            "table": read_image_bytes(config["table_img_path"]),
        }
        self._stats = {"throttles": 0, "retries": 0}
        self.qps = config.get("qps", 2.0)
        self.max_retries = config.get("max_retries", 6)
        self.backoff_base = config.get("backoff_base", 0.4)
        self.backoff_cap = config.get("backoff_cap", 8.0)
        self._rl = RateLimiter(self.qps)

    def determine_class(
        self, page: pymupdf.Page, page_number: int, context_builder: Callable = None, **kwargs
    ) -> PageClasses:
        """Determines the class of a document page using the Pixtral model.

        Falls back to baseline classifier if output is malformed or ClientError.

        Args:
            page: The page of th document that should be classified
            context_builder: Builds page context (e.g., text blocks, lines) for fallback classifier.
            page_number: the Page number of the page that should be classified
            **kwargs: Additionally passed unused arguments

        Returns:
            PageClasses: The predicted page class.
        """
        max_doc_size = self.config["max_document_size_mb"] - self.config["slack_size_mb"]
        image_bytes = get_page_image_bytes(page, page_number, max_mb=max_doc_size)

        conversation = self._build_conversation(image_bytes=image_bytes)

        try:
            response = self._send_conversation(conversation)
            raw_label = response["output"]["message"]["content"][0]["text"]

            label = clean_label(raw_label)
            category = map_string_to_page_class(label)
            if category == PageClasses.UNKNOWN and label not in ("unknown", ""):
                logger.warning("Falling back to baseline classification, due to malformed category.")
                if self.fallback_classifier:
                    return self.fallback_classifier.determine_class(page=page, context_builder=context_builder)

            return category

        except ClientError as e:
            logger.info(f"Pixtral classification failed due to ClientError: {e}. Fallback to baseline classification")
            if self.fallback_classifier:
                return self.fallback_classifier.determine_class(page=page, context_builder=context_builder)
            return PageClasses.UNKNOWN

        except Exception as e:
            logger.exception(f"Unexpected error during Pixtral classification: {e}")
            if self.fallback_classifier:
                return self.fallback_classifier.determine_class(page=page, context_builder=context_builder)
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
        """Sends the conversation to Bedrock with retry-on-throttle."""
        attempt = 0
        while True:
            self._rl.acquire()  # enusre we dont exceed QPS
            try:
                return self.client.converse(
                    modelId=self.model_id,
                    messages=conversation,
                    system=self.system_content,
                    inferenceConfig={
                        "maxTokens": self.config.get("max_tokens", 5),
                        "temperature": self.config.get("temperature", 0.0),
                    },
                )
            except ClientError as e:
                # Retry on throttling
                if is_throttle_error(e) and attempt < self.max_retries:
                    delay = min(self.backoff_cap, self.backoff_base * (2**attempt))
                    # full jitter
                    delay *= random.uniform(0.5, 1.5)
                    logger.warning(f"Bedrock throttled (attempt {attempt + 1}/{self.max_retries}); sleep {delay:.2f}s")
                    time.sleep(delay)
                    attempt += 1

                    self._stats["retries"] += 1
                    if "Throttl" in str(e):
                        self._stats["throttles"] += 1
                    continue
                raise  # not retryable or out of retries
            except Exception:
                # Non-ClientError; retry a couple of times
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    attempt += 1
                    continue
                raise
