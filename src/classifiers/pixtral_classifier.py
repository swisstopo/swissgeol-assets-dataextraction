import logging
import random
import threading
import time
from collections.abc import Callable

import boto3
import pymupdf
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from src.classifiers.classifier_types import Classifier, ClassifierTypes
from src.classifiers.utils import clean_label, map_string_to_page_class, read_image_bytes
from src.page_classes import PageClasses
from src.page_graphics import get_page_image_bytes
from src.page_structure import PageContext
from src.utils.utility import read_params

logger = logging.getLogger(__name__)


class PixtralImageSource(BaseModel):
    """Raw bytes payload for an image."""

    bytes_: bytes = Field(alias="bytes")


class PixtralImage(BaseModel):
    """Image content block containing its format and raw bytes source."""

    format_: str = Field(alias="format")
    source: PixtralImageSource


class PixtralMessage(BaseModel):
    """A single content block in a Pixtral conversation, either text or image."""

    text: str | None = None
    image: PixtralImage | None = None


class PixtralMessageStack(BaseModel):
    """A full conversation turn with a role (e.g. 'user') and a list of content blocks."""

    role: str
    content: list[PixtralMessage]


class PixtralResponseOutput(BaseModel):
    """The output field of response, wrapping the assistant message."""

    message: PixtralMessageStack


class PixtralResponse(BaseModel):
    """Top-level response, containing the model output."""

    output: PixtralResponseOutput


class RateLimiter:
    """Simple token bucket QPS limiter."""

    def __init__(self, qps: float):
        """Initialise the rate limiter with a target queries-per-second rate.

        Args:
            qps (float): Maximum number of requests allowed per second.
        """
        self.qps = max(0.1, qps)
        self.lock = threading.Lock()
        self.tokens = 0.0
        self.last = time.monotonic()

    def acquire(self):
        """Block until a token is available, then consume it."""
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


class PixtralConnector:
    """Low-level client for the Pixtral model.

    Handles authentication, rate limiting, and retries with exponential
    back-off and full jitter when API throttles requests.
    """

    def __init__(
        self,
        config: dict,
        aws_config: dict,
    ):
        """Initialise client and rate-limiting settings.

        Args:
            config (dict): Pixtral configuration dict.
            aws_config (dict): AWS settings dict.
        """
        self.config = config
        self.client = boto3.client("bedrock-runtime", region_name=aws_config["region"])
        self.model_id = aws_config["model_id"]
        self._stats = {"throttles": 0, "retries": 0}
        self.qps = config.get("qps", 2.0)
        self.max_retries = config.get("max_retries", 6)
        self.backoff_base = config.get("backoff_base", 0.4)
        self.backoff_cap = config.get("backoff_cap", 8.0)
        self._rl = RateLimiter(self.qps)
        self.max_doc_size = self.config["max_document_size_mb"] - self.config["slack_size_mb"]

    def _send_conversation(self, message: PixtralMessageStack, system: PixtralMessage) -> PixtralResponse:
        """Send a single-turn conversation to the Pixtral model.

        Args:
            message (PixtralMessageStack): The user message stack to send.
            system (PixtralMessage): The system prompt message.

        Returns:
            PixtralResponse: The validated model response.
        """
        attempt = 0
        while True:
            self._rl.acquire()  # ensure we don't exceed QPS
            try:
                answer = self.client.converse(
                    modelId=self.model_id,
                    messages=[message.model_dump(by_alias=True, exclude_none=True)],
                    system=[system.model_dump(by_alias=True, exclude_none=True)],
                    inferenceConfig={
                        "maxTokens": self.config.get("max_tokens", 5),
                        "temperature": self.config.get("temperature", 0.2),
                    },
                )
                return PixtralResponse.model_validate(answer)
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


class PixtralClassifier(PixtralConnector, Classifier):
    """Page classifier that uses the Pixtral vision model."""

    def __init__(
        self,
        config: dict,
        aws_config: dict,
        fallback_classifier: Callable = None,
    ):
        """Initialise the classifier, loading prompts and example images.

        Args:
            config (dict): Pixtral configuration dict.
            aws_config (dict): AWS settings dict.
            fallback_classifier (Callable): Optional classifier to use when Pixtral
                returns an unrecognised label or errors out.
        """
        # Create connection to remote model
        PixtralConnector.__init__(self, config=config, aws_config=aws_config)

        self.type = ClassifierTypes.PIXTRAL
        self.prompts_dict = read_params(config["prompt_path"])[config["prompt_version"]]
        self.fallback_classifier = fallback_classifier
        self.system_content = PixtralMessage(text=self.prompts_dict["system_prompt"])
        self.examples_bytes = {
            "borehole": read_image_bytes(config["borehole_img_path"]),
            "text": read_image_bytes(config["text_img_path"]),
            "map": read_image_bytes(config["map_img_path"]),
            "title": read_image_bytes(config["title_img_path"]),
            "geo_profile": read_image_bytes(config["geo_profile_img_path"]),
            "diagram": read_image_bytes(config["diagram_img_path"]),
            "table": read_image_bytes(config["table_img_path"]),
        }

    def determine_class(
        self, page: pymupdf.Page, page_number: int, context_builder: Callable[[], PageContext] = None, **kwargs
    ) -> PageClasses:
        """Determines the class of a document page using the Pixtral model.

        Falls back to treebased classifier if output is malformed or ClientError.

        Args:
            page: The page of th document that should be classified
            context_builder: Builds page context (e.g., text blocks, lines) for fallback classifier.
            page_number: the Page number of the page that should be classified
            **kwargs: Additionally passed unused arguments

        Returns:
            PageClasses: The predicted page class.
        """
        image_bytes = get_page_image_bytes(page, max_mb=self.max_doc_size)
        message = self._build_conversation(image_bytes=image_bytes)

        try:
            response = self._send_conversation(message=message, system=self.system_content)
            raw_label = response.output.message.content[0].text

            label = clean_label(raw_label)
            category = map_string_to_page_class(label)
            if category == PageClasses.UNKNOWN and label not in ("unknown", ""):
                logger.warning("Falling back to treebased classifier, due to malformed category.")
                if self.fallback_classifier:
                    return self.fallback_classifier.determine_class(
                        page=page, page_number=page_number, context_builder=context_builder
                    )

            return category

        except ClientError as e:
            logger.info(f"Pixtral classification failed due to ClientError: {e}. Fallback to treebased classifier")
            if self.fallback_classifier:
                return self.fallback_classifier.determine_class(
                    page=page, page_number=page_number, context_builder=context_builder
                )
            return PageClasses.UNKNOWN

        except Exception as e:
            logger.exception(f"Unexpected error during Pixtral classification: {e}")
            if self.fallback_classifier:
                return self.fallback_classifier.determine_class(
                    page=page, page_number=page_number, context_builder=context_builder
                )
            return PageClasses.UNKNOWN

    def _build_conversation(self, image_bytes: bytes) -> PixtralMessageStack:
        """Build the user message containing few-shot examples and the target image.

        Args:
            image_bytes: Encoded bytes of the page to classify.

        Returns:
            PixtralMessageStack: A user turn ready to send.
        """
        # List of examples for pixtral model
        content_examples = [
            PixtralMessage(
                image=PixtralImage(
                    format="jpeg",
                    source=PixtralImageSource(bytes=self.examples_bytes[text.strip("@")]),
                )
            )
            if text.startswith("@")
            else PixtralMessage(text=text)
            for text in self.prompts_dict.get("examples_prompt", [])
        ]

        # User prompt with content to classify
        content_user_text = PixtralMessage(text=self.prompts_dict["user_prompt"])
        content_user_img = PixtralMessage(
            image=PixtralImage(
                format="jpeg",
                source=PixtralImageSource(bytes=image_bytes),
            ),
        )

        return PixtralMessageStack(role="user", content=content_examples + [content_user_text, content_user_img])


class PixtralFeatureExtraction(PixtralConnector):
    """Uses the Pixtral vision model to extract features from PDF pages."""

    def __init__(self, config: dict, aws_config: dict, system_prompt: str):
        """Initialise the extractor with a custom system prompt.

        Args:
            config (dict): Pixtral configuration dict.
            aws_config (dict): AWS settings dict.
            system_prompt (str): Instruction text sent as the system message for
                every extraction request.
        """
        # Create connection to remote model
        PixtralConnector.__init__(self, config=config, aws_config=aws_config)
        self.system_prompt = PixtralMessage(text=system_prompt)

    def _build_conversation(self, image_bytes: bytes) -> PixtralMessageStack:
        """Build a minimal user message containing only the target page image.

        Args:
            image_bytes (bytes): Encoded bytes of the page to process.

        Returns:
            PixtralMessageStack: A 'user' turn with a single image content block.
        """
        return PixtralMessageStack(
            role="user",
            content=[
                PixtralMessage(
                    image=PixtralImage(
                        format="jpeg",
                        source=PixtralImageSource(bytes=image_bytes),
                    )
                )
            ],
        )

    def find(self, page: pymupdf.Page) -> str:
        """Extract a feature from a single PDF page using the Pixtral model.

        Args:
            page (pymupdf.Page): The PyMuPDF page object to process.

        Returns:
            str: The raw text returned by the model (e.g. an extracted title).
        """
        # User prompt with content to classify
        image_bytes = get_page_image_bytes(page, max_mb=self.max_doc_size)
        content_user = self._build_conversation(image_bytes=image_bytes)

        response = self._send_conversation(message=content_user, system=self.system_prompt)
        return response.output.message.content[0].text
