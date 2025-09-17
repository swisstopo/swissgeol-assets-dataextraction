import io
import logging
import re

from PIL import Image

from src.page_classes import ALIASES, PageClasses, label_mappings

logger = logging.getLogger(__name__)


def clean_label(label: str) -> str:
    """Cleans a raw string returned by Pixtral and standardizes formatting."""
    label = label.strip().lower()
    label = re.sub(r"[`\"']", "", label)  # remove backticks, quotes
    label = re.sub(r"[.:\s]+$", "", label)  # remove trailing punctuation/spaces
    label = re.sub(r"[*]", "", label)  # remove bold ticks
    return label


def normalize_label(label: str) -> str:
    """Normalize to canonical enum value string (e.g. 'geo profile' -> 'geo_profile')."""
    label = clean_label(label)
    return ALIASES.get(label, label)


def map_string_to_page_class(label: str) -> PageClasses:
    """Maps a string label to a PageClasses enum member."""
    norm = normalize_label(label)

    if norm in label_mappings:
        return label_mappings[norm]

    if norm != "unknown":
        logger.warning(f"Unexpected label:  {label!r} (normalized: {norm}), mapping it to unknown.")
    return PageClasses.UNKNOWN


def read_image_bytes(image_path: str, compress: bool = True) -> bytes:
    """Read image file as bytes."""
    if compress:
        return compress_image(image_path)
    else:
        with open(image_path, "rb") as f:
            return f.read()


def compress_image(image_path: str, max_size_kb: int = 500, quality: int = 85) -> bytes:
    """Compress image to reduce size while maintaining readability."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if too large (max dimension 1024px)
        max_dimension = 1024
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        # Save with compression
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()

        # Check size and reduce quality if needed
        while len(compressed_bytes) > max_size_kb * 1024 and quality > 30:
            quality -= 10
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            compressed_bytes = output.getvalue()

        return compressed_bytes
