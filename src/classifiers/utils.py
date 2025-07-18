import io
import logging
import re

import PIL

from src.page_classes import PageClasses

logger = logging.getLogger(__name__)


def clean_label(label: str) -> str:
    """
    Cleans a raw string returned by Pixtral and standardizes formatting.
    """
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
        case "boreprofile" | "borehole" | "boreholes":
            return PageClasses.BOREPROFILE
        case "map" | "maps":
            return PageClasses.MAP
        case "title page" | "title_page" | "title":
            return PageClasses.TITLE_PAGE
        case _:
            if label != "unknown":
                logger.warning(f"Unknown label: {label}, mapping it to unknown.")
            return PageClasses.UNKNOWN


def read_image_bytes(image_path: str, compress: bool = True) -> bytes:
    """Read image file as bytes"""
    if compress:
        return compress_image(image_path)
    else:
        with open(image_path, "rb") as f:
            return f.read()


def compress_image(image_path: str, max_size_kb: int = 500, quality: int = 85) -> bytes:
    """
    Compress image to reduce size while maintaining readability
    """
    with PIL.Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if too large (max dimension 1024px)
        max_dimension = 1024
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), PIL.Image.Resampling.LANCZOS)

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
