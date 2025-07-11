import logging
import math
from dataclasses import dataclass
from PIL import Image
import io
import pymupdf
logger = logging.getLogger(__name__)


@dataclass
class ImageRect:
    rect: pymupdf.Rect  # Transformed rectangle respecting page rotation
    rotation: int  # Page rotation in degrees (0, 90, 180, 270)
    xref: int  # Image xref ID (unique reference in PDF)

    def page_coverage(self, page_rect: pymupdf.Rect) -> int:
        """Computes how much of the text page area is covered by this image."""
        return self.rect.get_area() / page_rect.get_area()


def extract_page_graphics(page: pymupdf.Page, is_digital: bool):
    """Extract drawings and image bounding boxes from page"""
    if not is_digital:
        return [], []

    drawings = page.get_drawings()
    image_rects = get_images_from_page(page)

    return drawings, image_rects


def get_images_from_page(page: pymupdf.Page) -> list[ImageRect]:
    """Extracts all image bounding boxes from the page, transformed by page rotation."""
    page_rotation = page.rotation
    rotation_matrix = page.rotation_matrix

    extracted_images = []
    for image_info in page.get_images():
        xref = image_info[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            rotated_rect = rect * rotation_matrix if page.rotation else rect
            extracted_images.append(ImageRect(rect=rotated_rect, rotation=page_rotation, xref=xref))

    return extracted_images


def get_page_image_bytes(page: pymupdf.Page, page_number: int, max_mb: float = 4.5) -> bytes:
    """
    Returns JPEG image bytes of a single PDF page. Downscales if image exceeds allowed size.
    """
    max_bytes = int(max_mb * 1024 * 1024)
    scale = 1.0

    for attempt in range(10):
        # Render and convert to JPEG
        with pymupdf.open() as pdf_doc:
            pdf_doc.insert_pdf(page.parent, from_page=page_number, to_page=page_number)
            page_bytes = pdf_doc.tobytes(deflate=True, garbage=3, use_objstms=1)

        image_bytes = convert_pdf_to_jpeg(page_bytes, scale=scale)
        current_size = len(image_bytes)
        size_mb = current_size / 1024 / 1024

        if current_size <= max_bytes:
            return image_bytes

        logger.info(f"[image-bytes] Attempt {attempt + 1}: scale={scale:.2f}, size={size_mb:.2f} MB — too large")

        # Update scale
        ratio = max_bytes / current_size
        proposed = scale * math.sqrt(ratio)
        new_scale = (scale + proposed) / 2

        if abs(scale - new_scale) < 0.01:
            new_scale *= 0.95

        scale = max(min(new_scale, scale), 0.2)

    logger.warning(f"[image-bytes] Final size {size_mb:.2f} MB after 10 attempts.")
    return image_bytes

def convert_pdf_to_jpeg(page_bytes: bytes, scale: float = 1.0) -> bytes:
    """
    Converts a PDF page (as bytes) to JPEG image bytes using PyMuPDF and PIL.
    """
    with pymupdf.open(stream=page_bytes, filetype="pdf") as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)

    logger.info(f"[convert] PDF rendered to image: {pix.width}x{pix.height} at scale={scale:.2f}")

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return stream.getvalue()