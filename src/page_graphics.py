import io
import logging
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pymupdf
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageRect:
    """Represents an image rectangle on a PDF page."""

    rect: pymupdf.Rect  # Transformed rectangle respecting page rotation
    rotation: int  # Page rotation in degrees (0, 90, 180, 270)
    xref: int  # Image xref ID (unique reference in PDF)

    def page_coverage(self, page_rect: pymupdf.Rect) -> int:
        """Computes how much of the text page area is covered by this image."""
        return self.rect.get_area() / page_rect.get_area()


def extract_page_graphics(page: pymupdf.Page, is_digital: bool):
    """Extract drawings and image bounding boxes from page."""
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
    """Returns JPEG image bytes of a single PDF page. Downscales if image exceeds allowed size."""
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
    """Converts a PDF page (as bytes) to JPEG image bytes using PyMuPDF and PIL."""
    with pymupdf.open(stream=page_bytes, filetype="pdf") as doc:
        page = doc[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)

    logger.info(f"[convert] PDF rendered to image: {pix.width}x{pix.height} at scale={scale:.2f}")

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def get_color_proportion(
    page: pymupdf.Page, scale: int = 2, h_bins: int = 20, s_bins: int = 3, v_bins: int = 3, display: bool = False
) -> Counter:
    """Extract discretized colors from a PDF page (in HSV space).

    Args:
        page (fitz.Page): The page object.
        scale (int): Scaling factor for rendering.
        h_bins (int): Number of bins for hue (0-1 mapped to bins).
        s_bins (int): Number of bins for saturation.
        v_bins (int): Number of bins for value/brightness.
        display (bool): Whether to display the most common colors in RGB space.

    Returns:
        Counter: color code (in hsv discretized) and proportion of page covered
    """
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)

    # Convert to numpy array and map to HSV
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    pixels = img.reshape(-1, 3)
    pixels_norm = pixels.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)

    # Scale to bins to discretize
    h_bin = (hsv[:, 0] * h_bins).astype(int)
    s_bin = (hsv[:, 1] * s_bins).astype(int)
    v_bin = (hsv[:, 2] * v_bins).astype(int)
    hsv_bins = np.stack([h_bin, s_bin, v_bin], axis=1)

    # Count HSV bins and filter to keep only dominant colors (excluding greyish/dark tones)
    hsv_counter = Counter([tuple(map(int, row)) for row in hsv_bins])
    hsv_counter_prop = filter_color_counter(hsv_counter, pix.height * pix.width, 300, 1, 1)

    if display:
        display_colors(hsv_counter_prop, h_bins, s_bins, v_bins)
    return hsv_counter_prop


def filter_color_counter(
    counter: Counter, total_pixels: int, min_prop: int, min_saturation: int = 1, min_value: int = 1
) -> Counter:
    """Keep colors that pass the thresholds.

    A color passes the thresholds if it covers a sufficient proportion of the page and is not too greyish or dark.
    A higher min_saturation removes greyish colors, a higher min_value removes dark colors.

    Args:
        counter (Counter): Counter of discretized HSV colors and their pixel counts.
        total_pixels (int): Total number of pixels in the image.
        min_prop (int): Minimum proportion (1/n) of the page that a color must cover to be kept.
        min_saturation (int): Minimum saturation bin (0 to s_bins - 1).
        min_value (int): Minimum value/brightness bin (0 to v_bins - 1).
    """
    out = Counter()

    for hsv, cnt in counter.items():
        if cnt * min_prop < total_pixels:
            continue
        _, s, v = hsv
        if s >= min_saturation and v >= min_value:
            out[hsv] = cnt / total_pixels
    return out


def display_colors(filtered_colors: Counter, h_bins: int, s_bins: int, v_bins: int):
    """Display the most common colors in RGB space.

    The colors are converted back from discretized HSV to RGB for visualization. This conversion is not exact
    due to the discretization, but it is only used to display colors and provides a good approximation of the
    dominant colors on the page.

    Args:
        filtered_colors (Counter): Counter of discretized HSV colors and their proportions.
        h_bins (int): Number of bins for hue.
        s_bins (int): Number of bins for saturation.
        v_bins (int): Number of bins for value/brightness.
    """
    rgb_counter = Counter()
    for (h_bin, s_bin, v_bin), count in filtered_colors.items():
        h = min((h_bin + 0.5) / h_bins, 1.0)
        s = min((s_bin + 0.5) / s_bins, 1.0)
        v = min((v_bin + 0.5) / v_bins, 1.0)
        r, g, b = hsv_to_rgb(h, s, v)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        rgb_counter[rgb] = count

    rgb_counter = sorted(rgb_counter.items(), key=lambda x: x[1], reverse=True)

    labels = [str(c) for c, _ in rgb_counter]
    counts = [cnt for _, cnt in rgb_counter]
    rgb = [tuple(v / 255 for v in c) for c, _ in rgb_counter]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, counts, color=rgb)
    plt.gca().invert_yaxis()
    plt.title("Most Frequent Colors (HSV Discretized)")
    plt.xlabel("Pixel Count")
    plt.ylabel("Color (R,G,B)")
    plt.show()
