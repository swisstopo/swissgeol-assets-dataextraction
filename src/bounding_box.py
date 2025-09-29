import logging

import cv2
import numpy as np
import pymupdf

from src.geometric_objects import Line

logger = logging.getLogger(__name__)


def merge_bounding_boxes(rects):
    """Computes the smallest bbox that contains all input rectangles."""
    x0 = min(rect.x0 for rect in rects)
    y0 = min(rect.y0 for rect in rects)
    x1 = max(rect.x1 for rect in rects)
    y1 = max(rect.y1 for rect in rects)
    return pymupdf.Rect(x0, y0, x1, y1)


def _x_center(rect: pymupdf.Rect) -> float:
    """Finds middle x position of a rectangle."""
    return 0.5 * (rect.x0 + rect.x1)


def _y_center(rect: pymupdf.Rect) -> float:
    """Finds middle y position of a rectangle."""
    return 0.5 * (rect.y0 + rect.y1)


def bbox_of_lines(lines: list[Line]) -> pymupdf.Rect | None:
    """Returns bounding box of a list of lines."""
    if not lines:
        return None
    xs, ys = [], []
    for L in lines:
        xs.extend((L.start.x, L.end.x))
        ys.extend((L.start.y, L.end.y))
    return pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))


def rects_intersect(rect_a: pymupdf.Rect, rect_b: pymupdf.Rect) -> bool:
    """Checks if two rectangles intersect, given they share same axis."""
    ax0, ay0, ax1, ay1 = rect_a.x0, rect_a.y0, rect_a.x1, rect_a.y1
    bx0, by0, bx1, by1 = rect_b.x0, rect_b.y0, rect_b.x1, rect_b.y1

    x_overlap = max(ax0, bx0) < min(ax1, bx1)
    y_overlap = max(ay0, by0) < min(ay1, by1)

    return x_overlap and y_overlap


def is_line_below_box(line_rect: pymupdf.Rect, image_rect: pymupdf.Rect) -> bool:
    """Determines whether a text line rect is directly below an image rect and horizontally aligned.

    Args:
        line_rect (pymupdf.Rect): Bounding box of the text line.
        image_rect (pymupdf.Rect): Bounding box of the image (transformed according to page rotation).

    Returns:
        bool: True if the line is well aligned else False
    """
    if image_rect.y1 - line_rect.y0 > image_rect.height * 0.25:
        return False

    max_offset = image_rect.width * 0.2
    left_within = line_rect.x0 >= image_rect.x0 - max_offset
    right_within = line_rect.x1 <= image_rect.x1 + max_offset

    return left_within and right_within


def find_document_bounding_box(gray):
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)  # (x, y, w, h)


def get_page_bbox(page, dpi=150):
    # Render page to grayscale image
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    bbox = find_document_bounding_box(img)
    if bbox is None:
        logger.info("No bounding box detected via image. Falling back to page.rect")
        return page.rect

    x, y, w, h = bbox
    scale = 72 / dpi  # Convert from pixels to PDF points
    return pymupdf.Rect(x * scale, y * scale, (x + w) * scale, (y + h) * scale)
