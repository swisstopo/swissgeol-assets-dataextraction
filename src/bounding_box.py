import logging

import cv2
import numpy as np
import pymupdf

logger = logging.getLogger(__name__)


def merge_bounding_boxes(rects):
    """Computes the smallest bbox that contains all input rectangles."""
    x0 = min(rect.x0 for rect in rects)
    y0 = min(rect.y0 for rect in rects)
    x1 = max(rect.x1 for rect in rects)
    y1 = max(rect.y1 for rect in rects)
    return pymupdf.Rect(x0, y0, x1, y1)


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
