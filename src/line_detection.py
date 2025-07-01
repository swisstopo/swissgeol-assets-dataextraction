from __future__ import annotations

import logging

import cv2
import numpy as np
import pymupdf
from numpy.typing import NDArray

from src.geometric_objects import Line, Point

logger = logging.getLogger(__name__)


def turn_page_to_image(page: pymupdf.Page, zoom: float = 2.0) -> np.ndarray:
    """turns pdf page into an BGR image"""
    mat = pymupdf.Matrix(zoom, zoom)  # apply zoom
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)

    # Convert to NumPy array (RGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Convert RGB to BGR for OpenCV
    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def line_from_array(line: np.ndarray) -> Line:
    start = Point(int(line[0][0]), int(line[0][1]))
    end = Point(int(line[0][2]), int(line[0][3]))
    return Line(start, end)


def extract_geometric_lines(page: pymupdf.Page) -> list:
    """Extracts all lines on a page using line segment detection."""
    image = turn_page_to_image(page)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)

    lsd_lines = line_segment_detection(blurred)

    if lsd_lines is None:
        logger.warning(f"No geometric lines detected on page {page.number}")
        lines = []
    else:
        lines = [line_from_array(lsd_line) for lsd_line in lsd_lines]

    return lines


def line_segment_detection(
    preprocessed_image: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """detects straight lines using LineSegmentDetector from preprocessed image"""
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(preprocessed_image)[0]

    return lines
