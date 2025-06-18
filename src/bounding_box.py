import pymupdf

def merge_bounding_boxes(rects):
    """Computes the smallest bbox that contains all input rectangles."""
    x0 = min(rect.x0 for rect in rects)
    y0 = min(rect.y0 for rect in rects)
    x1 = max(rect.x1 for rect in rects)
    y1 = max(rect.y1 for rect in rects)
    return pymupdf.Rect(x0, y0, x1, y1)

def is_line_below_box(line_rect: pymupdf.Rect, image_rect: pymupdf.Rect) -> bool:
    """
      Determines whether a text line rect is directly below an image rect and horizontally aligned.
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