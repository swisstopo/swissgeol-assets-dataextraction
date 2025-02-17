import pymupdf

def bbox_overlap(rect1, rect2):
    """
    Check if two bboxes overlap
    """
    return rect1.intersects(rect2) or rect1.contains(rect2) or rect2.contains(rect1)

def merge_bounding_boxes(rects):
    """Computes the smallest bbox that contains all input rectangles."""
    x0 = min(rect.x0 for rect in rects)
    y0 = min(rect.y0 for rect in rects)
    x1 = max(rect.x1 for rect in rects)
    y1 = max(rect.y1 for rect in rects)
    return pymupdf.Rect(x0, y0, x1, y1)

def expand_bbox(rect, margin):
    """
    Expands a bounding box by a some margin.
    """
    return pymupdf.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)