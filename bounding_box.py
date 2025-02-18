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

def cluster_drawings(drawings):
    """
    Cluster overlapping drawings into groups
    """
    clusters = []
    expanded_bboxes = [expand_bbox(pymupdf.Rect(d["rect"]), 5) for d in drawings]    
    for bbox in expanded_bboxes:
        added = False
        
        for cluster in clusters:
            if any(bbox_overlap(bbox, pymupdf.Rect(existing_bbox)) for existing_bbox in cluster):
                cluster.append(bbox)
                added = True
                break
        
        if not added:
            clusters.append([bbox])  # new cluster
    
    # Merge bounding boxes inside each cluster
    merged_clusters = [merge_bounding_boxes(cluster) for cluster in clusters]
    
    return merged_clusters