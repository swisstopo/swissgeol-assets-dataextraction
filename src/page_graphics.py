import pymupdf
from dataclasses import dataclass

@dataclass
class ImageRect:
    rect: pymupdf.Rect         # Transformed rectangle respecting page rotation
    rotation: int           # Page rotation in degrees (0, 90, 180, 270)
    xref: int               # Image xref ID (unique reference in PDF)

    def page_coverage(self,text_rect: pymupdf.Rect) -> int:
        """Computes how much of the text page area is covered by this image."""
        return self.rect.get_area()/ text_rect.get_area()

def extract_page_graphics(page:pymupdf.Page, is_digital: bool):
    """Extract drawings and image bounding boxes from page"""
    if not is_digital:
        return [], []

    drawings = page.get_drawings()
    image_rects = get_images_from_page(page)

    return drawings, image_rects

def get_images_from_page(page:pymupdf.Page) ->list[ImageRect]:
    """Extracts all image bounding boxes from the page, transformed by page rotation."""
    page_rotation = page.rotation
    rotation_matrix = page.rotation_matrix

    extracted_images = []
    for image_info in page.get_images():
        xref = image_info[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            rotated_rect = rect * rotation_matrix if page.rotation else rect
            extracted_images.append(ImageRect(
                rect=rotated_rect,
                rotation=page_rotation,
                xref=xref
            ))

    return extracted_images


