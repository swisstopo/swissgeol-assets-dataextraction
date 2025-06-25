import logging
import pymupdf
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImageRect:
    rect: pymupdf.Rect         # Transformed rectangle respecting page rotation
    rotation: int           # Page rotation in degrees (0, 90, 180, 270)
    xref: int               # Image xref ID (unique reference in PDF)

    def page_coverage(self,page_rect: pymupdf.Rect) -> int:
        """Computes how much of the text page area is covered by this image."""
        return self.rect.get_area()/ page_rect.get_area()

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

def downscale_pdf_page_to_bytes(page: pymupdf.Page, scale: float = 1.0) -> bytes:
    """Render a page to an image and re-embed as a compressed PDF page."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)

    doc = pymupdf.open()
    img_rect = pymupdf.Rect(0, 0, pix.width, pix.height)
    pdf_page = doc.new_page(width=pix.width, height=pix.height)
    pdf_page.insert_image(img_rect, pixmap=pix)

    return doc.tobytes(deflate=True, garbage=3, use_objstms=1)

def get_page_bytes(page: pymupdf.Page, page_number: int, max_mb:float = 4.5) -> bytes:
    """Returns PDF bytes of a single page. Downscales only if it exceeds Bedrock size limit."""    # Step 1: Try extracting as-is
    max_bytes = int(max_mb * 1024 * 1024)

    single_page_pdf = pymupdf.open()
    single_page_pdf.insert_pdf(page.parent, from_page=page_number, to_page=page_number)
    page_bytes = single_page_pdf.tobytes(deflate=True, garbage=3, use_objstms=1)

    if len(page_bytes) < max_bytes:
        return page_bytes

    logger.info(f"Page {page_number} is {len(page_bytes) / 1024 / 1024:.2f} MB — downscaling.")
    scale = 1.0
    for attempt in range(5):
        if scale < 0.2:
            logger.warning(f"Scale dropped below 0.2. stopping downscaling.")
            break

        page_bytes = downscale_pdf_page_to_bytes(page, scale = scale)
        if len(page_bytes) <= max_bytes:
                logger.info(f"Successfully resized page at scale={scale:.2f}, size={len(page_bytes) / 1024 / 1024:.2f} MB")
                return page_bytes
        logger.info(f"Attempt {attempt + 1}: scale={scale:.2f}, size={len(page_bytes) / 1024 / 1024:.2f} MB — too large")
        scale *= 0.85

    logger.info(f"Final size {len(page_bytes) / 1024 / 1024:.2f} MB after 5 attempts. Returning last attempt.")
    return page_bytes