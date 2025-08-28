from pathlib import Path

import pymupdf


def convert_to_image(pdf_path, output_dir, dpi=300, page_range=None):
    """Convert PDF to PNG using PyMuPDF (fitz) - Fastest option."""
    with pymupdf.open(pdf_path) as pdf_document:
        pdf_name = Path(pdf_path).stem

        # Determine page range
        if page_range:
            start, end = page_range
            pages = range(start, min(end + 1, pdf_document.page_count))
        else:
            pages = range(pdf_document.page_count)

        for page_num in pages:
            page = pdf_document[page_num]

            # Create matrix for scaling (DPI)
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)

            # Render page as image
            pix = page.get_pixmap(matrix=mat)

            # Save as PNG
            output_path = Path(output_dir) / f"{pdf_name}_page_{page_num + 1}.png"
            pix.save(output_path)
            print(f"Saved: {output_path}")

    return True


if __name__ == "__main__":
    convert_to_image("examples/example_map_1252_2.pdf", "examples")
