"""Create areference document from all available classes."""

from pathlib import Path

import pymupdf

# Append one page from each class
REFERENCE_PAGES = [
    "data/single_pages/boreprofile/742_7.pdf",
    "data/single_pages/diagram/250_3.pdf",
    "data/single_pages/geo_profile/1630_114.pdf",
    "data/single_pages/map/1432_4.pdf",
    "data/single_pages/section_header/1630_393.pdf",
    "data/single_pages/table/7066_6.pdf",
    "data/single_pages/text/1062_7.pdf",
    "data/single_pages/title_page/440_02_1.pdf",
    "data/single_pages/unknown/250_10.pdf",
]

# Output poath for reference document
OUTPUT_PDF = Path("examples") / "reference_document.pdf"


def main() -> None:
    # Verify all source files exist.
    if any(path for path in REFERENCE_PAGES if not Path(path).exists()):
        raise FileNotFoundError("Make sure REFERENCE_PAGES are present")

    # Create empty document to happend pages
    out_doc = pymupdf.Document()

    # Append all pages
    for source_path in REFERENCE_PAGES:
        src_doc = pymupdf.Document(source_path)
        # Each single-page PDF contains exactly one page (page index 0).
        out_doc.insert_pdf(src_doc, from_page=0, to_page=0)
        src_doc.close()

    # Write ouput document
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(OUTPUT_PDF))
    out_doc.close()
    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
