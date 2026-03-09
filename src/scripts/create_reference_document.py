"""Create a reference document from all available classes."""

from pathlib import Path

import pymupdf

REFERENCE_PAGES = [
    "data/single_pages/boreprofile/742_7.pdf",
    "data/single_pages/diagram/250_3.pdf",
    "data/single_pages/geo_profile/24361_29.pdf",
    "data/single_pages/map/7066_11.pdf",
    "data/single_pages/section_header/1630_393.pdf",
    "data/single_pages/table/27898_16.pdf",
    "data/single_pages/text/1062_7.pdf",
    "data/single_pages/title_page/440_02_1.pdf",
    "data/single_pages/unknown/44179_195.pdf",
]


# Output path for reference document
OUTPUT_PDF = Path("examples") / "reference_document.pdf"


def main() -> None:
    # Verify all source files exist.
    if any(not Path(path).exists() for path in REFERENCE_PAGES):
        raise FileNotFoundError("Make sure REFERENCE_PAGES are present")

    # Create empty document to append pages
    out_doc = pymupdf.Document()

    # Append all pages
    for source_path in REFERENCE_PAGES:
        src_doc = pymupdf.Document(source_path)
        # Each single-page PDF contains exactly one page (page index 0).
        out_doc.insert_pdf(src_doc, from_page=0, to_page=0)
        src_doc.close()

    # Write output document
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(OUTPUT_PDF))
    out_doc.close()
    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
