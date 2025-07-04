import logging
from pathlib import Path

import pymupdf
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_layoutlm_data_from_pdf(doc: pymupdf.Document):
    all_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = extract_layoutlm_data_from_page(page)
        all_pages.append(page_data)
    return all_pages


def extract_layoutlm_data_from_page(page: pymupdf.Page) -> dict:
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block_no, line_no, word_no)

    word_texts = []
    bboxes = []
    for w in words:
        x0, y0, x1, y1, text = w[:5]
        word_texts.append(text)
        bboxes.append([x0, y0, x1, y1])

    # Render image (optional)
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    return {
        "words": word_texts,
        "boxes": bboxes,
        "image": img,
        "width": pix.width,
        "height": pix.height,
    }


def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def build_dataset_from_path_list(pdf_list: list[Path], ground_truth_map: dict | None = None) -> Dataset:
    all_samples = []

    for file_path in tqdm(pdf_list):
        if not file_path.name.endswith(".pdf"):
            continue
        doc = pymupdf.open(file_path)
        pages = extract_layoutlm_data_from_pdf(doc)

        for page_num, page in enumerate(pages, start=1):
            norm_boxes = [normalize_box(b, page["width"], page["height"]) for b in page["boxes"]]
            label = ground_truth_map[(file_path.name, page_num)] if ground_truth_map else None

            all_samples.append({"words": page["words"], "bboxes": norm_boxes, "image": page["image"], "label": label})
    print(
        f" There is {sum(1 for s in all_samples if s['label'] is None)} None labels."
        f"({[s for s in all_samples if s['label'] is None]})."
    )
    return Dataset.from_list(all_samples)


def build_dataset_from_page_list(page_list: list[pymupdf.Page], ground_truth_map: dict | None = None) -> Dataset:
    all_samples = []

    for page in page_list:
        page_num = page.number
        filename = Path(page.parent.name).name
        page_data = extract_layoutlm_data_from_page(page)

        norm_boxes = [normalize_box(b, page_data["width"], page_data["height"]) for b in page_data["boxes"]]
        label = ground_truth_map[(filename, page_num)] if ground_truth_map else None

        all_samples.append(
            {"words": page_data["words"], "bboxes": norm_boxes, "image": page_data["image"], "label": label}
        )

    return Dataset.from_list(all_samples)
