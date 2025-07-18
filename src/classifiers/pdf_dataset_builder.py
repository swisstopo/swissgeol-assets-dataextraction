import json
import logging
from pathlib import Path
from typing import Callable

import pymupdf
from datasets import Dataset, IterableDataset
from PIL import Image

from src.page_classes import label2id

logger = logging.getLogger(__name__)


def extract_layoutlm_data_from_pdf(doc: pymupdf.Document) -> list:
    """Extracts words, bounding boxes, and images from each page of a PDF document.

    Args:
        doc (pymupdf.Document): The PDF document to extract data from.

    Returns:
        list: A list of dictionaries, each containing words, bounding boxes, and images for each page.
    """
    all_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_data = extract_layoutlm_data_from_page(page)
        all_pages.append(page_data)
    return all_pages


def extract_layoutlm_data_from_page(page: pymupdf.Page) -> dict:
    """Extracts words, bounding boxes, and images from a single page of a PDF document.

    Args:
        page (pymupdf.Page): The page to extract data from.

    Returns:
        dict: A dictionary containing words, bounding boxes, and an image of the page.
    """
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


def normalize_box(box: list, width: int, height: int) -> list[int]:
    """Normalizes a bounding box to a range of 0 to 1000 based on the page dimensions.

    Args:
        box (list): A list containing the bounding box coordinates [x0, y0, x1, y1].
        width (int): The width of the page.
        height (int): The height of the page.

    Returns:
        list: A list containing the normalized bounding box coordinates [x0, y0, x1, y1],
              scaled to a range of 0 to 1000. Negative values are clamped to 0.
    """
    return [
        max(0, int(1000 * box[0] / width)),
        max(0, int(1000 * box[1] / height)),
        max(0, int(1000 * box[2] / width)),
        max(0, int(1000 * box[3] / height)),
    ]


def build_lazy_dataset(
    pdf_list: list[Path],
    preprocess_fn: Callable,
    ground_truth_map: dict | None = None,
) -> IterableDataset:
    """Builds an iterable dataset from a list of PDF files.

    This function processes each PDF file, extracts the necessary data from each page, normalizes the bounding boxes,
    and applies a preprocessing function to each page's data. The use of a generator allows for lazy loading of the
    data, which is efficient for large datasets.

    Args:
        pdf_list (list[Path]): List of paths to PDF files.
        preprocess_fn (Callable): Function to preprocess each page's data.
        ground_truth_map (dict, optional): A mapping of (filename, page_number) to label.
            If provided, it will be used to assign labels to the pages.

    Returns:
        IterableDataset: An iterable dataset that yields preprocessed page data.
    """

    # return LazyPDFDataset(pdf_list, preprocess_fn, ground_truth_map)
    def sample_generator():
        for file_path in pdf_list:
            if not file_path.name.lower().endswith(".pdf"):
                continue
            with pymupdf.open(file_path) as doc:
                pages = extract_layoutlm_data_from_pdf(doc)

                for page_num, page in enumerate(pages, start=1):
                    norm_boxes = [normalize_box(b, page["width"], page["height"]) for b in page["boxes"]]
                    label = ground_truth_map.get((file_path.name, page_num)) if ground_truth_map else None
                    if label is None:
                        logger.info(f"Label not found for {file_path.name}, page {page_num}. Skipping.")
                    yield preprocess_fn(
                        {
                            "words": page["words"],
                            "bboxes": norm_boxes,
                            "image": page["image"],
                            "label": label,
                        }
                    )

    return IterableDataset.from_generator(sample_generator)


def build_dataset_from_page_list(page_list: list[pymupdf.Page], ground_truth_map: dict | None = None) -> Dataset:
    """Builds a dataset from a list of pymupdf.Page objects.

    This function processes each page, extracts the necessary data, normalizes the bounding boxes,
    and constructs a dataset suitable for training or evaluation.

    Args:
        page_list (list[pymupdf.Page]): List of pymupdf.Page objects to process.
        ground_truth_map (dict): A mapping of (filename, page_number) to label. If provided, it will be used to
            assign labels to the pages.

    Returns:
        Dataset: A dataset containing the processed page data, including words, bounding boxes, images,
                 and labels (if available).
    """
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


def build_filename_to_label_map( gt_json_path: Path) -> dict[tuple[str, int], int]:
    """Build a map from filename to class ID based on the ground truth JSON."""
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)

    label_lookup = {}
    for entry in gt_data:
        filename = entry["filename"]
        for classification in entry["classification"]:
            page = classification["Page"]
            for label_name, value in classification.items():
                if label_name != "Page" and value == 1:
                    try:
                        label_id= label2id[label_name]
                        label_lookup[(filename, page)] = label_id
                    except KeyError:
                        raise ValueError(f"Unknown label: {label_name}")
    return label_lookup