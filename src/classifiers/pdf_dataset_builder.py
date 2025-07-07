import logging
from pathlib import Path
from typing import Callable

import pymupdf
from datasets import Dataset, IterableDataset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# class LazyPreprocessingDataset(IterableDataset):
#     def __init__(self, dataset, preprocess_fn, batch_size):
#         self.dataset = dataset
#         self.preprocess_fn = preprocess_fn
#         self.batch_size = batch_size
#         self._epoch = 0

#     def __iter__(self):
#         batch = []
#         for sample in self.dataset:
#             batch.append(sample)
#             if len(batch) >= self.batch_size:
#                 yield from self.preprocess_fn(batch)
#                 batch = []
#         if batch:
#             yield from self.preprocess_fn(batch)

#     def set_epoch(self, epoch: int):
#         """Sets the current epoch. Trainer will call this for each epoch when using distributed or IterableDataset."""
#         self._epoch = epoch


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
    return Dataset.from_list(all_samples)


def build_lazy_dataset(
    pdf_list: list[Path],
    preprocess_fn: Callable,
    ground_truth_map: dict | None = None,
) -> IterableDataset:
    def sample_generator():
        for file_path in pdf_list:
            if not file_path.name.lower().endswith(".pdf"):
                continue
            doc = pymupdf.open(file_path)
            pages = extract_layoutlm_data_from_pdf(doc)

            for page_num, page in enumerate(pages, start=1):
                norm_boxes = [normalize_box(b, page["width"], page["height"]) for b in page["boxes"]]
                label = ground_truth_map.get((file_path.name, page_num)) if ground_truth_map else None
                if label is None:
                    print("no label")
                yield preprocess_fn(
                    {
                        "words": page["words"],
                        "bboxes": norm_boxes,
                        "image": page["image"],
                        "label": label,
                    }
                )

    return IterableDataset.from_generator(sample_generator)

    # return LazyPreprocessingDataset(raw_dataset, wrapped_preprocess, batch_size=batch_size)

    # return IterableDataset.from_generator(sample_generator)


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
