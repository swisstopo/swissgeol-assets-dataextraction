"""Script to retrieve files from the S3 bucket."""

import logging
import random
from pathlib import Path

import boto3
import pymupdf
from tqdm import tqdm

from src.classifiers.baseline_classifier import BaselineClassifier
from src.classifiers.pixtral_classifier import PixtralClassifier
from src.pdf_processor import PDFProcessor
from src.utils import get_aws_config, read_params

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config and Constants ---
S3_PREFIX = "asset/asset_files_new_ocr/"
S3_PROFILE = "779726271945_swissgeol-assets-ro"
S3_BUCKET_NAME = "swissgeol-assets-swisstopo"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TMP_DIR = DATA_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = DATA_DIR / "single_pages_new"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_PARAMS = read_params(REPO_ROOT / "config" / "matching_params.yml")
# download_from_s3.py
PIXTRAL_CONFIG_FILE_PATH = REPO_ROOT / "config/pixtral_config.yml"
PIXTRAL_CONFIG = read_params(PIXTRAL_CONFIG_FILE_PATH)


def _resolve_pixtral_paths(cfg: dict, base: Path) -> dict:
    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p)

    # keys that contain file paths
    for key in (
        "prompt_path",
        "borehole_img_path",
        "text_img_path",
        "maps_img_path",
        "title_img_path",
        "unknown_img_path",
        "geo_profile_img_path",
        "table_img_path",
        "diagram_img_path",
    ):
        if key in cfg and cfg[key]:
            cfg[key] = str(_abs(cfg[key]))
    return cfg


PIXTRAL_CONFIG = _resolve_pixtral_paths(PIXTRAL_CONFIG, REPO_ROOT)

AWS_CONFIG = get_aws_config()

# Max number of pages to save per class per report
MAX_PER_CLASS = {
    "text": 1,
    "boreprofile": 2,
    "map": 2,
    "title_page": 2,
    "unknown": 5,
    "geo_profile": 5,
    "table": 5,
    "diagram": 5,
}


# --- Utility Functions ---
def key_to_filename(key: str) -> str:
    """Gets key of filename."""
    return key.split("/")[-1]


def group_pages_by_classification(pages: list[dict]) -> dict[str, list[int]]:
    """Group pages by classification."""
    grouped = {}
    for page_info in pages:
        page_num = page_info["page"]
        classification = page_info["classification"]
        page_class = next((cls for cls, val in classification.items() if val == 1), "unknown").lower()
        grouped.setdefault(page_class, []).append(page_num)
    return grouped


def save_new_pages(pdf_path: Path, page_by_class: dict[str, list[int]]) -> None:
    """Saves new pages to output directory."""
    try:
        with pymupdf.open(pdf_path) as doc:
            filename_base = pdf_path.stem

            for class_label, page_nums in page_by_class.items():
                max_pages = MAX_PER_CLASS.get(class_label, 1)
                selected_pages = random.sample(page_nums, min(len(page_nums), max_pages))

                class_output_dir = OUTPUT_DIR / class_label
                class_output_dir.mkdir(parents=True, exist_ok=True)

                for page_num in selected_pages:
                    if page_num < 1 or page_num > len(doc):
                        logger.warning(f"Skipping invalid page number {page_num} in {pdf_path.name}")
                        continue

                    out_filename = f"{filename_base}_{page_num}.pdf"
                    out_path = class_output_dir / out_filename

                    new_doc = pymupdf.open()
                    new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                    new_doc.save(out_path)
                    new_doc.close()
    except Exception as e:
        logger.error(f"Failed to save pages from {pdf_path.name}: {e}")


# --- Main Logic ---
def create_data(sample_size: int = 1) -> None:
    """Creates new data."""
    logger.info("Connecting to S3...")
    s3_session = boto3.Session(profile_name=S3_PROFILE)
    s3 = s3_session.resource("s3")
    bucket = s3.Bucket(S3_BUCKET_NAME)

    objs = list(bucket.objects.filter(Prefix=S3_PREFIX))
    if not objs:
        logger.warning("No PDF objects found in S3.")
        return

    sampled_objs = random.sample(objs, min(sample_size, len(objs)))

    processor = PDFProcessor(
        PixtralClassifier(
            config=PIXTRAL_CONFIG, aws_config=AWS_CONFIG, fallback_classifier=BaselineClassifier(MATCHING_PARAMS)
        )
    )

    for obj in tqdm(sampled_objs, desc="Processing PDFs"):
        filename = key_to_filename(obj.key)
        if not filename.endswith(".pdf"):
            continue

        local_path = TMP_DIR / filename
        try:
            bucket.download_file(obj.key, str(local_path))
        except Exception as e:
            logger.error(f"Failed to download {obj.key}: {e}")
            continue

        classification_result = processor.process(local_path)
        if not classification_result or "pages" not in classification_result:
            logger.warning(f"No classification result for {filename}")
            continue

        page_by_class = group_pages_by_classification(classification_result["pages"])
        save_new_pages(local_path, page_by_class)

        # Optional cleanup:
        # os.remove(local_path)


def main():
    """Main function to create new data."""
    create_data(sample_size=10)


if __name__ == "__main__":
    main()
