"""Split a directory of PDFs into train/validation/test using a deterministic hash."""

import hashlib
import logging
import shutil
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def deterministic_hash_ratio(text: str) -> float:
    """Map a string deterministically to a float in [0, 1).

    This is used to assign files to splits in a reproducible way, based only on
    their filename (or any stable string key).

    Args:
        text: Input string to hash (e.g., a filename).

    Returns:
        A float in the half-open interval [0, 1).
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Use the first 8 bytes (64 bits) to build a stable ratio in [0, 1).
    return int.from_bytes(h[:8], "big") / 2**64


@click.command(help="Split PDFs into train/validation/test folders deterministically.")
@click.option(
    "-i",
    "--input-directory",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing input *.pdf files (non-recursive).",
)
@click.option(
    "-o",
    "--output-directory",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory where train/, validation/, and test/ will be created.",
)
@click.option(
    "-rt",
    "--rtest",
    default=0.15,
    show_default=True,
    type=float,
    help="Fraction of files assigned to the test split.",
)
@click.option(
    "-rv",
    "--rvalid",
    default=0.15,
    show_default=True,
    type=float,
    help="Fraction of files assigned to the validation split.",
)
def split_data(input_directory: Path, output_directory: Path, rtest: float, rvalid: float) -> None:
    """Split PDF files into train/validation/test using filename hashing.

    The split is deterministic: each file is assigned based on a hash of its
    filename. This means adding new files later will **not** reshuffle existing
    assignments (as long as filenames don’t change).

    input_directory
    ├── file_1.pdf
    ├── ...
    └── file_n.pdf

    output_directory
    ├── train
    │   ├── file_1.pdf
    │   └── ...
    ├── validation
    │   ├── file_2.pdf
    │   └── ...
    └── test
        ├── file_3.pdf
        └── ...

    Args:
        input_directory (Path): Folder or subfolders containing the input PDF files.
        output_directory (Path): Destination folder for the split subfolders.
        rtest (float): Fraction assigned to the test set. Defaults to 0.15.
        rvalid (float): Fraction assigned to the validation set. Defaults to 0.15.
    """
    # Check that valid and test are coherent
    if rtest < 0 or rvalid < 0 or (rtest + rvalid) >= 1:
        raise click.BadParameter("Require 0 <= rtest, rvalid and rtest + rvalid < 1.")

    # Read input files and check if any
    paths = [p for p in input_directory.rglob("*.pdf")]
    if not paths:
        logger.info("No PDF files found in %s", input_directory)
        return

    # Get split into sets.
    splits = {"train": [], "validation": [], "test": []}
    for path in paths:
        # Extract filename for hash
        x_ratio = deterministic_hash_ratio(path.name)
        if x_ratio < rtest:
            splits["test"].append(path)
        elif x_ratio < rtest + rvalid:
            splits["validation"].append(path)
        else:
            splits["train"].append(path)

    logger.info(
        "Files train: {}, validation: {}, test: {}".format(
            len(splits["train"]), len(splits["validation"]), len(splits["test"])
        )
    )

    # Prepare outputs
    for split_name, split_paths in splits.items():
        # Check if at least a single file in split
        if len(split_paths) == 0:
            continue
        # Create output directory
        (output_directory / split_name).mkdir(parents=True, exist_ok=True)
        # Write to output
        for path in split_paths:
            shutil.copyfile(path, output_directory / split_name / path.name)

    logger.info("Done.")


if __name__ == "__main__":
    split_data()
