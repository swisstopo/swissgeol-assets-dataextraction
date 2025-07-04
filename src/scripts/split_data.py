import random
import shutil
from pathlib import Path

import click


@click.command()
@click.option("-i", "--input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--val-ratio", default=0.2, help="Proportion of validation set (default: 0.2)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def split_dataset(input_dir: Path, output_dir: Path, val_ratio: float, seed: int):
    """
    Recursively splits PDFs in INPUT_DIR into flat train/val folders under OUTPUT_DIR.
    Does NOT preserve original subdirectory structure.
    """
    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        click.echo("No PDF files found in the input directory.")
        return

    random.seed(seed)
    random.shuffle(pdf_files)

    n_val = int(len(pdf_files) * val_ratio)
    val_files = pdf_files[:n_val]
    train_files = pdf_files[n_val:]

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    for folder in [train_dir, val_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    # Copy files flatly
    for file in train_files:
        shutil.copy2(file, train_dir / file.name)

    for file in val_files:
        shutil.copy2(file, val_dir / file.name)

    print(f"Split {len(pdf_files)} files: {len(train_files)} train / {len(val_files)} val.")
    print(f"Output saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    split_dataset()
