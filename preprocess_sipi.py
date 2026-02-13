"""
Preprocess USC-SIPI Image Database files into grayscale 256x256 PNGs.
"""

import os
from typing import Tuple, Optional
from PIL import Image


def preprocess_sipi_dataset(
    input_root: str = "data/USC-SIPI Image Database",
    output_dir: str = "data/processed",
    size: int = 256,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Convert SIPI images to grayscale and resize to a fixed square size.

    Returns (processed_count, skipped_count).
    """
    if not os.path.exists(input_root):
        print(f"Input root not found: {input_root}")
        return 0, 0

    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    processed = 0
    skipped = 0

    for root, _, files in os.walk(input_root):
        rel_root = os.path.relpath(root, input_root)
        for fname in files:
            if not fname.lower().endswith(exts):
                continue

            src_path = os.path.join(root, fname)
            dest_dir = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.splitext(fname)[0] + ".png")

            try:
                img = Image.open(src_path).convert("L")
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                img.save(dest_path)
                processed += 1
            except Exception as exc:  # noqa: BLE001 - keep pipeline robust
                print(f"  ✗ Failed: {src_path} ({exc})")
                skipped += 1

            if limit and processed >= limit:
                print(f"Reached processing limit: {limit}")
                return processed, skipped

    print(f"Preprocessed {processed} image(s); skipped {skipped}. Output -> {output_dir}/")
    return processed, skipped


if __name__ == "__main__":
    preprocess_sipi_dataset()
