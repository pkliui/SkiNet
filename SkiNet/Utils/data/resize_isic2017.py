"""
This script is designed to download the ISIC 2017 dataset from Kaggle,
resize the images and masks to 256x256, and save them to a specified output directory.
It uses multithreading to speed up the resizing process.

Example usage:

    # Set environment variables as needed
    export KAGGLE_DATASET="johnchfr/isic-2017"
    export ISIC_OUT_DIR="/path/to/output/directory"
    export RESIZE_WORKERS=8
    python resize_isic2017.py

"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "johnchfr/isic-2017")
DST = Path(os.environ.get("ISIC_OUT_DIR", "/teamspace/lightning_storage/isic2017/ISIC2017DATA_256/"))
SIZE = (256, 256)
WORKERS = int(os.environ.get("RESIZE_WORKERS", "8"))

IMAGE_DIRS = {
    "ISIC-2017_Training_Data",
    "ISIC-2017_Validation_Data",
    "ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data",
}
MASK_DIRS = {
    "ISIC-2017_Training_Part1_GroundTruth",
    "ISIC-2017_Validation_Part1_GroundTruth",
    "ISIC-2017_Test_v2_Part1_GroundTruth",
}


def download(dst_dir: Path) -> None:
    print(f"Downloading {KAGGLE_DATASET} -> {dst_dir}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "--unzip", "-p", str(dst_dir)],
        check=True,
    )


def process_file(src_file: Path, dst_file: Path, resample: Image.Resampling) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if src_file.suffix.lower() == ".csv":
        shutil.copy2(src_file, dst_file)
    else:
        Image.open(src_file).resize(SIZE, resample).save(dst_file)


def resize_dir(src_dir: Path, src_root: Path, resample: Image.Resampling) -> None:
    files = [f for f in sorted(src_dir.rglob("*")) if f.is_file()]
    tasks = [(f, DST / f.relative_to(src_root)) for f in files]
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_file, s, d, resample): s for s, d in tasks}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {src_dir.name}: {done}/{len(tasks)}")
    print(f"  {src_dir.name}: done ({len(tasks)} files)")


def process(src_root: Path) -> None:
    for d in IMAGE_DIRS:
        src = src_root / d
        if src.exists():
            print(f"Resizing images: {d}")
            resize_dir(src, src_root, Image.Resampling.BILINEAR)
        else:
            print(f"  skipping (not found): {d}")

    for d in MASK_DIRS:
        src = src_root / d
        if src.exists():
            print(f"Resizing masks: {d}")
            resize_dir(src, src_root, Image.Resampling.NEAREST)
        else:
            print(f"  skipping (not found): {d}")

    for f in src_root.glob("*.csv"):
        out = DST / f.relative_to(src_root)
        shutil.copy2(f, out)
        print(f"Copied: {f.name}")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    src_dir = os.environ.get("ISIC_SRC_DIR")
    if src_dir:
        print(f"Using existing data at {src_dir}")
        process(Path(src_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="isic2017_raw_") as tmp:
            download(Path(tmp))
            process(Path(tmp))

    print(f"Done. Output: {DST}")


if __name__ == "__main__":
    main()
