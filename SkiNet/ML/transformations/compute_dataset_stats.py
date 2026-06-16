"""
Compute per-channel mean and std of the training split for use with
normalization_mode: "standard" in TRANSFORM_CONFIG.

Works with any dataset configured in main_config.yaml. For datasets that have
predefined splits (e.g. ISIC 2017), the official training split is used automatically.
For datasets without predefined splits (e.g. PH2), the random split defined by
split_train_size / split_random_seed is used.

Usage:
    python compute_dataset_stats.py --config main_config.yaml

Output is printed as YAML-ready values to paste into main_config.yaml:
    normalization_mean: [R, G, B]
    normalization_std:  [R, G, B]

Stats are computed on the raw uint8 images (before any augmentation) from the
training split only.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_stats(cfg_path: Path) -> tuple[list[float], list[float]]:
    """
    Produce the R/G/B mean and std of all pixel values in the training split, normalized to [0, 1].
    These are the values torchvision.transforms.Normalize expects.

    :param cfg_path: Path to the YAML config file containing the dataset configuration.
    :return: A tuple of two lists: (mean, std), where each is a list of three floats corresponding to the R, G, B channels.
    """
    config = load_config_from_yaml(cfg_path)
    datasets = create_segmentation_datasets_from_config(config)
    train_dataset = datasets.train

    n_samples = len(train_dataset)
    logger.info("Computing stats over %d training samples...", n_samples)

    # Initialise single-pass pixel-weighted mean and variance (no caching).
    # Uses the parallel/batch form of Welford's online algorithm so only one
    # image is in memory at a time.
    mean = np.zeros(3, dtype=np.float64)  # running mean per channel
    M2 = np.zeros(3, dtype=np.float64)   # running sum of squared deviations from running mean
    total_pixels = np.int64(0)

    # for each training image, update the running mean and M2 using the parallel Welford merge
    for i in range(n_samples):
        # ge an image before any augmenation or resizing
        sample = train_dataset.get_raw_sample(i)
        img = sample.image
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)

        # scale to 0-1 range
        img_f = img_np.astype(np.float64) / 255.0  # CHW or HWC

        # Normalise to CHW
        if img_f.ndim == 3 and img_f.shape[2] == 3:
            img_f = img_f.transpose(2, 0, 1)

        n_new = np.int64(img_f.shape[1] * img_f.shape[2])

        # Per-channel stats for this image (before any augmentation or resizing).
        img_mean = img_f.mean(axis=(1, 2))           # (3,)
        img_var = img_f.var(axis=(1, 2))            # (3,)

        # Parallel Welford merge: combine (total_pixels, mean, M2) with (n_new, img_mean, img_var*n_new)
        n_total = total_pixels + n_new
        delta = img_mean - mean
        mean = (total_pixels * mean + n_new * img_mean) / n_total
        # M2 tracks the total sum of squared deviations from the combined mean across all pixels seen so far.
        # img_var * n_new — the internal variance of the new image, scaled by its pixel count. This is how spread out pixels are within this image.
        # delta**2 * (total_pixels * n_new / n_total) — a correction for the fact that the two groups had different means.
        # Even if both groups had zero internal variance, combining them would produce variance if their means differ.
        # This term accounts for that inter-group spread.
        M2 += img_var * n_new + delta ** 2 * (total_pixels * n_new / n_total)
        total_pixels = n_total

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            logger.info("  %d / %d", i + 1, n_samples)

    std = np.sqrt(M2 / total_pixels)

    mean_list = [round(float(v), 4) for v in mean]
    std_list = [round(float(v), 4) for v in std]
    return mean_list, std_list


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute training-set channel mean and std for any configured dataset.")
    ap.add_argument("--config", type=Path, default=Path("main_config.yaml"),
                    help="Path to experiment YAML config (default: main_config.yaml)")
    args = ap.parse_args()

    mean, std = compute_stats(args.config)

    print("\n--- Paste into your config YAML under TRANSFORM_CONFIG ---")
    print("  normalization_mode: \"standard\"")
    print(f"  normalization_mean: {mean}")
    print(f"  normalization_std:  {std}")
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    main()
