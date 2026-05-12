"""
Compute per-channel mean and std of the training split for use with
normalization_mode: "standard" in TRANSFORM_CONFIG.

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

    # Welford online algorithm: numerically stable single-pass mean and variance.
    # Operates on per-channel pixel means per image to avoid loading all pixels at once.
    n = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)

    for i in range(n_samples):
        sample = train_dataset.get_raw_sample(i)
        img = sample.image
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()  # CHW uint8
        else:
            img_np = np.array(img)  # CHW uint8

        # CHW -> normalise to [0,1] float64 and compute per-channel pixel mean
        img_f = img_np.astype(np.float64) / 255.0  # (C, H, W)
        sample_mean = img_f.mean(axis=(1, 2))  # (C,)

        n += 1
        delta = sample_mean - mean
        mean += delta / n
        delta2 = sample_mean - mean
        M2 += delta * delta2

        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            logger.info("  %d / %d", i + 1, n_samples)

    # M2 / (n-1) is the variance of per-image channel means — not pixel-level std.
    # For normalization we want the std of pixel values, so we need a two-pass approach
    # for accuracy. Re-iterate to accumulate pixel-level variance.
    logger.info("Second pass: computing pixel-level std...")
    sum_sq_diff = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for i in range(n_samples):
        sample = train_dataset.get_raw_sample(i)
        img = sample.image
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = np.array(img)

        img_f = img_np.astype(np.float64) / 255.0  # (C, H, W)
        h, w = img_f.shape[1], img_f.shape[2]
        total_pixels += h * w
        # sum of (x - mean)^2 per channel
        diff = img_f - mean[:, None, None]
        sum_sq_diff += (diff ** 2).sum(axis=(1, 2))

    std = np.sqrt(sum_sq_diff / total_pixels)

    mean_list = [round(float(v), 4) for v in mean]
    std_list = [round(float(v), 4) for v in std]
    return mean_list, std_list


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute PH2 training-set channel mean and std.")
    ap.add_argument("--config", type=Path, default=Path("main_config.yaml"),
                    help="Path to experiment YAML config (default: main_config.yaml)")
    args = ap.parse_args()

    mean, std = compute_stats(args.config)

    print("\n--- Paste into main_config.yaml under TRANSFORM_CONFIG ---")
    print("  normalization_mode: \"standard\"")
    print(f"  normalization_mean: {mean}")
    print(f"  normalization_std:  {std}")
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    main()
