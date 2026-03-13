import argparse
import logging
from pathlib import Path

import torch

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader
from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset

logging.basicConfig(level=logging.INFO)


def worker_init_fn(worker_id: int) -> None:
    logging.getLogger(__name__).info("Worker %d started (pid=%d)", worker_id, torch.multiprocessing.current_process().pid)


def main(cfg_path: Path, batch_size: int, num_workers: int, max_num_to_repeat: int, max_batches: int) -> None:

    cfg = load_config_from_yaml(cfg_path)

    dataset = SegmentationDataset(cfg)
    dl = RepeatDataLoader(
        dataset,
        max_num_to_repeat=max_num_to_repeat,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    logging.info("Dataset length: %d, DataLoader len (batches per epoch): %d", len(dataset), len(dl))

    seen = 0
    for batch in dl:
        img = batch.get("image")
        mask = batch.get("mask")
        if isinstance(img, torch.Tensor):
            logging.info("Batch %d image shape: %s", seen, tuple(img.shape))
        else:
            logging.info("Batch %d image type: %s", seen, type(img))
        if mask is not None:
            logging.info("Batch %d mask shape: %s", seen, tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask))
        seen += 1
        if seen >= max_batches:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="Path to experiment YAML config")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-num-to-repeat", type=int, default=1, help="0 = infinite")
    ap.add_argument("--max-batches", type=int, default=5, help="How many batches to print then exit")
    args = ap.parse_args()

    main(args.config, args.batch_size, args.num_workers, args.max_num_to_repeat, args.max_batches)
