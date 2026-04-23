from typing import Any
import torch
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


def default_worker_init_fn(worker_id: int) -> None:
    """Seed each worker process for reproducibility."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collate_preserving_specs(batch: list[Any]) -> Any:
    """
    Collate a batch while preserving per-sample ``specs`` payloads as Python objects.

    PyTorch's default collation recurses into nested dicts and attempts to tensorize
    scalar leaves. That breaks for sample metadata that legitimately contains strings
    such as sample IDs or paths. For segmentation samples we want tensor collation for
    ``image``/``mask`` and a plain list for ``specs``.
    """
    if not batch:
        return batch

    first = batch[0]
    if isinstance(first, dict) and "specs" in first:
        collated = {
            key: default_collate([item[key] for item in batch])
            for key in first
            if key != "specs"
        }
        collated["specs"] = [item["specs"] for item in batch]
        return collated

    return default_collate(batch)


class RepeatDataLoader(DataLoader):
    """
    DataLoader with persistent workers to avoid respawning processes each epoch.
    Replaces the previous _RepeatSampler-based implementation.

    Preserves the same constructor signature so existing call sites are unaffected.
    `max_num_to_repeat` is accepted but ignored — Lightning controls epoch iteration.
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 **kwargs: Any):

        if not hasattr(dataset, '__len__'):
            raise TypeError("Dataset must be sized (have __len__ method).")

        if "worker_init_fn" not in kwargs:
            kwargs["worker_init_fn"] = default_worker_init_fn
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = collate_preserving_specs

        # Only enable persistent_workers when there are actually workers to persist.
        # With num_workers=0 the flag is invalid and PyTorch will raise.
        num_workers = kwargs.get("num_workers", 0)
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = num_workers > 0

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )
