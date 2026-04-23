"""Unit tests for SkiNet.ML.dataloaders.dataloaders"""
import pytest
from torch.utils.data import TensorDataset
import torch

from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader

DATASET = TensorDataset(torch.arange(10))


def test_persistent_workers_enabled_when_num_workers_positive() -> None:
    dl = RepeatDataLoader(DATASET, num_workers=2)
    assert dl.persistent_workers is True


def test_persistent_workers_disabled_when_num_workers_zero() -> None:
    dl = RepeatDataLoader(DATASET, num_workers=0)
    assert dl.persistent_workers is False


def test_caller_can_override_persistent_workers() -> None:
    dl = RepeatDataLoader(DATASET, num_workers=2, persistent_workers=False)
    assert dl.persistent_workers is False


def test_len_matches_dataset_size_divided_by_batch_size() -> None:
    dl = RepeatDataLoader(DATASET, batch_size=2)
    assert len(dl) == 5

def test_unsized_dataset_raises() -> None:
    class Unsized:
        def __getitem__(self, i: int) -> int:
            return i

    with pytest.raises(TypeError):
        RepeatDataLoader(Unsized())  # type: ignore[arg-type]
