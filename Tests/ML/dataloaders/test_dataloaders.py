"""Unit tests for SkiNet.ML.dataloaders.dataloaders"""
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader, default_worker_init_fn

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


def test_worker_init_sets_cv2_threads_to_zero() -> None:
    """
    Test that default_worker_init_fn sets the number of threads for OpenCV (cv2) to zero,
    which is important to prevent OpenCV from spawning per-worker thread pools that can cause
    contention and degrade performance when using DataLoader multiprocessing.
    """
    mock_cv2 = MagicMock()
    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        default_worker_init_fn(0)
    mock_cv2.setNumThreads.assert_called_once_with(0)


def test_worker_init_seeds_rng_deterministically() -> None:
    """
    Test that default_worker_init_fn seeds the random number generators (RNGs) for numpy and random
    in a deterministic way based on the worker ID, so that multiple calls with the same worker ID produce the same RNG state.
    """
    mock_cv2 = MagicMock()
    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        torch.manual_seed(42)
        default_worker_init_fn(0)
        np_state_a = int(np.random.get_state()[1][0])  # type: ignore[index]
        rng_state_a = random.getstate()[1][0]

        torch.manual_seed(42)
        default_worker_init_fn(0)
        np_state_b = int(np.random.get_state()[1][0])  # type: ignore[index]
        rng_state_b = random.getstate()[1][0]

    assert np_state_a == np_state_b
    assert rng_state_a == rng_state_b
