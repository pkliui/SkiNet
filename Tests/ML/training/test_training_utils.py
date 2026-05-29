import pytest
import torch

from SkiNet.ML.training.training_utils import find_best_threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_separation() -> tuple[torch.Tensor, torch.Tensor]:
    """Probs cleanly separated: negatives < 0.3, positives > 0.6."""
    probs = torch.tensor([0.1, 0.2, 0.7, 0.8])
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    return probs, targets


@pytest.fixture
def random_probs_targets() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    probs = torch.rand(200)
    targets = (torch.rand(200) > 0.5).float()
    return probs, targets


# ---------------------------------------------------------------------------
# Return-type / contract tests
# ---------------------------------------------------------------------------

def test_returns_basics(perfect_separation: tuple[torch.Tensor, torch.Tensor]) -> None:
    result = find_best_threshold(*perfect_separation)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"best_threshold", "best_dice"}
    assert isinstance(result["best_threshold"], float)
    assert isinstance(result["best_dice"], float)


def test_reurns_are_in_unit_interval(random_probs_targets: tuple[torch.Tensor, torch.Tensor]) -> None:
    result = find_best_threshold(*random_probs_targets)
    assert 0.0 <= result["best_threshold"] <= 1.0
    assert 0.0 <= result["best_dice"] <= 1.0


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_perfect_separation_yields_dice_one(
    perfect_separation: tuple[torch.Tensor, torch.Tensor]
) -> None:
    result = find_best_threshold(*perfect_separation)
    assert result["best_dice"] == pytest.approx(1.0)


def test_perfect_separation_threshold_between_classes(
    perfect_separation: tuple[torch.Tensor, torch.Tensor]
) -> None:
    # Any threshold in (0.2, 0.7] correctly separates the two classes.
    result = find_best_threshold(*perfect_separation)
    assert 0.2 < result["best_threshold"] <= 0.7


def test_highest_threshold_selected_on_tie() -> None:
    """
    probs=[0.1, 0.2, 0.7, 0.8], targets=[0, 0, 1, 1], n_thresholds=51
    linspace(1.0, 0.0, 51) step=-0.02; rows show where preds change:

    threshold   | preds          | tp | fp | fn | dice
    ------------|----------------|----|----|----|-----
    1.00-0.82   | [F, F, F, F]   |  0 |  0 |  2 | 0.00
    0.80-0.72   | [F, F, F, T]   |  1 |  0 |  1 | 0.67
    0.70        | [F, F, T, T]   |  2 |  0 |  0 | 1.00  ← first max
    0.68-0.22   | [F, F, T, T]   |  2 |  0 |  0 | 1.00
    0.20-0.12   | [F, T, T, T]   |  2 |  1 |  0 | 0.80
    0.10-0.00   | [T, T, T, T]   |  2 |  2 |  0 | 0.67

    Thresholds 0.70 through 0.22 all tie at dice=1.0; argmax picks the first
    in the descending list → best_threshold=0.70.
    """
    probs = torch.tensor([0.1, 0.2, 0.7, 0.8])
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    result = find_best_threshold(probs, targets, n_thresholds=51)
    assert result["best_threshold"] == pytest.approx(0.70, abs=1e-5)
    assert result["best_dice"] == pytest.approx(1.0)


def test_all_targets_positive() -> None:
    """
    probs=[0.6, 0.7, 0.8, 0.9], targets=[1, 1, 1, 1], n_thresholds=51 (default)
    linspace(1.0, 0.0, 51) step=-0.02; rows show where preds change:

    threshold | preds          | tp | fp | fn | dice
    ----------|----------------|----|----|----|-----
    1.00      | [F, F, F, F]   |  0 |  0 |  4 | 0.00
    0.90      | [F, F, F, T]   |  1 |  0 |  3 | 0.40
    0.80      | [F, F, T, T]   |  2 |  0 |  2 | 0.67
    0.70      | [F, T, T, T]   |  3 |  0 |  1 | 0.86
    0.60      | [T, T, T, T]   |  4 |  0 |  0 | 1.00  ← first max (min prob = 0.6)

    fp=0 throughout (all targets are positive). best_threshold=0.60, best_dice=1.0
    """
    probs = torch.tensor([0.6, 0.7, 0.8, 0.9])
    targets = torch.ones(4)
    result = find_best_threshold(probs, targets)
    assert result["best_dice"] == pytest.approx(1.0)
    assert result["best_threshold"] == pytest.approx(0.6, abs=1e-6)


def test_all_targets_negative_dice_is_zero() -> None:
    # No true positives possible → dice must be 0 at every threshold.
    probs = torch.rand(50)
    targets = torch.zeros(50)
    result = find_best_threshold(probs, targets)
    assert result["best_dice"] == pytest.approx(0.0, abs=1e-6)


def test_all_probs_zero() -> None:
    """
    probs=[0, ..., 0] (10), targets=[1, ..., 1] (10), n_thresholds=51 (default)

    threshold   | preds           | tp | fp | fn | dice
    ------------|-----------------|----|----|----|-----
    1.00-0.02   | [F, ..., F]     |  0 |  0 | 10 | 0.00
    0.00        | [T, ..., T]     | 10 |  0 |  0 | 1.00  ← only threshold reaching dice=1.0

    best_threshold=0.00, best_dice=1.0
    """
    probs = torch.zeros(10)
    targets = torch.ones(10)
    result = find_best_threshold(probs, targets)
    assert result["best_threshold"] == pytest.approx(0.0)
    assert result["best_dice"] == pytest.approx(1.0)


def test_all_probs_one() -> None:
    """
    probs=[1, ..., 1] (10), targets=[1, ..., 1] (10), n_thresholds=51 (default)
    1.0 >= t for every t in [0, 1], so all thresholds predict all samples positive.

    threshold   | preds           | tp | fp | fn | dice
    ------------|-----------------|----|----|----|-----
    1.00        | [T, ..., T]     | 10 |  0 |  0 | 1.00  ← argmax picks this (first in descending list)
    0.98-0.00   | [T, ..., T]     | 10 |  0 |  0 | 1.00

    best_threshold=1.00, best_dice=1.0
    """
    probs = torch.ones(10)
    targets = torch.ones(10)
    result = find_best_threshold(probs, targets)
    assert result["best_dice"] == pytest.approx(1.0)
    assert result["best_threshold"] == pytest.approx(1.0)


def test_single_element_true_positive() -> None:
    """
    probs=[0.9], targets=[1.0], n_thresholds=11
    thresholds=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    threshold | pred | tp | fp | fn | dice
    ----------|------|----|----|----|-----
    1.0       |  F   |  0 |  0 |  1 | 0.00
    0.9       |  T   |  1 |  0 |  0 | 1.00  ← first max
    0.8-0.0   |  T   |  1 |  0 |  0 | 1.00

    best_threshold=0.9, best_dice=1.0
    """
    probs = torch.tensor([0.9])
    targets = torch.tensor([1.0])
    result = find_best_threshold(probs, targets, n_thresholds=11)
    assert result["best_dice"] == pytest.approx(1.0)
    assert result["best_threshold"] == pytest.approx(0.9)


def test_single_element_true_negative() -> None:
    """
    probs=[0.1], targets=[0.0], n_thresholds=11
    thresholds=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    threshold | pred | tp | fp | fn | dice
    ----------|------|----|----|----|-----
    1.0-0.2   |  F   |  0 |  0 |  0 | 0.00  ← all tied; argmax → index 0 (t=1.0)
    0.1-0.0   |  T   |  0 |  1 |  0 | 0.00

    No positive targets → tp=0 always → dice=0 everywhere.
    best_threshold=1.0, best_dice=0.0
    """
    probs = torch.tensor([0.1])
    targets = torch.tensor([0.0])
    result = find_best_threshold(probs, targets, n_thresholds=11)
    assert result["best_dice"] == pytest.approx(0.0, abs=1e-6)
    assert result["best_threshold"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Parameter / configuration tests
# ---------------------------------------------------------------------------

def test_n_thresholds_one() -> None:
    """
    probs=[0.5, 0.8], targets=[1.0, 1.0], n_thresholds=1
    linspace(1.0, 0.0, 1) = [1.0] — only one threshold evaluated.

    threshold | preds  | tp | fp | fn | dice
    ----------|--------|----|----|----|-----
    1.0       | [F, F] |  0 |  0 |  2 | 0.00

    best_threshold=1.0, best_dice=0.0
    """
    probs = torch.tensor([0.5, 0.8])
    targets = torch.tensor([1.0, 1.0])
    result = find_best_threshold(probs, targets, n_thresholds=1)
    assert result["best_threshold"] == pytest.approx(1.0)


def test_n_thresholds_two() -> None:
    """
    probs=[0.5, 0.8], targets=[1.0, 1.0], n_thresholds=2
    linspace(1.0, 0.0, 2) = [1.0, 0.0] — only boundary thresholds evaluated.

    threshold | preds  | tp | fp | fn | dice
    ----------|--------|----|----|----|-----
    1.0       | [F, F] |  0 |  0 |  2 | 0.00
    0.0       | [T, T] |  2 |  0 |  0 | 1.00  ← max

    best_threshold=0.0, best_dice=1.0
    """
    probs = torch.tensor([0.5, 0.8])
    targets = torch.tensor([1.0, 1.0])
    result = find_best_threshold(probs, targets, n_thresholds=2)
    assert result["best_threshold"] == pytest.approx(0.0)
    assert result["best_dice"] == pytest.approx(1.0)


def test_coarser_grid_still_finds_good_threshold(
    random_probs_targets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    result_fine = find_best_threshold(*random_probs_targets, n_thresholds=101)
    result_coarse = find_best_threshold(*random_probs_targets, n_thresholds=11)
    # Coarser grid may miss the optimum, but must not exceed it.
    assert result_coarse["best_dice"] <= result_fine["best_dice"] + 1e-6


# ---------------------------------------------------------------------------
# Device test
# ---------------------------------------------------------------------------

def test_output_is_cpu_float_regardless_of_input_device(
    perfect_separation: tuple[torch.Tensor, torch.Tensor]
) -> None:
    probs, targets = perfect_separation
    result = find_best_threshold(probs.cpu(), targets.cpu())
    # .item() always returns a plain Python float, not a tensor.
    assert not isinstance(result["best_threshold"], torch.Tensor)
    assert not isinstance(result["best_dice"], torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_works_on_cuda(perfect_separation: tuple[torch.Tensor, torch.Tensor]) -> None:
    probs, targets = perfect_separation
    result = find_best_threshold(probs.cuda(), targets.cuda())
    assert result["best_dice"] == pytest.approx(1.0)
