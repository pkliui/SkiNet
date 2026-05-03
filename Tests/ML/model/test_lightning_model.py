from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from SkiNet.ML.configs.train_configs.train_config import ReduceOnPlateauConfig
from SkiNet.ML.model.lightning_model import LightningModel


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def lm() -> LightningModel:
    """Minimal LightningModel with nn.Identity backbone and BCE loss for unit testing."""
    return LightningModel(
        model=nn.Identity(),
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=1e-3,
        optimizer_name="adam",
        weight_decay=0.0,
        lr_scheduler_config=ReduceOnPlateauConfig(),
    )


# ---------------------------------------------------------------------------
# _tensor_debug_summary
# ---------------------------------------------------------------------------

def test_tensor_debug_summary_finite_tensor() -> None:
    """Checks that a fully finite tensor produces a summary containing the tensor
    name, finite count matching total count, and both min/max statistics."""
    t = torch.tensor([1.0, 2.0, 3.0])
    result = LightningModel._tensor_debug_summary("x", t)
    assert "x" in result
    assert "finite=3/3" in result
    assert "min=" in result
    assert "max=" in result


def test_tensor_debug_summary_all_inf() -> None:
    """Checks that a tensor containing only infinite values reports finite=0/N
    and omits min/max (which are undefined when no finite elements exist)."""
    t = torch.tensor([float("inf"), float("inf")])
    result = LightningModel._tensor_debug_summary("y", t)
    assert "y" in result
    assert "finite=0/2" in result
    assert "min=" not in result
    assert "max=" not in result


def test_tensor_debug_summary_mixed_finite_and_nan() -> None:
    """Checks that a tensor with a mix of finite values and NaN correctly counts
    only the finite elements and still reports min/max over the finite subset."""
    t = torch.tensor([1.0, float("nan"), 3.0])
    result = LightningModel._tensor_debug_summary("z", t)
    assert "finite=2/3" in result
    assert "min=" in result
    assert "max=" in result


def test_tensor_debug_summary_includes_shape_and_dtype() -> None:
    """Checks that the summary string encodes both tensor shape and dtype so that
    downstream error messages contain enough context for diagnosis."""
    t = torch.zeros(2, 3, dtype=torch.float32)
    result = LightningModel._tensor_debug_summary("t", t)
    assert "shape=(2, 3)" in result
    assert "float32" in result


# ---------------------------------------------------------------------------
# _prepare_mask
# ---------------------------------------------------------------------------

def test_prepare_mask_binary_float_passthrough() -> None:
    """Checks that a float32 mask with strictly binary values (0.0 and 1.0) is
    returned unchanged — no conversion or modification should occur."""
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
    out = LightningModel._prepare_mask(mask)
    assert out.dtype == torch.float32
    assert torch.equal(out, mask)


def test_prepare_mask_thresholds_soft_values() -> None:
    """Soft values from bilinear augmentation (e.g. Albumentations Affine) are
    thresholded at 0.5 rather than rejected, matching the validation sweep."""
    mask = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
    out = LightningModel._prepare_mask(mask)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_prepare_mask_int_converted_to_float() -> None:
    """Checks that integer masks (e.g. loaded from a PIL image as uint8/int64)
    are cast to float32 with values preserved, so downstream loss functions
    that require floating-point inputs don't error."""
    mask = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    out = LightningModel._prepare_mask(mask)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_prepare_mask_out_of_range_value_treated_as_foreground() -> None:
    """Values > 1 (e.g. un-normalised pixels) pass the >= 0.5 threshold and
    become 1.0; callers are responsible for normalising before this point."""
    mask = torch.tensor([0.0, 2.0])
    out = LightningModel._prepare_mask(mask)
    assert torch.equal(out, torch.tensor([0.0, 1.0]))


def test_prepare_mask_max_value_one_passes() -> None:
    """Checks that a mask with values exactly at the boundary [0, 1] is accepted
    without raising, i.e. the guard is inclusive of the valid range."""
    mask = torch.tensor([0.0, 1.0])
    out = LightningModel._prepare_mask(mask)
    assert torch.equal(out, mask)


# ---------------------------------------------------------------------------
# _get_probs_and_preds
# ---------------------------------------------------------------------------

def test_get_probs_and_preds_return_keys() -> None:
    """Checks that the returned dict contains exactly the two expected keys.
    target is no longer returned — callers compute mask.long() themselves."""
    logits = torch.randn(1, 1, 2, 2)
    result = LightningModel._get_probs_and_preds(logits, torch.tensor(0.5))
    assert set(result.keys()) == {"probs", "preds"}


def test_get_probs_and_preds_probs_in_unit_interval() -> None:
    """Checks that applying sigmoid to arbitrary logits always yields probabilities
    strictly inside [0, 1], which is required by the threshold comparison."""
    logits = torch.randn(2, 1, 4, 4)
    result = LightningModel._get_probs_and_preds(logits, torch.tensor(0.5))
    assert result["probs"].min().item() >= 0.0
    assert result["probs"].max().item() <= 1.0


@pytest.mark.parametrize("logit,threshold,expected_pred", [
    (2.0, 0.5, 1),   # sigmoid(2.0) ≈ 0.88 >= 0.5  → positive
    (-2.0, 0.5, 0),  # sigmoid(-2.0) ≈ 0.12 < 0.5  → negative
    (0.0, 0.4, 1),   # sigmoid(0.0) = 0.5 >= 0.4   → positive (threshold below boundary)
    (0.0, 0.6, 0),   # sigmoid(0.0) = 0.5 < 0.6    → negative (threshold above boundary)
])
def test_get_probs_and_preds_thresholding(logit: float, threshold: float, expected_pred: int) -> None:
    """Checks that the prediction is 1 when sigmoid(logit) >= threshold and 0 otherwise,
    including the boundary case where sigmoid output equals exactly 0.5."""
    logits = torch.tensor([logit])
    result = LightningModel._get_probs_and_preds(logits, torch.tensor(threshold))
    assert result["preds"].item() == expected_pred


def test_get_probs_and_preds_preds_are_long() -> None:
    """Checks that preds are returned as torch.long, which is the dtype expected
    by torchmetrics BinaryF1Score / BinaryJaccardIndex."""
    logits = torch.randn(4)
    result = LightningModel._get_probs_and_preds(logits, torch.tensor(0.5))
    assert result["preds"].dtype == torch.long


# ---------------------------------------------------------------------------
# _raise_if_non_finite
# ---------------------------------------------------------------------------

def test_raise_if_non_finite_passes_on_finite_tensor(lm: LightningModel) -> None:
    """Checks that a fully finite tensor does not raise, i.e. the guard is a no-op
    in the normal training case and adds no overhead-inducing exception path."""
    lm._raise_if_non_finite("img", torch.tensor([1.0, 2.0, 3.0]), batch_idx=0)


def test_raise_if_non_finite_raises_on_inf(lm: LightningModel) -> None:
    """Checks that a tensor containing at least one +inf or -inf triggers a ValueError.
    Inf values in activations indicate exploding gradients or loss instability."""
    with pytest.raises(ValueError, match="Non-finite"):
        lm._raise_if_non_finite("img", torch.tensor([1.0, float("inf")]), batch_idx=0)


def test_raise_if_non_finite_raises_on_nan(lm: LightningModel) -> None:
    """Checks that a tensor containing NaN raises a ValueError.
    NaN propagates silently through most operations and corrupts all downstream metrics."""
    with pytest.raises(ValueError, match="Non-finite"):
        lm._raise_if_non_finite("img", torch.tensor([float("nan")]), batch_idx=0)


def test_raise_if_non_finite_error_contains_batch_idx(lm: LightningModel) -> None:
    """Checks that the batch index is included in the error message so that a failing
    batch can be identified and reproduced during debugging."""
    with pytest.raises(ValueError, match="batch 7"):
        lm._raise_if_non_finite("img", torch.tensor([float("inf")]), batch_idx=7)


def test_raise_if_non_finite_error_contains_tensor_name(lm: LightningModel) -> None:
    """Checks that the tensor name (e.g. 'train/logits') appears in the error message
    so it is immediately clear which stage of the forward pass produced the bad values."""
    with pytest.raises(ValueError, match="train/logits"):
        lm._raise_if_non_finite("train/logits", torch.tensor([float("nan")]), batch_idx=0)


# ---------------------------------------------------------------------------
# _compute_and_log_threshold_search_metrics_for_sigmoid
# ---------------------------------------------------------------------------

def test_threshold_search_returns_early_on_empty_probs(lm: LightningModel) -> None:
    """Checks that the method exits immediately when no validation probabilities have
    been collected (empty _val_probs). Without this guard the method would call
    self.log, which raises without an attached trainer."""
    lm._compute_and_log_threshold_search_metrics_for_sigmoid()


def test_threshold_search_logs_sentinel_on_single_class(lm: LightningModel) -> None:
    """Checks that when all validation targets belong to a single class (all 1s here),
    the method logs val_best_dice_at_threshold=0.0 as a sentinel. This prevents
    KeyError in Optuna which expects the metric key to always exist."""
    lm._val_probs = [torch.ones(10)]
    lm._val_masks = [torch.ones(10)]
    with patch.object(lm, "log") as mock_log:
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    mock_log.assert_called_once_with(
        "val_best_dice_at_threshold", 0.0,
        on_step=False, on_epoch=True, prog_bar=False, logger=True,
    )


def test_threshold_search_clears_lists_on_single_class(lm: LightningModel) -> None:
    """Checks that _val_probs and _val_masks are cleared even in the single-class
    early-return branch, preventing stale data from bleeding into the next epoch."""
    lm._val_probs = [torch.ones(10)]
    lm._val_masks = [torch.ones(10)]
    with patch.object(lm, "log"):
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    assert lm._val_probs == []
    assert lm._val_masks == []


def test_threshold_search_updates_optimal_threshold(lm: LightningModel) -> None:
    """Checks that optimal_threshold is updated from its initial value of 0.5 to the
    threshold that maximises Dice on the given validation data. With perfect separation
    (negatives have prob 0.1/0.2, positives have prob 0.8/0.9) find_best_threshold
    returns the highest threshold at which Dice=1.0, which is the grid point nearest
    to 0.8 in linspace(1.0, 0.0, 51). pytest.approx handles float32 rounding."""
    probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
    masks = torch.tensor([0.0, 0.0, 1.0, 1.0])
    lm._val_probs = [probs]
    lm._val_masks = [masks]
    with patch.object(lm, "log"), \
            patch.object(lm.val_dice, "compute", return_value=torch.tensor(0.5)):
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    assert lm.optimal_threshold.item() == pytest.approx(0.8, abs=0.02)


def test_threshold_search_clears_lists_after_update(lm: LightningModel) -> None:
    """Checks that _val_probs and _val_masks are cleared after a successful threshold
    sweep so that the next validation epoch starts from an empty accumulation buffer."""
    lm._val_probs = [torch.tensor([0.1, 0.9])]
    lm._val_masks = [torch.tensor([0.0, 1.0])]
    with patch.object(lm, "log"), \
            patch.object(lm.val_dice, "compute", return_value=torch.tensor(0.5)):
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    assert lm._val_probs == []
    assert lm._val_masks == []


def test_threshold_search_logs_dice_threshold_gain(lm: LightningModel) -> None:
    """Checks that val_dice_threshold_gain is logged as best_dice - fixed_thr_dice.

    Probabilities are designed so the fixed 0.5 threshold yields Dice=1.0 (perfect
    separation) and the swept best Dice is also 1.0, giving gain=0.0. A second
    assertion uses a case where the fixed threshold misclassifies one sample so
    gain > 0 to verify the sign and computation are correct.
    """
    # Case 1: fixed threshold already perfect → gain == 0
    probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
    masks = torch.tensor([0.0, 0.0, 1.0, 1.0])
    lm._val_probs = [probs]
    lm._val_masks = [masks]
    logged: dict[str, float] = {}
    with patch.object(lm, "log", side_effect=lambda key, val, **_: logged.update({key: float(val)})):
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    assert "val_dice_threshold_gain" in logged
    assert logged["val_dice_threshold_gain"] == pytest.approx(0.0, abs=1e-5)

    # Case 2: positive prob (0.6) straddles 0.5 → fixed threshold classifies it as
    # positive when the true label is 0, so fixed-thr Dice < best Dice → gain > 0
    lm._val_probs = [torch.tensor([0.1, 0.6, 0.9])]
    lm._val_masks = [torch.tensor([0.0, 0.0, 1.0])]
    logged2: dict[str, float] = {}
    with patch.object(lm, "log", side_effect=lambda key, val, **_: logged2.update({key: float(val)})):
        lm._compute_and_log_threshold_search_metrics_for_sigmoid()
    assert logged2["val_dice_threshold_gain"] > 0.0
