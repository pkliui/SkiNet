import pytest
import torch
import segmentation_models_pytorch as smp

from SkiNet.Utils.experiment_keys import LossFunctionKey
from SkiNet.ML.training.build_loss import build_loss


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal batch of logits and binary masks for loss forward pass testing."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    return logits, mask


# --- Return type tests ---

def test_build_loss_bce_returns_correct_type() -> None:
    """build_loss(BCE) should return BCEWithLogitsLoss."""
    loss = build_loss(LossFunctionKey.BCE)
    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)


def test_build_loss_dice_returns_correct_type() -> None:
    """build_loss(DICE) should return smp DiceLoss."""
    loss = build_loss(LossFunctionKey.DICE)
    assert isinstance(loss, smp.losses.DiceLoss)


def test_build_loss_bce_dice_returns_nn_module() -> None:
    """build_loss(BCE_DICE) should return a torch.nn.Module."""
    loss = build_loss(LossFunctionKey.BCE_DICE)
    assert isinstance(loss, torch.nn.Module)


# --- Forward pass tests ---

def test_build_loss_bce_forward(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """BCE loss forward pass should return a finite scalar."""
    logits, mask = batch
    loss = build_loss(LossFunctionKey.BCE)
    result = loss(logits, mask)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_build_loss_dice_forward(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Dice loss forward pass should return a finite scalar."""
    logits, mask = batch
    loss = build_loss(LossFunctionKey.DICE)
    result = loss(logits, mask)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_build_loss_bce_dice_forward(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """BCE+Dice loss forward pass should return a finite scalar."""
    logits, mask = batch
    loss = build_loss(LossFunctionKey.BCE_DICE)
    result = loss(logits, mask)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_build_loss_bce_dice_is_average_of_components(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """BCE+Dice loss should equal 0.5*BCE + 0.5*Dice."""
    logits, mask = batch
    bce = torch.nn.BCEWithLogitsLoss()(logits, mask)
    dice = smp.losses.DiceLoss(mode="binary")(logits, mask)
    expected = 0.5 * bce + 0.5 * dice

    combined = build_loss(LossFunctionKey.BCE_DICE)(logits, mask)
    assert torch.isclose(combined, expected, atol=1e-6)


# --- Gradient flow tests ---

@pytest.mark.parametrize("loss_key", list(LossFunctionKey))
def test_build_loss_gradients_flow(
    loss_key: LossFunctionKey,
    batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """All loss functions should produce gradients on logits."""
    logits, mask = batch
    logits = logits.requires_grad_(True)
    loss = build_loss(loss_key)
    result = loss(logits, mask)
    result.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# --- Error handling ---

def test_build_loss_invalid_key_raises() -> None:
    """build_loss should raise ValueError for unsupported keys."""
    with pytest.raises(ValueError, match="Unsupported loss"):
        build_loss("invalid_key")  # type: ignore[arg-type]
