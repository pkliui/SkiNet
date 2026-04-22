import pytest
from SkiNet.Utils.experiment_keys import LossFunctionKey


# --- LossFunctionKey enum tests ---

def test_loss_function_key_values() -> None:
    """LossFunctionKey should have the expected string values."""
    assert LossFunctionKey.BCE.value == "bce"
    assert LossFunctionKey.DICE.value == "dice"
    assert LossFunctionKey.BCE_DICE.value == "bce_dice"


def test_loss_function_key_from_valid_string() -> None:
    """LossFunctionKey should be constructable from valid string values."""
    assert LossFunctionKey("bce") == LossFunctionKey.BCE
    assert LossFunctionKey("dice") == LossFunctionKey.DICE
    assert LossFunctionKey("bce_dice") == LossFunctionKey.BCE_DICE


def test_loss_function_key_from_invalid_string() -> None:
    """LossFunctionKey should raise ValueError for unknown string values."""
    with pytest.raises(ValueError):
        LossFunctionKey("focal")


def test_loss_function_key_exhaustive() -> None:
    """LossFunctionKey should have exactly 3 members."""
    assert len(LossFunctionKey) == 3
