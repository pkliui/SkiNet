import pytest
from SkiNet.Utils.experiment_keys import LossFunctionKey, MetricsKey


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


# --- MetricsKey ---

def test_metrics_key_values() -> None:
    assert MetricsKey.VAL_BEST_DICE_AT_THRESHOLD.value == "val_best_dice_at_threshold"


def test_metrics_key_from_valid_string() -> None:
    assert MetricsKey("val_best_dice_at_threshold") == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD


def test_metrics_key_from_invalid_string() -> None:
    with pytest.raises(ValueError):
        MetricsKey("val_loss")


def test_metrics_key_exhaustive() -> None:
    assert len(MetricsKey) == 1


def test_metrics_key_default_monitor_returns_correct_member() -> None:
    assert MetricsKey.default_monitor() == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD


def test_metrics_key_default_monitor_is_metrics_key_instance() -> None:
    assert isinstance(MetricsKey.default_monitor(), MetricsKey)
