import pytest
from pydantic import ValidationError
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig


def test_sweep_config_defaults() -> None:
    """SweepConfig should be constructable with no arguments and have sensible defaults."""
    cfg = SweepConfig()

    assert cfg.monitor == "val_best_dice_at_threshold"
    assert cfg.direction == "maximize"
    assert cfg.experiment_name == "optuna_sweep"
    assert 3e-4 in cfg.lr
    assert 1e-4 in cfg.weight_decay
    assert 16 in cfg.batch_size


def test_sweep_config_search_space_contains_all_keys() -> None:
    """search_space property should expose all three tunable parameters."""
    cfg = SweepConfig()
    space = cfg.search_space

    assert set(space.keys()) == {"lr", "weight_decay", "batch_size"}
    assert space["lr"] == cfg.lr
    assert space["weight_decay"] == cfg.weight_decay
    assert space["batch_size"] == cfg.batch_size


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_sweep_config_valid_directions(direction: str) -> None:
    cfg = SweepConfig(direction=direction)
    assert cfg.direction == direction


def test_sweep_config_invalid_direction_raises() -> None:
    """direction must be exactly 'maximize' or 'minimize'."""
    with pytest.raises(ValidationError, match="direction"):
        SweepConfig(direction="max")


def test_sweep_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SweepConfig(unknown_field="oops")  # type: ignore[call-arg]


def test_sweep_config_custom_search_space() -> None:
    """Custom values should be reflected in search_space."""
    cfg = SweepConfig(lr=[1e-3, 1e-4, 1e-5], batch_size=[8, 16])

    assert cfg.search_space["lr"] == [1e-3, 1e-4, 1e-5]
    assert cfg.search_space["batch_size"] == [8, 16]
