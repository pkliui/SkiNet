import pytest
from pydantic import ValidationError
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig
from SkiNet.Utils.experiment_keys import HyperparamKey


def test_sweep_config_defaults() -> None:
    """SweepConfig should build with no arguments and populate the default sweep settings.

    This covers the baseline contract used by the Optuna entrypoint: the default
    monitor, optimization direction, experiment name, and the built-in candidate
    values for weight decay and batch size must all be present.

    LR defaults to a single fixed value (3e-4, determined by the E1 LR search) so
    it is not varied in the sweep unless explicitly overridden in the YAML config.
    """
    cfg = SweepConfig()

    assert cfg.monitor == "val_best_dice_at_threshold"
    assert cfg.direction == "maximize"
    assert cfg.experiment_name == "optuna_sweep"
    assert cfg.lr == [3e-4]
    assert 1e-4 in cfg.weight_decay
    assert 16 in cfg.batch_size
    assert 4 in cfg.num_workers
    assert 2 in cfg.prefetch_factor


def test_sweep_config_search_space_contains_all_keys() -> None:
    """search_space should expose every tunable hyperparameter as copied lists.

    The property should map each ``HyperparamKey`` to the corresponding config
    values, but it must return copies rather than the original list objects so
    callers can inspect or mutate the returned search space safely.
    """
    cfg = SweepConfig()
    space = cfg.search_space

    assert set(space.keys()) == set(HyperparamKey)
    assert space[HyperparamKey.LR] == cfg.lr
    assert space[HyperparamKey.WEIGHT_DECAY] == cfg.weight_decay
    assert space[HyperparamKey.BATCH_SIZE] == cfg.batch_size
    assert space[HyperparamKey.NUM_WORKERS] == cfg.num_workers
    assert space[HyperparamKey.PREFETCH_FACTOR] == cfg.prefetch_factor
    assert space[HyperparamKey.SCHEDULER_TYPE] == cfg.scheduler_type
    #
    # assert hyperparameters in the search_space and in config are not the same objects
    assert space[HyperparamKey.LR] is not cfg.lr
    assert space[HyperparamKey.WEIGHT_DECAY] is not cfg.weight_decay
    assert space[HyperparamKey.BATCH_SIZE] is not cfg.batch_size
    assert space[HyperparamKey.NUM_WORKERS] is not cfg.num_workers
    assert space[HyperparamKey.PREFETCH_FACTOR] is not cfg.prefetch_factor
    assert space[HyperparamKey.SCHEDULER_TYPE] is not cfg.scheduler_type


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_sweep_config_valid_directions(direction: str) -> None:
    """SweepConfig should accept the only two valid Optuna study directions."""
    cfg = SweepConfig(direction=direction)
    assert cfg.direction == direction


def test_sweep_config_invalid_direction_raises() -> None:
    """SweepConfig should reject any direction outside ``maximize`` or ``minimize``.

    The regex constraint on ``direction`` is part of the public validation
    contract, so invalid values should fail model construction with a
    ``ValidationError`` mentioning the field.
    """
    with pytest.raises(ValidationError, match="direction"):
        SweepConfig(direction="max")


def test_sweep_config_forbids_extra_fields() -> None:
    """SweepConfig should reject unknown fields to keep the schema strict.

    This verifies the ``extra='forbid'`` model configuration so misspelled or
    unsupported keys do not silently pass through configuration loading.
    """
    with pytest.raises(ValidationError):
        SweepConfig(unknown_field="oops")  # type: ignore[call-arg]


def test_sweep_config_custom_search_space() -> None:
    """Custom hyperparameter candidates should be surfaced unchanged in search_space.

    When callers override parts of the sweep definition, the exported search
    space should reflect those exact values instead of falling back to defaults.
    """
    cfg = SweepConfig(lr=[1e-3, 1e-4, 1e-5], batch_size=[8, 16], prefetch_factor=[4, 8])

    assert cfg.search_space[HyperparamKey.LR] == [1e-3, 1e-4, 1e-5]
    assert cfg.search_space[HyperparamKey.BATCH_SIZE] == [8, 16]
    assert cfg.search_space[HyperparamKey.PREFETCH_FACTOR] == [4, 8]


def test_sweep_config_search_space_mutation_does_not_affect_config() -> None:
    """Mutating the returned search_space must not mutate the stored config lists.

    ``search_space`` is expected to return defensive copies. This test exercises
    that contract by changing the returned lists and confirming the underlying
    ``SweepConfig`` values remain unchanged.
    """
    cfg = SweepConfig()

    space = cfg.search_space
    space[HyperparamKey.LR].append(1e-5)
    space[HyperparamKey.BATCH_SIZE][0] = 99
    space[HyperparamKey.PREFETCH_FACTOR].append(16)

    assert cfg.lr == [3e-4]
    assert cfg.batch_size == [16, 32]
    assert cfg.prefetch_factor == [2, 4]
