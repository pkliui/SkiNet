import os
from typing import cast, Dict, Any
import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.Utils.experiment_keys import LossFunctionKey, MetricsKey


def test_train_config_defaults_are_valid() -> None:
    """
    Test that the default values of TrainConfig are valid and do not raise any validation errors.
    """
    cfg = TrainConfig()

    assert cfg.log_dir == "experiment_logs"
    assert cfg.experiment_name == "unet2d_experiment"
    assert cfg.run_test_after_fit is False
    assert cfg.test_on_val_split is False
    assert cfg.batch_size == 8
    assert cfg.num_workers is not None  # resolves to os.cpu_count()
    assert cfg.pin_memory is not None  # auto-resolved from accelerator
    assert cfg.prefetch_factor is None
    assert cfg.cache_in_ram is True
    assert cfg.loss_name == LossFunctionKey.BCE_DICE
    assert cfg.optimizer_name == "adamw"
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 1e-4
    assert cfg.seed == 42
    assert cfg.deterministic is True
    assert cfg.max_epochs == 1
    assert cfg.accelerator == "auto"
    assert cfg.devices == "auto"
    assert cfg.precision is not None  # auto-resolved from accelerator at construction time
    assert cfg.log_every_n_steps == 1
    assert cfg.check_val_every_n_epoch == 1
    assert cfg.num_sanity_val_steps == 0
    assert cfg.system_metrics_interval_sec == 5.0
    assert cfg.use_mlflow_logger is False
    assert cfg.use_checkpoint is False
    assert cfg.use_early_stopping is False
    assert cfg.use_litlogger_logger is False
    assert cfg.early_stopping_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD
    assert cfg.early_stopping_config.mode == "max"
    assert cfg.early_stopping_config.min_delta == 0.0
    assert cfg.early_stopping_config.patience == 5
    assert cfg.early_stopping_config.strict is True
    assert cfg.early_stopping_config.check_finite is True
    assert cfg.checkpoint_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD
    assert cfg.checkpoint_config.mode == "max"
    assert cfg.checkpoint_config.save_top_k == 1
    assert cfg.checkpoint_config.save_last is True
    assert cfg.checkpoint_config.filename == "epoch{epoch:03d}"
    assert cfg.litlogger_config.teamspace is None
    assert cfg.litlogger_config.log_model is False
    assert cfg.litlogger_config.save_logs is False
    assert cfg.mlflow_config.fallback_to_local_mlflow is False
    assert cfg.mlflow_config.tracking_uri is None
    assert cfg.mlflow_config.log_model == "all"
    assert cfg.mlflow_config.log_model_summary is True
    assert cfg.use_lr_scheduler is True
    assert cfg.lr_scheduler_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD
    assert cfg.lr_scheduler_config.mode == "max"
    assert cfg.lr_scheduler_config.patience == 5
    assert cfg.lr_scheduler_config.factor == 0.5


@pytest.mark.parametrize(
    ("kwargs", "expected_uri"),
    [
        (
            {
                "use_mlflow_logger": True,
                "mlflow_config": {"tracking_uri": "http://127.0.0.1:5000"},
            },
            "http://127.0.0.1:5000",
        ),
    ],
)
def test_train_config_mlflow_uri_accepted_when_flag_is_true(kwargs: dict, expected_uri: str | None) -> None:
    """
    Test that TrainConfig correctly accepts the mlflow tracking_uri based on the use_mlflow_logger flag.
    The test_train_config_defaults_are_valid covers the case for use_mlflow_logger=False
    """
    cfg = TrainConfig(**kwargs)
    assert cfg.mlflow_config.tracking_uri == expected_uri


def test_train_config_requires_tracking_uri_when_mlflow_enabled() -> None:
    """
    Test that TrainConfig raises a ValidationError when use_mlflow_logger is True
    but no tracking_uri is provided in mlflow_config.
    """
    with pytest.raises(ValidationError, match="tracking_uri"):
        TrainConfig(use_mlflow_logger=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown_field": True},
        {"mlflow_logger": {"tracking_uri": "http://127.0.0.1:5000"}},
        {"checkppint_config": {"filename": "best"}},
        {"use_litlogger": True},
    ],
)
def test_train_config_rejects_unknown_top_level_fields(kwargs: dict) -> None:
    """
    Test that TrainConfig raises a ValidationError when unknown top-level fields are provided in the input dictionary.
    """
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainConfig(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("batch_size", 0),
        ("num_workers", -1),
        ("lr", 0.0),
        ("weight_decay", -1e-4),
        ("seed", -1),
        ("max_epochs", 0),
        ("log_every_n_steps", 0),
        ("system_metrics_interval_sec", 0.0),
        ("prefetch_factor", 0)
    ],
)
def test_train_config_rejects_invalid_scalar_bounds(field_name: str, value: int | float) -> None:
    """
    Test that TrainConfig raises a ValidationError when scalar fields are set to values that violate
     their defined bounds (e.g., batch_size < 1, lr <= 0, etc.).
    """
    with pytest.raises(ValidationError, match=field_name):
        # cast the single-key dict to a loosely typed mapping so mypy accepts ** expansion
        TrainConfig(**cast(Dict[str, Any], {field_name: value}))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "checkpoint_config": {
                    "monitor": "val_best_dice_at_threshold",
                    "mode": "max",
                    "save_top_k": 1,
                    "save_last": False,
                    "filename": "best",
                }
            },
            {
                "monitor": MetricsKey.VAL_BEST_DICE_AT_THRESHOLD,
                "mode": "max",
                "save_top_k": 1,
                "save_last": False,
                "filename": "best",
            },
        ),
        (
            {
                "early_stopping_config": {
                    "monitor": "val_best_dice_at_threshold",
                    "patience": 9,
                    "min_delta": 0.2,
                }
            },
            {
                "monitor": MetricsKey.VAL_BEST_DICE_AT_THRESHOLD,
                "patience": 9,
                "min_delta": 0.2,
            },
        ),
        (
            {
                "mlflow_config": {
                    "tracking_uri": "http://127.0.0.1:5000",
                    "fallback_to_local_mlflow": True,
                    "log_model": False,
                    "log_model_summary": False,
                }
            },
            {
                "tracking_uri": "http://127.0.0.1:5000",
                "fallback_to_local_mlflow": True,
                "log_model": False,
                "log_model_summary": False,
            },
        ),
        (
            {
                "lr_scheduler_config": {
                    "monitor": "val_best_dice_at_threshold",
                    "mode": "max",
                    "patience": 2,
                    "factor": 0.25,
                }
            },
            {
                "monitor": MetricsKey.VAL_BEST_DICE_AT_THRESHOLD,
                "mode": "max",
                "patience": 2,
                "factor": 0.25,
            },
        ),
        (
            {
                "litlogger_config": {
                    "teamspace": "teamspace-1",
                    "log_model": True,
                    "save_logs": True,
                    "checkpoint_name": "latest",
                }
            },
            {
                "teamspace": "teamspace-1",
                "log_model": True,
                "save_logs": True,
                "checkpoint_name": "latest",
            },
        ),
    ],
)
def test_train_config_parses_nested_configs(kwargs: dict, expected: dict) -> None:
    """
    Test that TrainConfig correctly parses nested configuration dictionaries for checkpoint_config, early_stopping_config,
    mlflow_config, and litlogger_config.
    """
    cfg = TrainConfig(**kwargs)

    nested_config_name = next(iter(kwargs))
    nested_config = getattr(cfg, nested_config_name)
    for key, value in expected.items():
        assert getattr(nested_config, key) == value


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr_scheduler_config": {"patience": -1}}, "patience"),
        ({"lr_scheduler_config": {"factor": 0.0}}, "factor"),
        ({"lr_scheduler_config": {"factor": 1.0}}, "factor"),
    ],
)
def test_train_config_rejects_invalid_lr_scheduler_values(kwargs: dict, match: str) -> None:
    """
    Test that TrainConfig rejects invalid ReduceOnPlateau scheduler values.
    """
    with pytest.raises(ValidationError, match=match):
        TrainConfig(**kwargs)

# ------ Test loss config ------


def test_train_config_default_loss_name() -> None:
    """TrainConfig default loss_name should be BCE_DICE."""
    cfg = TrainConfig()
    assert cfg.loss_name == LossFunctionKey.BCE_DICE


@pytest.mark.parametrize("loss_name,expected_key", [
    ("bce", LossFunctionKey.BCE),
    ("dice", LossFunctionKey.DICE),
    ("bce_dice", LossFunctionKey.BCE_DICE),
])
def test_train_config_valid_loss_names(loss_name: str, expected_key: LossFunctionKey) -> None:
    """TrainConfig should accept all valid LossFunctionKey string values and coerce to enum.
    Pydantic coerces strings at runtime (loss_name: str will become loss_name: LossFunctionKey) """
    cfg = TrainConfig(loss_name=loss_name)  # type: ignore[arg-type]
    assert cfg.loss_name == expected_key


def test_train_config_invalid_loss_name_raises() -> None:
    """TrainConfig should reject unknown loss_name values with ValidationError.
    Pydantic coerces strings at runtime (loss_name: str will become loss_name: LossFunctionKey)"""
    with pytest.raises(ValidationError, match="loss_name"):
        TrainConfig(loss_name="some_loss")  # type: ignore[arg-type]


def test_train_config_loss_name_in_defaults_assertion() -> None:
    """test_train_config_defaults_are_valid should also cover loss_name — add this assertion there."""
    cfg = TrainConfig()
    assert cfg.loss_name == LossFunctionKey.BCE_DICE

# ------ Test cache_in_ram ------


@pytest.mark.parametrize("value", [True, False])
def test_train_config_cache_in_ram_accepts_bool(value: bool) -> None:
    """cache_in_ram should accept True and False and round-trip the value."""
    cfg = TrainConfig(cache_in_ram=value)
    assert cfg.cache_in_ram is value


# ------ Test use_lr_scheduler ------


@pytest.mark.parametrize("value", [True, False])
def test_train_config_use_lr_scheduler_accepts_bool(value: bool) -> None:
    """use_lr_scheduler should accept True and False and round-trip the value."""
    cfg = TrainConfig(use_lr_scheduler=value)
    assert cfg.use_lr_scheduler is value


# ------ Test prefetch_factor validator ------


def test_train_config_prefetch_factor_set_to_none_when_num_workers_is_zero() -> None:
    """
    validate_prefetch_factor should silently coerce prefetch_factor to None
    when num_workers=0, regardless of the supplied value.
    """
    cfg = TrainConfig(num_workers=0, prefetch_factor=4)
    assert cfg.prefetch_factor is None


def test_train_config_prefetch_factor_nulled_by_zero_workers(caplog: pytest.LogCaptureFixture) -> None:
    """
    TrainConfig got num_workers=0, so prefetch_factor=4
    should be coerced to None and a warning should be emitted.
    """
    import logging
    with caplog.at_level(logging.WARNING, logger="SkiNet.ML.configs.train_configs.train_config"):
        cfg = TrainConfig(num_workers=0, prefetch_factor=4)
    assert cfg.prefetch_factor is None
    assert "prefetch_factor" in caplog.text


def test_train_config_prefetch_factor_default_values() -> None:
    """
    Default TrainConfig has prefetch_factor=None
    """
    cfg = TrainConfig()  # prefetch_factor=None by default
    assert cfg.prefetch_factor is None


def test_train_config_prefetch_factor_minimum_valid_with_workers() -> None:
    """
    prefetch_factor=1 is the minimum valid value (ge=1) and should be accepted
    when num_workers > 0.
    """
    cfg = TrainConfig(num_workers=1, prefetch_factor=1)
    assert cfg.prefetch_factor == 1


# ------ Test precision auto-resolution ------


@pytest.mark.parametrize(
    ("accelerator", "expected_precision"),
    [
        ("gpu", "16-mixed"),
        ("cuda", "16-mixed"),
        ("mps", "16-mixed"),
        ("cpu", "32-true"),
    ],
)
def test_precision_auto_set_from_explicit_accelerator(accelerator: str, expected_precision: str) -> None:
    """
    When precision is not set, it should be resolved from the explicit accelerator value.
    """
    cfg = TrainConfig(accelerator=accelerator)
    assert cfg.precision == expected_precision


def test_explicit_precision_is_not_overridden() -> None:
    """
    When precision is explicitly set, the validator must not overwrite it.
    """
    cfg = TrainConfig(accelerator="gpu", precision="32-true")
    assert cfg.precision == "32-true"


def test_precision_auto_set_from_auto_accelerator_is_not_none() -> None:
    """
    accelerator='auto' should resolve to a non-None precision on any machine
    where torch can detect hardware.
    """
    cfg = TrainConfig(accelerator="auto")
    assert cfg.precision is not None


# ------ Test num_workers auto-resolution ------


@pytest.mark.parametrize("devices", [1, "auto"])
def test_num_workers_auto_set_from_cpu_count(devices: int | str) -> None:
    """
    When num_workers is not set and devices is 1 or 'auto' (no DDP), it resolves
    to os.cpu_count() with no per-device division.
    """
    cfg = TrainConfig(devices=devices)
    assert cfg.num_workers == os.cpu_count()


def test_num_workers_auto_set_ddp_aware() -> None:
    """
    When devices is an int > 1, num_workers is divided among DDP processes
    to avoid oversubscribing CPUs.
    """
    devices = 4
    cpu_count = os.cpu_count()
    cfg = TrainConfig(devices=devices)
    if cpu_count is not None:
        assert cfg.num_workers == max(1, cpu_count // devices)


def test_explicit_num_workers_is_not_overridden() -> None:
    """
    When num_workers is explicitly set, the validator must not overwrite it.
    """
    cfg = TrainConfig(num_workers=2)
    assert cfg.num_workers == 2


# ------ Test pin_memory auto-resolution ------


@pytest.mark.parametrize(
    ("accelerator", "expected_pin_memory"),
    [
        ("gpu", True),
        ("cuda", True),
        ("mps", False),
        ("cpu", False),
    ],
)
def test_pin_memory_auto_set_from_explicit_accelerator(
    accelerator: str, expected_pin_memory: bool
) -> None:
    """
    When pin_memory is not set, it should be True iff effective accelerator is GPU/CUDA.
    MPS and CPU do not benefit from pinned memory.
    """
    cfg = TrainConfig(accelerator=accelerator)
    assert cfg.pin_memory is expected_pin_memory


def test_explicit_pin_memory_is_not_overridden() -> None:
    """
    When pin_memory is explicitly set to False, the validator must not overwrite it
    even when the accelerator would otherwise yield True.
    """
    cfg = TrainConfig(accelerator="gpu", pin_memory=False)
    assert cfg.pin_memory is False
