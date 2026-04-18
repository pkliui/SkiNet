from typing import cast, Dict, Any
import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.train_configs.train_config import TrainConfig


def test_train_config_defaults_are_valid() -> None:
    """
    Test that the default values of TrainConfig are valid and do not raise any validation errors.
    """
    cfg = TrainConfig()

    assert cfg.log_dir == "experiment_logs"
    assert cfg.experiment_name == "unet2d_experiment"
    assert cfg.batch_size == 8
    assert cfg.num_workers == 0
    assert cfg.optimizer_name == "adamw"
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 1e-4
    assert cfg.seed == 42
    assert cfg.deterministic is True
    assert cfg.max_epochs == 1
    assert cfg.accelerator == "auto"
    assert cfg.devices == "auto"
    assert cfg.log_every_n_steps == 1
    assert cfg.system_metrics_interval_sec == 5.0
    assert cfg.use_mlflow_logger is False
    assert cfg.use_checkpoint is False
    assert cfg.use_early_stopping is False
    assert cfg.use_litlogger_logger is False
    assert cfg.checkpoint_config.filename == "epoch{epoch:03d}"
    assert cfg.mlflow_config.tracking_uri is None
    assert cfg.lr_scheduler_config.monitor == "val_dice"
    assert cfg.lr_scheduler_config.mode == "max"
    assert cfg.lr_scheduler_config.patience == 5
    assert cfg.lr_scheduler_config.factor == 0.5
    assert cfg.pin_memory is True
    assert cfg.prefetch_factor == 2
    assert cfg.run_test_after_fit is False
    assert cfg.test_on_val_split is False
    assert cfg.check_val_every_n_epoch == 1
    assert cfg.num_sanity_val_steps == 0


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
                    "monitor": "val_metric",
                    "mode": "max",
                    "save_top_k": 1,
                    "save_last": False,
                    "filename": "best",
                }
            },
            {
                "monitor": "val_metric",
                "mode": "max",
                "save_top_k": 1,
                "save_last": False,
                "filename": "best",
            },
        ),
        (
            {
                "early_stopping_config": {
                    "monitor": "val_metric",
                    "patience": 9,
                    "min_delta": 0.2,
                }
            },
            {
                "monitor": "val_metric",
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
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 2,
                    "factor": 0.25,
                }
            },
            {
                "monitor": "val_loss",
                "mode": "min",
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
