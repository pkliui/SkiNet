from pathlib import Path
from typing import Optional

import pytest
import yaml

from SkiNet.ML.config_keys import (DATA_CONFIG, DATASET, EXPERIMENT_TYPE, GENERAL_CONFIG, MODEL, MODEL_CONFIG,
                                   SEGMENTATION, SWEEP_CONFIG, TRAIN_CONFIG, TRANSFORM_CONFIG)
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.load_config_from_yaml import (_get_model_and_dataset_keys, _validate_yaml_config,
                                                     load_config_from_yaml)
from SkiNet.Utils.experiment_keys import DatasetKey, ExperimentType, ModelKey

# --------------------------------Tests for _validate_yaml_config -----------------------------------


def make_valid_yaml_dict() -> dict:
    """
    Generate a valid base YAML dict for testing.
    """
    return {
        GENERAL_CONFIG: {
            MODEL: ModelKey.UNET2D.value,
            DATASET: DatasetKey.PH2.value,
            EXPERIMENT_TYPE: SEGMENTATION,
        },
        DATA_CONFIG: {},
        TRANSFORM_CONFIG: {},
        MODEL_CONFIG: {},
        TRAIN_CONFIG: {},
        SWEEP_CONFIG: {},
    }


def missing_top_level_key_cases() -> list[tuple[dict, str]]:
    """
    Generate test cases for missing top-level keys in the YAML config.
    """
    required_keys = [GENERAL_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, TRANSFORM_CONFIG, SWEEP_CONFIG]
    cases = []
    for key in required_keys:
        d = make_valid_yaml_dict()
        del d[key]
        msg = f"Missing key in YAML config: {key}"
        cases.append((d, msg))
    return cases


@pytest.mark.parametrize(
    "yaml_dict,message",
    missing_top_level_key_cases()
)
def test_validate_yaml_config_top_level_keys(yaml_dict: dict, message: str) -> None:
    """
    Test that _validate_yaml_config raises KeyError for missing top-level keys in YAML input
    """
    with pytest.raises(KeyError, match=message):
        _validate_yaml_config(yaml_dict)


def missing_general_key_cases() -> list[tuple[dict, str, bool]]:
    """
    Generate test cases for missing GENERAL_CONFIG keys in the YAML config.
    """
    required_general_keys = [EXPERIMENT_TYPE, MODEL, DATASET]
    cases = []
    for key in required_general_keys:
        d = make_valid_yaml_dict()
        del d[GENERAL_CONFIG][key]
        msg = f"Missing key in YAML config under {GENERAL_CONFIG}: {key}"
        cases.append((d, msg, True))
    return cases


@pytest.mark.parametrize(
    "yaml_dict,message,should_raise",
    missing_general_key_cases()
)
def test_validate_yaml_config_general_keys(yaml_dict: dict, message: str, should_raise: bool) -> None:
    """
    Test that _validate_yaml_config raises KeyError for missing general config keys in YAML input
    """
    if should_raise:
        with pytest.raises(KeyError, match=message):
            _validate_yaml_config(yaml_dict)


def invalid_general_value_cases() -> list[tuple[str, Optional[str], str]]:
    """
    Generate test cases for invalid values in GENERAL_CONFIG fields.

    :return: List of tuples containing:
        field_key: which GENERAL_CONFIG field to mutate
        invalid_value: value to set
        expected_pattern: regex/substring to match in exception
    """
    return [
        # experiment_type
        (EXPERIMENT_TYPE, "invalid_type", r"Invalid experiment_type: invalid_type"),
        (EXPERIMENT_TYPE, "", r"Invalid experiment_type: "),
        (EXPERIMENT_TYPE, None, r"Invalid experiment_type: None"),

        # model
        (MODEL, "INVALID_MODEL", r"Invalid model name: INVALID_MODEL"),
        (MODEL, "", r"Invalid model name: "),
        (MODEL, None, r"Invalid model name: None"),

        # dataset
        (DATASET, "INVALID_DATASET", r"Invalid dataset name: INVALID_DATASET"),
        (DATASET, "", r"Invalid dataset name: "),
        (DATASET, None, r"Invalid dataset name: None"),
    ]


@pytest.mark.parametrize("field_key,invalid_value,expected_pattern", invalid_general_value_cases())
def test_validate_yaml_config_invalid_general_values(field_key: str, invalid_value: str, expected_pattern: str) -> None:
    """
    Test that _validate_yaml_config raises ValueError for invalid GENERAL_CONFIG field values.
    """
    d = make_valid_yaml_dict()
    d[GENERAL_CONFIG][field_key] = invalid_value

    with pytest.raises(ValueError, match=expected_pattern):
        _validate_yaml_config(d)


# ------------------------------ Tests for _get_model_and_dataset_keys ------------------------------


@pytest.mark.parametrize(
    "yaml_config,expected_model,expected_dataset",
    [
        (
            {GENERAL_CONFIG: {MODEL: ModelKey.UNET2D.value, DATASET: DatasetKey.PH2.value}},
            ModelKey.UNET2D,
            DatasetKey.PH2,
        ),
    ],
)
def test__get_model_and_dataset_keys_valid(yaml_config: dict, expected_model: ModelKey, expected_dataset: DatasetKey) -> None:
    """
    Test that _get_model_and_dataset_keys returns the correct model and dataset keys.
    """
    model_key, dataset_key = _get_model_and_dataset_keys(yaml_config)
    assert model_key == expected_model
    assert dataset_key == expected_dataset


@pytest.mark.parametrize(
    "yaml_config,expected_pattern",
    [
        # invalid model
        (
            {GENERAL_CONFIG: {MODEL: "INVALID_MODEL", DATASET: DatasetKey.PH2.value}},
            r"INVALID_MODEL",
        ),
        # invalid dataset
        (
            {GENERAL_CONFIG: {MODEL: ModelKey.UNET2D.value, DATASET: "INVALID_DATASET"}},
            r"INVALID_DATASET",
        ),
        # both invalid (ModelKey conversion fails first)
        (
            {GENERAL_CONFIG: {MODEL: "INVALID_MODEL", DATASET: "INVALID_DATASET"}},
            r"INVALID_MODEL",
        ),
    ],
)
def test__get_model_and_dataset_keys_invalid_values(yaml_config: dict, expected_pattern: str) -> None:
    """
    Test that _get_model_and_dataset_keys raises ValueError for invalid model and dataset keys
    (Enum-raised)
    """
    with pytest.raises(ValueError, match=expected_pattern):
        _get_model_and_dataset_keys(yaml_config)


@pytest.mark.parametrize(
    "yaml_config,missing_key_name",
    [
        ({GENERAL_CONFIG: {DATASET: DatasetKey.PH2.value}}, MODEL),
        ({GENERAL_CONFIG: {MODEL: ModelKey.UNET2D.value}}, DATASET),
        ({}, GENERAL_CONFIG),
    ],
)
def test__get_model_and_dataset_keys_missing_keys(yaml_config: dict, missing_key_name: str) -> None:
    """
    Test that _get_model_and_dataset_keys raises KeyError for missing model or dataset keys.
    """
    with pytest.raises(KeyError) as exc_info:
        _get_model_and_dataset_keys(yaml_config)

    msg = str(exc_info.value)
    assert "Missing key in YAML config under GENERAL_CONFIG:" in msg
    assert missing_key_name in msg


# ----------------------------Tests for load_config_from_yaml----------------------------


def test_load_config_from_yaml_valid(tmp_path: Path) -> None:
    """
    Test that load_config_from_yaml correctly loads a valid YAML config
    and returns an ExperimentConfig object
    """
    yaml_dict = make_valid_yaml_dict()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_dict))

    config = load_config_from_yaml(yaml_path)
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "unet2d_ph2_experiment"
    assert config.experiment_type == ExperimentType.SEGMENTATION
    assert config.description == "UNet2D on PH2 dataset"


@pytest.mark.parametrize("yaml_content", ["[]", "null", '"just a string"', "123"])
def test_load_config_from_yaml_non_mapping_yaml_raises(tmp_path: Path, yaml_content: str) -> None:
    """
    Test that load_config_from_yaml raises ValueError when YAML does not load to a mapping.
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_content)

    with pytest.raises(ValueError, match=r"did not load to a mapping"):
        load_config_from_yaml(yaml_path)


@pytest.mark.parametrize(
    "yaml_dict,expected_exception,expected_pattern",
    [
        # missing experiment type
        (
            {
                GENERAL_CONFIG: {
                    MODEL: ModelKey.UNET2D.value,
                    DATASET: DatasetKey.PH2.value,
                },
                DATA_CONFIG: {},
                TRANSFORM_CONFIG: {},
                MODEL_CONFIG: {},
                TRAIN_CONFIG: {},
                SWEEP_CONFIG: {},
            },
            KeyError,
            rf"Missing key in YAML config under {GENERAL_CONFIG}: {EXPERIMENT_TYPE}"
        ),
        # invalid model
        (
            {
                GENERAL_CONFIG: {
                    MODEL: "INVALID_MODEL",
                    DATASET: DatasetKey.PH2.value,
                    EXPERIMENT_TYPE: SEGMENTATION,
                },
                DATA_CONFIG: {},
                TRANSFORM_CONFIG: {},
                MODEL_CONFIG: {},
                TRAIN_CONFIG: {},
                SWEEP_CONFIG: {},
            },
            ValueError,
            r"INVALID_MODEL",
        ),
        # invalid dataset
        (
            {
                GENERAL_CONFIG: {
                    MODEL: ModelKey.UNET2D.value,
                    DATASET: "INVALID_DATASET",
                    EXPERIMENT_TYPE: SEGMENTATION,
                },
                DATA_CONFIG: {},
                TRANSFORM_CONFIG: {},
                MODEL_CONFIG: {},
                TRAIN_CONFIG: {},
                SWEEP_CONFIG: {},
            },
            ValueError,
            r"INVALID_DATASET",
        ),
        # unknown experiment type
        (
            {
                GENERAL_CONFIG: {
                    MODEL: ModelKey.UNET2D.value,
                    DATASET: DatasetKey.PH2.value,
                    EXPERIMENT_TYPE: "CLASSIFICATION",
                },
                DATA_CONFIG: {},
                TRANSFORM_CONFIG: {},
                MODEL_CONFIG: {},
                TRAIN_CONFIG: {},
                SWEEP_CONFIG: {},
            },
            ValueError,
            r"Invalid experiment_type: CLASSIFICATION",
        ),
    ],
)
def test_load_config_from_yaml_invalid(tmp_path: Path, yaml_dict: dict, expected_exception: type[Exception], expected_pattern: str) -> None:
    """
    Test that load_config_from_yaml raises the expected exception for invalid YAML configs.
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_dict))

    with pytest.raises(expected_exception, match=expected_pattern):
        load_config_from_yaml(yaml_path)


def test_validate_yaml_config_missing_transform_config_raises() -> None:
    """
    Test that _validate_yaml_config raises KeyError for missing TRANSFORM_CONFIG
    """
    d = make_valid_yaml_dict()
    del d[TRANSFORM_CONFIG]
    with pytest.raises(KeyError, match=rf"Missing key in YAML config: {TRANSFORM_CONFIG}"):
        _validate_yaml_config(d)


def test_load_config_from_yaml_yaml_values_override_all_config_defaults(tmp_path: Path) -> None:
    """
    Non-default values in every YAML sub-config section must land in the
    resulting ExperimentConfig, overriding pydantic model defaults.
    This catches YAML-key → config-field-name wiring bugs that unit tests
    on each layer independently would miss.
    """
    yaml_dict = {
        GENERAL_CONFIG: {
            MODEL: ModelKey.UNET2D.value,
            DATASET: DatasetKey.PH2.value,
            EXPERIMENT_TYPE: SEGMENTATION,
        },
        DATA_CONFIG: {
            "split_train_size": 0.7,
            "split_random_seed": 99,
        },
        TRANSFORM_CONFIG: {
            "augmentation_required": False,
            "seed_value": 7,
            "crop": {"crop_type": "center_crop", "size": [128, 128]},
            "photometric_augmentation": {"color_jitter_apply": True},
        },
        MODEL_CONFIG: {
            "out_channels_layer1": 32,
            "number_of_layers": 4,
        },
        TRAIN_CONFIG: {
            "batch_size": 16,
            "max_epochs": 10,
            "lr": 3e-4,
        },
        SWEEP_CONFIG: {
            "direction": "minimize",
            "experiment_name": "my_sweep",
        },
    }
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_dict))

    config = load_config_from_yaml(yaml_path)

    assert config.dataconfig.split_train_size == pytest.approx(0.7)
    assert config.dataconfig.split_random_seed == 99

    assert config.transformconfig.augmentation_required is False
    assert config.transformconfig.seed_value == 7
    assert config.transformconfig.crop.crop_type == "center_crop"
    assert config.transformconfig.crop.size == (128, 128)
    assert config.transformconfig.photometric_augmentation.color_jitter_apply is True

    assert config.modelconfig.out_channels_layer1 == 32
    assert config.modelconfig.number_of_layers == 4

    assert config.trainconfig.batch_size == 16
    assert config.trainconfig.max_epochs == 10
    assert config.trainconfig.lr == pytest.approx(3e-4)

    assert config.sweepconfig.direction == "minimize"
    assert config.sweepconfig.experiment_name == "my_sweep"
