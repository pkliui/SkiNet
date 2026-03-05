from pathlib import Path

import pytest
import yaml

from SkiNet.ML.config_keys import DATASET, MODEL
from SkiNet.ML.configs.configs_factory import (PH2_UNet_ConfigFactory, _get_config_factory, _get_model_and_dataset_keys,
                                               load_experiment_config_from_yaml)
from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.Utils.model_keys import ModelKey


@pytest.mark.parametrize(
    "yaml_config,expected_model,expected_dataset",
    [
        ({MODEL: "UNET2D_MODEL", DATASET: "PH2_DATASET"}, "UNET2D", "PH2"),
        ({MODEL: ModelKey.UNET2D.value, DATASET: DatasetKey.PH2.value}, ModelKey.UNET2D.name, DatasetKey.PH2.name),
    ]
)
def test__get_model_and_dataset_keys_valid(yaml_config: dict, expected_model: str, expected_dataset: str) -> None:
    """
    Test that _get_model_and_dataset_keys returns the expected model and dataset keys
    for valid input.
    """
    model_key, dataset_key = _get_model_and_dataset_keys(yaml_config)
    assert model_key.name == expected_model
    assert dataset_key.name == expected_dataset


@pytest.mark.parametrize(
    "yaml_config",
    [
        {MODEL: "INVALID_MODEL", DATASET: DatasetKey.PH2.value},
        {MODEL: ModelKey.UNET2D.value, DATASET: "INVALID_DATASET"},
        {MODEL: "INVALID_MODEL", DATASET: "PH2_DATASET"},
        {MODEL: "UNET2D_MODEL", DATASET: "INVALID_DATASET"},
        {MODEL: "INVALID_MODEL", DATASET: "INVALID_DATASET"},
    ]
)
def test__get_model_and_dataset_keys_invalid_values(yaml_config: dict) -> None:
    """
    Test that _get_model_and_dataset_keys raises ValueError for invalid values in YAML
    """
    with pytest.raises(ValueError, match="Invalid model or dataset name"):
        _get_model_and_dataset_keys(yaml_config)


@pytest.mark.parametrize(
    "yaml_config",
    [
        {"INVALIDKEY": ModelKey.UNET2D.value, DATASET: DatasetKey.PH2.value},
        {MODEL: ModelKey.UNET2D.value, "INVALIDKEY": DatasetKey.PH2.value},
    ]
)
def test__get_model_and_dataset_keys_invalid_keys(yaml_config: dict) -> None:
    """
    Test that _get_model_and_dataset_keys raises ValueError for invalid keys in YAML
    """
    with pytest.raises(KeyError, match="Missing key in YAML config:"):
        _get_model_and_dataset_keys(yaml_config)


@pytest.mark.parametrize(
    "yaml_dict,expected_factory_type,raises",
    [
        # Valid config: UNET2D + PH2
        (
            {MODEL: ModelKey.UNET2D.value, DATASET: DatasetKey.PH2.value},
            PH2_UNet_ConfigFactory,
            False
        ),
        (
            {MODEL: 'UNET2D_MODEL', DATASET: 'PH2_DATASET'},
            PH2_UNet_ConfigFactory,
            False
        )
    ]
)
def test_get_config_factory_and_config_valid_inputs(tmp_path: Path, yaml_dict: dict, expected_factory_type: type, raises: bool) -> None:
    """
    Test that get_config_factory returns the expected factory for valid model/dataset
    combinations using enum values and strings in YAML dictionary
    """
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    dataset_name = yaml_dict.get(DATASET)
    model_name = yaml_dict.get(MODEL)
    dataset_key = DatasetKey(dataset_name)
    model_key = ModelKey(model_name)

    factory = _get_config_factory(model_key, dataset_key)
    assert isinstance(factory, expected_factory_type)


@pytest.mark.parametrize(
    "model_name, dataset_name,expected_factory_type,raises",
    [
        ("unknown", DatasetKey.PH2.value, None, True),
        (ModelKey.UNET2D.value, "unknown", None, True),
        ("unknown", "PH2_DATASET", None, True),
        ("UNET2D_MODEL", "unknown", None, True),
    ]
)
def test_get_config_factory_and_config_invalid(tmp_path: Path, model_name: str, dataset_name: str, expected_factory_type: type, raises: bool) -> None:
    """
    Test that get_config_factory raises ValueError for invalid model/dataset combinations in YAML dictionary
    """
    with pytest.raises(ValueError, match="No factory found for "):
        dataset_key = DatasetKey(dataset_name) if dataset_name in DatasetKey._value2member_map_ else dataset_name
        model_key = ModelKey(model_name) if model_name in ModelKey._value2member_map_ else model_name
        _get_config_factory(model_key, dataset_key)  # type: ignore


@pytest.mark.parametrize(
    "yaml_dict,expected_type,should_raise",
    [
        # Valid config
        ({MODEL: "UNET2D_MODEL", DATASET: "PH2_DATASET"}, PH2_UNet_ConfigFactory, False),
        ({MODEL: ModelKey.UNET2D.value, DATASET: DatasetKey.PH2.value}, PH2_UNet_ConfigFactory, False),
        # Invalid model
        ({MODEL: "INVALID_MODEL", DATASET: "PH2_DATASET"}, None, True),
        # Invalid dataset
        ({MODEL: "UNET2D_MODEL", DATASET: "INVALID_DATASET"}, None, True),
    ]
)
def test_load_experiment_config_from_yaml(tmp_path: Path, yaml_dict: dict, expected_type: type, should_raise: bool) -> None:
    """
    Test that load_experiment_config_from_yaml returns the expected config type for valid YAML input
    """
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    if should_raise:
        with pytest.raises(ValueError, match="Invalid model or dataset name"):
            load_experiment_config_from_yaml(yaml_path)
    else:
        config = load_experiment_config_from_yaml(yaml_path)
        assert isinstance(config, config.__class__)


@pytest.mark.parametrize(
    "yaml_dict,expected_type,should_raise",
    [
        # IInvalid key
        ({"INVALIDKEY": "UNET2D_MODEL", DATASET: "PH2_DATASET"}, PH2_UNet_ConfigFactory, True),
    ]
)
def test_load_experiment_config_from_yaml_invalid_key(tmp_path: Path, yaml_dict: dict, expected_type: type, should_raise: bool) -> None:
    """
    Test that load_experiment_config_from_yaml raises KeyError for missing keys in YAML input
    """
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    if should_raise:
        with pytest.raises(KeyError, match="Missing key in YAML config:"):
            load_experiment_config_from_yaml(yaml_path)


@pytest.mark.parametrize(
    "yaml_dict,expected_type,should_raise",
    [
        # IInvalid key
        ({MODEL: "INVALID", DATASET: "PH2_DATASET"}, PH2_UNet_ConfigFactory, True),
    ]
)
def test_load_experiment_config_from_yaml_invalid_values(tmp_path: Path, yaml_dict: dict, expected_type: type, should_raise: bool) -> None:
    """
    Test load_experiment_config_from_yaml raises ValueError for invalid values in YAML input
    """
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    if should_raise:
        with pytest.raises(ValueError, match="Invalid model or dataset name"):
            load_experiment_config_from_yaml(yaml_path)
