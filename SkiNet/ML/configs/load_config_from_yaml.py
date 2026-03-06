
from pathlib import Path

import yaml

from SkiNet.ML.config_keys import (DATA_CONFIG, DATASET, EXPERIMENT_TYPE, GENERAL_CONFIG, MODEL, MODEL_CONFIG,
                                   SEGMENTATION, TRAIN_CONFIG)
from SkiNet.ML.configs.config_factory import ConfigFactory, _get_config_factory
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.Utils.experiment_keys import DatasetKey, ModelKey

# Allowed experiment types
ALLOWED_EXPERIMENT_TYPES = {SEGMENTATION}

def load_config_from_yaml(yaml_path: Path) -> ExperimentConfig:
    """
    Load an experiment configuration based on the provided YAML config.

    :param yaml_path: Path to the YAML config file.
    :return: An instance of ExperimentConfig corresponding to the model and dataset specified in the YAML config.
    """
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    if not isinstance(yaml_config, dict):
        raise ValueError(f"YAML file {yaml_path} did not load to a mapping (got {type(yaml_config)!r})")

    _validate_yaml_config(yaml_config)

    general_config = yaml_config[GENERAL_CONFIG]
    experiment_type = general_config[EXPERIMENT_TYPE]

    # get factory and prepare kwargs
    model_key, dataset_key = _get_model_and_dataset_keys(yaml_config)
    factory: ConfigFactory = _get_config_factory(model_key, dataset_key)

    dataconfig_kwargs = yaml_config.get(DATA_CONFIG, {})
    modelconfig_kwargs = yaml_config.get(MODEL_CONFIG, {})
    trainconfig_kwargs = yaml_config.get(TRAIN_CONFIG, {})

    if experiment_type == SEGMENTATION:
        config_creator = factory.get_config_creator()
        experiment_config = config_creator.create_config(dataconfig_kwargs=dataconfig_kwargs,
                                                         modelconfig_kwargs=modelconfig_kwargs,
                                                         trainconfig_kwargs=trainconfig_kwargs)
    else:
        # should be unreachable if _validate_yaml_config enforces allowed types
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return experiment_config


def _validate_yaml_config(yaml_config: dict) -> None:
    """
    Validate the structure and content of the experiment YAML config.
    Raises KeyError or ValueError with descriptive messages if invalid.
    """

    # Check for required top-level keys
    required_keys = [GENERAL_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG]
    # Check for required keys under GENERAL_CONFIG
    required_general_keys = [EXPERIMENT_TYPE, MODEL, DATASET]

    # Check for required top-level keys
    for key in required_keys:
        if key not in yaml_config:
            raise KeyError(f"Missing key in YAML config: {key}")

    # Check for required keys under GENERAL_CONFIGs
    general = yaml_config[GENERAL_CONFIG]
    for key in required_general_keys:
        if key not in general:
            raise KeyError(f"Missing key in YAML config under {GENERAL_CONFIG}: {key}")

    # Validate experiment_type
    if general[EXPERIMENT_TYPE] not in ALLOWED_EXPERIMENT_TYPES:
        raise ValueError(f"Invalid experiment_type: {general[EXPERIMENT_TYPE]}")

    # Validate model and dataset enums
    try:
        ModelKey(general[MODEL])
    except ValueError:
        raise ValueError(f"Invalid model name: {general[MODEL]}. Available models: {[model.value for model in ModelKey]}")
    try:
        DatasetKey(general[DATASET])
    except ValueError:
        raise ValueError(f"Invalid dataset name: {general[DATASET]}. Available datasets: {[dataset.value for dataset in DatasetKey]}")


def _get_model_and_dataset_keys(yaml_config: dict) -> tuple[ModelKey, DatasetKey]:
    """
    Using YAML config dictionary with MODEL and DATASET keys, get the corresponding ModelKey and DatasetKey enum members,
    corresponding to the model and dataset names defined in the yaml config.

    Example YAML config:
        {
            GENERAL_CONFIG: {
              MODEL: "UNET2D_MODEL",
              DATASET: "PH2_DATASET"
            }
        }
    """
    try:
        model_name = yaml_config[GENERAL_CONFIG][MODEL]
        dataset_name = yaml_config[GENERAL_CONFIG][DATASET]
        return ModelKey(model_name), DatasetKey(dataset_name)
    except KeyError as e:
        missing = e.args[0] if e.args else str(e)
        raise KeyError(f"Missing key in YAML config under {GENERAL_CONFIG}: {missing}")
