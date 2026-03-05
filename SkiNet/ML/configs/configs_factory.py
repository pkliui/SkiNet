import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import yaml

from SkiNet.ML.config_keys import DATASET, MODEL
from SkiNet.ML.configs.base_experiment_config import BaseExperimentConfig
from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.ML.configs.experiment_configs import PH2_UNet_Config
from SkiNet.Utils.model_keys import ModelKey


class ExperimentConfigFactory(ABC):
    """
    Abstract base class for all experiment configuration factories.
    """
    @abstractmethod
    def load_config(self) -> BaseExperimentConfig:
        pass


class PH2_UNet_ConfigFactory(ExperimentConfigFactory):
    """
    Config factory for an experiment based on UNet2D model and using PH2 dataset

    :param local_data_root: The root path to data on the local file system. The path should point to a directory that contains folders
        with samples of data uniquely identifiable by their ID. Only used when azure_data is set to False.
    :param azure_data: If True, local_data_root should not be provided and the script will look for data in Azure.
                       If False, local_data_root must be provided and the script will look for data on the local file system.
    """

    def __init__(self, local_data_root: Optional[str] = None, azure_data: bool = False) -> None:
        if azure_data and local_data_root is not None:
            raise ValueError("Do not provide local_data_root when using azure_data.")
        self.local_data_root = local_data_root
        self.azure_data = azure_data

    def load_config(self) -> BaseExperimentConfig:
        return PH2_UNet_Config.create(local_data_root=self.local_data_root, azure_data=self.azure_data)


def load_experiment_config_from_yaml(yaml_path: Path) -> BaseExperimentConfig:
    """
    Load an experiment configuration based on the provided YAML config.

    :param yaml_path: Path to the YAML config file containing at least the MODEL and DATASET keys,
    with values corresponding to valid ModelKey and DatasetKey enum members.

    :return: An instance of BaseExperimentConfig corresponding to the model and dataset specified in the YAML config.
    """
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    model_key, dataset_key = _get_model_and_dataset_keys(yaml_config)
    factory = _get_config_factory(model_key, dataset_key)

    experiment_config = factory.load_config()
    return experiment_config


def _get_config_factory(model_key: ModelKey, dataset_key: DatasetKey) -> ExperimentConfigFactory:
    """
    Get the configuration factory for an experiment using a specific model and dataset combination.

    :param model_key: The ModelKey enum member corresponding to the model used in the experiment (e.g., ModelKey.UNET2D).
    :param dataset_key: The DatasetKey enum member corresponding to the dataset used in the experiment (e.g., DatasetKey.PH2).
    :return: An instance of ExperimentConfigFactory corresponding to the specified model and dataset combination.
    """
    factories = {
        (ModelKey.UNET2D, DatasetKey.PH2): PH2_UNet_ConfigFactory()
    }

    factory = factories.get((model_key, dataset_key))
    if factory is None:
        logging.getLogger(__name__).error(f"No factory found for model key: {model_key}, dataset key: {dataset_key}")
        raise ValueError(f"No factory found for model key: {model_key}, dataset key: {dataset_key}")
    return factory


def _get_model_and_dataset_keys(yaml_config: dict) -> tuple[ModelKey, DatasetKey]:
    """
    Using YAML config dictionary with MODEL and DATASET keys, get the corresponding ModelKey and DatasetKey enum members,
    corresponding to the model and dataset names defined in the yaml config.

    Example YAML config:
        {
            "MODEL": "UNET2D_MODEL",
            "DATASET": "PH2_DATASET"
        }
    """
    try:
        model_name = yaml_config[MODEL]
        dataset_name = yaml_config[DATASET]
        return ModelKey(model_name), DatasetKey(dataset_name)
    except KeyError as e:
        raise KeyError(f"Missing key in YAML config: {e}")
    except ValueError:
        raise ValueError(f"Invalid model or dataset name: {yaml_config[MODEL]}, {yaml_config[DATASET]}")
