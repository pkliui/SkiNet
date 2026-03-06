import logging
from abc import ABC, abstractmethod
from typing import Any

from SkiNet.ML.configs.config_creator import ConfigCreator, PH2_UNet_ConfigCreator
from SkiNet.Utils.experiment_keys import DatasetKey, ModelKey


class ConfigFactory(ABC):
    """
    Abstract base class for all experiment configuration factories.
    """
    @abstractmethod
    def get_config_creator(self, **kwargs: Any) -> ConfigCreator:
        pass


class PH2_UNet_ConfigFactory(ConfigFactory):
    """
    Config factory for experiments based on UNet2D model and using PH2 dataset
    """

    def get_config_creator(self, **kwargs: Any) -> ConfigCreator:
        return PH2_UNet_ConfigCreator()


def _get_config_factory(model_key: ModelKey, dataset_key: DatasetKey) -> ConfigFactory:
    """
    Get the configuration factory for an experiment using a specific model and dataset combination.

    : param model_key: The ModelKey enum member corresponding to the model used in the experiment(e.g., ModelKey.UNET2D).
    : param dataset_key: The DatasetKey enum member corresponding to the dataset used in the experiment(e.g., DatasetKey.PH2).
    : return: An instance of ExperimentConfigFactory corresponding to the specified model and dataset combination.
    """
    _factories = {(ModelKey.UNET2D, DatasetKey.PH2): PH2_UNet_ConfigFactory}
    factory_cls = _factories.get((model_key, dataset_key))
    if factory_cls is None:
        logging.getLogger(__name__).error(f"No factory found for model key: {model_key}, dataset key: {dataset_key}")
        raise ValueError(f"No factory found for model key: {model_key}, dataset key: {dataset_key}")
    return factory_cls()
