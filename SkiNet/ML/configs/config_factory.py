import logging
from abc import ABC, abstractmethod
from typing import Any

from SkiNet.ML.configs.config_creator import ConfigCreator, PH2_UNet_ConfigCreator
from SkiNet.Utils.experiment_keys import DatasetKey, ModelKey

"""
ConfigFactory registry and helpers

Purpose
-------
This module defines the abstract ConfigFactory API and a small registry that maps
(ModelKey, DatasetKey) -> concrete factory class. Factories are responsible for
producing ConfigCreator instances which in turn create ExperimentConfig objects.

Public API
----------
- ConfigFactory (abstract): implement get_config_creator(**kwargs) -> ConfigCreator
    (subclasses create and return a ConfigCreator for a model/dataset pairing)
- PH2_UNet_ConfigFactory: example concrete factory (UNet2D + PH2)
- get_config_factory(model_key: ModelKey, dataset_key: DatasetKey) -> ConfigFactory
    Lookup registry (mapping of (ModelKey, DatasetKey) -> factory *class*) and
    instantiate a factory per-call. Returns a ConfigFactory instance.

Extension steps
----------------
1. Implement a new ConfigCreator subclass that returns ExperimentConfig instances.
2. Add a concrete ConfigFactory that returns your creator from get_config_creator().
3. Register the factory in the registry mapping (tuple -> factory class).
4. Add unit tests:
   - get_config_factory returns an instance of your factory class.
   - The factory's creator.create_config(...) returns a valid ExperimentConfig for expected kwargs.
"""

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


def get_config_factory(model_key: ModelKey, dataset_key: DatasetKey) -> ConfigFactory:
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
