from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig


class ConfigCreator(ABC):
    """
    Abstract base class for all classes creaing configurations for experiments.
    """
    @abstractmethod
    def create_config(self,
                      dataconfig_kwargs: Optional[Dict[str, Any]] = None,
                      modelconfig_kwargs: Optional[Dict[str, Any]] = None,
                      trainconfig_kwargs: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        pass


class PH2_UNet_ConfigCreator(ConfigCreator):
    """
    Return a concrete configuration for the segmentation of the PH2 dataset with UNet2D model.
    """

    def create_config(self,
                      dataconfig_kwargs: Optional[Dict[str, Any]] = None,
                      modelconfig_kwargs: Optional[Dict[str, Any]] = None,
                      trainconfig_kwargs: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        dataconfig_kwargs = dataconfig_kwargs or {}
        modelconfig_kwargs = modelconfig_kwargs or {}
        trainconfig_kwargs = trainconfig_kwargs or {}
        return ExperimentConfig(experiment_name="unet2d_ph2_experiment",
                                experiment_type="segmentation",
                                description="UNet2D on PH2 dataset",
                                dataconfig=PH2DatasetConfig(**dataconfig_kwargs),
                                modelconfig=UNet2DModelConfig(**modelconfig_kwargs),
                                trainconfig=BaseTrainConfig(**trainconfig_kwargs))
