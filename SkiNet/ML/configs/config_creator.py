from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig

"""
ConfigCreator: produce ExperimentConfig instances

Purpose
-------
ConfigCreator implementations encapsulate how to assemble an ExperimentConfig for a
specific (model, dataset) pairing. They accept the following optional kwargs dicts:
 - dataconfig_kwargs: passed to the dataset config constructor
 - transformconfig_kwargs: passed to the transform config constructor
 - modelconfig_kwargs: passed to the model config constructor
 - trainconfig_kwargs: passed to the train config constructor

Design notes
------------
- Default kwargs are Optional[Dict] = None and normalized to {} inside create_config.
- Model configs should be strict (extra="forbid") to catch hyperparameter typos.
- Data/train configs can be permissive (extra="ignore") during development for faster iteration;
  promote frequently used keys to the schema when they become stable.

How to add a new creator
----------------------------------
1. Implement a subclass of ConfigCreator implementing create_config(...).
2. Validate and coerce kwargs into the typed pydantic models.
3. Add tests for unknown kwargs behavior (forbid vs ignore) as per project policy.
"""


class ConfigCreator(ABC):
    """
    Abstract base class for all classes creaing configurations for experiments.
    """
    @abstractmethod
    def create_config(self,
                      dataconfig_kwargs: Optional[Dict[str, Any]] = None,
                      transformconfig_kwargs: Optional[Dict[str, Any]] = None,
                      modelconfig_kwargs: Optional[Dict[str, Any]] = None,
                      trainconfig_kwargs: Optional[Dict[str, Any]] = None,
                      sweepconfig_kwargs: Optional[Dict[str, Any]] = None,
                      experiment_name: str = "") -> ExperimentConfig:
        pass


class PH2_UNet_ConfigCreator(ConfigCreator):
    """
    Return a concrete configuration for the segmentation of the PH2 dataset with UNet2D model.
    """

    def create_config(self,
                      dataconfig_kwargs: Optional[Dict[str, Any]] = None,  # mypy - Optional because default is None
                      transformconfig_kwargs: Optional[Dict[str, Any]] = None,
                      modelconfig_kwargs: Optional[Dict[str, Any]] = None,
                      trainconfig_kwargs: Optional[Dict[str, Any]] = None,
                      sweepconfig_kwargs: Optional[Dict[str, Any]] = None,
                      experiment_name: str = "unet2d_ph2_experiment") -> ExperimentConfig:
        dataconfig_kwargs = dataconfig_kwargs or {}
        transformconfig_kwargs = transformconfig_kwargs or {}
        modelconfig_kwargs = modelconfig_kwargs or {}
        trainconfig_kwargs = trainconfig_kwargs or {}
        sweepconfig_kwargs = sweepconfig_kwargs or {}

        return ExperimentConfig(experiment_name=experiment_name,
                                experiment_type=ExperimentType.SEGMENTATION,
                                description="UNet2D on PH2 dataset",
                                dataconfig=PH2DatasetConfig(**dataconfig_kwargs),
                                transformconfig=TransformConfig(**transformconfig_kwargs),
                                modelconfig=UNet2DModelConfig(**modelconfig_kwargs),
                                trainconfig=TrainConfig(**trainconfig_kwargs),
                                sweepconfig=SweepConfig(**sweepconfig_kwargs))
