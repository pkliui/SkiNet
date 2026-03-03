from typing import Any

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.segmentation_configs.segmentation_config import SegmentationConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig


class LoadConfig:
    def __init__(self, main_config: Any) -> None:
        self.main_config = main_config

    def get_config(self) -> SegmentationConfig:
        return SegmentationConfig(experiment_name=self.main_config.experiment_name,
                                  description=self.main_config.description,
                                  dataconfig=PH2DatasetConfig(csv_path=self.main_config.csv_path,
                                                              azure_data=self.main_config.azure_data),
                                  trainconfig=BaseTrainConfig(),
                                  modelconfig=UNet2DModelConfig())
