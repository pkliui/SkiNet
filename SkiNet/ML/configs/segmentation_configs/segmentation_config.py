from typing import Annotated, Literal, Union

from pydantic import Field

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.segmentation_configs.base_segmentation_config import BaseSegmentationConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig

DataConfig = Annotated[Union[PH2DatasetConfig],
                       Field(discriminator="kind")]
ModelConfig = Annotated[Union[UNet2DModelConfig],
                        Field(discriminator="kind")]

class SegmentationConfig(BaseSegmentationConfig):
    """
    Configuration for segmentation experiments.
    """
    model_type: Literal["segmentation"] = "segmentation"

    dataconfig: DataConfig = Field(..., description="Data configuration for segmentation experiments")
    trainconfig: BaseTrainConfig = Field(..., description="Training configuration for segmentation experiments")
    modelconfig: ModelConfig = Field(..., description="Model configuration for segmentation experiments")
