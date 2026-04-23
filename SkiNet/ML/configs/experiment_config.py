import logging
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig
from SkiNet.Utils.experiment_keys import ExperimentType

DataConfig = Annotated[Union[PH2DatasetConfig], Field(discriminator="kind")]
ModelConfig = Annotated[Union[UNet2DModelConfig], Field(discriminator="kind")]

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """
    Base configuration for a ML experiment, containing common fields such as experiment name, description, and model type.

    """
    model_config = ConfigDict(extra="forbid")  # Forbid extra fields not defined in the model or its subclasses
    # Subclasses specify the experiment type, e.g. segmentation, classification, etc.
    experiment_type: ExperimentType = Field(...,
                                            description="Type of the experiment, e.g. 'segmentation', 'classification', etc. ")
    experiment_name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Description of the experiment")
    dataconfig: DataConfig = Field(..., description="Data configuration for ML experiments. "
                                   "Discriminated by 'kind' field to select the appropriate dataset configuration.")
    transformconfig: TransformConfig = Field(...,
                                             description="Transformation configuration for ML experiments,"
                                             "including cropping and augmentations.")
    trainconfig: TrainConfig = Field(..., description="Training configuration for ML experiments")
    sweepconfig: SweepConfig = Field(default_factory=SweepConfig,
                                     description="Optional configuration required only for optuna hyperparameter sweep")
    modelconfig: ModelConfig = Field(..., description="Model configuration for ML experiments. "
                                     "Discriminated by 'kind' field to select the appropriate model configuration.")

    cfg_path: str | None = Field(
        default=None, description="Resolved path to the YAML config used to create this config")

    @model_validator(mode="after")
    def _validate_crop_matches_model(self) -> "ExperimentConfig":
        """
        Check that the crop size is compatible with the model's required input size.
        """
        if not self.transformconfig.crop.crop_apply:
            return self

        crop_height, crop_width = self.transformconfig.crop.size
        mult = getattr(self.modelconfig, "required_input_multiple", None)
        if mult is None:
            logger.warning("Model config does not specify required_input_multiple; "
                           "skipping crop size validation against model downsampling.")
            return self

        if isinstance(mult, tuple):
            model_height_multiple, model_width_multiple = mult
        else:
            model_height_multiple = mult
            model_width_multiple = mult

        if (crop_height % model_height_multiple != 0 or crop_width % model_width_multiple != 0):
            raise ValueError(f"crop.size {self.transformconfig.crop.size} must be "
                             f"divisible by {(model_height_multiple, model_width_multiple)} as per model design.")
        return self
