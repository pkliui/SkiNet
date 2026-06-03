import logging
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.data_configs.isic2017dataset_config.isic2017dataset_config import ISIC2017DatasetConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig
from SkiNet.Utils.experiment_keys import ExperimentType, MetricsKey

DataConfig = Annotated[Union[PH2DatasetConfig, ISIC2017DatasetConfig], Field(discriminator="kind")]
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
    def _propagate_sweep_monitor(self) -> "ExperimentConfig":
        """
        ``SWEEP_CONFIG.monitor`` in the YAML is the single source of truth for
        the optimisation metric.  This validator fills that value into every
        callback that monitors a metric:

        - ``trainconfig.early_stopping_config.monitor``
        - ``trainconfig.checkpoint_config.monitor``
        - ``trainconfig.lr_scheduler_config.monitor``

        **Omit ``monitor`` from those sub-sections in the YAML** — it will be
        populated automatically.  If any sub-section has ``monitor`` explicitly
        set to a *different* value, a ``ValueError`` is raised so the mismatch
        is caught at config-load time rather than silently producing a sweep
        that optimises a different metric than the one that stopped training.
        """
        sweep_monitor = self.sweepconfig.monitor
        default_monitor = MetricsKey.default_monitor()
        conflicts: list[str] = []
        for attr, label in (
            ("early_stopping_config", "trainconfig.early_stopping_config.monitor"),
            ("checkpoint_config", "trainconfig.checkpoint_config.monitor"),
            ("lr_scheduler_config", "trainconfig.lr_scheduler_config.monitor"),
        ):
            sub_cfg = getattr(self.trainconfig, attr)
            # Only a conflict when the sub-config was explicitly set to a
            # non-default value that disagrees with the sweep monitor.
            # A sub-config at its default was never explicitly configured.
            if sub_cfg.monitor != sweep_monitor and sub_cfg.monitor != default_monitor:
                conflicts.append(f"  {label} = {sub_cfg.monitor!r}")
        if conflicts:
            raise ValueError(
                f"sweepconfig.monitor is {sweep_monitor!r} but the following "
                f"sub-configs have a different value — remove their explicit "
                f"'monitor' keys from the YAML and let sweepconfig.monitor be "
                f"the single source of truth:\n" + "\n".join(conflicts)
            )
        # Propagate canonical value to all sub-configs so callers can always
        # read monitor directly from them.
        self.trainconfig.early_stopping_config.monitor = sweep_monitor
        self.trainconfig.checkpoint_config.monitor = sweep_monitor
        self.trainconfig.lr_scheduler_config.monitor = sweep_monitor
        return self

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
