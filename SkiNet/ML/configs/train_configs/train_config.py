from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import logging

logger = logging.getLogger(__name__)


class ReduceOnPlateauConfig(BaseModel):
    """
    Learning rate ReduceOnPlateau scheduler configuration for PyTorch Lightning.
    """
    monitor: str = Field(default="val_dice")
    mode: Literal["min", "max"] = Field(default="max")
    patience: int = Field(default=5, ge=0)
    factor: float = Field(default=0.5, gt=0, lt=1)


class CheckpointConfig(BaseModel):
    """
    Configuration for model checkpointing.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    # ok For segmentation, sometimes Dice/IoU is a better “best model” criterion than validation loss, depending on what you care about.
    monitor: str = Field(default="val_loss")  # ok ModelCheckpoint
    mode: str = Field(default="min")  # ok ModelCheckpoint
    save_top_k: int = Field(default=3, ge=1)  # ok ModelCheckpoint
    save_last: bool = Field(default=True)  # ok ModelCheckpoint
    filename: str = Field(default="epoch{epoch:03d}")  # ok


class EarlyStoppingConfig(BaseModel):  # passed all, ok
    """
    Configuration for early stopping.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    monitor: str = Field(default="val_loss")
    mode: str = Field(default="min")
    min_delta: float = Field(default=0.0)
    patience: int = Field(default=5, ge=0)
    strict: bool = Field(default=True)
    check_finite: bool = Field(default=True)
    stopping_threshold: float | None = Field(default=None)
    divergence_threshold: float | None = Field(default=None)

    @model_validator(mode="after")
    def warn_monitor_is_default(self) -> "EarlyStoppingConfig":
        if self.monitor == "val_loss":
            logger.warning(
                "EarlyStopping monitor is set to default 'val_loss' — "
                "make sure your LightningModule logs this exact key."
            )
        return self


class MLflowConfig(BaseModel):
    """
    Configuration for MLflow logging.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    fallback_to_local_mlflow: bool = Field(default=False)
    tracking_uri: str | None = Field(default=None)  # ok logger
    log_model: bool | Literal["all"] = Field(default="all")  # ok logger
    log_model_summary: bool = Field(default=True)  # ok MLflowTrainingArtifactsCallback


class LitLoggerConfig(BaseModel):
    """
    Configuration for LitLogger logging.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    teamspace: str | None = Field(default=None)  # ok
    log_model: bool = Field(default=False)  # ok
    save_logs: bool = Field(default=False)  # ok
    checkpoint_name: str | None = Field(default=None)  # ok

    @model_validator(mode="after")
    def warn_if_no_teamspace(self) -> "LitLoggerConfig":
        if not self.teamspace:
            logger.warning("LitLogger: teamspace is not set, metrics may not appear in dashboard.")
        return self


class TrainConfig(BaseModel):
    """
    Configuration for training.
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    # --- General params ---
    log_dir: str = Field(default="experiment_logs")  # ok
    experiment_name: str = Field(default="unet2d_experiment")
    run_test_after_fit: bool = Field(default=False)
    test_on_val_split: bool = Field(default=False)  # do not run test on validation set by default unless required

    # --- Fit and optimizer params
    # DataLoader params
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=True)
    prefetch_factor: int = Field(default=2, ge=1)
    # LightningModel params
    optimizer_name: str = Field(default="adamw")
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=True)
    # L.Trainer params
    max_epochs: int = Field(default=1, ge=1)
    accelerator: str = Field(default="auto")
    devices: str | int = Field(default="auto")
    log_every_n_steps: int = Field(default=1, ge=1)
    check_val_every_n_epoch: int = Field(default=1, ge=1)
    num_sanity_val_steps: int = Field(default=0, ge=0)

    # --- Nested configs  for callbacks and loggers ---
    early_stopping_config: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint_config: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)
    litlogger_config: LitLoggerConfig = Field(default_factory=LitLoggerConfig)

    # --- Other configs ---
    lr_scheduler_config: ReduceOnPlateauConfig = Field(default_factory=ReduceOnPlateauConfig)

    #  --- Other callbacks params ---
    system_metrics_interval_sec: float = Field(default=5.0, gt=0)

    # --- Logger toggles ---
    use_mlflow_logger: bool = Field(default=False)
    use_checkpoint: bool = Field(default=False)
    use_early_stopping: bool = Field(default=False)
    use_litlogger_logger: bool = Field(default=False)

    @model_validator(mode="after")
    def require_tracking_uri_if_enabled(self) -> "TrainConfig":
        if self.use_mlflow_logger and not self.mlflow_config.tracking_uri:
            raise ValueError("TRAIN_CONFIG.mlflow_config.tracking_uri must be set when use_mlflow_logger=true.")
        return self

    @model_validator(mode="after")
    def warn_if_testing_on_val(self) -> "TrainConfig":
        if self.run_test_after_fit and self.test_on_val_split:
            logger.warning(
                "test_on_val_split=True: final test will run on the validation set, "
                "not a held-out test set. Metrics may be optimistic."
            )
        return self
