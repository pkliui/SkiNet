import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
import logging

from SkiNet.Utils.experiment_keys import LossFunctionKey, MetricsKey

logger = logging.getLogger(__name__)


class ReduceOnPlateauConfig(BaseModel):
    """
    Learning rate ReduceOnPlateau scheduler configuration for PyTorch Lightning.
    """
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor())
    mode: Literal["min", "max"] = Field(default="max")
    patience: int = Field(default=5, ge=0)
    factor: float = Field(default=0.5, gt=0, lt=1)


class CosineAnnealingConfig(BaseModel):
    """
    CosineAnnealingLR scheduler configuration.
    T_max is set to max_epochs at runtime when None.
    """
    T_max: int | None = Field(default=None, ge=1)
    eta_min: float = Field(default=1e-6, ge=0)


class CheckpointConfig(BaseModel):
    """
    Configuration for model checkpointing.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    # ok For segmentation, sometimes Dice/IoU is a better “best model” criterion than validation loss, depending on what you care about.
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor())
    mode: Literal["min", "max"] = Field(default="max")  # ok ModelCheckpoint
    save_top_k: int = Field(default=1, ge=0)  # ok ModelCheckpoint
    save_last: bool = Field(default=True)  # ok ModelCheckpoint
    filename: str = Field(default="epoch{epoch:03d}")  # ok


class EarlyStoppingConfig(BaseModel):  # passed all, ok
    """
    Configuration for early stopping.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor())
    mode: Literal["min", "max"] = Field(default="max")
    min_delta: float = Field(default=0.0)
    patience: int = Field(default=5, ge=0)
    strict: bool = Field(default=True)
    check_finite: bool = Field(default=True)
    stopping_threshold: float | None = Field(default=None)
    divergence_threshold: float | None = Field(default=None)

    @model_validator(mode="after")
    def warn_monitor_is_default(self) -> "EarlyStoppingConfig":
        if self.monitor == MetricsKey.default_monitor():
            logger.warning(
                f"EarlyStopping monitor is set to default {MetricsKey.default_monitor()} — "
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


PrecisionType = Literal[
    "16-mixed",
    "bf16-mixed",
    "32-true",
    "16-true",
    "bf16-true",
]


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
    num_workers: int | None = Field(default=None, ge=0)
    pin_memory: bool | None = Field(default=None)
    prefetch_factor: int | None = Field(default=None, ge=1)
    cache_in_ram: bool = Field(default=True)
    use_torch_compile: bool = Field(default=False)
    torch_compile_backend: str = Field(default="inductor")
    # LightningModel params
    loss_name: LossFunctionKey = Field(
        default=LossFunctionKey.BCE_DICE,
        description="Loss function name. Supported: 'bce', 'dice', 'bce_dice'."
    )
    optimizer_name: str = Field(default="adamw")
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    optimal_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Fixed sigmoid threshold to use instead of sweeping. "
                    "When None (default), the threshold is found via grid search each validation epoch."
    )
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=True)
    # L.Trainer params
    max_epochs: int = Field(default=1, ge=1)
    accelerator: str = Field(default="auto")
    devices: str | int = Field(default="auto")
    strategy: str = Field(default="auto")
    precision: PrecisionType | None = Field(default=None)
    log_every_n_steps: int = Field(default=1, ge=1)
    check_val_every_n_epoch: int = Field(default=1, ge=1)
    num_sanity_val_steps: int = Field(default=0, ge=0)

    # --- Nested configs  for callbacks and loggers ---
    early_stopping_config: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint_config: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)
    litlogger_config: LitLoggerConfig = Field(default_factory=LitLoggerConfig)

    # --- Other configs ---
    use_lr_scheduler: bool = Field(default=True)
    scheduler_type: Literal["reduce_on_plateau", "cosine_annealing"] = Field(default="reduce_on_plateau")
    lr_scheduler_config: ReduceOnPlateauConfig = Field(default_factory=ReduceOnPlateauConfig)
    cosine_annealing_config: CosineAnnealingConfig = Field(default_factory=CosineAnnealingConfig)

    #  --- Other callbacks params ---
    system_metrics_interval_sec: float = Field(default=5.0, gt=0)

    # --- Logger toggles ---
    use_mlflow_logger: bool = Field(default=False)
    use_checkpoint: bool = Field(default=False)
    use_early_stopping: bool = Field(default=False)
    use_litlogger_logger: bool = Field(default=False)

    def _resolve_accelerator(self) -> str | None:
        """
        Resolve `accelerator` to a concrete device name (gpu/cuda/mps/cpu).
        Returns None if 'auto' cannot be resolved (torch not importable).
        """
        accelerator = self.accelerator.lower()
        if accelerator != "auto":
            return accelerator
        try:
            import torch
        except ImportError:
            return None
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @model_validator(mode="after")
    def set_precision_from_accelerator(self) -> "TrainConfig":
        """
        Set precision format based on the currently used device type
        """
        if self.precision is not None:
            return self
        accelerator = self._resolve_accelerator()
        if accelerator is None:
            return self
        precision_map: dict[str, PrecisionType] = {
            "gpu": "16-mixed",
            "cuda": "16-mixed",
            "mps": "16-mixed",
            "cpu": "32-true",
        }
        resolved = precision_map.get(accelerator)
        if resolved:
            object.__setattr__(self, "precision", resolved)
            logger.info("precision auto-set to '%s' for accelerator='%s'", resolved, self.accelerator)
        return self

    @model_validator(mode="after")
    def set_num_workers_auto(self) -> "TrainConfig":
        """
        Auto-detect num_workers from os.cpu_count(), DDP-aware: when `devices`
        is an int > 1 (one process per device), divide the CPU budget among them
        to avoid oversubscription.
        """
        if self.num_workers is not None:
            return self
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if isinstance(self.devices, int) and self.devices > 1:
            num_workers = max(1, cpu_count // self.devices)
            logger.info(
                "num_workers auto-set to %d (os.cpu_count()=%d // devices=%d)",
                num_workers, cpu_count, self.devices,
            )
        else:
            num_workers = cpu_count
            logger.info("num_workers auto-set to %d (os.cpu_count())", num_workers)
        object.__setattr__(self, "num_workers", num_workers)
        return self

    @model_validator(mode="after")
    def set_pin_memory_from_accelerator(self) -> "TrainConfig":
        """
        Auto-set pin_memory: True iff effective accelerator is CUDA/GPU.
        pin_memory is a no-op (or unsupported) for MPS/CPU.
        """
        if self.pin_memory is not None:
            return self
        accelerator = self._resolve_accelerator()
        pin_memory = accelerator in ("gpu", "cuda")
        object.__setattr__(self, "pin_memory", pin_memory)
        logger.info("pin_memory auto-set to %s for accelerator='%s'", pin_memory, self.accelerator)
        return self

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

    @model_validator(mode="after")
    def validate_prefetch_factor(self) -> "TrainConfig":
        if self.num_workers == 0 and self.prefetch_factor is not None:
            logger.warning("prefetch_factor is ignored when num_workers=0; setting to None.")
            object.__setattr__(self, "prefetch_factor", None)
        return self
