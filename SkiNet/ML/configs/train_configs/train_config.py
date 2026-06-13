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
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor(),
                                description="Metric to watch; propagated from SWEEP_CONFIG — do not set in YAML.")
    mode: Literal["min", "max"] = Field(default="max", description="'max' or 'min' for the monitored metric.")
    patience: int = Field(default=5, ge=0, description="Epochs without improvement before reducing the LR.")
    factor: float = Field(default=0.5, gt=0, lt=1, description="Multiplicative LR reduction factor.")


class CosineAnnealingConfig(BaseModel):
    """
    CosineAnnealingLR scheduler configuration.
    T_max is set to max_epochs at runtime when None.
    """
    T_max: int | None = Field(default=None, ge=1,
                              description="Cosine period in epochs; resolved to max_epochs at runtime when None.")
    eta_min: float = Field(default=1e-6, ge=0, description="Minimum LR at the end of each cosine cycle.")


class CheckpointConfig(BaseModel):
    """
    Configuration for model checkpointing.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    # ok For segmentation, sometimes Dice/IoU is a better “best model” criterion than validation loss, depending on what you care about.
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor(),
                                description="Metric to watch; propagated from SWEEP_CONFIG — do not set in YAML.")
    mode: Literal["min", "max"] = Field(default="max", description="'max' or 'min' for the monitored metric.")
    save_top_k: int = Field(default=1, ge=0, description="Number of best checkpoints to keep.")
    save_last: bool = Field(default=True, description="Always save the last checkpoint.")
    filename: str = Field(default="epoch{epoch:03d}", description="Checkpoint filename template.")


class EarlyStoppingConfig(BaseModel):  # passed all, ok
    """
    Configuration for early stopping.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    monitor: MetricsKey = Field(default=MetricsKey.default_monitor(),
                                description="Metric to watch; propagated from SWEEP_CONFIG — do not set in YAML.")
    mode: Literal["min", "max"] = Field(default="max", description="'max' or 'min' for the monitored metric.")
    min_delta: float = Field(default=0.0, description="Minimum change to count as an improvement.")
    patience: int = Field(default=5, ge=0, description="Epochs without improvement before stopping.")
    strict: bool = Field(default=True, description="Raise an error if the monitored metric is missing.")
    check_finite: bool = Field(default=True, description="Stop if the metric becomes NaN or Inf.")
    stopping_threshold: float | None = Field(
        default=None, description="Stop immediately once the metric passes this value.")
    divergence_threshold: float | None = Field(
        default=None, description="Stop immediately if the metric falls below this value.")

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
    fallback_to_local_mlflow: bool = Field(
        default=False, description="Fall back to local MLflow when the remote server is unreachable.")
    tracking_uri: str | None = Field(
        default=None, description="MLflow tracking server URI, e.g. 'http://127.0.0.1:5000'.")
    log_model: bool | Literal["all"] = Field(default="all", description="Log model artifacts; 'all' logs every checkpoint.")
    log_model_summary: bool = Field(default=True, description="Log the model summary as an artifact.")


class LitLoggerConfig(BaseModel):
    """
    Configuration for LitLogger logging.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
    teamspace: str | None = Field(default=None, description="Lightning Studio teamspace name.")
    log_model: bool = Field(default=False, description="Log the model to LitLogger.")
    save_logs: bool = Field(default=False, description="Persist logs to disk.")
    checkpoint_name: str | None = Field(default=None, description="Checkpoint name for LitLogger.")

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
    log_dir: str = Field(default="experiment_logs", description="Local directory for logs and checkpoints.")
    experiment_name: str = Field(
        default="unet2d_experiment",
        description="MLflow experiment name and run-name prefix. Drives run naming "
                    "(run = '{experiment_name}_seed{seed}_{YYYYMMDD-HHMMSS}'); checkpoints "
                    "are written under '{log_dir}/checkpoints/{run_name}'.",
    )
    run_test_after_fit: bool = Field(default=False, description="Run test-set evaluation after training completes.")
    test_on_val_split: bool = Field(
        default=False,
        description="Use the validation split as the test set. Metrics may be optimistic; "
                    "leave False unless you have no held-out test set.",
    )

    # --- Fit and optimizer params
    # DataLoader params
    batch_size: int = Field(default=8, ge=1, description="Samples per batch.")
    num_workers: int | None = Field(
        default=None, ge=0,
        description="DataLoader worker processes. When None, auto-set to os.cpu_count() "
                    "(single GPU) or cpu_count // devices under DDP.",
    )
    pin_memory: bool | None = Field(
        default=None,
        description="Pin host memory for faster H2D copies. When None, auto-set True on "
                    "CUDA/GPU and False on MPS/CPU.",
    )
    prefetch_factor: int | None = Field(
        default=None, ge=1,
        description="Batches pre-loaded per worker. Ignored (forced None) when num_workers=0.",
    )
    cache_in_ram: bool = Field(default=True, description="Pre-load all images into RAM at startup.")
    use_torch_compile: bool = Field(default=False, description="Wrap the model with torch.compile.")
    torch_compile_backend: str = Field(
        default="inductor",
        description="torch.compile backend. Use 'eager' to avoid an nvcc dependency.",
    )
    # LightningModel params
    loss_name: LossFunctionKey = Field(
        default=LossFunctionKey.BCE_DICE,
        description="Loss function name. Supported: 'bce', 'dice', 'bce_dice'."
    )
    optimizer_name: str = Field(default="adamw", description="Optimizer name: 'adam' or 'adamw'.")
    lr: float = Field(default=1e-4, gt=0, description="Base learning rate.")
    weight_decay: float = Field(default=1e-4, ge=0, description="L2 regularisation (weight decay).")
    optimal_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Fixed sigmoid threshold to use instead of sweeping. "
                    "When None (default), the threshold is found via grid search each validation epoch."
    )
    seed: int = Field(default=42, ge=0, description="Global RNG seed passed to L.seed_everything.")
    deterministic: bool = Field(default=True, description="Enable cuDNN deterministic mode.")
    # L.Trainer params
    max_epochs: int = Field(default=1, ge=1, description="Maximum number of training epochs.")
    accelerator: str = Field(
        default="auto",
        description="Lightning accelerator: 'auto', 'gpu', 'mps', or 'cpu'.",
    )
    devices: str | int = Field(default="auto", description="Number of devices, or 'auto'.")
    strategy: str = Field(default="auto", description="Lightning strategy: 'auto', 'ddp', etc.")
    precision: PrecisionType | None = Field(
        default=None,
        description="Training precision. When None, auto-set to '16-mixed' on GPU/MPS and "
                    "'32-true' on CPU.",
    )
    log_every_n_steps: int = Field(default=1, ge=1, description="Logging frequency in steps.")
    check_val_every_n_epoch: int = Field(default=1, ge=1, description="Validation frequency in epochs.")
    num_sanity_val_steps: int = Field(default=0, ge=0, description="Sanity validation steps before training.")

    # --- Nested configs  for callbacks and loggers ---
    early_stopping_config: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
        description="Early-stopping callback config. Its 'monitor' is propagated from SWEEP_CONFIG.",
    )
    checkpoint_config: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Model-checkpoint callback config. Its 'monitor' is propagated from SWEEP_CONFIG.",
    )
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig, description="MLflow logger config.")
    litlogger_config: LitLoggerConfig = Field(
        default_factory=LitLoggerConfig, description="Lightning Studio LitLogger config.")

    # --- Other configs ---
    use_lr_scheduler: bool = Field(
        default=True, description="Master toggle; when False no LR scheduler is attached.")
    scheduler_type: Literal["reduce_on_plateau", "cosine_annealing"] = Field(
        default="reduce_on_plateau",
        description="Which scheduler sub-config is used: 'reduce_on_plateau' → lr_scheduler_config, "
                    "'cosine_annealing' → cosine_annealing_config.",
    )
    lr_scheduler_config: ReduceOnPlateauConfig = Field(
        default_factory=ReduceOnPlateauConfig,
        description="ReduceLROnPlateau config; used when scheduler_type='reduce_on_plateau'. "
                    "Its 'monitor' is propagated from SWEEP_CONFIG.",
    )
    cosine_annealing_config: CosineAnnealingConfig = Field(
        default_factory=CosineAnnealingConfig,
        description="CosineAnnealingLR config; used when scheduler_type='cosine_annealing'.",
    )

    #  --- Other callbacks params ---
    system_metrics_interval_sec: float = Field(
        default=5.0, gt=0, description="System-metrics logging interval in seconds.")

    # --- Logger toggles ---
    use_mlflow_logger: bool = Field(
        default=False,
        description="Enable MLflow logging. Requires mlflow_config.tracking_uri to be set.",
    )
    use_checkpoint: bool = Field(default=False, description="Enable model checkpointing.")
    use_early_stopping: bool = Field(default=False, description="Enable early stopping.")
    use_litlogger_logger: bool = Field(default=False, description="Enable the Lightning Studio LitLogger.")

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
