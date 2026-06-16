from pydantic import BaseModel, ConfigDict, Field
from SkiNet.Utils.experiment_keys import HyperparamKey, MetricsKey


class SweepConfig(BaseModel):
    """
    Configuration for the optuna HPO sweep.

    ``monitor`` and ``direction`` are the **single source of truth** for the
    optimisation objective.  ``ExperimentConfig`` propagates ``monitor`` into
    ``trainconfig.early_stopping_config``, ``trainconfig.checkpoint_config``,
    and ``trainconfig.lr_scheduler_config`` so all four fields are always in
    sync without manual YAML duplication.
    """
    model_config = ConfigDict(extra="forbid")

    monitor: MetricsKey = Field(
        default=MetricsKey.default_monitor(),
        description="Metric to optimise; the single source of truth, propagated into the "
                    "early-stopping, checkpoint, and LR-scheduler configs.",
    )
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$",
                           description="Optuna optimisation direction: 'maximize' or 'minimize'.")
    experiment_name: str = Field(default="optuna_sweep", description="MLflow experiment name for the sweep.")
    # Each field is a list of GridSampler candidates for one dimension. Defaults are a single value
    # per field, kept consistent with the SWEEP_CONFIG block in main_config.yaml, so SweepConfig()
    # yields a 1-combo (no-op) grid; widen a field in the YAML to sweep that dimension. In practice
    # tune one dimension at a time.
    lr: list[float] = Field(default_factory=lambda: [3e-4],
                            description="Learning-rate candidates (GridSampler dimension).")
    weight_decay: list[float] = Field(default_factory=lambda: [0.0],
                                      description="Weight-decay candidates (GridSampler dimension).")
    batch_size: list[int] = Field(default_factory=lambda: [8],
                                  description="Batch-size candidates; also rescales the LR via scale_lr.")
    num_workers: list[int] = Field(default_factory=lambda: [2],
                                   description="DataLoader worker-count candidates (GridSampler dimension).")
    prefetch_factor: list[int] = Field(default_factory=lambda: [4],
                                       description="Batches pre-loaded per worker (GridSampler dimension).")
    scheduler_type: list[str] = Field(
        default_factory=lambda: ["none"],
        description=(
            "LR scheduler variants to sweep. Use 'none' to disable the scheduler "
            "for a given trial; 'cosine_annealing' and 'reduce_on_plateau' map "
            "directly to TrainConfig.scheduler_type."
        ),
    )

    @property
    def search_space(self) -> dict[str, list]:
        return {
            HyperparamKey.LR: self.lr.copy(),
            HyperparamKey.WEIGHT_DECAY: self.weight_decay.copy(),
            HyperparamKey.BATCH_SIZE: self.batch_size.copy(),
            HyperparamKey.NUM_WORKERS: self.num_workers.copy(),
            HyperparamKey.PREFETCH_FACTOR: self.prefetch_factor.copy(),
            HyperparamKey.SCHEDULER_TYPE: self.scheduler_type.copy(),
        }
