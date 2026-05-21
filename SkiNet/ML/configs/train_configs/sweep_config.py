from pydantic import BaseModel, ConfigDict, Field
from SkiNet.Utils.experiment_keys import HyperparamKey


class SweepConfig(BaseModel):
    """
    Configuration for the optuna HPO sweep.
    """
    model_config = ConfigDict(extra="forbid")

    monitor: str = Field(default="val_best_dice_at_threshold")
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$")
    experiment_name: str = Field(default="optuna_sweep")
    lr: list[float] = Field(default_factory=lambda: [3e-4, 1e-4])
    weight_decay: list[float] = Field(default_factory=lambda: [1e-4, 1e-3])
    batch_size: list[int] = Field(default_factory=lambda: [16, 32])
    num_workers: list[int] = Field(default_factory=lambda: [4, 8])
    prefetch_factor: list[int] = Field(default_factory=lambda: [2, 4])
    scheduler_type: list[str] = Field(
        default_factory=lambda: ["none", "cosine_annealing", "reduce_on_plateau"],
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
