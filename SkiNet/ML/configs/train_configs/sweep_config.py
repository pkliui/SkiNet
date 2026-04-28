from pydantic import BaseModel, ConfigDict, Field


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

    @property
    def search_space(self) -> dict[str, list]:
        return {
            "lr": self.lr.copy(),
            "weight_decay": self.weight_decay.copy(),
            "batch_size": self.batch_size.copy(),
        }
