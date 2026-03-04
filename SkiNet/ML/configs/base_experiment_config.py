from pydantic import BaseModel, ConfigDict, Field


class BaseExperimentConfig(BaseModel):
    """
    Base configuration for an experiment, containing common fields such as experiment name, description, and model type.
    """
    model_config = ConfigDict(extra="forbid")
    experiment_name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Description of the experiment")
    model_type: str  # Subclasses specify the model type
