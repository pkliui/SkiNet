from pydantic import BaseModel, ConfigDict, Field


class BaseExperimentConfig(BaseModel):
    """
    Base configuration for an experiment, containing common fields such as experiment name, description, and model type.

    """
    model_config = ConfigDict(extra="forbid")  # Forbid extra fields not defined in the model or its subclasses
    experiment_type: str  # Subclasses specify the experiment type, e.g. segmentation, classification, etc.
    experiment_name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Description of the experiment")
