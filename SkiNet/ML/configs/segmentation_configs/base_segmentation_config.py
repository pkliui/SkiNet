from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseSegmentationConfig(BaseModel):
    """
    Base configuration for ML segmentation experiments.
    """
    model_config = ConfigDict(extra="forbid")
    experiment_name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Description of the experiment")
    model_type: Literal["segmentation"] = "segmentation"
