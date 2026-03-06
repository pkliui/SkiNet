from pydantic import BaseModel, ConfigDict


class BaseTrainConfig(BaseModel):
    """
    Base configuration for training.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)
