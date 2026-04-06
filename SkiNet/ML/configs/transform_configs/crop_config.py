from pydantic import BaseModel, Field
from typing import Literal


class CropConfig(BaseModel):
    """
    Default configuration for cropping in segmentation.
    """
    crop_apply: bool = Field(default=True, description="Apply cropping.")
    crop_type: Literal["center_crop", "random_crop", "random_resized_crop"] = "random_resized_crop"
    size: tuple[int, int] = Field(default=(512, 512), description="Crop size (height, width).")
    scale: tuple[float, float] = Field(default=(0.8, 1.0), description="Scale range for random resized crop. Required if crop_type is 'random_resized_crop'.")
