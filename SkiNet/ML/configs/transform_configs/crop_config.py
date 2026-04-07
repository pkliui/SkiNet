from pydantic import BaseModel, Field, model_validator
from typing import Literal


class CropConfig(BaseModel):
    """
    Default configuration for cropping in segmentation.
    """
    crop_apply: bool = Field(default=True, description="Apply cropping.")
    crop_type: Literal["center_crop", "random_crop", "random_resized_crop"] = Field(default="random_resized_crop",
                                                                                    description="Type of cropping to apply. Required if crop_apply is True.")
    size: tuple[int, int] = Field(default=(512, 512), description="Crop size (height, width).")
    scale: tuple[float, float] = Field(
        default=(0.8, 1.0), description="Scale range for random resized crop. Required if crop_type is 'random_resized_crop'.")

    @model_validator(mode="after")
    def _validate_crop_params(self) -> "CropConfig":
        h, w = self.size
        if h <= 0 or w <= 0:
            raise ValueError("size must be (height, width) with height>0 and width>0")

        s0, s1 = self.scale
        if s0 > s1:
            raise ValueError("scale must be (min, max) with min <= max")
        if s0 <= 0.0 or s1 <= 0.0:
            raise ValueError("scale values must be > 0.0")
        if s0 > 1.0 or s1 > 1.0:
            raise ValueError("scale values must be <= 1.0")

        return self
